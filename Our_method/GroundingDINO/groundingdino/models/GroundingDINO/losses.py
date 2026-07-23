import torch
import torch.nn as nn
import torch.nn.functional as F
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from util import box_ops
from .utils import sigmoid_focal_loss
from scipy.optimize import linear_sum_assignment
import os
import copy
from torchvision.ops import roi_pool
from .matcher import search_query_pos
import math
import numpy as np
import random
import bisect

from torchvision.ops.boxes import batched_nms, box_iou
from .relation_calibration import adaptive_relation_calibration


def estimate_image_size(boxes_1, boxes_2):

    # Concatenate these two tensors.
    boxes = torch.cat((boxes_1[0], boxes_2[0]), dim=0).squeeze(0)

    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    # Find the leftmost/rightmost x coordinates and top/bottom y coordinates across all boxes.
    min_x = torch.min(x1)
    max_x = torch.max(x2)
    min_y = torch.min(y1)
    max_y = torch.max(y2)

    width = max_x - min_x
    height = max_y - min_y

    return width, height

def compute_spatial_encodings(boxes_1, boxes_2):
    """
    Parameters:
    -----------
    boxes_1: List[Tensor]
        First set of bounding boxes (M, 4)
    boxes_1: List[Tensor]
        Second set of bounding boxes (M, 4)
    shapes: List[Tuple[int, int]]
        Image shapes, heights followed by widths
    eps: float
        A small constant used for numerical stability

    Returns:
    --------
    Tensor
        Computed spatial encodings between the boxes (N, 36)
    """
    features = []
    w,h = estimate_image_size(boxes_1, boxes_2)
    # (center_x, center_y, w, h)
    for b1, b2 in zip(boxes_1, boxes_2):
         #box_ops.box_cxcywh_to_xyxy(src_boxes),
        # box_ops.box_cxcywh_to_xyxy(target_boxes)))

        c1_x = b1[:, 0]
        c1_y = b1[:, 1]
        c2_x = b2[:, 0]
        c2_y = b2[:, 1]

        b1_w = b1[:, 2]
        b1_h = b1[:, 3]
        b2_w = b2[:, 2]
        b2_h = b2[:, 3]

        d_x = torch.abs(c2_x - c1_x) / (b1_w + 1e-10)
        d_y = torch.abs(c2_y - c1_y) / (b1_h + 1e-10)

        iou = torch.diag(box_iou(b1, b2))

        # Construct spatial encoding
        f = torch.stack([
            # Relative position of box centre    ---CD (dif)
            c1_x / w, c1_y / h, c2_x / w, c2_y / h,  # 4
            # Relative box width and height RA
            b1_w / w, b1_h / h, b2_w / w, b2_h / h,  # 4
            # Relative box area    ---RA
            b1_w * b1_h / (h * w), b2_w * b2_h / (h * w),
            b2_w * b2_h / (b1_w * b1_h + 1e-10),  # 3
            # Box aspect ratio   ---AR
            b1_w / (b1_h + 1e-10), b2_w / (b2_h + 1e-10),  # 2
            # Intersection over union   ---IU
            iou,  # 1
            # Relative distance and direction of the object w.r.t. the person  --- RP(dif)
            (c2_x > c1_x).float() * d_x,  # 4
            (c2_x < c1_x).float() * d_x,
            (c2_y > c1_y).float() * d_y,
            (c2_y < c1_y).float() * d_y,
        ], 1)
        features.append(
            torch.cat([f, torch.log(f + 1e-10)], 1)
        )
    return torch.cat(features)


def padding_last(logits, max_len=2048):
    logits_pad = torch.full((*logits.shape[:-1], max_len),
                            float("-inf"), device=logits.device)
    logits_pad[..., : logits.shape[-1]] = logits

    return logits_pad


def shrink_sigmoid(x, scale=1.):
    return 1.0 / (1.0 + torch.exp(-scale * x))


def tensor_to_list(x):
    if isinstance(x, torch.Tensor):
        return x.tolist()
    return x


class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, weight_dict, focal_alpha, losses,
                 **kwargs):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_loss_for_edges = kwargs.get("focal_loss_for_edges", False)

        self.rln_proj = kwargs.get("rln_proj", None)
        self.rln_proj_teacher = kwargs.get("rln_proj_teacher", None)
        self.rln_classifier = kwargs.get("rln_classifier", None)
        self.rln_freq_bias = kwargs.get("rln_freq_bias", None)
        self.use_relation_adaptive_calibration = kwargs.get("use_relation_adaptive_calibration", False)
        self.relation_adaptive_delta = kwargs.get("relation_adaptive_delta", 1.0)
        self.relation_adaptive_eps = kwargs.get("relation_adaptive_eps", 1e-6)
        self.relation_visual_weight = kwargs.get("relation_visual_weight", 0.5)
        self.relation_spatial_weight = kwargs.get("relation_spatial_weight", 0.5)
        self.register_buffer(
            "predicate_counts",
            torch.as_tensor(kwargs.get("predicate_counts", []), dtype=torch.float32),
            persistent=False,
        )

        self.rln_pretraining = kwargs.get("rln_pretraining", False)
        self.tokenizer = kwargs.get("tokenizer", None)
        self.ind_to_predicates = kwargs.get("ind_to_predicates", None)
        self.global_iter = -1
        hidden_dim = kwargs.get("hidden_dim", 256)
        self.min_obj = -hidden_dim * math.log(0.9)
        self.obj_temp = kwargs.get("obj_temp", 1.3 / hidden_dim)
        self.obj_start_iter = kwargs.get("obj_start_iter", 1000)
        self.obj_threshold = kwargs.get("obj_threshold", 0.5)

        self.rel_proposals_threshold = kwargs.get("rel_proposals_threshold", 0.5)
        self.rel_proposals_threshold_enabled = kwargs.get("rel_proposals_threshold_enabled", False)

        self.rel_batch_per_image = kwargs.get("rel_batch_per_image", 64)
        self.unsupervised_distill = kwargs.get("unsupervised_distill", False)
        self.is_closed_set = (self.rln_classifier is not None) and (not self.focal_loss_for_edges)
        self.ablation_mode = kwargs.get("ablation_mode", -1)
        self.fix_rel_batch = kwargs.get("fix_rel_batch", False)

        self.spatial_head = kwargs.get("spatial_head", None)
        self.spatial_head_ovr = kwargs.get("spatial_head_ovr", None)

        self.occ_dict = {12: [31, 48, 21, 22, 20, 50, 7, 30, 40, 49, 8, 43, 9, 38, 29, 1, 16, 41, 46, 36, 19, 23, 6, 33, 11, 42, 35, 44, 45, 4, 13, 25, 10, 17, 5, 3, 27, 47, 12, 24, 14, 2, 28, 26, 39, 32, 34, 18, 37, 15], 10: [31, 48, 30, 21, 20, 50, 41, 22, 7, 40, 49, 44, 8, 16, 23, 33, 38, 46, 42, 29, 1, 5, 43, 6, 45, 9, 34, 35, 25, 19, 26, 28, 3, 13, 12, 47, 2, 11, 36, 14, 10, 4, 24, 39, 32, 17, 18, 27, 37, 15], 1: [20, 50, 31, 19, 30, 21, 38, 41, 48, 29, 1, 22, 8, 23, 40, 49, 44, 16, 10, 43, 7, 25, 9, 47, 6, 42, 11, 35, 46, 5, 28, 45, 33, 17, 36, 32, 24, 4, 26, 3, 14, 13, 2, 37, 12, 18, 34, 27, 39, 15], 0: [20, 31, 29, 19, 30, 50, 7, 41, 21, 48, 22, 40, 8, 23, 1, 38, 28, 43, 18, 32, 49, 45, 25, 9, 12, 16, 24, 35, 11, 6, 46, 13, 47, 3, 33, 42, 36, 4, 26, 37, 5, 44, 17, 10, 14, 2, 34, 39, 27, 15], 13: [20, 21, 48, 31, 22, 40, 30, 7, 5, 49, 29, 50, 44, 9, 16, 46, 8, 1, 45, 38, 6, 41, 14, 23, 35, 11, 24, 3, 43, 13, 12, 17, 33, 19, 2, 4, 37, 42, 10, 25, 47, 32, 36, 26, 34, 28, 18, 27, 15, 39], 5: [20, 31, 30, 22, 48, 50, 29, 8, 1, 46, 40, 43, 21, 6, 49, 10, 41, 33, 13, 18, 44, 23, 19, 9, 16, 32, 7, 11, 45, 5, 14, 38, 47, 35, 2, 24, 36, 28, 25, 4, 42, 26, 12, 3, 17, 37, 27, 34, 39, 15], 7: [21, 50, 31, 48, 22, 20, 43, 45, 30, 9, 40, 16, 6, 33, 29, 7, 46, 1, 38, 44, 41, 8, 49, 13, 47, 25, 28, 19, 11, 23, 17, 14, 4, 3, 35, 12, 10, 2, 34, 39, 27, 42, 5, 24, 26, 32, 36, 37, 18], 11: [50, 48, 21, 31, 29, 43, 30, 23, 20, 40, 22, 33, 18, 49, 41, 1, 7, 8, 38, 46, 19, 11, 45, 14, 6, 35, 9, 24, 25, 13, 28, 16, 32, 17, 26, 10, 44, 42, 47, 5, 27, 4, 3, 12, 39, 2, 36, 37, 34, 15], 8: [50, 30, 21, 31, 48, 22, 40, 29, 23, 20, 43, 45, 9, 8, 38, 16, 19, 35, 49, 18, 7, 41, 34, 13, 46, 11, 1, 28, 10, 6, 44, 36, 3, 47, 24, 14, 32, 33, 12, 25, 4, 42, 5, 17, 2, 39, 26, 27, 37], 3: [22, 31, 48, 30, 20, 50, 29, 41, 46, 21, 40, 8, 19, 35, 43, 1, 34, 32, 4, 13, 38, 23, 49, 7, 47, 6, 16, 9, 2, 33, 24, 17, 11, 27, 42, 10, 25, 45, 26, 28, 14, 44, 36, 12, 3, 5, 39, 37, 18, 15], 6: [22, 1, 21, 31, 50, 38, 20, 7, 48, 40, 23, 41, 30, 8, 9, 6, 4, 43, 28, 29, 46, 13, 16, 49, 25, 35, 33, 24, 34, 11, 17, 14, 19, 44, 32, 45, 37, 10, 42, 3, 2, 47, 27, 5, 12, 36, 26, 39, 18, 15], 2: [50, 31, 29, 48, 21, 41, 1, 30, 20, 46, 40, 22, 23, 8, 6, 10, 43, 49, 14, 9, 34, 7, 3, 32, 45, 38, 19, 33, 16, 24, 35, 28, 25, 11, 4, 17, 42, 12, 47, 13, 36, 44, 2, 27, 5, 26, 18, 37, 39, 15], 4: [31, 30, 48, 22, 41, 20, 29, 8, 1, 45, 40, 50, 21, 25, 4, 5, 7, 9, 38, 43, 6, 23, 49, 16, 35, 19, 11, 46, 33, 3, 24, 28, 44, 32, 27, 10, 47, 37, 14, 13, 42, 17, 2, 12, 26, 18, 34, 36, 39, 15], 20: [48, 31, 40, 29, 50, 30, 20, 7, 49, 22, 8, 45, 46, 41, 21, 11, 19, 43, 13, 44, 9, 1, 10, 4, 14, 33, 16, 38, 42, 27, 2, 23, 6, 25, 17, 28, 3, 24, 47, 12, 32, 5, 35, 37, 26, 36, 34, 18], 15: [48, 31, 50, 29, 41, 21, 25, 22, 20, 30, 49, 7, 23, 6, 9, 38, 11, 46, 40, 35, 8, 43, 47, 1, 19, 4, 24, 12, 33, 5, 10, 44, 16, 27, 14, 45, 3, 18, 13, 26, 28, 36, 37, 32, 17, 39, 34, 2, 42, 15], 9: [48, 21, 31, 22, 20, 30, 41, 29, 50, 49, 40, 6, 8, 7, 1, 43, 33, 3, 38, 9, 46, 16, 23, 35, 11, 13, 45, 28, 5, 19, 44, 24, 4, 47, 25, 17, 37, 14, 10, 42, 32, 26, 2, 34, 39, 12, 27, 36, 18, 15], 21: [48, 31, 22, 20, 30, 44, 8, 38, 50, 49, 23, 43, 7, 29, 21, 46, 1, 40, 28, 11, 25, 35, 41, 5, 6, 14, 45, 42, 19, 16, 12, 4, 13, 32, 9, 47, 36, 24, 10, 33, 17, 18, 37, 34, 26, 2, 27], 22: [50, 48, 31, 20, 22, 38, 21, 23, 49, 29, 8, 43, 19, 30, 46, 40, 11, 32, 44, 6, 35, 1, 47, 14, 7, 42, 9, 25, 5, 12, 45, 26, 2, 3, 16, 41, 36, 24, 4, 34, 13, 33, 28, 37, 10, 17, 27, 18], 19: [48, 31, 21, 22, 30, 49, 20, 50, 23, 40, 38, 33, 12, 8, 7, 29, 46, 9, 19, 6, 41, 11, 13, 43, 1, 4, 3, 14, 42, 10, 27, 44, 17, 28, 25, 5, 45, 35, 24, 36, 2, 32, 26, 47, 16, 18, 34, 37], 16: [48, 50, 31, 22, 21, 30, 49, 20, 40, 7, 38, 46, 29, 41, 23, 8, 16, 13, 9, 45, 1, 35, 28, 43, 11, 12, 19, 14, 47, 6, 5, 24, 44, 39, 37, 33, 4, 25, 36, 10, 3, 17, 32, 42, 18, 27, 2, 26, 15, 34], 18: [50, 48, 31, 9, 21, 49, 20, 29, 40, 33, 13, 22, 30, 41, 7, 23, 6, 46, 45, 38, 8, 11, 19, 1, 43, 35, 10, 4, 24, 17, 47, 14, 25, 2, 3, 42, 5, 27, 26, 28, 39, 16, 12, 44, 32, 36, 18, 37, 34], 14: [48, 50, 31, 46, 21, 30, 49, 20, 6, 22, 7, 14, 29, 38, 44, 8, 40, 1, 41, 45, 23, 9, 11, 43, 24, 4, 13, 2, 3, 35, 12, 19, 25, 33, 47, 16, 27, 5, 39, 10, 26, 42, 18, 17, 36, 37, 32, 28, 34, 15], 23: [48, 31, 29, 21, 50, 20, 22, 30, 6, 23, 9, 40, 5, 16, 8, 13, 11, 7, 28, 43, 38, 49, 1, 46, 41, 33, 12, 25, 10, 36, 47, 26, 2, 45, 35, 19, 3, 14, 18, 32, 4, 34, 24], 17: [1, 22, 48, 21, 50, 31, 7, 25, 30, 49, 20, 6, 40, 32, 44, 45, 8, 38, 41, 43, 10, 23, 29, 11, 46, 35, 28, 9, 14, 19, 3, 42, 24, 33, 5, 4, 17, 13, 16, 27, 47, 2, 12, 18, 34, 26, 36, 15, 37], 24: [21, 31, 48, 22, 20, 49, 38, 8, 23, 1, 36, 46, 30, 14, 5, 16, 50, 29, 43, 41, 13, 6, 9, 4, 44, 40, 11, 47, 12, 7, 35, 25, 19, 26, 32, 3, 42, 33, 18, 45, 24, 17, 28, 2, 10], 25: [21, 31, 50, 20, 48, 8, 22, 16, 29, 6, 30, 46, 7, 35, 49, 13, 40, 41, 1, 33, 43, 44, 25, 23, 11, 38, 4, 32, 10, 12, 42, 47, 2, 45, 14, 36, 9, 19, 26, 24, 3], 26: [48, 31, 20, 22, 30, 21, 49, 44, 50, 7, 1, 40, 38, 29, 6, 16, 32, 46, 33, 8, 19, 12, 25, 45, 41, 43, 10, 13, 23, 28, 11, 36, 35, 14, 4, 2, 42, 26, 9, 47], 28: [48, 31, 33, 22, 21, 20, 1, 40, 50, 30, 38, 6, 16, 43, 29, 49, 8, 13, 7, 46, 3, 14, 41, 25, 23, 47, 45, 4, 44, 24, 12, 35, 32, 27, 5, 19, 9, 26, 10, 28, 34, 2, 11, 37, 36], 29: [22, 31, 20, 8, 48, 11, 21, 50, 1, 46, 38, 23, 30, 41, 40, 49, 24, 6, 7, 43, 29, 3, 10, 2, 45, 12, 26, 4, 19, 25, 33, 18, 16, 14, 39, 44, 17, 36, 9], 30: [30, 46, 22, 48, 21, 31, 38, 23, 20, 29, 1, 33, 50, 41, 49, 43, 9, 40, 6, 8, 11, 26, 4, 13, 44, 47, 35, 7, 5, 28, 19, 14, 42, 45], 31: [30, 50, 31, 48, 21, 46, 1, 38, 22, 49, 40, 29, 8, 20, 35, 4, 6, 41, 23, 43, 34, 14, 47, 32, 25, 33, 9, 10, 11, 28, 44], 27: [31, 30, 50, 48, 11, 20, 40, 38, 21, 29, 6, 22, 16, 43, 49, 8, 7, 1, 47, 23, 25, 9, 46, 19, 41, 4, 44, 35, 45, 27, 33, 12, 34, 24, 5, 36, 28, 13], 35: [31, 20, 22, 50, 48, 30, 38, 1, 6, 21, 40, 43, 29, 41, 46, 8, 33, 13, 23, 27, 7, 49, 18, 44, 12], 32: [20, 31, 21, 48, 11, 18, 40, 50, 30, 22, 49, 38, 8, 6, 29, 41, 45, 1, 35, 23, 3, 43, 33, 7, 10, 46, 25, 44], 33: [20, 31, 21, 22, 48, 30, 40, 38, 1, 6, 50, 41, 11, 49, 29, 8, 43, 10, 23, 13, 46, 25, 44, 7], 36: [47, 22, 31, 20, 50, 48, 30, 8, 40, 6, 29, 21, 49, 1, 7, 43, 26, 23, 27, 2, 41, 33, 16, 5, 38, 44, 46], 34: [47, 48, 31, 50, 29, 40, 30, 41, 20, 11, 22, 6, 21, 1, 24, 43, 7, 49, 8, 38, 25, 33], 37: [31, 48, 40, 38, 30, 22, 20, 8, 43, 1, 47, 50, 29, 45, 49, 5, 21, 9, 25, 41, 7, 44, 46, 36], 38: [22, 48, 31, 49, 20, 1, 50, 21, 8, 29, 30, 40, 19, 25, 16, 23, 7, 42, 3, 28, 36, 33, 43, 34, 41, 32, 17, 10, 6, 38, 11, 44, 45, 9], 39: [48, 22, 45, 20, 31, 30, 8, 1, 50, 49, 29, 43, 7, 6, 21, 46], 40: [48, 22, 31, 23, 29, 1, 30, 20, 40, 50, 9, 7, 36, 6, 21, 49], 41: [31, 48, 30, 1, 49, 22, 29, 40, 20, 23, 21, 8, 50, 47, 9, 38, 14, 28, 43], 42: [31, 48, 1, 22, 21, 38, 41, 6, 43, 20, 19, 30, 32, 16, 7, 50, 29], 47: [31, 8, 30, 22, 19, 16, 29, 1, 33, 23, 50, 20, 42, 21, 7, 11, 48], 43: [31, 30, 1, 22, 8, 48, 41, 40, 20, 50, 9, 36, 43, 25, 29], 44: [48, 22, 30, 1, 8, 25, 31, 29, 50, 40, 20, 9, 13, 34], 46: [48, 50, 30, 22, 29, 31, 20, 21], 48: [8, 30, 16, 22, 20, 31, 29, 50, 17, 21, 43, 1, 6, 41, 23, 18, 7, 24, 33, 34, 40], 49: [31, 30, 22, 48, 29, 50, 14, 20, 1, 43, 12, 21, 23, 8, 24, 5, 49], 52: [29, 31, 30, 22, 38, 7, 20, 8, 50, 23], 53: [29, 30, 31, 20, 50, 8, 22, 48, 25, 40, 11, 49, 21, 24, 9, 38, 14, 47, 23, 10], 45: [30, 22, 1, 29, 25, 49, 20, 31, 8, 3, 16, 23, 2, 21, 4, 18, 42, 50, 5, 10, 6], 54: [30, 48, 20, 31, 29, 50, 40, 1, 33, 8, 22, 43, 49, 41, 23, 27, 21], 50: [8, 22, 31, 20, 50, 23, 29, 1, 30, 33, 43, 24, 44, 5, 40, 49, 45, 46], 55: [31, 48, 50, 49, 13, 20, 21], 57: [48, 38, 20, 50, 6, 30, 31, 9, 1, 18, 22, 7, 36], 58: [29, 20, 25, 6, 30, 31, 22, 21, 1, 13, 50, 33, 9, 43], 51: [49, 31, 30, 22, 23, 50, 29, 40, 14, 25, 43, 21], 61: [48, 20, 30, 50, 31, 1, 18, 9, 7, 22, 8], 56: [31, 48, 11, 49, 46, 8, 50, 29, 38], 60: [38, 48, 49, 50, 20, 31, 19, 22, 1, 30, 21], 78: [49, 20, 48, 31, 22, 8, 46, 29, 50, 11, 25, 21, 6, 40, 30, 38, 41, 44, 9, 43, 2, 1, 47, 23, 3, 26, 24, 45, 5, 36, 37, 33, 14], 120: [49, 48, 31, 30], 115: [31, 20, 19, 50, 29, 21, 22, 1, 7, 28, 16, 23, 43, 8, 36, 33, 42, 34, 40, 17, 6, 27, 10, 5, 4, 3, 30, 39, 25, 24], 111: [20, 48, 31, 22, 49, 50, 6, 29, 5, 30, 32, 33, 25, 43], 114: [29, 35, 31, 46, 1, 40, 3, 33, 22, 41, 10, 23, 20, 18, 50, 13, 30, 4, 8, 6, 2, 42, 43], 124: [29, 35, 46, 31, 4, 20, 22, 1, 2, 45, 23, 11, 10, 50, 40, 30, 6, 41, 43, 33, 12, 49, 8, 5], 112: [20, 48, 30, 43, 31, 22, 29, 50, 5, 49, 1], 87: [49, 20, 48, 50, 22, 31, 30, 19, 33, 5], 136: [29, 31, 4, 20, 8, 23, 1, 22, 43, 50, 10, 33, 30, 7, 41, 17, 18, 36, 19, 48, 13, 12, 49, 6, 3, 21, 16], 145: [50, 20, 29, 22, 17, 31, 8, 23, 1, 30, 32, 43, 7, 33, 40, 9, 2, 4, 11, 36, 48, 13, 12, 3, 25, 42, 10, 19, 6, 34], 126: [31, 29, 1, 27, 50, 20, 40, 6, 43, 22, 23, 33, 5, 30, 8, 24, 21, 3, 36, 26, 10, 41, 7, 13, 37, 12, 48], 59: [20, 31, 50, 30, 22, 16, 7, 43, 17, 29, 28, 42, 21], 110: [31, 30, 43, 20, 29, 23, 1, 22, 40, 8, 33, 6, 50, 21, 26, 7], 97: [31, 50, 43, 29, 20, 1, 30, 22, 7, 26, 40, 8, 23, 12, 21, 28, 33, 24, 34, 13, 47, 5], 74: [43, 20, 31, 30, 50, 7, 9, 36, 48, 22, 21], 77: [20, 31, 30, 34, 50, 7], 70: [50, 11, 29, 20, 48, 30, 21, 49, 31, 40, 46, 22, 23, 44, 8], 149: [48, 40, 29, 50, 1, 11, 20, 31, 6, 22, 30, 23, 8, 46, 49, 21, 41, 2, 9, 5, 25, 45, 7, 38, 43, 10, 44, 33, 14, 47], 72: [1, 31, 20, 8, 44, 16, 40, 6, 29, 50, 30, 43, 25, 21], 88: [31, 50, 43, 40, 26, 1, 21, 6, 29, 7, 22, 33, 16, 19, 13, 30, 20], 106: [20, 22, 41, 40, 31, 6, 42, 29, 30, 8, 50], 76: [20, 22, 17, 31, 33, 1, 7, 28, 29, 30, 43, 32, 19, 21, 23, 50, 10, 8, 16, 42, 40, 6], 133: [43, 1, 31, 20, 30, 23, 29, 8, 50, 33, 22, 16, 6], 101: [20, 22, 29, 31, 1, 16, 50, 21, 8, 43, 30, 33, 10], 96: [31, 20, 1, 22, 29, 8, 16, 23, 21, 50, 30, 43, 40, 7, 12, 18, 4, 33, 14], 71: [31, 22, 1, 23, 8, 20, 29, 7, 33, 40, 21, 17, 43, 50, 5, 30, 28, 2], 90: [31, 6, 46, 2, 1, 23, 29, 40, 14, 48, 38, 22, 33, 10, 43, 24, 47, 41, 8, 21, 45, 5, 25, 13, 20, 16, 50, 32], 67: [20, 48, 49, 31, 22, 30, 50, 5, 29, 1, 21], 66: [20, 31, 48, 50, 49, 22, 30, 19, 5, 21, 43, 32, 8], 99: [31, 1, 17, 29, 7, 22, 8, 21, 50, 43, 20, 19, 16, 23, 42, 4, 28, 41, 10, 5, 27, 30, 33, 35], 144: [20, 30, 43, 31, 22, 29, 28, 50, 36, 7, 32], 123: [31, 20, 22, 29, 1, 40, 16, 30, 8, 50, 43], 62: [31, 48, 20, 29, 50, 21, 1, 49, 7, 22], 91: [20, 48, 31, 50, 38, 22, 30, 40, 2, 46, 45, 29, 23, 11, 21, 41, 49, 8, 1, 9, 43, 6, 13, 10, 42, 25, 44, 33, 24, 47, 14], 135: [20, 40, 31, 22, 41, 43, 1, 30, 29, 16, 8, 7, 23, 33, 42, 9, 34, 32], 130: [20, 31, 30, 22, 7, 29, 28, 36, 27, 42, 50, 17], 140: [20, 31, 30, 40, 22, 23, 21, 1, 43, 50, 41, 29], 93: [22, 1, 40, 31, 8, 20, 24, 29, 21, 43, 26, 50, 5], 113: [48, 22, 31, 49, 20, 1, 50], 65: [1, 31, 29, 20, 30, 50, 23, 22, 8, 7, 3, 42, 13, 9, 2, 33, 12, 43], 108: [1, 29, 31, 30, 22, 21, 50, 20, 32, 40, 8], 116: [29, 31, 43, 1, 22, 33, 30, 50, 20, 8, 7], 73: [31, 20, 30, 29, 50, 33, 22, 7, 17, 23, 18, 36, 19, 1, 43, 48, 8, 41], 142: [29, 31, 35, 22, 23, 30, 20, 40, 12, 43, 50, 8, 16, 42, 4, 1], 139: [50, 11, 31, 8, 1, 23, 29, 43, 21, 33, 7, 20, 28, 13, 44, 16, 22, 30, 19, 48], 80: [38, 31, 29, 40, 35, 1, 20, 8, 23, 30, 3], 92: [31, 8, 1, 6, 29, 30, 17, 40, 22, 50, 21, 20], 146: [20, 30, 31, 29, 7, 9], 75: [50, 31, 20, 36, 1, 27, 32, 7, 30, 39, 16, 29, 34], 137: [31, 8, 35, 29, 30, 20, 23, 22, 6, 50, 32, 43, 1, 7, 34, 40], 128: [48, 20, 31, 29, 22, 49, 21, 50], 107: [31, 8, 20, 1, 47, 32, 50, 22, 40, 29, 23, 25, 30], 104: [40, 20, 23, 29, 1, 22, 31, 41, 30, 8, 46, 13, 33, 10], 86: [31, 30, 22, 1, 29, 50, 40, 49, 48, 21], 105: [20, 31, 30, 50, 1, 19, 7, 8, 33, 43, 13, 36, 29, 12, 22], 103: [29, 23, 20, 7, 31, 8, 50, 19, 30, 1, 18, 40, 22, 41, 27, 21], 141: [22, 21, 31, 50, 30], 150: [30, 23, 8, 43, 31, 20, 50, 29, 9], 84: [30, 20, 31, 9, 36, 1, 50, 43, 22], 138: [31, 20, 30, 21, 29, 50, 8, 23, 1, 22, 48, 16, 43, 9], 83: [20, 30, 7, 29, 31], 63: [31, 8, 43, 1, 29, 22, 13, 12, 41, 18, 23, 50], 82: [20, 30, 22, 31, 50, 1, 43], 100: [31, 20, 8, 19, 7, 23, 1, 22, 29, 10, 28, 41, 17, 50, 16, 43, 21], 127: [30, 20, 31, 32, 7, 22, 50, 9, 21], 121: [31, 22, 20, 1, 12, 29, 41, 26, 13, 50, 43, 33, 5, 40, 24, 30, 25], 148: [5, 8, 16, 31, 29, 2, 1, 17, 21, 43, 19, 7, 50, 20, 33, 22], 131: [40, 20, 31, 1, 33, 8, 22, 44, 29, 50], 95: [19, 7, 31, 30, 29, 20, 32, 22, 47, 1, 43, 8, 50, 25], 132: [1, 19, 31, 29, 22, 50, 24, 32, 13], 129: [29, 41, 46, 31, 22, 20, 8], 64: [1, 29, 30, 31, 20, 38, 40, 21, 22, 43, 5, 8, 50, 23], 147: [7, 31, 20, 43, 19, 30, 29], 81: [29, 1, 20, 31, 8, 23, 50, 13, 33, 46, 43, 22, 12, 30, 41, 25], 118: [11, 21, 49, 31, 50, 22, 20, 43, 44, 48, 41, 24, 16], 134: [22, 20, 31, 8, 29, 43, 6, 30, 1, 16, 40, 17], 85: [31, 50, 20, 7, 42, 23, 34, 32, 30], 125: [11, 50, 29, 31, 43, 22, 21, 41, 38, 24], 102: [50, 22, 21, 20, 31, 30, 7, 11], 122: [31, 48, 5, 30, 20, 1, 50], 79: [41, 1, 40, 48, 31, 5, 29, 8, 38, 50, 20, 49, 22, 45, 47], 89: [30, 31, 20], 94: [31, 22, 29, 14, 43, 1, 21], 68: [50, 29, 22, 48, 20, 16, 31, 38], 109: [29, 20, 31, 50, 8], 143: [31, 23, 22, 50, 8, 33, 38, 20], 98: [50, 31, 22, 21, 30, 37, 47, 48, 20, 8], 69: [20, 47, 1, 31, 22, 33, 21, 50, 29], 117: [31, 8, 38, 50, 22, 21, 40, 43, 20, 23], 119: [49, 30, 20, 31, 40, 8, 48, 22, 41]}
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_logits = outputs['pred_logits']  # (bsz, num_queries, 256)
        src_mask = src_logits == float('-inf')
        # 
        tgt_pos_seg = []
        for bid, (target, (_, J)) in enumerate(zip(targets, indices)):
            gt_names = target['gt_names']
            all_ids = target['input_ids']
            pos_seg = []
            for name in gt_names:
                ids = self.tokenizer(name + '.').input_ids[1:-1]
                start_i, end_i = search_query_pos(all_ids.tolist(), ids)
                assert start_i != end_i, "cannot find query:{} from input_ids:{}".format(ids,
                                                                                         self.tokenizer.decode(all_ids))
                pos_seg.append((start_i, end_i))

            for j in J.tolist():
                tgt_pos_seg.append(pos_seg[j])

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)

        # set positive labels
        for bid, qid, seg in zip(idx[0], idx[1], tgt_pos_seg):
            target_classes_onehot[bid, qid, seg[0]:seg[1]].fill_(1.0)

        target_classes_onehot = target_classes_onehot[:, :, :-1]

        alpha = self.focal_alpha
        gamma = 2.0

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes,
                                     alpha=alpha, gamma=gamma, reduction='none') * src_logits.shape[1]
        loss_ce.masked_fill_(src_mask, 0.)

        nb_pos = target_classes_onehot.sum(-1)
        nb_pos[nb_pos == 0] = 1.0

        loss_ce = (loss_ce / nb_pos.unsqueeze(2)).mean(1).sum() / num_boxes
        losses = {'loss_ce': loss_ce}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))

        losses['loss_giou'] = loss_giou.sum() / num_boxes

        ## calculate the x,y and h,w loss
        # with torch.no_grad():
        #    losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
        #    losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'edges': self.loss_edges,
            'obj_likelihood': self.loss_obj_likelihood,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def loss_obj_likelihood(self, outputs, targets, indices, num_boxes):
        assert "pred_obj" in outputs, "pred_obj does not exist in outputs, outputs.keys:{}".format(outputs.keys())

        idx = self._get_src_permutation_idx(indices)
        pred_obj = outputs["pred_obj"][idx]
        return {'loss_obj_ll': torch.clamp(pred_obj, min=self.min_obj).sum() / num_boxes}

    @torch.no_grad()
    def get_objectness(self, pred_obj):
        return torch.exp(-self.obj_temp * pred_obj)

    def loss_edges(self, outputs, targets, indices, num_boxes,
                   object_token, relation_token, rel_text_dict,
                   rel_text_dict_t=None, outputs_t=None, indices_t=None):
        """
          compute loss for relations 
        """
        self.is_closed_set = (self.rln_classifier is not None) and (not self.focal_loss_for_edges)
        device = outputs['pred_logits'].device
        bs, num_queries = outputs['pred_logits'].shape[:2]
        relation_wo_labels = 'edges' not in targets[0]
        relation_feature_t = None
        relation_is_str = False
        sample_cross_frame = False
        rel_tgt = []

        # Check whether targets contain edge information.
        for target in targets:
            if 'edges' not in target:
                break
            if len(target['edges']) > 0 and isinstance(target['edges'][0][2], str) and (not relation_is_str):
                relation_is_str = True
                sample_cross_frame = relation_is_str
                break

        # If edge labels exist.
        if not relation_wo_labels:  #
            all_edge_lbl = []
            freq_dist = []
            batch_ids = []

            sid, oid = [], []
            s_labels, o_labels = [], []
            sid_t, oid_t = [], []
            # Iterate over annotations for each image.
            for bid, target in enumerate(targets):
                tgt_edges = target['edges']
                if len(tgt_edges) == 0:
                    continue
                matched = {}
                # Prediction indices matched to each ground-truth object index.
                for src, dst in zip(indices[bid][0].tolist(), indices[bid][1].tolist()):
                    matched[dst] = src
                # Number of positive samples and total samples (positive + negative).
                num_pos = num_total = 0
                n = len(matched)
                # Fully connected ground-truth/prediction adjacency matrix with zero diagonal.
                full_adj = torch.ones((n, n)) - torch.diag(torch.ones(n))
                # Set positions with matched relations in the adjacency matrix to 0.
                for edge in tensor_to_list(tgt_edges):
                    if edge[0] in matched and edge[1] in matched:
                        full_adj[edge[0], edge[1]] = 0

                # teacher's matched nodes (optional)
                if outputs_t is not None:
                    matched_t = {}
                    for src, dst in zip(indices_t[bid][0].tolist(), indices_t[bid][1].tolist()):
                        matched_t[dst] = src

                # This branch is not entered.
                if not self.is_closed_set and self.fix_rel_batch:
                    pos_num_per_img = int(self.rel_batch_per_image * 0.25)
                    if len(tgt_edges) > pos_num_per_img:
                        if isinstance(tgt_edges, list):
                            tgt_edges = random.sample(tgt_edges, pos_num_per_img)
                        else:
                            tgt_edges = tgt_edges[torch.randperm(len(tgt_edges))][:pos_num_per_img]

                # Access the number of relation edges in each image.
                # Process positive samples.
                for edge in tensor_to_list(tgt_edges):
                    # If both targets in an edge are correctly matched to detections.
                    # Positive sample.
                    if edge[0] in matched and edge[1] in matched:
                        # Batch image ID and predicted box index for the left node of the edge.
                        sid.append([bid, matched[edge[0]]])
                        # Batch image ID and predicted box index for the right node of the edge.
                        oid.append([bid, matched[edge[1]]])
                        if relation_is_str:
                            rel_tgt.append(edge[2])
                        else:
                            # Edge category.
                            all_edge_lbl.append(edge[2])
                        # Increment positive sample count by 1.
                        num_pos += 1

                        if self.rln_freq_bias is not None:
                            # Category information for the left and right edge nodes.
                            s_labels.append(target['labels'][edge[0]])
                            o_labels.append(target['labels'][edge[1]])

                        if outputs_t is not None:
                            sid_t.append([bid, matched_t[edge[0]]])
                            oid_t.append([bid, matched_t[edge[1]]])

                # Indices of object-node pairs with no paired relation [M, 2].
                # Process negative samples.
                neg_edges = torch.nonzero(full_adj)
                if self.is_closed_set or (relation_is_str and not self.fix_rel_batch):
                    # Use three times as many negative samples as positive samples.
                    num_neg_per_img = max(1, num_pos * 3)
                else:
                    # Each image has at most 64 samples; negatives are the remainder after positives.
                    num_neg_per_img = max(1, self.rel_batch_per_image - num_pos)

                # Sample negative edges.
                if len(neg_edges) >= num_neg_per_img:
                    if outputs_t is not None and not self.unsupervised_distill:
                        # sample
                        hs_obj_t = outputs_t['hs_obj'][bid]
                        nsid, noid = [], []
                        for edge in neg_edges.tolist():
                            nsid.append(matched_t[edge[0]])
                            noid.append(matched_t[edge[1]])
                        nsid = torch.as_tensor(nsid)
                        noid = torch.as_tensor(noid)

                        feat = torch.cat((hs_obj_t[nsid], hs_obj_t[noid],
                                          outputs_t['hs_rln'][bid].flatten(1).repeat(nsid.shape[0], 1)),
                                         1)
                        with torch.no_grad():
                            feat = self.rln_proj_teacher(feat)
                            encoded_text = rel_text_dict_t['encoded_text'][bid]
                            feat = (feat @ encoded_text.T).sigmoid()

                        feat_score = feat.max(-1)[0]

                        if self.rel_proposals_threshold_enabled:
                            keep1 = torch.where(feat_score > self.rel_proposals_threshold)[0]
                            keep2 = feat_score.topk(num_neg_per_img)[1]
                            keep = torch.cat((keep1, keep2)).unique()
                        else:
                            keep = feat_score.topk(num_neg_per_img)[1]

                        neg_edges = neg_edges[keep.to(neg_edges.device)]
                    else:
                        if sample_cross_frame:
                            idx_ = torch.randperm(neg_edges.shape[0])[: max(1, num_pos * 3)]
                            neg_edges = neg_edges[idx_]
                        else:
                            # Randomly select several negative edges.
                            idx_ = torch.randperm(neg_edges.shape[0])[:num_neg_per_img]
                            neg_edges = neg_edges[idx_]

                # Negative sample.
                for edge in neg_edges.tolist():
                    sid.append([bid, matched[edge[0]]])
                    oid.append([bid, matched[edge[1]]])

                    if self.rln_freq_bias is not None:
                        s_labels.append(target['labels'][edge[0]])
                        o_labels.append(target['labels'][edge[1]])

                    if relation_is_str:
                        rel_tgt.append('[UNK]')
                    else:
                        # Set all negative edge labels to 0.
                        all_edge_lbl.append(0)

                    num_total += 1
                    if outputs_t is not None:
                        sid_t.append([bid, matched_t[edge[0]]])
                        oid_t.append([bid, matched_t[edge[1]]])

                # This branch is not entered.
                if sample_cross_frame:
                    sample_num = num_neg_per_img - len(neg_edges)
                    if sample_num > 0 and len(targets) > 1:
                        p = np.array([1] * len(targets))
                        p[bid] = 0
                        p = p / p.sum()
                        sample_bid = np.random.choice(range(len(targets)), sample_num, p=p)
                        sample_tid = np.random.choice(range(object_token.shape[1]), sample_num)

                        for c_bid, c_tid in zip(sample_bid, sample_tid):
                            if len(tgt_edges) > 0:
                                pos_edge = random.choice(tgt_edges)
                                pos_edge = (matched[pos_edge[0]], matched[pos_edge[1]])
                            else:
                                if len(matched) > 0:
                                    pos_edge = np.random.choice(list(matched.keys()), 2)
                                    pos_edge = (matched[pos_edge[0]], matched[pos_edge[1]])
                                else:
                                    pos_edge = np.random.choice(range(object_token.shape[1]), 2)

                            if random.random() > 0.5:
                                sid.append([bid, pos_edge[0]])
                                oid.append([c_bid, c_tid])
                            else:
                                oid.append([bid, pos_edge[1]])
                                sid.append([c_bid, c_tid])
                            rel_tgt.append('[UNK]')
                            num_total += 1

                num_total += num_pos
                batch_ids.extend([bid] * num_total)

            # Triplet prediction samples, including positives and negatives; negatives only indicate whether an edge relation exists.
            assert len(sid) == len(oid) and len(sid) > 0, " Error: len(sid):%s, len(oid):%s" % (len(sid), len(oid))
            sid = torch.as_tensor(sid)
            oid = torch.as_tensor(oid)
            if not relation_is_str:
                all_edge_lbl = torch.as_tensor(all_edge_lbl)

            batch_ids = torch.as_tensor(batch_ids)
            if self.ablation_mode == 'wo_rln':
                relation_feature = torch.cat((object_token[sid[:, 0], sid[:, 1]],
                                              object_token[oid[:, 0], oid[:, 1]]), 1)
            elif self.ablation_mode == 'avg_rln':
                relation_feature = 0
                for k in range(relation_token.shape[1]):
                    relation_feature += self.rln_proj(torch.cat((object_token[sid[:, 0], sid[:, 1]],
                                                                 object_token[oid[:, 0], oid[:, 1]],
                                                                 relation_token[batch_ids][:, k, :]), 1))

                relation_feature /= relation_token.shape[1]
            else:
                # Relation feature for each sample, formed by concatenating two object-box features and the image-level relation feature.
                # [52,768]
                relation_feature = torch.cat((object_token[sid[:, 0], sid[:, 1]],
                                              object_token[oid[:, 0], oid[:, 1]],
                                              relation_token[batch_ids].flatten(1)), 1)
            # [52,4]
            spatial_l = outputs['pred_boxes'][sid[:, 0], sid[:, 1]]
            spatial_r = outputs['pred_boxes'][oid[:, 0], oid[:, 1]]
            spatial_feature = compute_spatial_encodings([spatial_l], [spatial_r])

            if self.rln_freq_bias is not None:
                s_labels = torch.as_tensor(s_labels).to(object_token.device)
                o_labels = torch.as_tensor(o_labels).to(object_token.device)
                # Accumulate object occurrence counts in positive and negative samples.
                freq_dist.append(self.rln_freq_bias( \
                    torch.stack((s_labels, o_labels), 1)
                ))
            if outputs_t is not None:
                sid_t = torch.as_tensor(sid_t)
                oid_t = torch.as_tensor(oid_t)
                relation_feature_t = torch.cat((outputs_t['hs_obj'][sid_t[:, 0], sid_t[:, 1]],
                                                outputs_t['hs_obj'][oid_t[:, 0], oid_t[:, 1]],
                                                outputs_t['hs_rln'][batch_ids].flatten(1)), 1)

        # If there are no edge label IDs.
        else:  # In open-set mode, sample collection differs from full supervision by using edge label names instead of edge label IDs.
            rel_tgt = []
            batch_ids = []
            sid, oid = [], []

            for bid, target in enumerate(targets):
                if len(target['relations']) == 0:
                    continue

                grounded = {}
                cur_num = 0
                for src, dst in zip(indices[bid][0], indices[bid][1]):
                    grounded[dst.item()] = src.item()

                q_nouns = target['gt_names']
                for rel in target['relations']:
                    if rel[0] not in q_nouns or rel[1] not in q_nouns:
                        continue
                    si = q_nouns.index(rel[0])
                    oi = q_nouns.index(rel[1])

                    rel_tgt.append(rel[2])
                    sid.append([bid, grounded[si]])
                    oid.append([bid, grounded[oi]])
                    cur_num += 1

                # random sample negatives
                n = len(q_nouns)
                full_adj = torch.ones((n, n)) - torch.diag(torch.ones(n))
                for rel in target['relations']:
                    if rel[0] not in q_nouns or rel[1] not in q_nouns:
                        continue
                    si = q_nouns.index(rel[0])
                    oi = q_nouns.index(rel[1])
                    full_adj[si, oi] = 0
                neg_edges = torch.nonzero(full_adj)

                num_neg_per_img = max(1, len(target['relations']) * 3)
                if len(neg_edges) >= num_neg_per_img:
                    idx_ = torch.randperm(neg_edges.shape[0])[:num_neg_per_img]
                    neg_edges = neg_edges[idx_, :]

                for item in neg_edges.tolist():
                    sid.append([bid, grounded[item[0]]])
                    oid.append([bid, grounded[item[1]]])
                    rel_tgt.append('[UNK]')
                    cur_num += 1

                # random sampling from other images in a batch 
                if len(neg_edges) < num_neg_per_img:
                    for rel in target['relations']:
                        if rel[0] not in q_nouns or rel[1] not in q_nouns:
                            continue
                        si = q_nouns.index(rel[0])
                        oi = q_nouns.index(rel[1])

                        next_id = bid
                        assert len(targets) >= 1, "batch size must be greather than 1!"
                        if len(targets) == 1:
                            continue

                        while next_id == bid:
                            next_id = random.randint(0, len(targets) - 1)

                        rid = random.randint(0, object_token.shape[1] - 1)
                        if random.randint(0, 1) == 0:
                            sid.append([bid, grounded[si]])
                            oid.append([next_id, rid])
                        else:
                            sid.append([next_id, rid])
                            oid.append([bid, grounded[oi]])

                        rel_tgt.append('[UNK]')
                        cur_num += 1

                batch_ids.extend([bid] * cur_num)

            batch_ids = torch.as_tensor(batch_ids)
            sid = torch.as_tensor(sid)
            oid = torch.as_tensor(oid)
            relation_feature = torch.cat((object_token[sid[:, 0], sid[:, 1]],
                                          object_token[oid[:, 0], oid[:, 1]],
                                          relation_token[batch_ids].flatten(1)
                                          ), 1)
            spatial_l = outputs['pred_boxes'][sid[:, 0], sid[:, 1]]
            spatial_r = outputs['pred_boxes'][oid[:, 0], oid[:, 1]]
            spatial_feature = compute_spatial_encodings([spatial_l], [spatial_r])
        assert len(relation_feature) > 0, "No relation features !"

        # Shuffle relation feature order.
        _idx_ = torch.randperm(len(relation_feature))
        batch_ids = batch_ids[_idx_]
        relation_feature = relation_feature[_idx_]
        spatial_feature = spatial_feature[_idx_]
        if self.ablation_mode == 'avg_rln':
            pass
        else:
            # Reduce relation feature dimension with a linear layer.
            relation_feature = self.rln_proj(relation_feature)

        if relation_feature_t is not None:
            with torch.no_grad():
                relation_feature_t = self.rln_proj_teacher(relation_feature_t[_idx_])

        if (not relation_wo_labels) and (not relation_is_str):
            # Shuffle relation labels accordingly.
            all_edge_lbl = all_edge_lbl[_idx_]
        else:
            rel_tgt = [rel_tgt[e] for e in _idx_.tolist()]

        # Predict relation labels from relation features.
        # Closed-set mode directly uses the classifier.
        # Feed relation features into the classifier to obtain category information.
        if self.rln_classifier is not None:
            assert not relation_wo_labels, "rln_classifier should be None for open vocabulary !"
            rel_logits = self.rln_classifier(relation_feature)
            spa_logits = self.spatial_head(spatial_feature)
            if self.use_relation_adaptive_calibration:
                rel_logits = (
                    self.relation_visual_weight * rel_logits.softmax(-1)
                    + self.relation_spatial_weight * spa_logits.softmax(-1)
                )
            else:
                rel_logits = self.relation_visual_weight * rel_logits + self.relation_spatial_weight * spa_logits
            # Fuse statistical priors into relation logits.
            if self.rln_freq_bias is not None:
                freq_dist = torch.cat(freq_dist, 0)[_idx_]
                rel_logits += freq_dist
            if self.use_relation_adaptive_calibration:
                rel_logits = adaptive_relation_calibration(
                    rel_logits,
                    self.predicate_counts,
                    delta=self.relation_adaptive_delta,
                    eps=self.relation_adaptive_eps,
                    scores_are_prob=True,
                )

            # Constraints can be applied to rel_logits here.
            # Convert positive/negative sample labels to one-hot form so their shape matches relation logits.
            rel_tgt_onehot = torch.zeros([rel_logits.shape[0], rel_logits.shape[1]],
                                         dtype=rel_logits.dtype,
                                         layout=rel_logits.layout,
                                         device=rel_logits.device)

            all_edge_lbl = all_edge_lbl.to(rel_logits.device)
            rel_tgt_onehot.scatter_(-1, all_edge_lbl.unsqueeze(-1), 1)
        # In the open-set setting, obtain relation logits based on similarity.
        else:
            encoded_text = rel_text_dict['encoded_text'][batch_ids]
            text_mask = rel_text_dict['text_token_mask'][batch_ids]
            input_ids = rel_text_dict['input_ids'][batch_ids]

            # relation feature [234,256]
            # spatial feature [234, 36]
            # 234,88
            rel_logits = torch.einsum("a d, a b d -> a b", relation_feature, encoded_text)
            # [234,51]
            spa_logits = torch.einsum("a d, a b d -> a b", self.spatial_head_ovr(spatial_feature), encoded_text)

            rel_logits.masked_fill_(~text_mask, float('-inf'))
            spa_logits.masked_fill_(~text_mask, float('-inf'))

            # padding to max_text_len
            rel_logits = padding_last(rel_logits, 512)  # 512 ? 2048
            rel_tgt_onehot = torch.zeros_like(rel_logits)

            spa_logits = padding_last(spa_logits, 512)  # 512 ? 2048
            # spa_tgt_onehot = torch.zeros_like(spa_logits)

            if relation_feature_t is not None and not self.unsupervised_distill:
                with torch.no_grad():
                    encoded_text_t = rel_text_dict_t['encoded_text'][batch_ids]
                    rel_logits_t = torch.einsum("a d, a b d -> a b", relation_feature_t, encoded_text_t)
                    rel_logits_t.masked_fill_(~text_mask, float('-inf'))
                    rel_logits_t = padding_last(rel_logits_t, rel_logits.shape[-1])

                rel_logits_t = shrink_sigmoid(rel_logits_t, 2.0)  # suitable for OvR
                # rel_logits_t.sigmoid_()

            if relation_wo_labels or relation_is_str:
                for ii, name in enumerate(rel_tgt):
                    if name == '[UNK]':
                        continue

                    ids = self.tokenizer(name + '.').input_ids[1:-1]
                    all_ids = input_ids[ii]

                    start_i, end_i = search_query_pos(all_ids.tolist(), ids)
                    assert start_i != end_i, "cannot find query:{} from input_ids: {}".format(
                        name, self.tokenizer.decode(all_ids))
                    assert start_i < rel_tgt_onehot.shape[1] and end_i < rel_tgt_onehot.shape[
                        1], "start_i:{}, end_i:{}, tgt shape:{}".format(start_i, end_i, rel_tgt_onehot.shape)

                    rel_tgt_onehot[ii, start_i: end_i] = 1.0
                    # spa_tgt_onehot[ii, start_i: end_i] = 1.0

            else:
                for ii, label in enumerate(all_edge_lbl.tolist()):
                    name = self.ind_to_predicates[label]
                    if name == '[UNK]' and relation_feature_t is not None and not self.unsupervised_distill:  # use teacher's output
                        rel_tgt_onehot[ii] = rel_logits_t[ii]
                        continue

                    if name == '[UNK]':
                        continue

                    ids = self.tokenizer(name + '.').input_ids[1:-1]
                    all_ids = input_ids[ii]
                    start_i, end_i = search_query_pos(all_ids.tolist(), ids)
                    assert start_i != end_i, "cannot find query:{} from input_ids".format(ids)

                    rel_tgt_onehot[ii, start_i: end_i] = 1.0
                    # spa_tgt_onehot[ii, start_i: end_i] = 1.0
            if self.use_relation_adaptive_calibration:
                rel_logits = (
                    self.relation_visual_weight * rel_logits.softmax(-1)
                    + self.relation_spatial_weight * spa_logits.softmax(-1)
                )
            else:
                rel_logits = self.relation_visual_weight * rel_logits + self.relation_spatial_weight * spa_logits
            if self.use_relation_adaptive_calibration:
                rel_logits = adaptive_relation_calibration(
                    rel_logits,
                    self.predicate_counts,
                    delta=self.relation_adaptive_delta,
                    eps=self.relation_adaptive_eps,
                    scores_are_prob=True,
                )

        # Total number of positive and negative samples.
        rel_num = torch.as_tensor(rel_tgt_onehot.shape[0], device=rel_tgt_onehot.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(rel_num)
        rel_num = torch.clamp(rel_num / get_world_size(), min=1).item()

        if not self.focal_loss_for_edges:  # CE
            # Use standard cross-entropy loss.
            # Note that class 0 is the background class.
            loss = F.cross_entropy(rel_logits, all_edge_lbl, reduction='sum') / rel_num
            losses = dict(loss_edges=loss)
        else:  # focal loss
            alpha, gamma = 0.25, 2.0
            eps = 1e-5
            rel_prob = rel_logits.sigmoid().clamp(min=eps, max=1.0 - eps)
            rel_mask = (rel_logits != float('-inf')).float()

            rel_weight = (rel_tgt_onehot > 0.5).sum(1, keepdim=True)
            rel_weight[rel_weight == 0] = 1.0

            pos_loss = - torch.log(rel_prob) * ((1.0 - rel_prob) ** gamma) * rel_tgt_onehot * rel_mask / rel_weight
            neg_loss = - torch.log(1.0 - rel_prob) * (rel_prob ** gamma) * (
                        1.0 - rel_tgt_onehot) * rel_mask / rel_weight

            pos_loss = pos_loss.sum()
            neg_loss = neg_loss.sum()

            loss = (pos_loss + neg_loss) / rel_num

            losses = dict(loss_edges=loss)
            with torch.no_grad():
                losses['loss_edges_pos'] = pos_loss.detach() / rel_num
                losses['loss_edges_neg'] = neg_loss.detach() / rel_num

            if outputs_t is not None and self.unsupervised_distill:
                ids = torch.where(all_edge_lbl == 0)[0]
                loss_distill = F.l1_loss(relation_feature[ids], relation_feature_t[ids], reduction='sum') / rel_num

                losses['loss_distill'] = loss_distill

        losses['rel_batch'] = torch.as_tensor(rel_num).to(losses['loss_edges'].device)
        if os.environ.get("DEBUG") == '1':
            import pdb
            pdb.set_trace()

        return losses

    def _focal_loss(self, logits, tgt_onehot, gamma=2.0, eps=1e-5):
        prob = logits.sigmoid().clamp(min=eps, max=1.0 - eps)
        mask = (logits != float('-inf')).float()

        pos_loss = -torch.log(prob) * torch.pow(1.0 - prob, gamma) * tgt_onehot * mask
        neg_loss = -torch.log(1.0 - prob) * torch.pow(prob, gamma) * (1 - tgt_onehot) * mask

        num_pos = tgt_onehot.sum()
        if num_pos == 0:
            return neg_loss.sum()

        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos

        return loss

    def forward(self, outputs, targets, outputs_t=None, return_indices=False,
                global_iter=-1):

        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.
        """
        input_ids = outputs['input_ids']
        for bid, target in enumerate(targets):
            target['input_ids'] = input_ids[bid]
        self.global_iter = global_iter
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        device = next(iter(outputs.values())).device
        # Indices in predictions that match ground truth.
        indices = self.matcher(outputs_without_aux, targets)
        # Skip the following content.
        if outputs_t is not None:
            outputs_without_aux_t = {k: v for k, v in outputs_t.items() if k != 'aux_outputs'}
            indices_t = self.matcher(outputs_without_aux_t, targets)
        else:
            indices_t = None

        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Number of ground-truth boxes in the batch.
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        # prepare for dn loss
        dn_meta = outputs['dn_meta'] if 'dn_meta' in outputs else None

        if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            output_known_lbs_bboxes, single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['boxes']) > 0:
                    t = torch.arange(0, len(targets[i]['boxes']) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    # Repeat the box-index list 11 times [11, n].
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    # 18*0+[1-7],18*1+[1-7]...,18*10+[1-7]
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
            l_dict = {}
            for loss in self.losses:
                if 'edges' == loss:
                    continue
                if 'obj_likelihood' == loss:
                    continue

                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}
                l_dict.update(
                    self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes * scalar, **kwargs))

            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            pass

        for loss in self.losses:
            kwargs = {}
            if 'edges' in loss:
                kwargs = {'object_token': outputs['hs_obj'],
                          'relation_token': outputs['hs_rln'],
                          'rel_text_dict': outputs.get('rel_text_dict', None),
                          'rel_text_dict_t': outputs_t.get('rel_text_dict', None) if outputs_t is not None else None
                          }
                if outputs_t is not None:
                    kwargs['outputs_t'] = outputs_t
                    kwargs['indices_t'] = indices_t

            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                if outputs_t is not None:
                    aux_outputs_t = outputs_t['aux_outputs'][idx]
                    indices_t = self.matcher(aux_outputs_t, targets)

                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    if 'edges' in loss:
                        kwargs = {'object_token': aux_outputs['hs_obj'],
                                  'relation_token': aux_outputs['hs_rln'],
                                  'rel_text_dict': outputs.get('rel_text_dict', None),
                                  'rel_text_dict_t': outputs_t.get('rel_text_dict',
                                                                   None) if outputs_t is not None else None
                                  }
                        if outputs_t is not None:
                            kwargs['outputs_t'] = aux_outputs_t
                            kwargs['indices_t'] = indices_t

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
                    l_dict = {}
                    for loss in self.losses:
                        if 'edges' == loss:
                            continue

                        kwargs = {}
                        if 'labels' in loss:
                            kwargs = {'log': False}

                        l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes * scalar,
                                                    **kwargs))

                    l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    pass

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)

            if outputs_t is not None:
                interm_outputs_t = outputs_t['interm_outputs']
                indices_t = self.matcher(interm_outputs_t, targets)

            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                if 'obj_likelihood' == loss:
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}

                if 'edges' in loss:
                    kwargs = {'object_token': interm_outputs['hs_obj'],
                              'relation_token': interm_outputs['hs_rln'],
                              'rel_text_dict': outputs.get('rel_text_dict', None),
                              'rel_text_dict_t': outputs_t.get('rel_text_dict', None) if outputs_t is not None else None
                              }
                    if outputs_t is not None:
                        kwargs['outputs_t'] = interm_outputs_t
                        kwargs['indices_t'] = indices_t

                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # enc output loss
        if 'enc_outputs' in outputs:
            for i, enc_outputs in enumerate(outputs['enc_outputs']):
                indices = self.matcher(enc_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    if 'obj_likelihood' == loss:
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    if 'edges' == loss:
                        continue

                    l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def prep_for_dn(self, dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups, pad_size = dn_meta['num_dn_group'], dn_meta['pad_size']
        assert pad_size % num_dn_groups == 0
        single_pad = pad_size // num_dn_groups

        return output_known_lbs_bboxes, single_pad, num_dn_groups
