import torch
from torchvision.ops.boxes import box_iou

def estimate_image_size(boxes_1, boxes_2):
    # 拼接这两个张量
    boxes = torch.cat((boxes_1[0], boxes_2[0]), dim=0).squeeze(0)

    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    # 找到所有边界框的最左边和最右边的x坐标，以及最顶部和最底部的y坐标
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