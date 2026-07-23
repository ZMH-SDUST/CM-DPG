import json
import os

import torch
from util.phys_scene_split import resolve_phys_scene_train_annotation


def load_predicate_counts(args, num_rel_categories):
    """Load train-set predicate counts for adaptive relation calibration."""
    count_file = getattr(args, "predicate_counts_file", None)
    if count_file:
        return _load_counts_file(count_file, num_rel_categories)

    dataset_file = getattr(args, "dataset_file", None)
    data_path = getattr(args, "data_path", None)

    if dataset_file == "phys_scene" and data_path is not None:
        ann_file = resolve_phys_scene_train_annotation(args)
        if os.path.exists(ann_file):
            return _counts_from_phys_scene(ann_file, num_rel_categories)

    rln_freq_bias_file = getattr(args, "rln_freq_bias", None)
    if rln_freq_bias_file and os.path.exists(rln_freq_bias_file):
        statistics = torch.load(rln_freq_bias_file, map_location="cpu")
        fg_matrix = statistics.get("fg_matrix")
        if fg_matrix is not None:
            counts = fg_matrix.sum((0, 1)).float()
            return _fit_counts_length(counts, num_rel_categories)

    return None


def adaptive_relation_calibration(scores, predicate_counts, delta=1.0, eps=1e-6, scores_are_prob=False):
    """Eq.(15)-style adaptive weighting over relation predicate logits.

    The returned tensor is log(p_hat). It can be fed to cross_entropy or
    softmax exactly like logits while preserving stable computation.
    """
    if predicate_counts is None or predicate_counts.numel() == 0:
        return scores

    counts = predicate_counts.to(device=scores.device, dtype=scores.dtype)
    counts = _fit_counts_length(counts, scores.shape[-1]).clamp_min(eps)

    finite_mask = torch.isfinite(scores)
    safe_scores = scores.masked_fill(~finite_mask, 0.0)
    if scores_are_prob:
        prob_scores = safe_scores.clamp_min(eps)
        exp_scores = torch.exp(prob_scores)
    else:
        safe_logits = safe_scores
        prob_scores = safe_logits.sigmoid().clamp_min(eps)
        exp_scores = torch.exp(safe_logits)

    score_ratio = prob_scores.unsqueeze(-2) / (prob_scores.unsqueeze(-1) + eps)
    count_ratio = counts.view(1, -1) / (counts.view(-1, 1) + eps)
    weights = (delta * score_ratio * count_ratio).clamp_min(eps)

    weighted_exp_scores = weights * exp_scores.unsqueeze(-2)
    weighted_exp_scores = weighted_exp_scores.masked_fill(~finite_mask.unsqueeze(-2), 0.0)
    denom = weighted_exp_scores.sum(-1).clamp_min(eps)

    calibrated = exp_scores / denom
    calibrated = calibrated.clamp_min(eps).log()
    return calibrated.masked_fill(~finite_mask, float("-inf"))


def _load_counts_file(count_file, num_rel_categories):
    if count_file.endswith(".pt") or count_file.endswith(".pth"):
        data = torch.load(count_file, map_location="cpu")
    else:
        with open(count_file, "r", encoding="utf-8") as f:
            data = json.load(f)

    if isinstance(data, dict):
        if "predicate_counts" in data:
            data = data["predicate_counts"]
        elif "fg_matrix" in data:
            data = torch.as_tensor(data["fg_matrix"]).sum((0, 1))
        else:
            counts = torch.zeros(num_rel_categories, dtype=torch.float32)
            for key, value in data.items():
                idx = int(key)
                if 0 <= idx < num_rel_categories:
                    counts[idx] = float(value)
            return counts

    return _fit_counts_length(torch.as_tensor(data, dtype=torch.float32), num_rel_categories)


def _counts_from_phys_scene(ann_file, num_rel_categories):
    counts = torch.zeros(num_rel_categories, dtype=torch.float32)
    with open(ann_file, "r", encoding="utf-8") as f:
        items = json.load(f)

    for item in items:
        if item.get("split", "train") != "train":
            continue
        for edge in item.get("edges", []):
            pred_id = int(edge[2])
            if 0 <= pred_id < num_rel_categories:
                counts[pred_id] += 1.0
    return counts


def _fit_counts_length(counts, length):
    counts = counts.flatten().float()
    if counts.numel() == length:
        return counts
    fitted = torch.zeros(length, dtype=torch.float32)
    n = min(counts.numel(), length)
    fitted[:n] = counts[:n]
    return fitted
