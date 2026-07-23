import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _ordered_values(mapping):
    return [mapping[str(i)] for i in range(len(mapping))]


def build_phys_scene_stats(annotation_file, dict_file, eps=1e-3):
    with open(annotation_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    with open(dict_file, "r", encoding="utf-8") as f:
        dictionary = json.load(f)

    obj_classes = _ordered_values(dictionary["idx_to_label"])
    rel_classes = _ordered_values(dictionary["idx_to_predicate"])
    num_obj_classes = len(obj_classes)
    num_rel_classes = len(rel_classes)

    fg_matrix = np.zeros(
        (num_obj_classes, num_obj_classes, num_rel_classes),
        dtype=np.int64,
    )
    bg_matrix = np.zeros((num_obj_classes, num_obj_classes), dtype=np.int64)

    for item in annotations:
        labels = item.get("labels", [])
        for sid in range(len(labels)):
            for oid in range(len(labels)):
                if sid != oid:
                    bg_matrix[labels[sid], labels[oid]] += 1

        for edge in item.get("edges", []):
            sid, oid, predicate = edge[:3]
            if sid >= len(labels) or oid >= len(labels):
                continue
            subj = labels[sid]
            obj = labels[oid]
            if 0 <= subj < num_obj_classes and 0 <= obj < num_obj_classes and 0 <= predicate < num_rel_classes:
                fg_matrix[subj, obj, predicate] += 1

    bg_matrix += 1
    fg_matrix[:, :, 0] = bg_matrix
    pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)

    return {
        "fg_matrix": torch.from_numpy(fg_matrix),
        "pred_dist": torch.from_numpy(pred_dist).float(),
        "obj_classes": obj_classes,
        "rel_classes": rel_classes,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a VG-style frequency-bias statistics file for PhysScene."
    )
    parser.add_argument(
        "--annotation-file",
        default="data/phys_scene/annotations/phys_scene_train.json",
        help="PhysScene training annotation JSON. Only the training split should be used.",
    )
    parser.add_argument(
        "--dict-file",
        default="data/phys_scene/annotations/phys_scene_dict.json",
        help="PhysScene dictionary JSON.",
    )
    parser.add_argument(
        "--output-file",
        default="data/phys_scene/phys_scene_stats.pt",
        help="Output .pt file compatible with rln_freq_bias.",
    )
    parser.add_argument("--eps", default=1e-3, type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    stats = build_phys_scene_stats(args.annotation_file, args.dict_file, args.eps)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, output_file)

    fg_matrix = stats["fg_matrix"]
    predicate_counts = fg_matrix[:, :, 1:].sum((0, 1))
    print(f"Saved: {output_file}")
    print(f"fg_matrix: {tuple(fg_matrix.shape)}")
    print(f"pred_dist: {tuple(stats['pred_dist'].shape)}")
    print(f"obj_classes: {len(stats['obj_classes'])}")
    print(f"rel_classes: {len(stats['rel_classes'])}")
    print(f"foreground predicate instances: {int(predicate_counts.sum())}")


if __name__ == "__main__":
    main()
