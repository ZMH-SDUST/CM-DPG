import argparse
import json
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from util.phys_scene_categories import (
    build_phys_scene_open_vocab_splits,
    object_names_for_count,
    predicate_names_for_count,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Update PhysScene dictionary names and open-vocabulary splits.")
    parser.add_argument(
        "--dict-file",
        default="data/phys_scene/annotations/phys_scene_dict.json",
        help="PhysScene dictionary JSON.",
    )
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--novel-ratio", default=0.2, type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.dict_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    num_object_classes = len(data["idx_to_label"])
    num_predicate_classes = len(data["idx_to_predicate"])
    object_names = object_names_for_count(num_object_classes)
    predicate_names = predicate_names_for_count(num_predicate_classes)

    idx_to_label = {"0": "__background__"}
    idx_to_label.update({str(idx + 1): name for idx, name in enumerate(object_names)})
    label_to_idx = {name: int(idx) for idx, name in idx_to_label.items()}

    idx_to_predicate = {"0": "[UNK]"}
    idx_to_predicate.update({str(idx + 1): name for idx, name in enumerate(predicate_names)})
    predicate_to_idx = {name: int(idx) for idx, name in idx_to_predicate.items()}

    data["idx_to_label"] = idx_to_label
    data["label_to_idx"] = label_to_idx
    data["idx_to_predicate"] = idx_to_predicate
    data["predicate_to_idx"] = predicate_to_idx
    data["open_vocab_splits"] = build_phys_scene_open_vocab_splits(
        num_object_classes,
        num_predicate_classes,
        seed=args.seed,
        novel_ratio=args.novel_ratio,
    )
    data["note"] = (
        "Object category ids are shifted by +1 so index 0 remains background. "
        "Predicate ids keep the source verb ids; 0 is [UNK]. "
        "Open-vocabulary base/novel splits are generated from the configured seed."
    )

    with open(args.dict_file, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    splits = data["open_vocab_splits"]
    print(f"Updated: {args.dict_file}")
    print(f"base objects: {len(splits['base_objects'])}, novel objects: {len(splits['novel_objects'])}")
    print(f"base predicates: {len(splits['base_predicates'])}, novel predicates: {len(splits['novel_predicates'])}")


if __name__ == "__main__":
    main()
