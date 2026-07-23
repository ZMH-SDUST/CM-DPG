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
    predicate_names_from_source_ids,
)


DEFAULT_FILES = [
    "phys_scene_all.json",
    "phys_scene_train.json",
    "phys_scene_val.json",
    "phys_scene_test.json",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compact PhysScene predicate ids to a contiguous 1..K range.")
    parser.add_argument("--annotation-dir", default="data/phys_scene/annotations")
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--novel-ratio", default=0.2, type=float)
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def collect_predicate_ids(items):
    return sorted({int(edge[2]) for item in items for edge in item.get("edges", [])})


def compact_items(items, old_to_new):
    for item in items:
        for edge in item.get("edges", []):
            edge[2] = old_to_new[int(edge[2])]
        for hoi in item.get("hoi_annotations", []):
            old_predicate = int(hoi["predicate_id"])
            hoi["source_predicate_id"] = hoi.get("source_predicate_id", old_predicate)
            hoi["predicate_id"] = old_to_new[old_predicate]
    return items


def compact_hoi_map(hoi_map, old_to_new):
    compacted = []
    for row in hoi_map:
        old_predicate = int(row["predicate_id"])
        if old_predicate not in old_to_new:
            continue
        new_row = dict(row)
        new_row["source_predicate_id"] = new_row.get("source_predicate_id", old_predicate)
        new_row["predicate_id"] = old_to_new[old_predicate]
        compacted.append(new_row)
    return compacted


def main():
    args = parse_args()
    all_path = os.path.join(args.annotation_dir, "phys_scene_all.json")
    dict_path = os.path.join(args.annotation_dir, "phys_scene_dict.json")
    all_items = load_json(all_path)
    dictionary = load_json(dict_path)

    source_predicate_ids = collect_predicate_ids(all_items)
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(source_predicate_ids, start=1)}
    predicate_names = predicate_names_from_source_ids(source_predicate_ids)

    for name in DEFAULT_FILES:
        path = os.path.join(args.annotation_dir, name)
        if not os.path.exists(path):
            continue
        items = compact_items(load_json(path), old_to_new)
        write_json(path, items)
        print(f"Updated {path}: {len(items)} items")

    num_object_classes = len(dictionary["idx_to_label"])
    object_names = object_names_for_count(num_object_classes)
    idx_to_label = {"0": "__background__"}
    idx_to_label.update({str(idx + 1): name for idx, name in enumerate(object_names)})
    idx_to_predicate = {"0": "[UNK]"}
    idx_to_predicate.update({str(idx + 1): name for idx, name in enumerate(predicate_names)})

    dictionary["idx_to_label"] = idx_to_label
    dictionary["label_to_idx"] = {name: int(idx) for idx, name in idx_to_label.items()}
    dictionary["idx_to_predicate"] = idx_to_predicate
    dictionary["predicate_to_idx"] = {name: int(idx) for idx, name in idx_to_predicate.items()}
    dictionary["source_predicate_ids"] = source_predicate_ids
    dictionary["predicate_id_map"] = {str(old): new for old, new in old_to_new.items()}
    dictionary["hoi_map"] = compact_hoi_map(dictionary.get("hoi_map", []), old_to_new)
    dictionary["open_vocab_splits"] = build_phys_scene_open_vocab_splits(
        num_object_classes,
        len(idx_to_predicate),
        seed=args.seed,
        novel_ratio=args.novel_ratio,
        predicate_names=predicate_names,
    )
    dictionary["note"] = (
        "Object category ids are shifted by +1 so index 0 remains background. "
        "Predicate ids are compacted to a contiguous 1..K range; 0 is [UNK]. "
        "The original predicate ids are stored in source_predicate_ids and predicate_id_map."
    )
    write_json(dict_path, dictionary)

    print(f"source predicate ids: {source_predicate_ids}")
    print(f"compacted predicate count: {len(predicate_names)}")
    print(f"num_rln_cat should be: {len(predicate_names) + 1}")


if __name__ == "__main__":
    main()
