import argparse
import csv
import json
import os
import sys
import shutil
from pathlib import Path

from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from util.phys_scene_categories import object_names_for_count, predicate_names_from_source_ids


def parse_args():
    parser = argparse.ArgumentParser(description="Convert raw PhysScene HOI annotations to OvSGTR format.")
    parser.add_argument("--raw_root", required=True, help="Raw PhysScene dataset root.")
    parser.add_argument("--output_root", default="data/phys_scene", help="Converted PhysScene dataset root.")
    parser.add_argument("--copy-images", dest="copy_images", action="store_true", default=True)
    parser.add_argument("--no-copy-images", dest="copy_images", action="store_false")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_hoi_map(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 4:
                continue
            rows.append(
                {
                    "subject_raw_id": int(row[0]),
                    "predicate_id": int(row[1]),
                    "object_raw_id": int(row[2]),
                    "hoi_category_id": int(row[3]),
                }
            )
    return rows


def image_size(path):
    with Image.open(path) as img:
        return img.size


def convert_items(raw_items, image_dir, output_image_dir, copy_images):
    converted = []
    for item in raw_items:
        original_name = item["file_name"]
        src_image = image_dir / original_name
        dst_image = output_image_dir / original_name
        if copy_images:
            dst_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_image, dst_image)

        width, height = image_size(src_image)
        boxes = []
        labels = []
        raw_labels = []
        for ann in item.get("annotations", []):
            boxes.append([float(v) for v in ann["bbox"]])
            raw_label = int(ann["category_id"])
            raw_labels.append(raw_label)
            labels.append(raw_label + 1)

        edges = []
        hoi_annotations = []
        for hoi in item.get("hoi_annotation", []):
            subject_id = int(hoi["subject_id"])
            object_id = int(hoi["object_id"])
            predicate_id = int(hoi["category_id"])
            hoi_category_id = int(hoi.get("hoi_category_id") or 0)
            edges.append([subject_id, object_id, predicate_id])
            hoi_annotations.append(
                {
                    "subject_id": subject_id,
                    "object_id": object_id,
                    "predicate_id": predicate_id,
                    "hoi_category_id": hoi_category_id,
                }
            )

        converted.append(
            {
                "image_id": int(item["img_id"]),
                "image_key": original_name,
                "source_img_id": int(item["img_id"]),
                "original_file_name": original_name,
                "file_name": f"images/{original_name}",
                "width": int(width),
                "height": int(height),
                "boxes": boxes,
                "labels": labels,
                "raw_labels": raw_labels,
                "edges": edges,
                "hoi_annotations": hoi_annotations,
            }
        )
    return converted


def collect_predicate_ids(all_items):
    return sorted({int(edge[2]) for item in all_items for edge in item.get("edges", [])})


def compact_predicates(all_items, hoi_map):
    source_predicate_ids = collect_predicate_ids(all_items)
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(source_predicate_ids, start=1)}
    for item in all_items:
        for edge in item.get("edges", []):
            edge[2] = old_to_new[int(edge[2])]
        for hoi in item.get("hoi_annotations", []):
            old_predicate = int(hoi["predicate_id"])
            hoi["source_predicate_id"] = old_predicate
            hoi["predicate_id"] = old_to_new[old_predicate]

    compacted_hoi_map = []
    for row in hoi_map:
        old_predicate = int(row["predicate_id"])
        if old_predicate not in old_to_new:
            continue
        new_row = dict(row)
        new_row["source_predicate_id"] = old_predicate
        new_row["predicate_id"] = old_to_new[old_predicate]
        compacted_hoi_map.append(new_row)
    return all_items, compacted_hoi_map, source_predicate_ids, old_to_new


def build_dictionary(all_items, hoi_map, source_predicate_ids, old_to_new):
    max_raw_obj = max(label for item in all_items for label in item.get("raw_labels", [0]))

    object_names = object_names_for_count(max_raw_obj + 2)
    predicate_names = predicate_names_from_source_ids(source_predicate_ids)

    idx_to_label = {"0": "__background__"}
    idx_to_label.update({str(idx + 1): name for idx, name in enumerate(object_names)})
    label_to_idx = {name: int(idx) for idx, name in idx_to_label.items()}

    idx_to_predicate = {"0": "[UNK]"}
    idx_to_predicate.update({str(idx + 1): name for idx, name in enumerate(predicate_names)})
    predicate_to_idx = {name: int(idx) for idx, name in idx_to_predicate.items()}

    return {
        "idx_to_label": idx_to_label,
        "label_to_idx": label_to_idx,
        "idx_to_predicate": idx_to_predicate,
        "predicate_to_idx": predicate_to_idx,
        "hoi_map": hoi_map,
        "source_predicate_ids": source_predicate_ids,
        "predicate_id_map": {str(old): new for old, new in old_to_new.items()},
        "note": (
            "Object category ids are shifted by +1 so index 0 remains background. "
            "Predicate ids are compacted to a contiguous 1..K range; 0 is [UNK]. "
            "The original predicate ids are stored in source_predicate_ids and predicate_id_map."
        ),
    }


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main():
    args = parse_args()
    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    annotation_dir = raw_root / "annotations"
    output_annotation_dir = output_root / "annotations"
    output_image_dir = output_root / "images"

    trainval = load_json(annotation_dir / "trainval_hico.json")
    test = load_json(annotation_dir / "test_hico.json")
    trainval_items = convert_items(trainval, raw_root / "images" / "train", output_image_dir, args.copy_images)
    test_items = convert_items(test, raw_root / "images" / "test", output_image_dir, args.copy_images)
    all_items = trainval_items + test_items
    all_items, hoi_map, source_predicate_ids, old_to_new = compact_predicates(
        all_items,
        load_hoi_map(annotation_dir / "HOI对应列表.csv"),
    )
    dictionary = build_dictionary(all_items, hoi_map, source_predicate_ids, old_to_new)

    write_json(output_annotation_dir / "phys_scene_all.json", all_items)
    write_json(output_annotation_dir / "phys_scene_dict.json", dictionary)
    print(f"converted images: {len(all_items)}")
    print(f"output root: {output_root}")


if __name__ == "__main__":
    main()
