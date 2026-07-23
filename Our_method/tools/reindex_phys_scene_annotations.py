import argparse
import json
import os


DEFAULT_FILES = [
    "phys_scene_all.json",
    "phys_scene_train.json",
    "phys_scene_val.json",
    "phys_scene_test.json",
]


def image_key(item):
    file_name = item.get("file_name", "")
    if file_name:
        return os.path.basename(file_name)
    return item.get("image_key") or item.get("original_file_name")


def parse_args():
    parser = argparse.ArgumentParser(description="Assign stable unique image ids to PhysScene annotations.")
    parser.add_argument(
        "--annotation-dir",
        default="data/phys_scene/annotations",
        help="Directory containing PhysScene JSON annotation files.",
    )
    parser.add_argument("--start-id", default=1, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    all_path = os.path.join(args.annotation_dir, "phys_scene_all.json")
    with open(all_path, "r", encoding="utf-8") as f:
        all_items = json.load(f)

    ordered_keys = sorted({image_key(item) for item in all_items})
    key_to_id = {key: idx for idx, key in enumerate(ordered_keys, start=args.start_id)}

    for name in DEFAULT_FILES:
        path = os.path.join(args.annotation_dir, name)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            old_id = item.get("image_id")
            item["source_img_id"] = item.get("source_img_id", old_id)
            item["image_id"] = key_to_id[image_key(item)]
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"Updated {path}: {len(items)} items")

    print(f"unique image ids: {len(key_to_id)}")


if __name__ == "__main__":
    main()
