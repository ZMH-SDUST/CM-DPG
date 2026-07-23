import json
import os
import random


DEFAULT_ALL_FILE = "phys_scene_all.json"
DEFAULT_TRAIN_FILE = "phys_scene_train.json"
DEFAULT_VAL_FILE = "phys_scene_val.json"
DEFAULT_TEST_FILE = "phys_scene_test.json"


def ensure_phys_scene_splits(
    data_path,
    seed=1234,
    test_ratio=0.2,
    val_ratio=0.1,
    all_file=None,
    train_file=None,
    val_file=None,
    test_file=None,
    force=False,
):
    ann_dir = os.path.join(data_path, "annotations")
    all_path = all_file or os.path.join(ann_dir, DEFAULT_ALL_FILE)
    train_path = train_file or os.path.join(ann_dir, DEFAULT_TRAIN_FILE)
    val_path = val_file or os.path.join(ann_dir, DEFAULT_VAL_FILE)
    test_path = test_file or os.path.join(ann_dir, DEFAULT_TEST_FILE)

    if (not force) and os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        _strip_split_from_all_file(all_path)
        return train_path, val_path, test_path

    with open(all_path, "r", encoding="utf-8") as f:
        all_items = json.load(f)

    all_items = [_without_split(item) for item in all_items]
    image_order = _annotation_image_order(data_path, all_items)

    rng = random.Random(int(seed))
    shuffled = list(image_order)
    rng.shuffle(shuffled)

    total_count = len(shuffled)
    test_count = int(round(total_count * float(test_ratio)))
    val_count = int(round(total_count * float(val_ratio)))
    if test_count + val_count > total_count:
        raise ValueError(
            "phys_scene_test_ratio + phys_scene_val_ratio must not exceed 1.0"
        )
    test_names = set(shuffled[:test_count])
    val_names = set(shuffled[test_count:test_count + val_count])

    train_items = []
    val_items = []
    test_items = []
    for item in all_items:
        split_item = dict(item)
        if _image_key(item) in test_names:
            split_item["split"] = "test"
            test_items.append(split_item)
        elif _image_key(item) in val_names:
            split_item["split"] = "val"
            val_items.append(split_item)
        else:
            split_item["split"] = "train"
            train_items.append(split_item)

    _write_json(all_path, all_items)
    _write_json(train_path, train_items)
    _write_json(val_path, val_items)
    _write_json(test_path, test_items)
    return train_path, val_path, test_path


def resolve_phys_scene_annotation(args, image_set):
    data_path = getattr(args, "data_path")
    split = image_set
    seed = getattr(args, "phys_scene_split_seed", getattr(args, "seed", 1234))
    test_ratio = getattr(args, "phys_scene_test_ratio", 0.2)
    val_ratio = getattr(args, "phys_scene_val_ratio", 0.1)
    force = getattr(args, "phys_scene_regenerate_split", False)

    ann_file = getattr(args, "phys_scene_ann_file", None)
    if ann_file:
        return ann_file

    train_path, val_path, test_path = ensure_phys_scene_splits(
        data_path,
        seed=seed,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        force=force,
    )
    if split == "train":
        return train_path
    if split == "val":
        return val_path
    return test_path


def resolve_phys_scene_train_annotation(args):
    data_path = getattr(args, "data_path")
    seed = getattr(args, "phys_scene_split_seed", getattr(args, "seed", 1234))
    test_ratio = getattr(args, "phys_scene_test_ratio", 0.2)
    val_ratio = getattr(args, "phys_scene_val_ratio", 0.1)
    force = getattr(args, "phys_scene_regenerate_split", False)
    train_path, _, _ = ensure_phys_scene_splits(
        data_path,
        seed=seed,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        force=force,
    )
    return train_path


def _strip_split_from_all_file(all_path):
    if not os.path.exists(all_path):
        return
    with open(all_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    if any("split" in item for item in items):
        _write_json(all_path, [_without_split(item) for item in items])


def _without_split(item):
    clean = dict(item)
    clean.pop("split", None)
    return clean


def _annotation_image_order(data_path, items):
    item_names = {_image_key(item) for item in items}
    image_dir = os.path.join(data_path, "images")
    image_names = []
    if os.path.isdir(image_dir):
        for name in sorted(os.listdir(image_dir)):
            if name in item_names:
                image_names.append(name)
    missing = sorted(item_names - set(image_names))
    return image_names + missing


def _image_key(item):
    file_name = item.get("file_name", "")
    if file_name:
        return os.path.basename(file_name)
    return item.get("image_key") or item.get("original_file_name")


def _write_json(path, data):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
