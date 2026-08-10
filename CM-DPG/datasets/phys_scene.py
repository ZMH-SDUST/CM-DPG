import csv
import json
import os
import random
import re
from collections import Counter, OrderedDict

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO

from datasets.coco import make_coco_transforms


PREDICATE_BG = "[UNK]"

def _clean_name(name):
    name = name.strip().replace("_", " ").lower()
    return re.sub(r"\s+", " ", name)


def _make_unique(names):
    counts = Counter(names)
    seen = Counter()
    unique = []
    for name in names:
        seen[name] += 1
        if counts[name] == 1:
            unique.append(name)
        else:
            unique.append(name if seen[name] == 1 else f"{name} {seen[name]}")
    return unique


def _read_categories(csv_file, id_key, name_key):
    rows = []
    with open(csv_file, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append((int(row[id_key]), _clean_name(row[name_key])))

    rows = sorted(rows, key=lambda x: x[0])
    names = _make_unique([name for _, name in rows])
    return [raw_id for raw_id, _ in rows], names


def _read_relation_categories(csv_file):
    rows = []
    with open(csv_file, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append((
                int(row["relation_category_id"]),
                _clean_name(row["relation_category_name_en"]),
                row.get("relation_type", "").strip().lower(),
            ))

    rows = sorted(rows, key=lambda x: x[0])
    names = _make_unique([name for _, name, _ in rows])
    raw_ids = [raw_id for raw_id, _, _ in rows]
    rel_types = {raw_id: rel_type for raw_id, _, rel_type in rows}
    return raw_ids, names, rel_types


def _preprocess_caption(words):
    caption = ". ".join(words).lower().strip()
    if not caption.endswith("."):
        caption += "."
    return caption


class PhysSceneDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
        root,
        transforms=None,
        split_seed=42,
        novel_ratio=0.3,
        ovd_mode=False,
        ovr_mode=False,
        use_text_labels=True,
        exclude_self_relations=True,
        relation_types=None,
    ):
        assert split in {"train", "val", "test"}

        self.dataset_name = "phys_scene"
        self.split = split
        self.root = root
        self.img_dir = os.path.join(root, "image")
        self.ann_dir = os.path.join(root, "annotation")
        self.transforms = transforms
        self.ovd_mode = ovd_mode
        self.ovr_mode = ovr_mode
        self.use_text_labels = use_text_labels
        self.exclude_self_relations = exclude_self_relations
        self.relation_types = set(relation_types) if relation_types else None
        self._coco = None

        obj_raw_ids, obj_names = _read_categories(
            os.path.join(self.ann_dir, "object_categories.csv"),
            "category_id",
            "category_name_en",
        )
        rel_raw_ids, rel_names, self.rel_raw_to_type = _read_relation_categories(
            os.path.join(self.ann_dir, "relation_categories.csv")
        )

        self.obj_raw_to_label = {raw_id: i + 1 for i, raw_id in enumerate(obj_raw_ids)}
        self.rel_raw_to_label = {raw_id: i + 1 for i, raw_id in enumerate(rel_raw_ids)}

        self.ind_to_classes = ["__background__"] + obj_names
        self.ind_to_predicates = [PREDICATE_BG] + rel_names

        self.name2classes = OrderedDict(
            (name, idx) for idx, name in enumerate(self.ind_to_classes) if idx != 0
        )
        self.class2name = OrderedDict((v, k) for k, v in self.name2classes.items())
        self.name2predicates = OrderedDict(
            (name, idx) for idx, name in enumerate(self.ind_to_predicates) if idx != 0
        )

        self.categories = [
            {"supercategory": "phys_scene", "id": idx, "name": name}
            for idx, name in enumerate(self.ind_to_classes)
            if idx != 0
        ]

        # rng = random.Random(split_seed)
        # obj_labels = list(range(1, len(self.ind_to_classes)))
        # rel_labels = list(range(1, len(self.ind_to_predicates)))
        # self.unseen_obj_cats = set(
        #     rng.sample(obj_labels, max(1, round(len(obj_labels) * novel_ratio)))
        # )
        # self.unseen_rels = set(
        #     rng.sample(rel_labels, max(1, round(len(rel_labels) * novel_ratio)))
        # )

        rng = random.Random(split_seed)
        obj_labels = list(range(1, len(self.ind_to_classes)))
        rel_labels = list(range(1, len(self.ind_to_predicates)))
        default_unseen_obj_cats = [2, 21, 24, 27, 28, 30, 31, 32, 33, 34]
        if self.exclude_self_relations:
            default_unseen_rels = [4, 5, 9, 10, 12, 13, 16, 25, 28]
        else:
            default_unseen_rels = [4, 5, 9, 10, 12, 13, 16, 25, 28, 30, 35, 39]
            # default_unseen_rels = [4, 5, 9, 10, 12, 13, 16, 25, 28]
        self.unseen_obj_cats = set(default_unseen_obj_cats)
        self.unseen_rels = set(default_unseen_rels)
        self.unseen_obj_cats = self.unseen_obj_cats & set(obj_labels)
        self.unseen_rels = self.unseen_rels & set(rel_labels)

        self.base_obj_cats = set(obj_labels) - self.unseen_obj_cats
        self.base_rels = set(rel_labels) - self.unseen_rels

        ann_file = os.path.join(self.ann_dir, "annotation.json")
        with open(ann_file, "r", encoding="utf-8-sig") as f:
            all_records = json.load(f)

        all_records = sorted(all_records, key=lambda x: x["file_name"])

        idxs = list(range(len(all_records)))
        rng.shuffle(idxs)

        train_end = int(len(idxs) * 0.7)
        val_end = train_end + int(len(idxs) * 0.1)

        split_idxs = {
            "train": idxs[:train_end],
            "val": idxs[train_end:val_end],
            "test": idxs[val_end:],
        }[split]

        self.images = []
        self.annotations = []
        self.ids = []

        for image_id, record_idx in enumerate(split_idxs, start=1):
            record = all_records[record_idx]
            parsed = self._parse_record(record, for_training=(split == "train"))
            if parsed is None:
                continue

            image_info, ann = parsed
            image_info["id"] = image_id
            image_info["image_id"] = image_id
            ann["image_id"] = image_id

            self.images.append(image_info)
            self.annotations.append(ann)
            self.ids.append(image_id)

        self.zeroshot_triplet = np.zeros((0, 3), dtype=np.int64)

        print(
            "PhysScene split={} images={} objects={} predicates={} "
            "unseen_objects={} unseen_relations={}".format(
                self.split,
                len(self.images),
                len(self.ind_to_classes) - 1,
                len(self.ind_to_predicates) - 1,
                sorted(self.unseen_obj_cats),
                sorted(self.unseen_rels),
            )
        )

    def _keep_non_self_relation(self, rel_type, oid1, oid2):
        if self.relation_types and rel_type not in self.relation_types:
            return False
        return oid1 != oid2

    def _keep_relation_with_self(self, rel_type, oid1, oid2):
        if self.relation_types and rel_type not in self.relation_types:
            return False

        if oid1 == oid2:
            return rel_type == "attribute"

        return rel_type != "attribute"

    def _keep_relation(self, rel_type, oid1, oid2):
        if self.exclude_self_relations:
            return self._keep_non_self_relation(rel_type, oid1, oid2)
        return self._keep_relation_with_self(rel_type, oid1, oid2)

    def _parse_record(self, record, for_training=False):
        file_name = record["file_name"]
        img_path = os.path.join(self.img_dir, file_name)
        if not os.path.exists(img_path):
            return None

        with Image.open(img_path) as img:
            w, h = img.size

        obj_id_to_new = {}
        boxes = []
        labels = []

        for obj in record.get("objects", []):
            label = self.obj_raw_to_label[int(obj["category_id"])]

            if for_training and self.ovd_mode and label in self.unseen_obj_cats:
                continue

            box = np.asarray(obj["bbox"], dtype=np.float32)
            box[[0, 2]] = np.clip(box[[0, 2]], 0, w)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, h)

            if box[2] <= box[0] or box[3] <= box[1]:
                continue

            obj_id_to_new[int(obj["object_id"])] = len(boxes)
            boxes.append(box)
            labels.append(label)

        edges = []
        for rel in record.get("relations", []):
            # if self.relation_types and rel.get("type") not in self.relation_types:
            #     continue
            #
            # oid1 = int(rel["object_id1"])
            # oid2 = int(rel["object_id2"])
            #
            # if self.exclude_self_relations and oid1 == oid2:
            #     continue
            rel_type = rel.get("type", "").strip().lower()
            oid1 = int(rel["object_id1"])
            oid2 = int(rel["object_id2"])

            if not self._keep_relation(rel_type, oid1, oid2):
                continue

            if oid1 not in obj_id_to_new or oid2 not in obj_id_to_new:
                continue

            rel_label = self.rel_raw_to_label[int(rel["category_id"])]

            if for_training and self.ovr_mode and rel_label in self.unseen_rels:
                continue

            edges.append([obj_id_to_new[oid1], obj_id_to_new[oid2], rel_label])

        if len(boxes) < 2 or len(edges) == 0:
            return None

        image_info = {
            "file_name": img_path,
            "width": w,
            "height": h,
        }
        ann = {
            "boxes": np.stack(boxes).astype(np.float32),
            "labels": np.asarray(labels, dtype=np.int64),
            "edges": np.asarray(edges, dtype=np.int64),
        }
        return image_info, ann

    @property
    def coco(self):
        if self._coco is None:
            coco_dict = {
                "images": self.images,
                "annotations": [],
                "categories": self.categories,
            }

            ann_id = 0
            for ann in self.annotations:
                image_id = ann["image_id"]

                for cls, box in zip(ann["labels"], ann["boxes"]):
                    x1, y1, x2, y2 = box.tolist()
                    coco_dict["annotations"].append(
                        {
                            "area": float((x2 - x1) * (y2 - y1)),
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "category_id": int(cls),
                            "image_id": image_id,
                            "id": ann_id,
                            "iscrowd": 0,
                        }
                    )
                    ann_id += 1

            coco = COCO()
            coco.dataset = coco_dict
            coco.createIndex()
            self._coco = coco

        return self._coco

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        item = self.images[index]
        ann = self.annotations[index]

        img = Image.open(item["file_name"]).convert("RGB")

        boxes = torch.as_tensor(ann["boxes"], dtype=torch.float32)
        labels = torch.as_tensor(ann["labels"], dtype=torch.int64)
        edges = torch.as_tensor(ann["edges"], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "edges": edges,
            "image_id": item["image_id"],
            "orig_size": torch.as_tensor([int(item["height"]), int(item["width"])]),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.float32),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if self.use_text_labels:
            gt_names = [
                self.ind_to_classes[label]
                for label in target["labels"].cpu().numpy().tolist()
            ]

            relations = []
            rel_names = []

            for edge in target["edges"].cpu().numpy().tolist():
                sub = gt_names[edge[0]]
                obj = gt_names[edge[1]]
                pred = self.ind_to_predicates[edge[2]]
                relations.append((sub, obj, pred))
                rel_names.append(pred)

            if self.split == "train" and self.ovd_mode:
                object_ids = sorted(self.base_obj_cats)
            else:
                object_ids = list(range(1, len(self.ind_to_classes)))

            if self.split == "train" and self.ovr_mode:
                relation_ids = sorted(self.base_rels)
            else:
                relation_ids = list(range(1, len(self.ind_to_predicates)))

            target["caption"] = _preprocess_caption(
                [self.ind_to_classes[i] for i in object_ids]
            )
            target["rel_caption"] = _preprocess_caption(
                [self.ind_to_predicates[i] for i in relation_ids]
            )

            target["relations"] = relations
            target["gt_names"] = gt_names
            target["gt_rels"] = list(set(rel_names))

        return img, target

    def get_groundtruth(self, index):
        ann = self.annotations[index]
        return (
            torch.as_tensor(ann["boxes"], dtype=torch.float32),
            torch.as_tensor(ann["labels"], dtype=torch.int64),
            np.asarray(ann["edges"], dtype=np.int64),
        )


def build_phys_scene(image_set, args, disable_transforms=False):
    data_path = os.path.join(args.data_path, "PhysScene")

    transforms = (
        make_coco_transforms(
            image_set,
            fix_size=getattr(args, "fix_size", False),
            strong_aug=getattr(args, "strong_aug", False),
            args=args,
        )
        if not disable_transforms
        else None
    )

    return PhysSceneDataset(
        split=image_set,
        root=data_path,
        transforms=transforms,
        split_seed=getattr(args, "phys_scene_split_seed", 42),
        novel_ratio=getattr(args, "phys_scene_novel_ratio", 0.3),
        ovd_mode=getattr(args, "sg_ovd_mode", False),
        ovr_mode=getattr(args, "sg_ovr_mode", False),
        use_text_labels=getattr(args, "use_text_labels", True),
        exclude_self_relations=getattr(args, "phys_scene_exclude_self_relations", True),
        relation_types=getattr(args, "phys_scene_relation_types", None),
    )
