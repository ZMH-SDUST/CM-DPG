import copy
import json
import os
import random
from collections import OrderedDict

import torch
import torch.utils.data
from PIL import Image
from pycocotools.coco import COCO

from datasets.coco import make_coco_transforms
from util.phys_scene_categories import build_phys_scene_open_vocab_splits
from util.phys_scene_split import resolve_phys_scene_annotation


PREDICATE_BG = "[UNK]"


def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."


class PhysSceneDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
        data_path,
        ann_file,
        dict_file,
        transforms=None,
        filter_empty_rels=True,
        filter_duplicate_rels=True,
        num_obj=100,
        use_text_labels=True,
        ovd_mode=False,
        ovr_mode=False,
        ov_split_seed=1234,
        ov_novel_ratio=0.2,
    ):
        assert split in {"train", "val", "test"}
        self.dataset_name = "phys_scene"
        self.split = split
        self.data_path = data_path
        self.transforms = transforms
        self.filter_empty_rels = filter_empty_rels
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == "train"
        self.num_obj = num_obj
        self.use_text_labels = use_text_labels
        self.ovd_mode = ovd_mode
        self.ovr_mode = ovr_mode

        with open(dict_file, "r", encoding="utf-8") as f:
            info = json.load(f)
        self.ind_to_classes = self._indexed_list(info["idx_to_label"])
        self.ind_to_predicates = self._indexed_list(info["idx_to_predicate"])
        if self.ind_to_predicates[0] == "__background__":
            self.ind_to_predicates[0] = PREDICATE_BG

        self.name2classes = OrderedDict(
            (name, idx) for idx, name in enumerate(self.ind_to_classes) if name != "__background__"
        )
        self.class2name = OrderedDict((v, k) for k, v in self.name2classes.items())
        self.name2predicates = {name: idx for idx, name in enumerate(self.ind_to_predicates)}
        self.categories = [
            {"supercategory": "phys_scene", "id": idx, "name": name}
            for idx, name in enumerate(self.ind_to_classes)
            if name != "__background__"
        ]
        self.open_vocab_splits = build_phys_scene_open_vocab_splits(
            len(self.ind_to_classes),
            len(self.ind_to_predicates),
            seed=ov_split_seed,
            novel_ratio=ov_novel_ratio,
        )
        self.base_object_ids = set(self.open_vocab_splits["base_object_ids"])
        self.novel_object_ids = set(self.open_vocab_splits["novel_object_ids"])
        self.base_predicate_ids = set(self.open_vocab_splits["base_predicate_ids"])
        self.novel_predicate_ids = set(self.open_vocab_splits["novel_predicate_ids"])
        self.base_object_names = self.open_vocab_splits["base_objects"]
        self.base_predicate_names = self.open_vocab_splits["base_predicates"]
        self.novel_object_names = self.open_vocab_splits["novel_objects"]
        self.novel_predicate_names = self.open_vocab_splits["novel_predicates"]

        with open(ann_file, "r", encoding="utf-8") as f:
            items = json.load(f)
        self.items = [item for item in items if item.get("split", self.split) == self.split]
        if self.split == "train" and (self.ovd_mode or self.ovr_mode):
            self.items = [self._filter_open_vocab_training_item(item) for item in self.items]
        if filter_empty_rels:
            self.items = [item for item in self.items if len(item.get("edges", [])) > 0]
        self.ids = [int(item["image_id"]) for item in self.items]
        self._coco = None

    @staticmethod
    def _indexed_list(mapping):
        max_id = max(int(k) for k in mapping.keys())
        values = [None] * (max_id + 1)
        for key, value in mapping.items():
            values[int(key)] = value
        return [v if v is not None else f"unused_{i}" for i, v in enumerate(values)]

    @property
    def coco(self):
        if self._coco is None:
            coco = COCO()
            coco_dict = {"images": [], "annotations": [], "categories": self.categories}
            ann_id = 0
            for item in self.items:
                coco_dict["images"].append(
                    {
                        "id": int(item["image_id"]),
                        "file_name": item["file_name"],
                        "width": item["width"],
                        "height": item["height"],
                    }
                )
                for label, box in zip(item["labels"], item["boxes"]):
                    x1, y1, x2, y2 = box
                    coco_dict["annotations"].append(
                        {
                            "id": ann_id,
                            "image_id": int(item["image_id"]),
                            "category_id": int(label),
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "area": max(0.0, x2 - x1) * max(0.0, y2 - y1),
                            "iscrowd": 0,
                        }
                    )
                    ann_id += 1
            coco.dataset = coco_dict
            coco.createIndex()
            self._coco = coco
        return self._coco

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        img = Image.open(os.path.join(self.data_path, item["file_name"])).convert("RGB")
        boxes, labels, edges = self.get_groundtruth(index)

        if self.filter_duplicate_rels and len(edges) > 0:
            rel_sets = {}
            for sub_id, obj_id, pred_id in edges.tolist():
                rel_sets.setdefault((sub_id, obj_id), []).append(pred_id)
            edges = torch.as_tensor(
                [[sub_id, obj_id, random.choice(pred_ids)] for (sub_id, obj_id), pred_ids in rel_sets.items()],
                dtype=torch.int64,
            )

        width, height = item["width"], item["height"]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=height)

        target = {
            "iscrowd": torch.zeros(boxes.shape[0], dtype=torch.float32),
            "boxes": boxes,
            "labels": labels,
            "edges": edges,
            "image_id": torch.as_tensor(int(item["image_id"])),
            "image_key": item.get("image_key", item.get("original_file_name", os.path.basename(item["file_name"]))),
            "file_name": item["file_name"],
            "orig_size": torch.as_tensor([int(height), int(width)]),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if self.use_text_labels:
            label_names = [self.ind_to_classes[int(label)] for label in target["labels"].cpu().tolist()]
            triples = []
            nouns = []
            rels = []
            for edge in target["edges"]:
                sub = label_names[int(edge[0])]
                obj = label_names[int(edge[1])]
                pred = self.ind_to_predicates[int(edge[2])]
                triples.append((sub, obj, pred))
                nouns.extend([sub, obj])
                rels.append(pred)

            target["relations"] = triples
            target["gt_names"] = copy.deepcopy(label_names)
            target["gt_rels"] = copy.deepcopy(list(set(rels)))
            all_nouns = self.base_object_names if self.split == "train" and self.ovd_mode else self.ind_to_classes[1:]
            all_rels = self.base_predicate_names if self.split == "train" and self.ovr_mode else self.ind_to_predicates[1:]
            if self.split == "train":
                all_nouns = list(all_nouns)
                all_rels = list(all_rels)
                random.shuffle(all_nouns)
                random.shuffle(all_rels)
            target["caption"] = preprocess_caption(". ".join(all_nouns))
            target["rel_caption"] = preprocess_caption(". ".join(all_rels))

        if len(target["edges"]) == 0 and self.split == "train":
            return self[(index - random.randint(1, min(10, max(1, len(self.items) - 1)))) % len(self.items)]
        return img, target

    def get_groundtruth(self, index):
        item = self.items[index]
        boxes = torch.as_tensor(item["boxes"], dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(item["labels"], dtype=torch.int64)
        edges = torch.as_tensor(item["edges"], dtype=torch.int64).reshape(-1, 3)

        if len(boxes) > self.num_obj:
            keep = torch.randperm(len(boxes))[: self.num_obj]
            old_to_new = {int(old): new for new, old in enumerate(keep.tolist())}
            boxes = boxes[keep]
            labels = labels[keep]
            edges = torch.as_tensor(
                [
                    [old_to_new[int(sub)], old_to_new[int(obj)], int(pred)]
                    for sub, obj, pred in edges.tolist()
                    if int(sub) in old_to_new and int(obj) in old_to_new
                ],
                dtype=torch.int64,
            ).reshape(-1, 3)
        return boxes, labels, edges

    def _filter_open_vocab_training_item(self, item):
        boxes = item.get("boxes", [])
        labels = item.get("labels", [])
        raw_labels = item.get("raw_labels", [])
        edges = item.get("edges", [])

        old_to_new = {}
        new_boxes = []
        new_labels = []
        new_raw_labels = []
        for old_idx, (box, label) in enumerate(zip(boxes, labels)):
            if self.ovd_mode and int(label) not in self.base_object_ids:
                continue
            old_to_new[old_idx] = len(new_boxes)
            new_boxes.append(box)
            new_labels.append(label)
            if old_idx < len(raw_labels):
                new_raw_labels.append(raw_labels[old_idx])

        if not self.ovd_mode:
            old_to_new = {idx: idx for idx in range(len(labels))}
            new_boxes = list(boxes)
            new_labels = list(labels)
            new_raw_labels = list(raw_labels)

        new_edges = []
        for edge in edges:
            sub_id, obj_id, pred_id = [int(v) for v in edge[:3]]
            if sub_id not in old_to_new or obj_id not in old_to_new:
                continue
            if self.ovr_mode and pred_id not in self.base_predicate_ids:
                continue
            new_edges.append([old_to_new[sub_id], old_to_new[obj_id], pred_id])

        filtered = dict(item)
        filtered["boxes"] = new_boxes
        filtered["labels"] = new_labels
        if raw_labels:
            filtered["raw_labels"] = new_raw_labels
        filtered["edges"] = new_edges
        return filtered


def build_phys_scene(image_set, args, disable_transforms=False):
    data_path = getattr(args, "data_path")
    ann_file = resolve_phys_scene_annotation(args, image_set)
    dict_file = getattr(args, "phys_scene_dict_file", None) or os.path.join(
        data_path, "annotations", "phys_scene_dict.json"
    )
    transforms = None if disable_transforms else make_coco_transforms(image_set, fix_size=getattr(args, "fix_size", False))
    return PhysSceneDataset(
        split=image_set,
        data_path=data_path,
        ann_file=ann_file,
        dict_file=dict_file,
        transforms=transforms,
        filter_empty_rels=True,
        filter_duplicate_rels=True,
        num_obj=getattr(args, "num_obj", 100),
        use_text_labels=getattr(args, "use_text_labels", True),
        ovd_mode=getattr(args, "sg_ovd_mode", False),
        ovr_mode=getattr(args, "sg_ovr_mode", False),
        ov_split_seed=getattr(args, "phys_scene_ov_split_seed", getattr(args, "seed", 1234)),
        ov_novel_ratio=getattr(args, "phys_scene_ov_novel_ratio", 0.2),
    )
