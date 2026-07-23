# -*- coding: utf-8 -*-
"""
@Time ： 2024/12/18 20:13
@Auther ： Zzou
@File ：dataset_coor.py
@IDE ：PyCharm
@Motto ：ABC(Always Be Coding)
@Info ：
"""
import argparse
import h5py
from datasets import build_dataset, get_coco_api_from_dataset
from main import get_args_parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.data_path = "./data"
    args.dataset_file = 'vg'

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    dataset_test = build_dataset(image_set='test', args=args)

    occ_dict = dict()

    train_anno = dataset_train.annotations
    val_anno = dataset_val.annotations
    test_anno = dataset_test.annotations

    for item in train_anno:
        edges = item['edges']
        for edge in edges:
            p1 = edge[0]
            p2 = edge[1]
            p3 = edge[2]
            if p1 not in occ_dict.keys():
                occ_dict[p1] = list()
            if p2 not in occ_dict.keys():
                occ_dict[p2] = list()
            if p3 not in occ_dict[p1]:
                occ_dict[p1].append(p3)
            if p3 not in occ_dict[p2]:
                occ_dict[p2].append(p3)

    for item in val_anno:
        edges = item['edges']
        object_labels = item['labels']
        for i,edge in enumerate(edges):
            # This is incorrect: use the object category, not the object index.
            p1 = object_labels[edge[0]]
            p2 = object_labels[edge[1]]
            p3 = edge[2]
            if p1 not in occ_dict.keys():
                occ_dict[p1] = list()
            if p2 not in occ_dict.keys():
                occ_dict[p2] = list()
            if p3 not in occ_dict[p1]:
                occ_dict[p1].append(p3)
            if p3 not in occ_dict[p2]:
                occ_dict[p2].append(p3)

    for item in test_anno:
        edges = item['edges']
        for edge in edges:
            p1 = edge[0]
            p2 = edge[1]
            p3 = edge[2]
            if p1 not in occ_dict.keys():
                occ_dict[p1] = list()
            if p2 not in occ_dict.keys():
                occ_dict[p2] = list()
            if p3 not in occ_dict[p1]:
                occ_dict[p1].append(p3)
            if p3 not in occ_dict[p2]:
                occ_dict[p2].append(p3)

    print(occ_dict)
