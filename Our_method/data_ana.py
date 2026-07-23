# -*- coding: utf-8 -*-
"""
@Time ： 2024/9/22 19:49
@Auther ： Zzou
@File ：data_ana.py
@IDE ：PyCharm
@Motto ：ABC(Always Be Coding)
@Info ：
"""

import json
import os
import h5py
import torch

if __name__ == "__main__":
    folder = "./data/stanford_filtered"
    file_name = "VG-SGG-dicts-with-attri.json"
    file_path = os.path.join(folder, file_name)

    # # h5 file
    # with h5py.File(file_path, 'r') as file:
    #     keys = list(file.keys())

    # # torch file
    # data = torch.load(file_path)

    # json file
    with open(file_path, "r") as file:
        data_dict = json.load(file)
    file.close()
    predicate_count = data_dict["predicate_count"]
    predicates = list(predicate_count.keys())
    print(predicates)
    print(len(predicates))  # 50
    attribute_count = data_dict["attribute_count"]
    attributes = list(attribute_count.keys())
    print(attributes)
    print(len(attributes))  # 200
    pa = list(set(predicates + attributes))
    print(pa)
    print(len(pa))

    # A_minus_B = [item for item in predicates if item not in attributes]
    # B_minus_A = [item for item in attributes if item not in predicates]
    # print(A_minus_B)
    # print(B_minus_A)


