# modified from https://github.com/suprosanna/relationformer/blob/scene_graph/inference.py
import torch
import numpy as np
import torch.nn.functional as F
from typing import List, Dict
import copy
from groundingdino.models.GroundingDINO.losses import compute_spatial_encodings
from .relation_calibration import adaptive_relation_calibration

occ_dict = {
    12: [31, 48, 21, 22, 20, 50, 7, 30, 40, 49, 8, 43, 9, 38, 29, 1, 16, 41, 46, 36, 19, 23, 6, 33, 11, 42, 35, 44, 45,
         4, 13, 25, 10, 17, 5, 3, 27, 47, 12, 24, 14, 2, 28, 26, 39, 32, 34, 18, 37, 15],
    10: [31, 48, 30, 21, 20, 50, 41, 22, 7, 40, 49, 44, 8, 16, 23, 33, 38, 46, 42, 29, 1, 5, 43, 6, 45, 9, 34, 35, 25,
         19, 26, 28, 3, 13, 12, 47, 2, 11, 36, 14, 10, 4, 24, 39, 32, 17, 18, 27, 37, 15],
    1: [20, 50, 31, 19, 30, 21, 38, 41, 48, 29, 1, 22, 8, 23, 40, 49, 44, 16, 10, 43, 7, 25, 9, 47, 6, 42, 11, 35, 46,
        5, 28, 45, 33, 17, 36, 32, 24, 4, 26, 3, 14, 13, 2, 37, 12, 18, 34, 27, 39, 15],
    0: [20, 31, 29, 19, 30, 50, 7, 41, 21, 48, 22, 40, 8, 23, 1, 38, 28, 43, 18, 32, 49, 45, 25, 9, 12, 16, 24, 35, 11,
        6, 46, 13, 47, 3, 33, 42, 36, 4, 26, 37, 5, 44, 17, 10, 14, 2, 34, 39, 27, 15],
    13: [20, 21, 48, 31, 22, 40, 30, 7, 5, 49, 29, 50, 44, 9, 16, 46, 8, 1, 45, 38, 6, 41, 14, 23, 35, 11, 24, 3, 43,
         13, 12, 17, 33, 19, 2, 4, 37, 42, 10, 25, 47, 32, 36, 26, 34, 28, 18, 27, 15, 39],
    5: [20, 31, 30, 22, 48, 50, 29, 8, 1, 46, 40, 43, 21, 6, 49, 10, 41, 33, 13, 18, 44, 23, 19, 9, 16, 32, 7, 11, 45,
        5, 14, 38, 47, 35, 2, 24, 36, 28, 25, 4, 42, 26, 12, 3, 17, 37, 27, 34, 39, 15],
    7: [21, 50, 31, 48, 22, 20, 43, 45, 30, 9, 40, 16, 6, 33, 29, 7, 46, 1, 38, 44, 41, 8, 49, 13, 47, 25, 28, 19, 11,
        23, 17, 14, 4, 3, 35, 12, 10, 2, 34, 39, 27, 42, 5, 24, 26, 32, 36, 37, 18],
    11: [50, 48, 21, 31, 29, 43, 30, 23, 20, 40, 22, 33, 18, 49, 41, 1, 7, 8, 38, 46, 19, 11, 45, 14, 6, 35, 9, 24, 25,
         13, 28, 16, 32, 17, 26, 10, 44, 42, 47, 5, 27, 4, 3, 12, 39, 2, 36, 37, 34, 15],
    8: [50, 30, 21, 31, 48, 22, 40, 29, 23, 20, 43, 45, 9, 8, 38, 16, 19, 35, 49, 18, 7, 41, 34, 13, 46, 11, 1, 28, 10,
        6, 44, 36, 3, 47, 24, 14, 32, 33, 12, 25, 4, 42, 5, 17, 2, 39, 26, 27, 37],
    3: [22, 31, 48, 30, 20, 50, 29, 41, 46, 21, 40, 8, 19, 35, 43, 1, 34, 32, 4, 13, 38, 23, 49, 7, 47, 6, 16, 9, 2, 33,
        24, 17, 11, 27, 42, 10, 25, 45, 26, 28, 14, 44, 36, 12, 3, 5, 39, 37, 18, 15],
    6: [22, 1, 21, 31, 50, 38, 20, 7, 48, 40, 23, 41, 30, 8, 9, 6, 4, 43, 28, 29, 46, 13, 16, 49, 25, 35, 33, 24, 34,
        11, 17, 14, 19, 44, 32, 45, 37, 10, 42, 3, 2, 47, 27, 5, 12, 36, 26, 39, 18, 15],
    2: [50, 31, 29, 48, 21, 41, 1, 30, 20, 46, 40, 22, 23, 8, 6, 10, 43, 49, 14, 9, 34, 7, 3, 32, 45, 38, 19, 33, 16,
        24, 35, 28, 25, 11, 4, 17, 42, 12, 47, 13, 36, 44, 2, 27, 5, 26, 18, 37, 39, 15],
    4: [31, 30, 48, 22, 41, 20, 29, 8, 1, 45, 40, 50, 21, 25, 4, 5, 7, 9, 38, 43, 6, 23, 49, 16, 35, 19, 11, 46, 33, 3,
        24, 28, 44, 32, 27, 10, 47, 37, 14, 13, 42, 17, 2, 12, 26, 18, 34, 36, 39, 15],
    20: [48, 31, 40, 29, 50, 30, 20, 7, 49, 22, 8, 45, 46, 41, 21, 11, 19, 43, 13, 44, 9, 1, 10, 4, 14, 33, 16, 38, 42,
         27, 2, 23, 6, 25, 17, 28, 3, 24, 47, 12, 32, 5, 35, 37, 26, 36, 34, 18],
    15: [48, 31, 50, 29, 41, 21, 25, 22, 20, 30, 49, 7, 23, 6, 9, 38, 11, 46, 40, 35, 8, 43, 47, 1, 19, 4, 24, 12, 33,
         5, 10, 44, 16, 27, 14, 45, 3, 18, 13, 26, 28, 36, 37, 32, 17, 39, 34, 2, 42, 15],
    9: [48, 21, 31, 22, 20, 30, 41, 29, 50, 49, 40, 6, 8, 7, 1, 43, 33, 3, 38, 9, 46, 16, 23, 35, 11, 13, 45, 28, 5, 19,
        44, 24, 4, 47, 25, 17, 37, 14, 10, 42, 32, 26, 2, 34, 39, 12, 27, 36, 18, 15],
    21: [48, 31, 22, 20, 30, 44, 8, 38, 50, 49, 23, 43, 7, 29, 21, 46, 1, 40, 28, 11, 25, 35, 41, 5, 6, 14, 45, 42, 19,
         16, 12, 4, 13, 32, 9, 47, 36, 24, 10, 33, 17, 18, 37, 34, 26, 2, 27],
    22: [50, 48, 31, 20, 22, 38, 21, 23, 49, 29, 8, 43, 19, 30, 46, 40, 11, 32, 44, 6, 35, 1, 47, 14, 7, 42, 9, 25, 5,
         12, 45, 26, 2, 3, 16, 41, 36, 24, 4, 34, 13, 33, 28, 37, 10, 17, 27, 18],
    19: [48, 31, 21, 22, 30, 49, 20, 50, 23, 40, 38, 33, 12, 8, 7, 29, 46, 9, 19, 6, 41, 11, 13, 43, 1, 4, 3, 14, 42,
         10, 27, 44, 17, 28, 25, 5, 45, 35, 24, 36, 2, 32, 26, 47, 16, 18, 34, 37],
    16: [48, 50, 31, 22, 21, 30, 49, 20, 40, 7, 38, 46, 29, 41, 23, 8, 16, 13, 9, 45, 1, 35, 28, 43, 11, 12, 19, 14, 47,
         6, 5, 24, 44, 39, 37, 33, 4, 25, 36, 10, 3, 17, 32, 42, 18, 27, 2, 26, 15, 34],
    18: [50, 48, 31, 9, 21, 49, 20, 29, 40, 33, 13, 22, 30, 41, 7, 23, 6, 46, 45, 38, 8, 11, 19, 1, 43, 35, 10, 4, 24,
         17, 47, 14, 25, 2, 3, 42, 5, 27, 26, 28, 39, 16, 12, 44, 32, 36, 18, 37, 34],
    14: [48, 50, 31, 46, 21, 30, 49, 20, 6, 22, 7, 14, 29, 38, 44, 8, 40, 1, 41, 45, 23, 9, 11, 43, 24, 4, 13, 2, 3, 35,
         12, 19, 25, 33, 47, 16, 27, 5, 39, 10, 26, 42, 18, 17, 36, 37, 32, 28, 34, 15],
    23: [48, 31, 29, 21, 50, 20, 22, 30, 6, 23, 9, 40, 5, 16, 8, 13, 11, 7, 28, 43, 38, 49, 1, 46, 41, 33, 12, 25, 10,
         36, 47, 26, 2, 45, 35, 19, 3, 14, 18, 32, 4, 34, 24],
    17: [1, 22, 48, 21, 50, 31, 7, 25, 30, 49, 20, 6, 40, 32, 44, 45, 8, 38, 41, 43, 10, 23, 29, 11, 46, 35, 28, 9, 14,
         19, 3, 42, 24, 33, 5, 4, 17, 13, 16, 27, 47, 2, 12, 18, 34, 26, 36, 15, 37],
    24: [21, 31, 48, 22, 20, 49, 38, 8, 23, 1, 36, 46, 30, 14, 5, 16, 50, 29, 43, 41, 13, 6, 9, 4, 44, 40, 11, 47, 12,
         7, 35, 25, 19, 26, 32, 3, 42, 33, 18, 45, 24, 17, 28, 2, 10],
    25: [21, 31, 50, 20, 48, 8, 22, 16, 29, 6, 30, 46, 7, 35, 49, 13, 40, 41, 1, 33, 43, 44, 25, 23, 11, 38, 4, 32, 10,
         12, 42, 47, 2, 45, 14, 36, 9, 19, 26, 24, 3],
    26: [48, 31, 20, 22, 30, 21, 49, 44, 50, 7, 1, 40, 38, 29, 6, 16, 32, 46, 33, 8, 19, 12, 25, 45, 41, 43, 10, 13, 23,
         28, 11, 36, 35, 14, 4, 2, 42, 26, 9, 47],
    28: [48, 31, 33, 22, 21, 20, 1, 40, 50, 30, 38, 6, 16, 43, 29, 49, 8, 13, 7, 46, 3, 14, 41, 25, 23, 47, 45, 4, 44,
         24, 12, 35, 32, 27, 5, 19, 9, 26, 10, 28, 34, 2, 11, 37, 36],
    29: [22, 31, 20, 8, 48, 11, 21, 50, 1, 46, 38, 23, 30, 41, 40, 49, 24, 6, 7, 43, 29, 3, 10, 2, 45, 12, 26, 4, 19,
         25, 33, 18, 16, 14, 39, 44, 17, 36, 9],
    30: [30, 46, 22, 48, 21, 31, 38, 23, 20, 29, 1, 33, 50, 41, 49, 43, 9, 40, 6, 8, 11, 26, 4, 13, 44, 47, 35, 7, 5,
         28, 19, 14, 42, 45],
    31: [30, 50, 31, 48, 21, 46, 1, 38, 22, 49, 40, 29, 8, 20, 35, 4, 6, 41, 23, 43, 34, 14, 47, 32, 25, 33, 9, 10, 11,
         28, 44],
    27: [31, 30, 50, 48, 11, 20, 40, 38, 21, 29, 6, 22, 16, 43, 49, 8, 7, 1, 47, 23, 25, 9, 46, 19, 41, 4, 44, 35, 45,
         27, 33, 12, 34, 24, 5, 36, 28, 13],
    35: [31, 20, 22, 50, 48, 30, 38, 1, 6, 21, 40, 43, 29, 41, 46, 8, 33, 13, 23, 27, 7, 49, 18, 44, 12],
    32: [20, 31, 21, 48, 11, 18, 40, 50, 30, 22, 49, 38, 8, 6, 29, 41, 45, 1, 35, 23, 3, 43, 33, 7, 10, 46, 25, 44],
    33: [20, 31, 21, 22, 48, 30, 40, 38, 1, 6, 50, 41, 11, 49, 29, 8, 43, 10, 23, 13, 46, 25, 44, 7],
    36: [47, 22, 31, 20, 50, 48, 30, 8, 40, 6, 29, 21, 49, 1, 7, 43, 26, 23, 27, 2, 41, 33, 16, 5, 38, 44, 46],
    34: [47, 48, 31, 50, 29, 40, 30, 41, 20, 11, 22, 6, 21, 1, 24, 43, 7, 49, 8, 38, 25, 33],
    37: [31, 48, 40, 38, 30, 22, 20, 8, 43, 1, 47, 50, 29, 45, 49, 5, 21, 9, 25, 41, 7, 44, 46, 36],
    38: [22, 48, 31, 49, 20, 1, 50, 21, 8, 29, 30, 40, 19, 25, 16, 23, 7, 42, 3, 28, 36, 33, 43, 34, 41, 32, 17, 10, 6,
         38, 11, 44, 45, 9], 39: [48, 22, 45, 20, 31, 30, 8, 1, 50, 49, 29, 43, 7, 6, 21, 46],
    40: [48, 22, 31, 23, 29, 1, 30, 20, 40, 50, 9, 7, 36, 6, 21, 49],
    41: [31, 48, 30, 1, 49, 22, 29, 40, 20, 23, 21, 8, 50, 47, 9, 38, 14, 28, 43],
    42: [31, 48, 1, 22, 21, 38, 41, 6, 43, 20, 19, 30, 32, 16, 7, 50, 29],
    47: [31, 8, 30, 22, 19, 16, 29, 1, 33, 23, 50, 20, 42, 21, 7, 11, 48],
    43: [31, 30, 1, 22, 8, 48, 41, 40, 20, 50, 9, 36, 43, 25, 29],
    44: [48, 22, 30, 1, 8, 25, 31, 29, 50, 40, 20, 9, 13, 34], 46: [48, 50, 30, 22, 29, 31, 20, 21],
    48: [8, 30, 16, 22, 20, 31, 29, 50, 17, 21, 43, 1, 6, 41, 23, 18, 7, 24, 33, 34, 40],
    49: [31, 30, 22, 48, 29, 50, 14, 20, 1, 43, 12, 21, 23, 8, 24, 5, 49], 52: [29, 31, 30, 22, 38, 7, 20, 8, 50, 23],
    53: [29, 30, 31, 20, 50, 8, 22, 48, 25, 40, 11, 49, 21, 24, 9, 38, 14, 47, 23, 10],
    45: [30, 22, 1, 29, 25, 49, 20, 31, 8, 3, 16, 23, 2, 21, 4, 18, 42, 50, 5, 10, 6],
    54: [30, 48, 20, 31, 29, 50, 40, 1, 33, 8, 22, 43, 49, 41, 23, 27, 21],
    50: [8, 22, 31, 20, 50, 23, 29, 1, 30, 33, 43, 24, 44, 5, 40, 49, 45, 46], 55: [31, 48, 50, 49, 13, 20, 21],
    57: [48, 38, 20, 50, 6, 30, 31, 9, 1, 18, 22, 7, 36], 58: [29, 20, 25, 6, 30, 31, 22, 21, 1, 13, 50, 33, 9, 43],
    51: [49, 31, 30, 22, 23, 50, 29, 40, 14, 25, 43, 21], 61: [48, 20, 30, 50, 31, 1, 18, 9, 7, 22, 8],
    56: [31, 48, 11, 49, 46, 8, 50, 29, 38], 60: [38, 48, 49, 50, 20, 31, 19, 22, 1, 30, 21],
    78: [49, 20, 48, 31, 22, 8, 46, 29, 50, 11, 25, 21, 6, 40, 30, 38, 41, 44, 9, 43, 2, 1, 47, 23, 3, 26, 24, 45, 5,
         36, 37, 33, 14], 120: [49, 48, 31, 30],
    115: [31, 20, 19, 50, 29, 21, 22, 1, 7, 28, 16, 23, 43, 8, 36, 33, 42, 34, 40, 17, 6, 27, 10, 5, 4, 3, 30, 39, 25,
          24], 111: [20, 48, 31, 22, 49, 50, 6, 29, 5, 30, 32, 33, 25, 43],
    114: [29, 35, 31, 46, 1, 40, 3, 33, 22, 41, 10, 23, 20, 18, 50, 13, 30, 4, 8, 6, 2, 42, 43],
    124: [29, 35, 46, 31, 4, 20, 22, 1, 2, 45, 23, 11, 10, 50, 40, 30, 6, 41, 43, 33, 12, 49, 8, 5],
    112: [20, 48, 30, 43, 31, 22, 29, 50, 5, 49, 1], 87: [49, 20, 48, 50, 22, 31, 30, 19, 33, 5],
    136: [29, 31, 4, 20, 8, 23, 1, 22, 43, 50, 10, 33, 30, 7, 41, 17, 18, 36, 19, 48, 13, 12, 49, 6, 3, 21, 16],
    145: [50, 20, 29, 22, 17, 31, 8, 23, 1, 30, 32, 43, 7, 33, 40, 9, 2, 4, 11, 36, 48, 13, 12, 3, 25, 42, 10, 19, 6,
          34],
    126: [31, 29, 1, 27, 50, 20, 40, 6, 43, 22, 23, 33, 5, 30, 8, 24, 21, 3, 36, 26, 10, 41, 7, 13, 37, 12, 48],
    59: [20, 31, 50, 30, 22, 16, 7, 43, 17, 29, 28, 42, 21],
    110: [31, 30, 43, 20, 29, 23, 1, 22, 40, 8, 33, 6, 50, 21, 26, 7],
    97: [31, 50, 43, 29, 20, 1, 30, 22, 7, 26, 40, 8, 23, 12, 21, 28, 33, 24, 34, 13, 47, 5],
    74: [43, 20, 31, 30, 50, 7, 9, 36, 48, 22, 21], 77: [20, 31, 30, 34, 50, 7],
    70: [50, 11, 29, 20, 48, 30, 21, 49, 31, 40, 46, 22, 23, 44, 8],
    149: [48, 40, 29, 50, 1, 11, 20, 31, 6, 22, 30, 23, 8, 46, 49, 21, 41, 2, 9, 5, 25, 45, 7, 38, 43, 10, 44, 33, 14,
          47], 72: [1, 31, 20, 8, 44, 16, 40, 6, 29, 50, 30, 43, 25, 21],
    88: [31, 50, 43, 40, 26, 1, 21, 6, 29, 7, 22, 33, 16, 19, 13, 30, 20],
    106: [20, 22, 41, 40, 31, 6, 42, 29, 30, 8, 50],
    76: [20, 22, 17, 31, 33, 1, 7, 28, 29, 30, 43, 32, 19, 21, 23, 50, 10, 8, 16, 42, 40, 6],
    133: [43, 1, 31, 20, 30, 23, 29, 8, 50, 33, 22, 16, 6], 101: [20, 22, 29, 31, 1, 16, 50, 21, 8, 43, 30, 33, 10],
    96: [31, 20, 1, 22, 29, 8, 16, 23, 21, 50, 30, 43, 40, 7, 12, 18, 4, 33, 14],
    71: [31, 22, 1, 23, 8, 20, 29, 7, 33, 40, 21, 17, 43, 50, 5, 30, 28, 2],
    90: [31, 6, 46, 2, 1, 23, 29, 40, 14, 48, 38, 22, 33, 10, 43, 24, 47, 41, 8, 21, 45, 5, 25, 13, 20, 16, 50, 32],
    67: [20, 48, 49, 31, 22, 30, 50, 5, 29, 1, 21], 66: [20, 31, 48, 50, 49, 22, 30, 19, 5, 21, 43, 32, 8],
    99: [31, 1, 17, 29, 7, 22, 8, 21, 50, 43, 20, 19, 16, 23, 42, 4, 28, 41, 10, 5, 27, 30, 33, 35],
    144: [20, 30, 43, 31, 22, 29, 28, 50, 36, 7, 32], 123: [31, 20, 22, 29, 1, 40, 16, 30, 8, 50, 43],
    62: [31, 48, 20, 29, 50, 21, 1, 49, 7, 22],
    91: [20, 48, 31, 50, 38, 22, 30, 40, 2, 46, 45, 29, 23, 11, 21, 41, 49, 8, 1, 9, 43, 6, 13, 10, 42, 25, 44, 33, 24,
         47, 14], 135: [20, 40, 31, 22, 41, 43, 1, 30, 29, 16, 8, 7, 23, 33, 42, 9, 34, 32],
    130: [20, 31, 30, 22, 7, 29, 28, 36, 27, 42, 50, 17], 140: [20, 31, 30, 40, 22, 23, 21, 1, 43, 50, 41, 29],
    93: [22, 1, 40, 31, 8, 20, 24, 29, 21, 43, 26, 50, 5], 113: [48, 22, 31, 49, 20, 1, 50],
    65: [1, 31, 29, 20, 30, 50, 23, 22, 8, 7, 3, 42, 13, 9, 2, 33, 12, 43],
    108: [1, 29, 31, 30, 22, 21, 50, 20, 32, 40, 8], 116: [29, 31, 43, 1, 22, 33, 30, 50, 20, 8, 7],
    73: [31, 20, 30, 29, 50, 33, 22, 7, 17, 23, 18, 36, 19, 1, 43, 48, 8, 41],
    142: [29, 31, 35, 22, 23, 30, 20, 40, 12, 43, 50, 8, 16, 42, 4, 1],
    139: [50, 11, 31, 8, 1, 23, 29, 43, 21, 33, 7, 20, 28, 13, 44, 16, 22, 30, 19, 48],
    80: [38, 31, 29, 40, 35, 1, 20, 8, 23, 30, 3], 92: [31, 8, 1, 6, 29, 30, 17, 40, 22, 50, 21, 20],
    146: [20, 30, 31, 29, 7, 9], 75: [50, 31, 20, 36, 1, 27, 32, 7, 30, 39, 16, 29, 34],
    137: [31, 8, 35, 29, 30, 20, 23, 22, 6, 50, 32, 43, 1, 7, 34, 40], 128: [48, 20, 31, 29, 22, 49, 21, 50],
    107: [31, 8, 20, 1, 47, 32, 50, 22, 40, 29, 23, 25, 30],
    104: [40, 20, 23, 29, 1, 22, 31, 41, 30, 8, 46, 13, 33, 10], 86: [31, 30, 22, 1, 29, 50, 40, 49, 48, 21],
    105: [20, 31, 30, 50, 1, 19, 7, 8, 33, 43, 13, 36, 29, 12, 22],
    103: [29, 23, 20, 7, 31, 8, 50, 19, 30, 1, 18, 40, 22, 41, 27, 21], 141: [22, 21, 31, 50, 30],
    150: [30, 23, 8, 43, 31, 20, 50, 29, 9], 84: [30, 20, 31, 9, 36, 1, 50, 43, 22],
    138: [31, 20, 30, 21, 29, 50, 8, 23, 1, 22, 48, 16, 43, 9], 83: [20, 30, 7, 29, 31],
    63: [31, 8, 43, 1, 29, 22, 13, 12, 41, 18, 23, 50], 82: [20, 30, 22, 31, 50, 1, 43],
    100: [31, 20, 8, 19, 7, 23, 1, 22, 29, 10, 28, 41, 17, 50, 16, 43, 21], 127: [30, 20, 31, 32, 7, 22, 50, 9, 21],
    121: [31, 22, 20, 1, 12, 29, 41, 26, 13, 50, 43, 33, 5, 40, 24, 30, 25],
    148: [5, 8, 16, 31, 29, 2, 1, 17, 21, 43, 19, 7, 50, 20, 33, 22], 131: [40, 20, 31, 1, 33, 8, 22, 44, 29, 50],
    95: [19, 7, 31, 30, 29, 20, 32, 22, 47, 1, 43, 8, 50, 25], 132: [1, 19, 31, 29, 22, 50, 24, 32, 13],
    129: [29, 41, 46, 31, 22, 20, 8], 64: [1, 29, 30, 31, 20, 38, 40, 21, 22, 43, 5, 8, 50, 23],
    147: [7, 31, 20, 43, 19, 30, 29], 81: [29, 1, 20, 31, 8, 23, 50, 13, 33, 46, 43, 22, 12, 30, 41, 25],
    118: [11, 21, 49, 31, 50, 22, 20, 43, 44, 48, 41, 24, 16], 134: [22, 20, 31, 8, 29, 43, 6, 30, 1, 16, 40, 17],
    85: [31, 50, 20, 7, 42, 23, 34, 32, 30], 125: [11, 50, 29, 31, 43, 22, 21, 41, 38, 24],
    102: [50, 22, 21, 20, 31, 30, 7, 11], 122: [31, 48, 5, 30, 20, 1, 50],
    79: [41, 1, 40, 48, 31, 5, 29, 8, 38, 50, 20, 49, 22, 45, 47], 89: [30, 31, 20], 94: [31, 22, 29, 14, 43, 1, 21],
    68: [50, 29, 22, 48, 20, 16, 31, 38], 109: [29, 20, 31, 50, 8], 143: [31, 23, 22, 50, 8, 33, 38, 20],
    98: [50, 31, 22, 21, 30, 37, 47, 48, 20, 8], 69: [20, 47, 1, 31, 22, 33, 21, 50, 29],
    117: [31, 8, 38, 50, 22, 21, 40, 43, 20, 23], 119: [49, 30, 20, 31, 40, 8, 48, 22, 41]}

def graph_infer(outputs: List[Dict],
                rln_proj, rln_classifier, spatial_head, spatial_head_ovr,
                rln_freq_bias,
                text_dict,
                name2predicates,
                tokenizer,
                use_sigmoid=False,
                use_classifier=False,
                use_relation_adaptive_calibration=False,
                predicate_counts=None,
                relation_adaptive_delta=1.0,
                relation_adaptive_eps=1e-6,
                relation_visual_weight=0.5,
                relation_spatial_weight=0.5,
                save_features=False):
    dst = []
    if rln_freq_bias is not None:
        use_sigmoid = False

    for batch_id, output in enumerate(outputs):
        obj_token = output['obj_token']  # (#obj, dim)
        rln_token = output['rln_token']  # (#query, dim)

        if not use_classifier and text_dict is not None:
            encoded_text = text_dict['encoded_text'][batch_id]
            text_mask = text_dict['text_token_mask'][batch_id]
            input_ids = text_dict['input_ids'][batch_id]
            sep_idx = [i for i in range(len(input_ids)) if input_ids[i] in [101, 102, 1012]]

        boxes = copy.deepcopy(output['boxes'])  # (#obj, 4)
        scores = copy.deepcopy(output['scores'])  # (#obj)
        labels = copy.deepcopy(output['labels'])  # (#obj)

        node_id = torch.nonzero(labels).squeeze()

        # Features, classes, and class scores for each object (all scores are 1).
        obj_token = obj_token[node_id]
        pred_classes = labels[node_id]
        pred_cls_score = scores[node_id]

        pred_boxes = boxes[node_id]
        pred_boxes_score = pred_cls_score
        pred_boxes_class = pred_classes

        if node_id.dim() != 0 and node_id.nelement() != 0 and node_id.shape[0] > 1:
            # all possible node pairs in all token ordering
            tmp = torch.arange(len(node_id))
            # Enumerate all possible instance pairs.
            node_pairs = torch.cat((torch.combinations(tmp),
                                    torch.combinations(tmp)[:, [1, 0]]), 0)

            id_rel = torch.tensor(list(range(len(node_id))))
            node_pairs_rel = torch.cat((torch.combinations(id_rel), torch.combinations(id_rel)[:, [1, 0]]), 0)

            # feature
            # Concatenate instance-pair visual features with the global relation query as relation features [9900, 256].
            relation_feat = torch.cat((
                obj_token[node_pairs[:, 0], :],
                obj_token[node_pairs[:, 1], :],
                rln_token.flatten().repeat(len(node_pairs), 1),
            ),
                dim=1)
            relation_feat = rln_proj(relation_feat)

            # spatial feature
            spatial_l = pred_boxes[node_pairs[:, 0]]
            spatial_r = pred_boxes[node_pairs[:, 1]]
            spatial_feature = compute_spatial_encodings([spatial_l], [spatial_r])

            if use_classifier:
                relation_logits = rln_classifier(relation_feat)
                spa_logits = F.softmax(spatial_head(spatial_feature))
                if use_relation_adaptive_calibration:
                    relation_logits = (
                        relation_visual_weight * relation_logits.softmax(-1)
                        + relation_spatial_weight * spa_logits
                    )
                else:
                    relation_logits = relation_visual_weight * relation_logits + relation_spatial_weight * spa_logits
                if rln_freq_bias is not None:
                    bias = rln_freq_bias( \
                        torch.stack((pred_classes[node_pairs[:, 0]],
                                     pred_classes[node_pairs[:, 1]]), 1))

                    relation_logits += bias
                if use_relation_adaptive_calibration:
                    relation_logits = adaptive_relation_calibration(
                        relation_logits,
                        predicate_counts,
                        delta=relation_adaptive_delta,
                        eps=relation_adaptive_eps,
                        scores_are_prob=True,
                    )
            else:
                # Relation vector [pairs*124].
                # Around 10.
                relation_logits = torch.einsum("a d, b d -> a b", relation_feat, encoded_text)
                relation_logits.masked_fill(~text_mask, float('-inf'))
                spatial_feature = spatial_head_ovr(spatial_feature)
                spa_logits = torch.einsum("a d, b d -> a b", spatial_feature, encoded_text)
                spa_logits.masked_fill_(~text_mask, float('-inf'))
                if use_relation_adaptive_calibration:
                    relation_logits = (
                        relation_visual_weight * relation_logits.softmax(-1)
                        + relation_spatial_weight * spa_logits.softmax(-1)
                    )
                else:
                    relation_logits = relation_visual_weight * relation_logits + relation_spatial_weight * spa_logits
                if use_relation_adaptive_calibration:
                    relation_logits = adaptive_relation_calibration(
                        relation_logits,
                        predicate_counts,
                        delta=relation_adaptive_delta,
                        eps=relation_adaptive_eps,
                        scores_are_prob=True,
                    )

            all_node_pairs = node_pairs_rel.cpu()
            if use_sigmoid:
                # Apply sigmoid to the relation vector.
                relation_prob = relation_logits.sigmoid().detach().cpu()
            else:
                relation_prob = relation_logits.softmax(-1).detach().cpu()

            if use_classifier:
                all_relation = relation_prob
            else:
                # Map tokens to classes.
                all_relation = torch.zeros((relation_prob.shape[0], len(name2predicates)))
                for ii in range(1, len(sep_idx)):
                    right_idx = sep_idx[ii]
                    left_idx = sep_idx[ii - 1] + 1
                    if left_idx >= right_idx:
                        continue
                    name = tokenizer.decode(input_ids[left_idx:right_idx])
                    all_relation[:, name2predicates[name]] = relation_prob[:, left_idx:right_idx].mean(-1)

            classes = all_relation.shape[1]
            items = all_relation.shape[0]
            cls_mask = torch.zeros((items, classes), device=all_node_pairs.device)
            for i in range(0, items):
                s_class = pred_boxes_class[all_node_pairs[i][0]].item()
                o_class = pred_boxes_class[all_node_pairs[i][1]].item()
                s_valid_ind = occ_dict[s_class]
                o_valid_ind = occ_dict[o_class]
                # s_valid_ind.append(0)
                # o_valid_ind.append(0)
                set_s = set(s_valid_ind)
                set_o = set(o_valid_ind)
                com_list = list(set_s.intersection(set_o))
                cls_mask[i, com_list] = 1

            cls_mask = cls_mask.bool()
            all_relation = all_relation.masked_fill(~cls_mask, float(0))

            # Sort triplets by score.
            rel_score = all_relation[:, 1:].max(1)[0]

            obj_score0 = pred_boxes_score[all_node_pairs[:, 0]]
            obj_score1 = pred_boxes_score[all_node_pairs[:, 1]]
            rel_score = rel_score.to(obj_score0.device) * obj_score0 * obj_score1

            rel_idx = rel_score.sort(descending=True)[1].to(all_relation.device)
            # Sorted triplets including object1 index, object2 index, and relation logit.
            all_relation = all_relation[rel_idx]
            all_node_pairs = all_node_pairs[rel_idx]

        else:
            assert node_id.nelement() == 1, "#obj != 1"

            print("Warning: #obj==1!")
            all_node_pairs = torch.zeros(1, 2).long()
            all_relation = torch.zeros(1, 51)  #
            relation_feat = None
            pred_boxes = pred_boxes.view(1, -1).repeat(2, 1)
            pred_boxes_score = pred_boxes_score.view(1, -1).repeat(2, 1)
            pred_boxes_class = pred_boxes_class.view(1, -1).repeat(2, 1)
            # pred_boxes_score.fill_(0.)

        out = {}
        if all_relation is not None:
            out['node_id'] = node_id.cpu()  # 0-99
            out['pred_boxes'] = pred_boxes.cpu()  # Boxes for 100 objects.
            out['pred_boxes_score'] = pred_boxes_score.cpu()  # Scores for 100 objects.
            out['pred_boxes_class'] = pred_boxes_class.cpu()  # Classes for 100 objects.

            out['all_node_pairs'] = all_node_pairs  # Object indices for target pairs.
            out['all_relation'] = all_relation  # Class logits corresponding to target pairs.
            if save_features and relation_feat is not None:
                out['rln_features'] = relation_feat.data.cpu()

        dst.append(out)

    return dst
