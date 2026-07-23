# -*- coding: utf-8 -*-
"""
@Time ： 2025/1/14 10:30
@Auther ： Zzou
@File ：statics.py
@IDE ：PyCharm
@Motto ：ABC(Always Be Coding)
@Info ：
"""

import pandas as pd

df = pd.read_excel('your_file.xlsx')

verbs = df['verb'].tolist()
quantities = df['number'].tolist()

verb_quantities = {}
for verb, quantity in zip(verbs, quantities):
    if verb in verb_quantities:
        verb_quantities[verb] += quantity
    else:
        verb_quantities[verb] = quantity

for verb, total_quantity in verb_quantities.items():
    print(f"verb '{verb}' 's number if: {total_quantity}")
