# Modeling Scientific Experiment Scenes: Dataset and Model

This repository contains the official implementation and resources for the paper **Modeling Scientific Experiment Scenes: Dataset and Model**.

Building upon the **PhysScene** dataset, this work further investigates two key challenges in scientific experiment scene graph generation: **long-tailed predicate distributions** and the **visual-textual semantic gap**. Based on this analysis, we propose **CM-DPG**, a cross-modal dual-path framework designed to improve relation reasoning and open-vocabulary generalization in physics experiment scenes.

The repository consists of three connected parts:

- **PhysScene Dataset**: the dataset, annotations, category definitions, and data documentation for scientific scene graph generation in physics experiment scenarios.
- **PhysScene Baselines**: baseline implementations, configurations, and evaluation protocols used to benchmark representative scene graph generation methods on PhysScene.
- **CM-DPG Model**: the proposed cross-modal dual-path framework designed to address long-tailed predicate distributions and the visual-textual semantic gap.

The **PhysScene dataset paper** has been accepted by the **ACM International Conference on Multimedia (ACM MM 2026)**.

If you use the PhysScene dataset, please cite:

```bibtex
@article{zou2026physscene_arxiv,
  title={PhysScene: A Scene Graph Dataset for Scientific Visual Reasoning in Physics Experiments},
  author={Zou, Minghao and Zeng, Qingtian and Liu, Shangkun and Meng, Yanda and Yue, Guanghui and Zhao, Baoquan and Saddik, Abdulmotaleb El and Zhou, Wei},
  journal={arXiv preprint arXiv:2606.09368},
  pages={1--6},
  year={2026}
}
@inproceedings{zou2026physscene_acmmm,
  title={PhysScene: A Scene Graph Dataset for Scientific Visual Reasoning in Physics Experiments},
  author={Zou, Minghao and Zeng, Qingtian and Liu, Shangkun and Meng, Yanda and Yue, Guanghui and Zhao, Baoquan and Saddik, Abdulmotaleb El and Zhou, Wei},
  booktitle={Proceedings of the 34th ACM International Conference on Multimedia},
  year={2026},
  note={Accepted}
}
```

## 📥 1. PhysScene dataset

### 📘 Experimental Manual

We adopt the textbook “University Physics Experiments” published by China University of Mining and Technology Press, a widely used “13th Five-Year Plan” higher-education textbook in China.

The manual includes comprehensive descriptions of physics experiments:

- 🧰 **Experimental instrument**
- 🎯 **Experimental purpose**
- 🧪 **Experimental content**
- ▶️ **Operational procedures**
- ⚠️ **Precautions and notes**

The following figure shows the cover of the *University Physics Experiments* manual and a summary page of one representative experiment. These pages illustrate the structured descriptions—purposes, instruments, procedures, and operation flows—that serve as the foundation for the PhysScene dataset design and annotation.

<p align="center">
  <img src="imgs/ins.svg" alt="Dataset Overview" width="900">
</p>

###   🔍 Sample Manual for the Object Density Measurement Experiment

To view the full instructions for all experiments, please refer to the complete manual.

Below is a sample set of pages from the manual.

<p align="center">
  <img src="imgs/1.jpg" alt="Dataset Overview" width="900">
</p>
<p align="center">
  <img src="imgs/2.jpg" alt="Dataset Overview" width="900">
</p>
<p align="center">
  <img src="imgs/3.jpg" alt="Dataset Overview" width="900">
</p>

### 🧪 Experimental Categories

Our dataset covers 4 fundamental physics experiments:

- **Object Density Measurement**
- **Spectrometer-Based Measurement**
- **Surface Tension Measurement**
- **Rigid Body Inertia Determination**

### 🧱 Objects & Relations Overview

<p align="center">
  <img src="imgs/1.png" alt="Dataset Overview" width="600">
</p>
<p align="center">
  <img src="imgs/2.png" alt="Dataset Overview" width="500">
</p>
<p align="center">
  <img src="imgs/3.png" alt="Dataset Overview" width="600">
</p>

### 🎥 Collection Settings
- To ensure high-quality and diverse visual data, we collected images across various laboratory environments with differences in angle, lighting, operator behavior, and experiment configurations.
- A multi-stage data cleaning pipeline was applied to remove blurry, obstructed, out-of-focus, and duplicated samples.
- The figure below provides a visual overview of these components.

<p align="center">
  <img src="imgs/物理实验多样性采集.jpg" alt="Dataset Overview" width="700">
</p>

### ⬇️ Download Links

| Type        | Link |
|-------------|------|
| **Images**      | 👉 [Download Here](https://drive.google.com/file/d/1TFqrm0HJSXRIGNiXXGiQIR3Qp6FmW4M9/view?usp=sharing) |
| **Annotations** | 👉 [Download Here](https://drive.google.com/file/d/15YuzKB8s9RPYN30GzbgEJHAmcseA0aWP/view?usp=drive_link) |

The annotation package contains one JSON file and two CSV mapping files.
- **annotation.json** records the scene graph annotations in an image-level format. For each image, it provides object annotations, including object IDs, object category IDs, and bounding boxes, as well as relation annotations, including subject/object object IDs, relation category IDs, and relation types.
- **object_categories.csv** provides the ID-to-category mapping for object classes.
- **relation_categories.csv** provides the ID-to-category mapping for relation classes.

### 👪 Team
Data collection and annotation were led by Minghao Zou, with contributions from Aihang Jiang, Chenxi Zhao, Rui Cao, Fan Zhao, Xianwei Lu, Xiangdong Long, Wenjing Liu, Canchen Zhang, Wenbo Bai, Ruijun Guo, Rongkun Wang, and Jia Pu. We also extend our sincere gratitude to Yongping Miao, director of the Physics Experiment Center, for his invaluable guidance throughout the entire process.

### 🛡️ Ethics & Data Governance

This dataset involves human participants in both data collection and recording processes. We take ethical considerations, privacy protection, and responsible data usage seriously.

#### License

PhysScene is released under the CC BY-NC 4.0 License. The dataset may be used for non-commercial research and academic purposes with appropriate attribution. Commercial use is prohibited without prior permission. Users are expected to:

* **Use the data responsibly and ethically**
* **Avoid attempts to identify or contact participants**
* **Comply with all applicable laws and institutional guidelines**

#### Human Subjects

Participants were informed about the purpose and procedure of the data collection in advance. Participation was voluntary, and individuals had the right to decline or withdraw at any time.

#### Privacy & Anonymization

All data has been reviewed to minimize personally identifiable information.

#### Ethical Compliance

The data collection process follows standard ethical guidelines for human-subject research.

#### Release Status

The current version of the dataset has been fully released. Additional experimental types and more comprehensive annotations will be continuously updated and made available at this repository.

## 📥  2. Baselines

We benchmarked 11 open-source Scene Graph Generation (SGG) models on the PhysScene dataset.

The `./Baselines` directory provides the official resources for all evaluated models, including:

- project websites or code repositories;
- corresponding papers;
- citation information.

All baseline models were evaluated using the benchmark settings specified in their official implementations.

The `./Baselines/Experimental Settings.md` file documents the details of our experimental setup, including:

- hyperparameter configuration;
- experimental hardware;
- data splitting.

## 📥 3. CM-DPG Model

### Setup

Install PyTorch and project dependencies:

```bash
conda create -n env_name python==3.8
conda activate env_name
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
cd ./cmdpg
python setup.py install
cd ..
```

Download pretrained weights:

```bash
mkdir checkpoints
Download the pretrained checkpoints from the following links:

| Backbone | Checkpoint |
|---|---|
| Swin-T | [Download](https://drive.google.com/file/d/1ePSx_qtXqkh-zZAUpvwltZwUY1Ugrwwv/view?usp=drive_link) |
| Swin-B | [Download](https://drive.google.com/file/d/1-AJp180OmsLrNQICdi6oyoEfC1zDs2Iz/view?usp=drive_link) |

After downloading, the project should be organized as follows:
CM-DPG/
├── checkpoints/
│   ├── groundingdino_swint_ogc.pth
│   └── groundingdino_swinb_cogcoor.pth
├── config/
├── data/
├── datasets/
├── cmdpg/
├── main.py
└── setup.py
```

## Dataset Preparation

### Annotation Conversion

Convert raw PhysScene annotations into this project's scene-graph format (take HOI as an example):

```bash
python tools/convert_phys_scene.py \
  --raw_root /path/to/raw_phys_scene \
  --output_root data/phys_scene
```

The conversion should generate:

```text
data/phys_scene/images/
data/phys_scene/annotations/phys_scene_all.json
data/phys_scene/annotations/phys_scene_dict.json
```

Each item in `phys_scene_all.json` should contain fields like:

```json
{
  "image_id": 1,
  "image_key": "10_13_1.jpg",
  "source_img_id": 1,
  "original_file_name": "10_13_1.jpg",
  "file_name": "images/10_13_1.jpg",
  "width": 640,
  "height": 480,
  "boxes": [[x1, y1, x2, y2], ......],
  "labels": [1, ......],
  "edges": [[0, 1, 3], ......]
}
```

Where:

- `boxes` are object bounding boxes in `[x1, y1, x2, y2]` format.
- `labels` are object category IDs.
- `edges` are relation triplets in `[subject_index, object_index, predicate_id]` format.
- `file_name` is relative to `data/phys_scene/`.

### Train/Test Split

PhysScene train/val/test splits are generated from the image list using a deterministic random seed.

Default settings:

```python
phys_scene_split_seed = 1234
phys_scene_test_ratio = 0.2
phys_scene_val_ratio = 0.1
phys_scene_regenerate_split = True
```

The split generation produces:

```text
data/phys_scene/annotations/phys_scene_train.json
data/phys_scene/annotations/phys_scene_val.json
data/phys_scene/annotations/phys_scene_test.json
```

The split is generated by `util/phys_scene_split.py`. The dataset loader calls this automatically when PhysScene is selected.

After preparation, PhysScene should follow this structure:

```text
data/phys_scene/
+-- images/
|   +-- 10_13_1.jpg
|   +-- 10_13_2.jpg
|   +-- ...
+-- annotations/
    +-- phys_scene_all.json
    +-- phys_scene_train.json
    +-- phys_scene_val.json
    +-- phys_scene_test.json
    +-- phys_scene_dict.json
```

PhysScene has 24 valid HOI predicates. The raw source predicate ids are not contiguous because ids 12 and 23 are unused. Compact them before training:

```bash
python tools/compact_phys_scene_predicates.py \
  --annotation-dir data/phys_scene/annotations \
  --seed 1234 \
  --novel-ratio 0.3
```

After compaction, predicate ids are contiguous from 1 to 24, while 0 remains `[UNK]`. Therefore `num_rln_cat` should be 25.

### Frequency Bias Statistics

OvSGTR can use a VG-style relation frequency-bias file through `rln_freq_bias`.
For PhysScene, generate a dataset-specific file from the training split:

```bash
python tools/generate_phys_scene_stats.py \
  --annotation-file data/phys_scene/annotations/phys_scene_train.json \
  --dict-file data/phys_scene/annotations/phys_scene_dict.json \
  --output-file data/phys_scene/phys_scene_stats.pt
```

The generated file contains:

```text
fg_matrix
pred_dist
obj_classes
rel_classes
```

`pred_dist` follows the VG logic:

```python
pred_dist = log(fg_matrix / fg_matrix.sum(2)[:, :, None] + 1e-3)
```

## Training

### CS-SGG on PhysScene

Use the common CS-SGG config and select PhysScene through command-line arguments:

```bash
python main.py \
  --output_dir ./logs/phys_scene_cs_sgg \
  -c ./config/GroundingDINO_SwinT_Cs-SGG.py \
  --data_path ./data/phys_scene \
  --dataset_file phys_scene \
  --pretrain_model_path ./GroundingDINO/weights/groundingdino_swint_ogc.pth \
  --num_workers 0 \
    --seed 1234 \
  --options \
    batch_size=4 \
    num_rln_cat=25 \
    rln_freq_bias=data/phys_scene/phys_scene_stats.pt \
    phys_scene_split_seed=1234 \
    phys_scene_test_ratio=0.2 \
    phys_scene_val_ratio=0.1 \
    phys_scene_regenerate_split=False
```

By switching the configuration file from Cs-SGG to OvD-SGG or OvR-SGG, training or testing under other forms of supervision can be performed.

For PhysScene OVD training, add:

```bash
--options \
  sg_ovd_mode=True \
  phys_scene_ov_split_seed=1234 \
  phys_scene_ov_novel_ratio=0.3
```

For PhysScene OVR training, add:

```bash
--options \
  sg_ovr_mode=True \
  sgg_mode=ovr \
  phys_scene_ov_split_seed=1234 \
  phys_scene_ov_novel_ratio=0.3
```

## Evaluation

Evaluate a trained PhysScene checkpoint:

```bash
python main.py \
  --eval \
  --resume ./logs/phys_scene_cs_sgg/checkpoint.pth \
  --output_dir ./logs/phys_scene_cs_sgg_eval \
  -c ./config/GroundingDINO_SwinT_Cs-SGG.py \
  --data_path ./data/phys_scene \
  --dataset_file phys_scene \
  --num_workers 0 \
  --seed 1234 \
  --options \
    rln_freq_bias=data/phys_scene/phys_scene_stats.pt \
    phys_scene_split_seed=1234 \
    phys_scene_test_ratio=0.2 \
    phys_scene_val_ratio=0.1 \
    phys_scene_regenerate_split=False
```
## Acknowledgement

We thank:

- [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [OvSGTR](https://github.com/gpt4vision/OvSGTR)

for their awesome open-source codes and models.
