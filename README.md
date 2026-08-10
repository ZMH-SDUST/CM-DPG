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

### (1) Setup

Install PyTorch and project dependencies:

```bash
conda create -n env_name python==3.8
conda activate env_name
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
cd ./cmdpg
python setup.py install
cd ..
mkdir checkpoints
```
Download the pretrained checkpoints from the following links:

| Backbone | Checkpoint |
|---|---|
| Swin-T | [Download](https://drive.google.com/file/d/1ePSx_qtXqkh-zZAUpvwltZwUY1Ugrwwv/view?usp=drive_link) |
| Swin-B | [Download](https://drive.google.com/file/d/1-AJp180OmsLrNQICdi6oyoEfC1zDs2Iz/view?usp=drive_link) |

After downloading, the project should be organized as follows:
```bash
CM-DPG/
├── checkpoints/
│   ├── swint_ogc.pth
│   └── swinb_cogcoor.pth
├── config/
├── datasets/
├── cmdpg/
├── main.py
└── setup.py
```

### (2) Dataset Preparation

```bash
mkdir data
```
#### Visual Genome

Download the processed Visual Genome data from [here](https://huggingface.co/JosephZ/OvSGTR/resolve/main/vg_data.tar.gz).

After downloading, extract `vg_data.tar.gz` and place the extracted `visual_genome/` folder under `data/`.

The Visual Genome directory should be organized as follows:

```text
data/
└── visual_genome/
    ├── VG_100K/
    ├── stanford_filtered/
    │   ├── VG-SGG.h5
    │   ├── VG-SGG-dicts.json
    │   └── image_data.json
    ├── final_mixed_train_no_coco.json
    ├── vg_stats.pt
    └── zeroshot_triplet.pytorch
```

#### PhysScene

Place the PhysScene dataset under `data/PhysScene/`.

The PhysScene directory should be organized as follows:

```text
data/
└── PhysScene/
    ├── image/
    └── annotation/
        ├── annotation.json
        ├── object_categories.csv
        └── relation_categories.csv
```

After preparing both datasets, the final `data/` directory should be organized as follows:

```text
CM-DPG/
└── data/
    ├── PhysScene/
    │   ├── image/
    │   └── annotation/
    │       ├── annotation.json
    │       ├── object_categories.csv
    │       └── relation_categories.csv
    └── visual_genome/
        ├── VG_100K/
        ├── stanford_filtered/
        │   ├── VG-SGG.h5
        │   ├── VG-SGG-dicts.json
        │   └── image_data.json
        ├── final_mixed_train_no_coco.json
        ├── vg_stats.pt
        └── zeroshot_triplet.pytorch
```

### (3) Experimental Settings and Reproducibility

#### Configuration Files

We provide complete configuration files for two datasets, Visual Genome and PhysScene, under three supervision settings: closed-set scene graph generation (CS-SGG), open-vocabulary object detection scene graph generation (OVD-SGG), and open-vocabulary relation scene graph generation (OVR-SGG).

The configuration files are located in the `config/` directory and can be used directly for training and evaluation.

```text
config/
├── SwinT_vg_full.py
├── SwinT_vg_ovd.py
├── SwinT_vg_ovr.py
├── SwinB_vg_full.py
├── SwinB_vg_ovd.py
├── SwinB_vg_ovr.py
├── SwinT_phys_full.py
├── SwinT_phys_ovd.py
├── SwinT_phys_ovr.py
├── SwinB_phys_full.py
├── SwinB_phys_ovd.py
└── SwinB_phys_ovr.py
```

Here, `full` corresponds to the CS-SGG setting, `ovd` corresponds to the OVD-SGG setting, and `ovr` corresponds to the OVR-SGG setting.

#### Dataset Splits

* ##### Visual Genome

For Visual Genome, we follow the processed VG150 split used in OvSGTR. Since part of the original VG150 test images may have been seen during GroundingDINO pretraining, we use the cleaned split field `split_GLIPunseen` in `VG-SGG.h5`.

The split is controlled by:

```python
vg_roidb_key = "split_GLIPunseen"
```

The dataset split is defined as follows:

| Split | Protocol |
|---|---|
| Train | Images with `split_GLIPunseen == 0`, excluding the first `num_val_im` images |
| Validation | The first `num_val_im` images from `split_GLIPunseen == 0`; by default, `num_val_im = 5000` |
| Test | Images with `split_GLIPunseen == 2` |

In the CS-SGG setting, all object categories and relation categories are used for training and evaluation.

In the OVD-SGG setting, approximately 30% of the object categories are treated as unseen object classes. During training, annotations are restricted to base object categories, while validation and test keep the full object label space. The unseen object category IDs used in OVD-SGG are:
```text
[9, 10, 11, 20, 22, 23, 26, 27, 28, 37, 38, 40, 41, 45, 52, 53, 54, 57, 58, 60, 61, 64, 66, 74, 78, 87, 91, 95, 97, 99, 111, 112, 113, 115, 121, 124, 126, 127, 134, 135, 136, 144, 145, 149, 150]
```

In the OVR-SGG setting, approximately 30% of the relation categories are treated as unseen relation classes. During training, annotations containing unseen relation categories are removed, while validation and test keep the full relation label space. For Visual Genome, the unseen relation category IDs used in OVR-SGG are:
```text
[7, 9, 14, 16, 22, 30, 31, 34, 36, 37, 38, 39, 45, 46, 49]
```

The exact base/unseen label sets follow the configuration and label mapping files provided with the project.

* ##### PhysScene

For PhysScene, the image-level annotations are sorted by file name and then split with a fixed random seed.

```python
split_seed = 42
```

The default split ratio is:

| Split | Ratio |
|---|---|
| Train | 70% |
| Validation | 10% |
| Test | 20% |

In the CS-SGG setting, all object categories and relation categories are used for training and evaluation.

In the OVD-SGG setting, approximately 30% of the object categories are treated as unseen object classes. During training, objects belonging to unseen categories are removed from the training annotations, while validation and test keep the full object label space. The unseen object category IDs used in OVD-SGG are:
```text
[2, 21, 24, 27, 28, 30, 31, 32, 33, 34]
```

In the OVR-SGG setting, approximately 30% of the relation categories are treated as unseen relation classes. During training, relations belonging to unseen categories are removed from the training annotations, while validation and test keep the full relation label space. The unseen relation category IDs used in OVR-SGG are:
```text
[4, 5, 9, 10, 12, 13, 16, 25, 28, 30, 35, 39]
```

The exact base/unseen label sets are defined in the configuration files and the label mapping files released with PhysScene.

We also provide video-level split lists for PhysScene to support reproducible evaluation under video-level separation:

| Split List | Link |
|---|---|
| Video-level split list 1 | [Download](https://drive.google.com/file/d/1ajiVJV5ckh43GZWTkC1EQFBMaJmg233W/view?usp=drive_link) |
| Video-level split list 2 | [Download](https://drive.google.com/file/d/1Q38ds7et0LjlFyLv8clHX9gfLC0rldX4/view?usp=drive_link) |

**The dataset loading and split settings for both Visual Genome and PhysScene can be accessed in the files under the `datasets/` directory.**

### (4) Mapping Between Paper Modules and Code
The core implementation files of CM-DPG are located under the model and dataset modules. The following table maps the main paper components to their implementation.

| Paper module / equation | Implementation |
|---|---|
| Text self-attention in D-SGOD | `transformer.py`, `TransformerEncoder.text_layers` |
| C-MME | `transformer.py`, `BiAttentionBlock` / `TransformerEncoder.fusion_layers` |
| Deformable visual self-attention | `transformer.py`, `DeformableTransformerEncoderLayer` |
| D-SGOD decoder | `transformer.py`, `TransformerDecoder` |
| G-VRD module definition | `groundingdino.py`, `GroundingDINO.__init__` |
| G-VRD visual branch / Eq. (10) | `losses.py`, `SetCriterion.build_pair_relation_feature`; `graph_infer.py`, `graph_infer` |
| G-VRD geometric branch | `spatial.py`, `compute_spatial_encodings`; `groundingdino.py`, `spatial_head` / `spatial_head_ovr` |
| Visual-geometric fusion | `losses.py`, `SetCriterion.loss_edges`; `graph_infer.py`, `graph_infer` |
| Open-vocabulary relation-text similarity | `losses.py`, `SetCriterion.loss_edges`; `graph_infer.py`, `graph_infer` |
| G-VARR adaptive predicate calibration | `losses.py`, `SetCriterion.gvarr_calibrate` |
| OVD/OVR label filtering for VG | `datasets/vg.py`, `VGDataset` / `load_graphs` |
| OVD/OVR label filtering for PhysScene | `datasets/phys_scene.py`, `PhysSceneDataset._parse_record` |

**Module definitions and their corresponding implementations are also annotated in the code and described in the paper to facilitate understanding and reproducibility.**

### (5) Training

#### CS-SGG on PhysScene

Use the common CS-SGG config and select PhysScene through command-line arguments:

```bash
python main.py \
 --output_dir
./logs/phys_scene_swinb_full
-c
./config/SwinB_phys_full.py
--data_path
./data
--dataset_file
phys_scene
--pretrain_model_path
./checkpoints/swinb_cogcoor.pth
--num_workers
4
--options
dn_scalar=100
embed_init_tgt=TRUE
dn_label_coef=1.0
dn_bbox_coef=1.0
use_ema=False
dn_box_noise_scale=1.0
eval_before_train=False
batch_size=4
```

#### OVD-SGG on PhysScene

Use the common OVD-SGG config and select PhysScene through command-line arguments:

```bash
python main.py \
--output_dir
./logs/phys_scene_swinb_ovd
-c
./config/SwinB_phys_ovd.py
--data_path
./data
--dataset_file
phys_scene
--pretrain_model_path
./checkpoints/swinb_cogcoor.pth
--num_workers
4
--options
dn_scalar=100
embed_init_tgt=TRUE
dn_label_coef=1.0
dn_bbox_coef=1.0
use_ema=False
dn_box_noise_scale=1.0
eval_before_train=False
batch_size=4
```

#### OVR-SGG on PhysScene

```bash
python main.py \
--output_dir
./logs/phys_scene_swinb_ovr
-c
./config/SwinB_phys_ovr.py
--data_path
./data
--dataset_file
phys_scene
--pretrain_model_path
./checkpoints/swinb_cogcoor.pth
--num_workers
4
--options
dn_scalar=100
embed_init_tgt=TRUE
dn_label_coef=1.0
dn_bbox_coef=1.0
use_ema=False
dn_box_noise_scale=1.0
eval_before_train=False
batch_size=4
```
### (6) Inference

#### CS-SGG on PhysScene

Use the common CS-SGG config and select PhysScene through command-line arguments:

```bash
python main.py \
 --output_dir
./logs/phys_scene_swinb_full_eval
-c
./config/SwinB_phys_full.py
--data_path
./data
--eval
--resume
./logs/phys_scene_swinb_full/checkpoint.pth
--dataset_file
phys_scene
--num_workers
0
--options
dn_scalar=100
embed_init_tgt=TRUE
dn_label_coef=1.0
dn_bbox_coef=1.0
use_ema=False
use_test_set=True
```

#### OVD-SGG on PhysScene

Use the common OVD-SGG config and select PhysScene through command-line arguments:

```bash
python main.py \
--output_dir
./logs/phys_scene_swinb_ovd_eval
-c
./config/SwinB_phys_ovd.py
--data_path
./data
--eval
--resume
./logs/phys_scene_swinb_ovd/checkpoint.pth
--dataset_file
phys_scene
--num_workers
0
--options
dn_scalar=100
embed_init_tgt=TRUE
dn_label_coef=1.0
dn_bbox_coef=1.0
use_ema=False
use_test_set=True
```

#### OVR-SGG on PhysScene

```bash
python main.py \
--output_dir
./logs/phys_scene_swinb_ovr_eval
-c
./config/SwinB_phys_ovr.py
--data_path
./data
--eval
--resume
./logs/phys_scene_swinb_ovr/checkpoint.pth
--dataset_file
phys_scene
--num_workers
0
--options
dn_scalar=100
embed_init_tgt=TRUE
dn_label_coef=1.0
dn_bbox_coef=1.0
use_ema=False
use_test_set=True
```

## Acknowledgement

We thank:

- [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [OvSGTR](https://github.com/gpt4vision/OvSGTR)

for their awesome open-source codes and models.

If you use the CM-DPG model or code, please cite:

```bibtex
@article{zou2026cmdpg,
  title={Modeling Scientific Experiment Scenes: Dataset and Model},
  author={Zou, Minghao and Zeng, Qingtian and Liu, Shangkun and Meng, Yanda and Yue, Guanghui and Zhao, Baoquan and Saddik, Abdulmotaleb El and Zhou, Wei},
  journal={arXiv preprint arXiv:2608.02892},
  year={2026}
}
