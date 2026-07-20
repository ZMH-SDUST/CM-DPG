# 📥 1. PhysScene dataset

## 📘 Experimental Manual

**We adopt the textbook “University Physics Experiments” published by China University of Mining and Technology Press, a widely used “13th Five-Year Plan” higher-education textbook in China.**

**The manual includes comprehensive descriptions of physics experiments:**

- 🧰 **Experimental instrument**
- 🎯 **Experimental purpose**
- 🧪 **Experimental content**
- ▶️ **Operational procedures**
- ⚠️ **Precautions and notes**

**The following figure shows the cover of the *University Physics Experiments* manual and a summary page of one representative experiment. These pages illustrate the structured descriptions—purposes, instruments, procedures, and operation flows—that serve as the foundation for the PhysScene dataset design and annotation.**

<p align="center">
  <img src="imgs/ins.svg" alt="Dataset Overview" width="900">
</p>

## 🔍 Sample Manual for the Object Density Measurement Experiment

**To view the full instructions for all experiments, please refer to the complete manual.**

**Below is a sample set of pages from the manual.**

<p align="center">
  <img src="imgs/1.jpg" alt="Dataset Overview" width="900">
</p>
<p align="center">
  <img src="imgs/2.jpg" alt="Dataset Overview" width="900">
</p>
<p align="center">
  <img src="imgs/3.jpg" alt="Dataset Overview" width="900">
</p>

## 🧪 Experimental Categories

**Our dataset covers 4 fundamental physics experiments:**

- **Object Density Measurement**
- **Spectrometer-Based Measurement**
- **Surface Tension Measurement**
- **Rigid Body Inertia Determination**

## 🧱 Objects & Relations Overview

<p align="center">
  <img src="imgs/1.png" alt="Dataset Overview" width="600">
</p>
<p align="center">
  <img src="imgs/2.png" alt="Dataset Overview" width="500">
</p>
<p align="center">
  <img src="imgs/3.png" alt="Dataset Overview" width="600">
</p>

## 🎥 Collection Settings
- **To ensure high-quality and diverse visual data, we collected images across various laboratory environments with differences in angle, lighting, operator behavior, and experiment configurations.**
- **A multi-stage data cleaning pipeline was applied to remove blurry, obstructed, out-of-focus, and duplicated samples.**
- **The figure below provides a visual overview of these components.**

<p align="center">
  <img src="imgs/物理实验多样性采集.jpg" alt="Dataset Overview" width="700">
</p>


## ⬇️ Download Links

| Type        | Link |
|-------------|------|
| **Images**      | 👉 [Download Here](https://drive.google.com/file/d/1TFqrm0HJSXRIGNiXXGiQIR3Qp6FmW4M9/view?usp=sharing) |
| **Annotations** | 👉 [Download Here](https://drive.google.com/file/d/1yhih6c3b5LQTz54PTSNnML-18Myx7Psu/view?usp=drive_link) |

## 👪 Team
**Data collection and annotation were led by Minghao Zou, with contributions from Aihang Jiang, Chenxi Zhao, Rui Cao, Fan Zhao, Xianwei Lu, Xiangdong Long, Wenjing Liu, Canchen Zhang, Wenbo Bai, Ruijun Guo, Rongkun Wang, and Jia Pu.**

**We also extend our sincere gratitude to Yongping Miao, director of the Physics Experiment Center, for his invaluable guidance throughout the entire process.**

- # 📥  Baselines

**We benchmarked 11 open-source Scene Graph Generation (SGG) models on the PhysScene dataset.**

**The `./Baselines` directory provides the official resources for all evaluated models, including:**

- **project websites or code repositories;**
- **corresponding papers;**
- **citation information.**

**All baseline models were evaluated using the benchmark settings specified in their official implementations.**

**The `Experimental Settings.md` file documents the details of our experimental setup, including:**

- **hyperparameter configuration;**
- **experimental hardware;**
- **data splitting details for all supervised settings.**

- # 📥  CM-DPG model

