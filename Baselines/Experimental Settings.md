# Experimental Settings

## Hyperparameter Configuration

For all experiments, we used the default parameter settings provided by the official implementations of the corresponding baseline models. No special hyperparameter tuning was conducted.

For detailed model configurations, please refer to the related code repositories and paper links provided in this project.

## Experimental Hardware

All experiments were conducted on the following hardware:

- 4 NVIDIA RTX 3090 GPUs

## Data Splitting Strategy

We used `random.shuffle` to randomly shuffle the data before splitting. The random seed was fixed as:

```text
seed = 1234
```

### 1. CS-SGG Setting

For the **CS-SGG** setting, we performed a image-level random split.

We also provided a video-level random split, please refer to **trainval_video.txt** and **test_video.txt**.

### 2. OvD-SGG Setting

For the **OvD-SGG** setting, we performed a random split based on the object type list.

The complete object list is:

```text
[
  "Weight Box",
    "Hand",
    "Optical Axis",
    "Beaker",
    "Fixed Pulley",
    "Paper",
    "Zero Adjustment Knob",
    "Disk",
    "Vernier Caliper",
    "Base Screw_v2",
    "Tweezers",
    "Circular Level",
    "Regular Object",
    "Eyepiece Focusing Knob",
    "Weight_v1",
    "Ring",
    "Rotational Inertia Apparatus",
    "Adjustment Knob",
    "Electronic Balance",
    "Objective Focusing Knob",
    "Person",
    "Specimen Stage_v2",
    "Lifting Screw",
    "Rope",
    "Irregular Object",
    "Glass Dish",
    "Weight_v2",
    "Fixed Stage_v1",
    "Triangular Prism",
    "Hanging Ring",
    "Stand Screw",
    "Vernier Dial",
    "Base Screw_v1",
    "Pen",
]
```

### 3. OvR-SGG Setting

For the **OvR-SGG** setting, we performed a random split based on the relation type list.

The complete relation list is:

```text
[
  "Transparent",
    "Pinch",
    "On",
    "Rotate_v1",
    "Plastic",
    "Clip",
    "Near",
    "Measure",
    "Lift",
    "Black",
    "Read",
    "Frosted",
    "Record",
    "Glass",
    "Suspend",
    "Hold",
    "Adjust",
    "Inside",
    "Push-Pull",
    "Blurry",
    "Unpack",
    "Reflective",
    "White",
    "Touch",
    "Fix",
    "Mixed Color",
    "Grab",
    "In Front Of",
    "Move",
    "Drag",
    "Fold",
    "Tap",
    "Part Of",
    "Press",
    "Rotate_v2",
    "Observe",
    "Pave Out",
    "Metallic",
    "Measure",
]
```
