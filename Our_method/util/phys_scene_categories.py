import random


PHYS_SCENE_OBJECT_CATEGORIES = [
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


PHYS_SCENE_PREDICATE_CATEGORIES = [
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


def object_names_for_count(num_object_classes):
    count = num_object_classes - 1
    if count > len(PHYS_SCENE_OBJECT_CATEGORIES):
        raise ValueError(f"PhysScene object list has {len(PHYS_SCENE_OBJECT_CATEGORIES)} names, need {count}.")
    return PHYS_SCENE_OBJECT_CATEGORIES[:count]


def predicate_names_for_count(num_predicate_classes):
    count = num_predicate_classes - 1
    if count > len(PHYS_SCENE_PREDICATE_CATEGORIES):
        raise ValueError(f"PhysScene predicate list has {len(PHYS_SCENE_PREDICATE_CATEGORIES)} names, need {count}.")
    return PHYS_SCENE_PREDICATE_CATEGORIES[:count]


def predicate_names_from_source_ids(source_ids):
    names = []
    for source_id in source_ids:
        idx = int(source_id) - 1
        if idx < 0 or idx >= len(PHYS_SCENE_PREDICATE_CATEGORIES):
            raise ValueError(f"Invalid PhysScene source predicate id: {source_id}.")
        names.append(PHYS_SCENE_PREDICATE_CATEGORIES[idx])
    return names


def random_base_novel_split(names, seed=1234, novel_ratio=0.2):
    names = list(names)
    if not names:
        return [], []
    rng = random.Random(int(seed))
    shuffled = list(names)
    rng.shuffle(shuffled)
    novel_count = max(1, int(round(len(shuffled) * float(novel_ratio))))
    novel_names = set(shuffled[:novel_count])
    base = [name for name in names if name not in novel_names]
    novel = [name for name in names if name in novel_names]
    return base, novel


def build_phys_scene_open_vocab_splits(
    num_object_classes,
    num_predicate_classes,
    seed=1234,
    novel_ratio=0.2,
    predicate_names=None,
):
    object_names = object_names_for_count(num_object_classes)
    predicate_names = predicate_names or predicate_names_for_count(num_predicate_classes)
    base_objects, novel_objects = random_base_novel_split(object_names, seed=seed, novel_ratio=novel_ratio)
    base_predicates, novel_predicates = random_base_novel_split(predicate_names, seed=seed, novel_ratio=novel_ratio)

    object_name_to_id = {name: idx + 1 for idx, name in enumerate(object_names)}
    predicate_name_to_id = {name: idx + 1 for idx, name in enumerate(predicate_names)}

    return {
        "base_objects": base_objects,
        "novel_objects": novel_objects,
        "base_object_ids": [object_name_to_id[name] for name in base_objects],
        "novel_object_ids": [object_name_to_id[name] for name in novel_objects],
        "base_predicates": base_predicates,
        "novel_predicates": novel_predicates,
        "base_predicate_ids": [predicate_name_to_id[name] for name in base_predicates],
        "novel_predicate_ids": [predicate_name_to_id[name] for name in novel_predicates],
    }
