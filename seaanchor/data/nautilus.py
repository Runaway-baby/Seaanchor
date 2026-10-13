# -*- coding: utf-8 -*-
"""NAUTILUS merged-COCO generation and Detectron2 registration.

This module intentionally follows OCIA' NAUTILUS data contract:

* source annotations live in ``NAUTILUS-dataset_classified_annotations``;
* images live in ``NAUTILUS-dataset_classified2/<domain>``;
* four source domains are merged into one training COCO JSON;
* two held-out domains are merged into one test COCO JSON;
* dataset records are created exclusively from the merged JSON ``images``
  entries. The image directory is never scanned to create extra records.
"""

import copy
import json
import math
import os

from PIL import Image


IMAGE_DIR = "NAUTILUS-dataset_classified2"
ANNOTATION_DIR = "NAUTILUS-dataset_classified_annotations"
MERGED_ANNOTATION_DIR = "annotations_ocia"
DEFAULT_DATASET_ROOT = "/root/autodl-tmp/newdata"

TRAIN_GROUPS = ("1A2A", "1A2C", "1B2A", "1B2C")
TEST_GROUPS = ("1C2A", "1C2C")
ALL_GROUPS = TRAIN_GROUPS + TEST_GROUPS

TRAIN_JSON_NAME = "nautilus_train_1A2A_1A2C_1B2A_1B2C.json"
TEST_JSON_NAME = "nautilus_test_1C2A_1C2C.json"

DOMAIN_TEXT_BY_GROUP = {
    "1A2A": "no color cast and good exposure",
    "1A2C": "no color cast and poor exposure",
    "1B2A": "moderate color cast and good exposure",
    "1B2C": "moderate color cast and poor exposure",
    "1C2A": "severe color cast and good exposure",
    "1C2C": "severe color cast and poor exposure",
}

# Keep the dataset/evaluation IDs identical to OCIA. Natural-language CLIP
# aliases are applied later and never alter this annotation order.
NAUTILUS_CLASSES = (
    "corals", "fish", "cuttlefish", "diver", "turtle", "jellyfish",
    "starfish", "echinus", "scallop", "mollusk", "plants", "reptiles",
    "arthropoda", "human", "garbage", "gbottle", "pbag", "stingray",
    "ruins", "net", "shark", "puffin", "robots", "holothurian", "glove",
    "tire", "pbottle", "can", "Mask", "plastic", "misc", "penguin",
    "rod", "metal", "cellphone", "electronics", "small_fish", "crab",
    "shrimp", "sunglasses",
)


_EXIF_ORIENTATION_TAG = 274


def _display_image_size(image_path):
    """Return the dimensions Detectron2 sees after applying JPEG EXIF rotation."""
    with Image.open(image_path) as image:
        width, height = image.size
        try:
            orientation = image.getexif().get(_EXIF_ORIENTATION_TAG)
        except (AttributeError, TypeError, ValueError):
            orientation = None
    # EXIF orientations 5--8 transpose the image axes. Detectron2 applies
    # this orientation in read_image(), so COCO metadata must describe the
    # displayed tensor rather than the encoded JPEG raster.
    if orientation in {5, 6, 7, 8}:
        width, height = height, width
    return int(width), int(height)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _dump_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False)


def _source_annotation_path(dataset_root, group):
    return os.path.join(
        dataset_root,
        ANNOTATION_DIR,
        "{}_annotations.json".format(group),
    )


def merged_annotation_paths(dataset_root, output_dir=None):
    output_dir = output_dir or os.path.join(dataset_root, MERGED_ANNOTATION_DIR)
    return {
        "train": os.path.join(output_dir, TRAIN_JSON_NAME),
        "test": os.path.join(output_dir, TEST_JSON_NAME),
    }


def _collect_categories(dataset_root, groups):
    categories_by_id = {}
    for group in groups:
        annotation_path = _source_annotation_path(dataset_root, group)
        data = _load_json(annotation_path)
        for category in data.get("categories", []):
            category_id = category["id"]
            if category_id in categories_by_id:
                previous = categories_by_id[category_id]
                if previous.get("name") != category.get("name"):
                    raise ValueError(
                        "Category id {} has inconsistent names: {} vs {}".format(
                            category_id,
                            previous.get("name"),
                            category.get("name"),
                        )
                    )
            else:
                categories_by_id[category_id] = copy.deepcopy(category)
    return [categories_by_id[key] for key in sorted(categories_by_id)]


def _validate_categories(categories):
    sorted_categories = sorted(categories, key=lambda item: item["id"])
    category_ids = tuple(category["id"] for category in sorted_categories)
    names = tuple(
        str(category.get("name", "")).strip()
        for category in sorted_categories
    )
    expected_ids = tuple(range(1, len(NAUTILUS_CLASSES) + 1))
    if category_ids != expected_ids or names != NAUTILUS_CLASSES:
        raise ValueError(
            "NAUTILUS categories differ from OCIA. Expected ids/names {} / {}, "
            "got {} / {}".format(
                expected_ids, NAUTILUS_CLASSES, category_ids, names
            )
        )


def _merge_groups_to_coco(
    dataset_root,
    groups,
    output_file,
    strict_images=False,
):
    """Create the same merged COCO structure used by OCIA."""
    image_root = os.path.join(dataset_root, IMAGE_DIR)
    categories = _collect_categories(dataset_root, groups)
    _validate_categories(categories)

    images = []
    annotations = []
    next_image_id = 1
    next_annotation_id = 1
    missing_images = 0
    unreadable_images = 0
    corrected_image_sizes = 0
    skipped_annotations = 0
    invalid_annotations = 0

    for group in groups:
        data = _load_json(_source_annotation_path(dataset_root, group))
        image_id_map = {}

        # Iterate over JSON declarations only. Files present on disk but absent
        # from this list are deliberately invisible to training/evaluation.
        for image in data.get("images", []):
            file_name = str(image["file_name"])
            image_path = os.path.join(image_root, group, file_name)
            if not os.path.isfile(image_path):
                missing_images += 1
                if strict_images:
                    raise FileNotFoundError(image_path)
                continue

            try:
                actual_width, actual_height = _display_image_size(image_path)
            except Exception as error:
                unreadable_images += 1
                if strict_images:
                    raise ValueError(
                        "Cannot read image dimensions for {}: {}".format(
                            image_path, error
                        )
                    ) from error
                continue

            merged_image = copy.deepcopy(image)
            try:
                declared_size = (
                    int(merged_image.get("width")),
                    int(merged_image.get("height")),
                )
            except (TypeError, ValueError):
                declared_size = None
            actual_size = (actual_width, actual_height)
            if declared_size != actual_size:
                corrected_image_sizes += 1
            # Never trust stale/raw JPEG dimensions from source annotations.
            # DatasetMapper checks these fields against the EXIF-oriented image.
            merged_image["width"] = actual_width
            merged_image["height"] = actual_height
            source_image_id = merged_image["id"]
            merged_image["id"] = next_image_id
            merged_image["file_name"] = "{}/{}".format(
                group, file_name.replace("\\", "/")
            )
            merged_image["source_split"] = group
            merged_image["source_image_id"] = source_image_id
            images.append(merged_image)
            image_id_map[source_image_id] = next_image_id
            next_image_id += 1

        for annotation in data.get("annotations", []):
            source_image_id = annotation["image_id"]
            if source_image_id not in image_id_map:
                skipped_annotations += 1
                continue

            bbox = annotation.get("bbox", [])
            if (
                len(bbox) != 4
                or not all(math.isfinite(float(value)) for value in bbox)
                or float(bbox[2]) <= 0
                or float(bbox[3]) <= 0
            ):
                invalid_annotations += 1
                continue

            merged_annotation = copy.deepcopy(annotation)
            merged_annotation["id"] = next_annotation_id
            merged_annotation["image_id"] = image_id_map[source_image_id]
            merged_annotation["area"] = float(
                merged_annotation.get("area", bbox[2] * bbox[3])
            )
            merged_annotation["iscrowd"] = int(
                merged_annotation.get("iscrowd", 0)
            )
            annotations.append(merged_annotation)
            next_annotation_id += 1

    merged = {
        "info": {
            "description": "Merged NAUTILUS newdata annotations for OCIA",
            "source_root": str(dataset_root),
            "splits": list(groups),
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    _dump_json(merged, output_file)
    statistics = {
        "path": output_file,
        "images": len(images),
        "annotations": len(annotations),
        "categories": len(categories),
        "missing_images": missing_images,
        "unreadable_images": unreadable_images,
        "corrected_image_sizes": corrected_image_sizes,
        "skipped_annotations": skipped_annotations,
        "invalid_annotations": invalid_annotations,
    }
    print(
        "Wrote {path}: images={images}, annotations={annotations}, "
        "categories={categories}, missing_images={missing_images}, "
        "unreadable_images={unreadable_images}, "
        "corrected_image_sizes={corrected_image_sizes}, "
        "skipped_annotations={skipped_annotations}, "
        "invalid_annotations={invalid_annotations}".format(**statistics)
    )
    return statistics


def prepare_merged_coco(dataset_root, output_dir=None, strict_images=False):
    """Generate OCIA-compatible train/test COCO files and return their paths."""
    dataset_root = os.path.abspath(os.path.expanduser(dataset_root))
    paths = merged_annotation_paths(dataset_root, output_dir)
    _merge_groups_to_coco(
        dataset_root,
        TRAIN_GROUPS,
        paths["train"],
        strict_images=strict_images,
    )
    _merge_groups_to_coco(
        dataset_root,
        TEST_GROUPS,
        paths["test"],
        strict_images=strict_images,
    )
    # OCIA' converter also emits per-target-domain test JSON files.
    for group in TEST_GROUPS:
        split_path = os.path.join(
            os.path.dirname(paths["test"]),
            "nautilus_test_{}.json".format(group),
        )
        _merge_groups_to_coco(
            dataset_root,
            (group,),
            split_path,
            strict_images=strict_images,
        )
    return paths


def discover_thing_classes(dataset_root):
    """Read ID-ordered classes from COCO JSON, falling back to OCIA metadata."""
    paths = merged_annotation_paths(dataset_root)
    candidates = [
        paths["train"],
        _source_annotation_path(dataset_root, TRAIN_GROUPS[0]),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        categories = _load_json(path).get("categories", [])
        _validate_categories(categories)
        return [
            str(category["name"]).strip()
            for category in sorted(categories, key=lambda item: item["id"])
        ]
    return list(NAUTILUS_CLASSES)


def summarize_coco_json(json_file):
    data = _load_json(json_file)
    annotated_image_ids = {
        annotation["image_id"]
        for annotation in data.get("annotations", [])
        if int(annotation.get("iscrowd", 0)) == 0
    }
    domain_counts = {}
    for image in data.get("images", []):
        group = image.get("source_split")
        if group is None:
            group = str(image.get("file_name", "")).replace("\\", "/").split("/", 1)[0]
        domain_counts[group] = domain_counts.get(group, 0) + 1
    return {
        "images": len(data.get("images", [])),
        "images_with_boxes": sum(
            1 for image in data.get("images", []) if image["id"] in annotated_image_ids
        ),
        "annotations": len(data.get("annotations", [])),
        "categories": len(data.get("categories", [])),
        "domain_counts": domain_counts,
    }


def _load_registered_nautilus_coco(json_file, image_root, dataset_name):
    from detectron2.data.datasets import load_coco_json

    records = load_coco_json(json_file, image_root, dataset_name)
    for record in records:
        relative_path = os.path.relpath(record["file_name"], image_root)
        group = relative_path.replace("\\", "/").split("/", 1)[0]
        if group not in DOMAIN_TEXT_BY_GROUP:
            raise ValueError(
                "COCO image '{}' does not start with a known NAUTILUS domain".format(
                    relative_path
                )
            )
        record["domain_group"] = group
        record["domain_text"] = DOMAIN_TEXT_BY_GROUP[group]
    return records


def _register_coco_split(name, json_file, image_root):
    from detectron2.data import DatasetCatalog, MetadataCatalog

    def load_split(
        json_file=json_file,
        image_root=image_root,
        dataset_name=name,
    ):
        return _load_registered_nautilus_coco(
            json_file, image_root, dataset_name
        )

    DatasetCatalog.register(name, load_split)
    MetadataCatalog.get(name).set(
        json_file=json_file,
        image_root=image_root,
        evaluator_type="coco",
        thing_classes=list(NAUTILUS_CLASSES),
        thing_dataset_id_to_contiguous_id={
            dataset_id: dataset_id - 1 for dataset_id in range(1, 41)
        },
    )


def register_dataset(datasets_root):
    del datasets_root
    dataset_root = os.path.abspath(os.path.expanduser(
        os.getenv("NAUTILUS_DATASET_ROOT", DEFAULT_DATASET_ROOT)
    ))
    image_root = os.path.join(dataset_root, IMAGE_DIR)
    paths = merged_annotation_paths(dataset_root)
    _register_coco_split("nautilus_train", paths["train"], image_root)
    _register_coco_split("nautilus_test", paths["test"], image_root)
