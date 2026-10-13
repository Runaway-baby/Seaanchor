import argparse
import math
import os
import sys
from types import SimpleNamespace


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_ROOT = "/root/autodl-tmp/newdata"
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "nautilus")
DEFAULT_CONFIG_FILE = os.path.join(PROJECT_ROOT, "configs", "nautilus.yaml")
DEFAULT_DOMAIN_TEXT_FILE = os.path.join(PROJECT_ROOT, "configs", "domain_text.txt")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="Train Seaanchor (OCIA/RDM) on the NAUTILUS underwater dataset.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config-file", default=DEFAULT_CONFIG_FILE)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-epoch-period", type=int, default=10)
    parser.add_argument("--ims-per-batch", type=int, default=4)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-rank", type=int, default=0)
    parser.add_argument("--dist-url", default="auto")
    parser.add_argument("--offset-start-epoch", type=int, default=20)
    parser.add_argument("--offset-end-epoch", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--weights", default="")
    parser.add_argument(
        "--dinov3-repo",
        default=os.environ.get("DINOV3_REPO", ""),
        help="Local clone of facebookresearch/dinov3 containing hubconf.py.",
    )
    parser.add_argument(
        "--dinov3-weights",
        default=os.environ.get("DINOV3_WEIGHTS", ""),
        help="Approved DINOv3 backbone checkpoint path (not a detector checkpoint).",
    )
    parser.add_argument(
        "--dinov3-model",
        default="dinov3_vitb16",
        help="Official torch.hub backbone entry; default: dinov3_vitb16.",
    )
    parser.add_argument(
        "--strict-images",
        action="store_true",
        help=(
            "Fail while generating merged COCO JSON if a declared image is "
            "missing or unreadable."
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Generate and validate the OCIA-compatible COCO JSON files, then exit.",
    )
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.opts and args.opts[0] == "--":
        args.opts = args.opts[1:]
    return args


def main():
    args = parse_args()
    if args.num_gpus < 1:
        raise ValueError("--num-gpus must be at least 1")
    if args.num_machines < 1:
        raise ValueError("--num-machines must be at least 1")
    if not 0 <= args.machine_rank < args.num_machines:
        raise ValueError("--machine-rank must be within the machine count")
    if min(args.epochs, args.eval_epoch_period, args.ims_per_batch) < 1:
        raise ValueError("Epochs, evaluation period, and batch size must be positive")
    if args.offset_start_epoch < 0 or args.offset_end_epoch <= args.offset_start_epoch:
        raise ValueError("Offset epochs must satisfy 0 <= start < end")
    world_size = args.num_gpus * args.num_machines
    if args.ims_per_batch % world_size != 0:
        raise ValueError(
            "--ims-per-batch={} must be divisible by world size {}".format(
                args.ims_per_batch, world_size
            )
        )
    data_root = os.path.abspath(os.path.expanduser(args.data_root))
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    log_file = open(os.path.join(output_dir, "train.log"), "a", buffering=1, encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    os.environ["NAUTILUS_DATASET_ROOT"] = data_root

    from seaanchor.data.nautilus import (
        TEST_GROUPS,
        TRAIN_GROUPS,
        discover_thing_classes,
        prepare_merged_coco,
        summarize_coco_json,
    )

    merged_paths = prepare_merged_coco(
        data_root,
        strict_images=args.strict_images,
    )
    class_names = discover_thing_classes(data_root)
    if not class_names:
        raise RuntimeError(
            "No COCO categories were found in the merged NAUTILUS annotations."
        )

    train_summary = summarize_coco_json(merged_paths["train"])
    test_summary = summarize_coco_json(merged_paths["test"])
    train_records_with_boxes = train_summary["images_with_boxes"]
    train_boxes = train_summary["annotations"]
    if train_boxes <= 0:
        raise RuntimeError(
            "The merged training COCO JSON contains no valid boxes: {}".format(
                merged_paths["train"]
            )
        )

    total_train_images = train_summary["images"]
    test_images = test_summary["images"]
    if total_train_images <= 0:
        raise RuntimeError(
            "The merged training COCO JSON contains no declared images: {}".format(
                merged_paths["train"]
            )
        )

    iters_per_epoch = max(1, int(math.ceil(float(train_records_with_boxes) / float(args.ims_per_batch))))
    max_iter = max(1, args.epochs * iters_per_epoch)
    eval_period = max(1, args.eval_epoch_period * iters_per_epoch)
    offset_start_iter = max(0, args.offset_start_epoch * iters_per_epoch)
    offset_end_iter = max(offset_start_iter + 1, args.offset_end_epoch * iters_per_epoch)
    offset_end_iter = min(max_iter, offset_end_iter)
    step_iter = int(max_iter * 0.8)
    solver_steps = "[]" if step_iter <= 0 else "[{}]".format(step_iter)

    print("NAUTILUS data root: {}".format(data_root))
    print("Output dir: {}".format(output_dir))
    print("Train groups: {}".format(", ".join(TRAIN_GROUPS)))
    print("Test groups: {}".format(", ".join(TEST_GROUPS)))
    print("Train COCO JSON: {}".format(merged_paths["train"]))
    print("Test COCO JSON: {}".format(merged_paths["test"]))
    print("Detection classes ({}): {}".format(len(class_names), ", ".join(class_names)))
    print("Train images: {}, train images with boxes: {}, test images: {}".format(
        total_train_images, train_records_with_boxes, test_images
    ))
    print("Train records with boxes: {}, train boxes: {}".format(train_records_with_boxes, train_boxes))
    print("Train JSON domain counts: {}".format(train_summary["domain_counts"]))
    print("Test JSON domain counts: {}".format(test_summary["domain_counts"]))
    print("Epochs: {}, ims_per_batch: {}, eval every {} epoch(s)".format(
        args.epochs, args.ims_per_batch, args.eval_epoch_period
    ))
    print(
        "Distributed training: num_gpus={}, num_machines={}, world_size={}, "
        "images_per_gpu={}".format(
            args.num_gpus,
            args.num_machines,
            world_size,
            args.ims_per_batch // world_size,
        )
    )
    print("Iters/epoch: {}, max_iter: {}, eval_period: {}".format(iters_per_epoch, max_iter, eval_period))
    print("Offset optimization: iter {} to {}".format(offset_start_iter, offset_end_iter))
    print("Backbone: DINOv3 {} (backbone only)".format(args.dinov3_model))

    if args.prepare_only:
        print("Merged COCO preparation completed; --prepare-only requested, exiting.")
        return

    script_opts = [
        "OUTPUT_DIR", output_dir,
        "MODEL.ROI_HEADS.NUM_CLASSES", str(len(class_names)),
        "SOLVER.IMS_PER_BATCH", str(args.ims_per_batch),
        "SOLVER.MAX_ITER", str(max_iter),
        "SOLVER.STEPS", solver_steps,
        "SOLVER.CHECKPOINT_PERIOD", str(eval_period),
        "TEST.EVAL_PERIOD", str(eval_period),
        "TEST.EVAL_SAVE_PERIOD", str(eval_period),
        "OFFSET_OPT_INTERVAL", "[{}]".format(offset_start_iter),
        "OFFSET_OPT_ITERS", str(offset_end_iter),
        "TRAIN_TOTAL_EPOCHS", str(args.epochs),
        "TRAIN_ITERS_PER_EPOCH", str(iters_per_epoch),
        "TRAIN_EPOCH_LOG_PERIOD", "1",
        "DOMAIN_TEXT_FILE", DEFAULT_DOMAIN_TEXT_FILE,
        "SOURCE_DOMAIN_TEXT", "no color cast and good exposure",
        "MODEL.DINOV3.MODEL_NAME", args.dinov3_model,
    ]
    if args.dinov3_repo:
        script_opts.extend(["MODEL.DINOV3.REPO_DIR", args.dinov3_repo])
    if args.dinov3_weights:
        script_opts.extend(["MODEL.DINOV3.WEIGHTS", args.dinov3_weights])
    if args.weights:
        script_opts.extend(["MODEL.WEIGHTS", args.weights])
    script_opts.extend(args.opts)

    from detectron2.engine import launch
    from seaanchor.engine.trainer import main as train_main

    train_args = SimpleNamespace(
        config_file=args.config_file,
        resume=args.resume,
        eval_only=args.eval_only,
        opts=script_opts,
        num_gpus=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
    )
    launch(
        train_main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(train_args,),
    )


if __name__ == "__main__":
    main()
