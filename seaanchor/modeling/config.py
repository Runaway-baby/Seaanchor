from pathlib import Path
from detectron2.config import CfgNode as CN


def add_seaanchor_config(cfg):
    cfg.OFFSET_DOMAIN = ''
    cfg.OFFSET_FROZENBN = False
    cfg.OFFSET_DOMAIN_TEXT = ''
    cfg.DOMAIN_TEXT_FILE = str(Path(__file__).resolve().parents[2] / "configs/domain_text.txt")
    cfg.SOURCE_DOMAIN_TEXT = 'sunny day'
    cfg.OFFSET_NAME = 0
    cfg.OFFSET_OPT_INTERVAL = [10]
    cfg.OFFSET_OPT_ITERS = 0
    cfg.TRAIN_TOTAL_EPOCHS = 0
    cfg.TRAIN_ITERS_PER_EPOCH = 0
    cfg.TRAIN_EPOCH_LOG_PERIOD = 1
    cfg.AUG_PROB = 0.5
    cfg.DOMAIN_NAME = ""
    cfg.TEST.EVAL_SAVE_PERIOD = 5000
    cfg.TEST.RUN_SECOND_EVAL = True
    cfg.INPUT.CLIP_WITH_IMG = False
    cfg.INPUT.CLIP_RANDOM_CROPS = False
    cfg.INPUT.IMAGE_JITTER = False
    cfg.INPUT.RANDOM_CROP_SIZE = 224
    # Per-image weight is count(domain)^(-power). Zero preserves the original
    # image-frequency sampler; one makes source domains uniform.
    cfg.DATALOADER.DOMAIN_BALANCE_POWER = 0.0
    cfg.SOLVER.GLOBAL_GRAD_CLIP_NORM = 5.0
    cfg.SOLVER.MAX_CONSECUTIVE_NONFINITE_GRADS = 3
    cfg.SOLVER.ALLOW_TF32 = True
    cfg.MODEL.GLOBAL_GND = False
    cfg.BASE_YAML = "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"
    cfg.MODEL.RENAME = list()
    cfg.MODEL.CLIP_IMAGE_ENCODER_NAME = 'ViT-B/32'
    cfg.MODEL.CLIP_PROMPT_DOMAIN = 'generic'
    cfg.MODEL.BACKBONE.UNFREEZE = ['layer3', 'layer4', 'attnpool']
    cfg.MODEL.DINOV3 = CN()
    cfg.MODEL.DINOV3.REPO_DIR = ''
    cfg.MODEL.DINOV3.WEIGHTS = ''
    cfg.MODEL.DINOV3.MODEL_NAME = 'dinov3_vitb16'
    cfg.MODEL.DINOV3.FREEZE = False
    cfg.MODEL.DINOV3.UNFREEZE_LAST_N_BLOCKS = 4
    cfg.MODEL.DINOV3.OUT_CHANNELS = 1024
    cfg.MODEL.DINOV3.ROI_CHANNELS = 2048
    cfg.MODEL.DINOV3.CLIP_EMBED_DIM = 512
    cfg.MODEL.USE_PROJ = True
    cfg.MODEL.OCIA = CN()
    cfg.MODEL.OCIA.ENABLED = False
    cfg.MODEL.OCIA.EMBED_DIMS = 512
    cfg.MODEL.OCIA.TOKEN_LENGTH = 64
    cfg.MODEL.OCIA.SCALE_INIT = 0.001
    cfg.MODEL.OCIA.RESIDUAL_SCALE_INIT = 0.0
    cfg.MODEL.OCIA.FREQUENCY_RATIO = 0.2
    cfg.MODEL.OCIA.MAX_RESIDUAL_SCALE = 0.1
    cfg.MODEL.OCIA.ROI_CONSISTENCY_WEIGHT = 0.1
    cfg.MODEL.OCIA.LOGIT_CONSISTENCY_WEIGHT = 0.1
    cfg.MODEL.RDM = CN()
    cfg.MODEL.RDM.MULTI_SOURCE = False
    cfg.MODEL.RDM.IMAGE_DOMAIN_LOSS_WEIGHT = 1.0
    cfg.MODEL.RDM.INSTANCE_DOMAIN_LOSS_WEIGHT = 1.0
    cfg.MODEL.RDM.CONTENT_LOSS_WEIGHT = 0.1
    cfg.MODEL.RDM.DIRECTION_LOSS_WEIGHT = 1.0
