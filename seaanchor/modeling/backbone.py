import os

import torch
import torch.nn as nn
import torch.nn.functional as F


from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec




@BACKBONE_REGISTRY.register()
class DINOv3Backbone(Backbone):
    """DINOv3 ViT backbone adapted to RDM's C4-style detector interface.

    DINOv3 supplies the dense stride-16 feature map.  The small convolutional
    adapters replace the ResNet ``res4 -> res5`` shape conversion expected by
    the existing SBRPN/ROI heads; they do not replace either detection head.
    The CLIP text classifier is retained, so ROI features are projected to its
    512-dimensional embedding space before cosine classification.
    """

    uses_clip_visual_encoder = False
    input_mean = (0.485, 0.456, 0.406)
    input_std = (0.229, 0.224, 0.225)

    def __init__(self, cfg, input_shape):
        super().__init__()
        del input_shape
        dino_cfg = cfg.MODEL.DINOV3
        repo_dir = os.path.abspath(os.path.expanduser(dino_cfg.REPO_DIR))
        weights = os.path.abspath(os.path.expanduser(dino_cfg.WEIGHTS))
        if not dino_cfg.REPO_DIR or not os.path.isfile(
            os.path.join(repo_dir, "hubconf.py")
        ):
            raise FileNotFoundError(
                "MODEL.DINOV3.REPO_DIR must point to a local clone of the "
                "official facebookresearch/dinov3 repository (hubconf.py was "
                "not found): {!r}".format(dino_cfg.REPO_DIR)
            )
        if not dino_cfg.WEIGHTS or not os.path.isfile(weights):
            raise FileNotFoundError(
                "MODEL.DINOV3.WEIGHTS must point to an approved DINOv3 "
                "backbone checkpoint: {!r}".format(dino_cfg.WEIGHTS)
            )

        try:
            self.enc = torch.hub.load(
                repo_dir,
                dino_cfg.MODEL_NAME,
                source="local",
                weights=weights,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load DINOv3 backbone {!r} from {!r}. The official "
                "DINOv3 code must be compatible with the active Python/PyTorch "
                "environment.".format(dino_cfg.MODEL_NAME, repo_dir)
            ) from exc

        self.patch_size = int(getattr(self.enc, "patch_size", 16))
        if self.patch_size != 16:
            raise ValueError(
                "This C4 detector is configured for stride 16, but {} reports "
                "patch_size={}. Use a DINOv3 */16 ViT backbone.".format(
                    dino_cfg.MODEL_NAME, self.patch_size
                )
            )
        self.embed_dim = int(
            getattr(self.enc, "embed_dim", getattr(self.enc, "num_features", 0))
        )
        if self.embed_dim <= 0:
            raise AttributeError(
                "The loaded DINOv3 model does not expose embed_dim/num_features"
            )

        self.augmentation_channels = self.embed_dim
        self.out_channels = int(dino_cfg.OUT_CHANNELS)
        self.roi_channels = int(dino_cfg.ROI_CHANNELS)
        self.clip_embed_dim = int(dino_cfg.CLIP_EMBED_DIM)
        self._size_divisibility = self.patch_size

        self.res4_adapter = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.out_channels, kernel_size=1),
            nn.GroupNorm(32, self.out_channels),
            nn.GELU(),
        )
        self.res5_adapter = nn.Sequential(
            nn.Conv2d(self.out_channels, self.roi_channels, kernel_size=1),
            nn.GroupNorm(32, self.roi_channels),
            nn.GELU(),
            nn.Conv2d(
                self.roi_channels,
                self.roi_channels,
                kernel_size=3,
                padding=1,
                groups=self.roi_channels,
            ),
            nn.GroupNorm(32, self.roi_channels),
            nn.GELU(),
        )
        self.clip_projection = nn.Linear(
            self.roi_channels, self.clip_embed_dim
        )

        for parameter in self.enc.parameters():
            parameter.requires_grad = not bool(dino_cfg.FREEZE)
        if not bool(dino_cfg.FREEZE):
            self._freeze_early_blocks(int(dino_cfg.UNFREEZE_LAST_N_BLOCKS))

    def _freeze_early_blocks(self, last_n):
        """Fine-tune only the requested tail blocks, matching old RN101 policy."""
        blocks = list(getattr(self.enc, "blocks", []))
        if not blocks or last_n < 0:
            return
        cutoff = max(0, len(blocks) - last_n)
        for parameter in self.enc.parameters():
            parameter.requires_grad = False
        for block in blocks[cutoff:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for module_name in ("norm", "cls_norm", "local_cls_norm"):
            module = getattr(self.enc, module_name, None)
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad = True

    def set_backbone_model(self, model):
        # Compatibility with the old meta-architecture constructor. DINOv3 is
        # loaded independently; the passed object is CLIP's text-head visual
        # encoder and must not become the detector backbone.
        del model

    def _dense_patch_features(self, image):
        output = self.enc.forward_features(image)
        if not isinstance(output, dict) or "x_norm_patchtokens" not in output:
            raise TypeError(
                "Expected official DINOv3 forward_features() output containing "
                "'x_norm_patchtokens'"
            )
        patch_tokens = output["x_norm_patchtokens"]
        height = image.shape[-2] // self.patch_size
        width = image.shape[-1] // self.patch_size
        expected_tokens = height * width
        if patch_tokens.shape[1] != expected_tokens:
            raise ValueError(
                "DINOv3 returned {} patch tokens, expected {} for input {}x{}"
                .format(
                    patch_tokens.shape[1],
                    expected_tokens,
                    image.shape[-2],
                    image.shape[-1],
                )
            )
        return patch_tokens.transpose(1, 2).reshape(
            image.shape[0], self.embed_dim, height, width
        ).contiguous()

    def forward_l1(self, image):
        # RDM applies its learned source-to-target modulation to this dense
        # DINOv3 feature map.
        return self._dense_patch_features(image)

    def forward_l2(self, x):
        return x

    def forward_l3(self, x):
        return {"res4": self.res4_adapter(x)}

    def forward_l23(self, x):
        return self.forward_l3(self.forward_l2(x))

    def forward(self, image):
        return self.forward_l23(self.forward_l1(image))

    def output_shape(self):
        return {
            "res4": ShapeSpec(
                channels=self.out_channels, stride=self.patch_size
            )
        }

    @property
    def size_divisibility(self):
        return self.patch_size

    def forward_res5(self, x):
        return self.res5_adapter(x)

    def attention_global_pool(self, input):
        pooled = F.adaptive_avg_pool2d(input, output_size=1).flatten(1)
        embedding = self.clip_projection(pooled)
        return embedding, input
