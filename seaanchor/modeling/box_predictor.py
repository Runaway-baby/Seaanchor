
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from .classifier import ClipPredictor


class ClipFastRCNNOutputLayers(FastRCNNOutputLayers):

    def __init__(self, cfg, input_shape, clsnames) -> None:
        super().__init__(cfg, input_shape)
        self.cls_score = ClipPredictor(
            cfg.MODEL.CLIP_IMAGE_ENCODER_NAME,
            input_shape.channels,
            cfg.MODEL.DEVICE,
            clsnames,
            prompt_domain=cfg.MODEL.CLIP_PROMPT_DOMAIN,
        )

    def forward(self, x, gfeat=None):
        if isinstance(x, list):
            # scores = self.cls_score(x[0], text, gfeat)
            scores = self.cls_score(x[0], gfeat)
            proposal_deltas = self.bbox_pred(x[1])  # self.bbox_pred(self.proj(x[0]/x[0].norm(dim=-1,keepdim=True)))
        else:
            scores = self.cls_score(x, gfeat)
            proposal_deltas = self.bbox_pred(x)

        return scores, proposal_deltas
