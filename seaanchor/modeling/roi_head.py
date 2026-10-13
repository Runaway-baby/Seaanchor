from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F


from .prompt_learning import PromptLearner, TextEncoder
from detectron2.layers import ShapeSpec
from detectron2.data import MetadataCatalog
from detectron2.layers import cat
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, Res5ROIHeads
from detectron2.structures import ImageList, Instances
from .box_predictor import ClipFastRCNNOutputLayers
from .ocia import OCIAROIAdapter


def select_foreground_proposals(
        proposals: List[Instances], bg_label: int
) -> Tuple[List[Instances], List[torch.Tensor]]:
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.
    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.
    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


@ROI_HEADS_REGISTRY.register()
class ClipRes5ROIHeads(Res5ROIHeads):
    def __init__(self, cfg, input_shape) -> None:
        super().__init__(cfg, input_shape)
        cls = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("thing_classes")
        if cls is None:
            if cfg.DATASETS.TRAIN[0] == 'daytime_clear_train':
                clsnames = ['bus', 'bike', 'car', 'motor', 'person', 'rider', 'truck']
            elif cfg.DATASETS.TRAIN[0] == 'cityscape_train':
                clsnames = ['bus', 'bike', 'car', 'motor', 'person', 'rider', 'truck']
            elif cfg.DATASETS.TRAIN[0] == 'cityscape_car_train':
                clsnames = ['car']
            elif cfg.DATASETS.TRAIN[0] == 'sim10k_train':
                clsnames = ['car']
            else:
                raise KeyError(
                    "Define cls according to '{}' category!".format(cfg.DATASETS.TRAIN[0])
                )
        else: clsnames = cls
        # import pdb;pdb.set_trace()
        ##change the labels to represent the objects correctly
        for name in cfg.MODEL.RENAME:
            ind = clsnames.index(name[0])
            clsnames[ind] = name[1]
        self.class_names = clsnames
        out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * (2 ** 3)  ### copied
        self.box_predictor = ClipFastRCNNOutputLayers(cfg, ShapeSpec(channels=out_channels, height=1, width=1),
                                                      clsnames)
        self.clip_im_predictor = self.box_predictor.cls_score  # should call it properly
        self.device = cfg.MODEL.DEVICE

    def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            targets: Optional[List[Instances]] = None,
            crops: Optional[List[Tuple]] = None,
    ):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
            # import pdb;pdb.set_trace()
            loss_crop_im = None
            if crops is not None:
                crop_im = list()  # [x[0] for x in crops] #bxcropx3x224x224
                crop_boxes = list()  # [x[1].to(self.device) for x in crops] #bxcropsx4
                keep = torch.ones(len(crops)).bool()

                for ind, x in enumerate(crops):
                    if len(x) == 0:
                        keep[ind] = False
                        continue
                    crop_im.append(x[0])
                    crop_boxes.append(x[1].to(self.device))

                c = self._shared_roi_transform(
                    [features[f][keep] for f in self.in_features], crop_boxes)  # (b*crops)x2048x7x7
                loss_crop_im, _ = self.clip_im_predictor.forward_crops(crop_im, c.mean(dim=[2, 3]))

        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        predictions = self.box_predictor(box_features.mean(dim=[2, 3]))
        # import pdb;pdb.set_trace()
        if self.training:
            del features
            losses = self.box_predictor.losses(predictions, proposals)
            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                # Since the ROI feature transform is shared between boxes and masks,
                # we don't need to recompute features. The mask loss is only defined
                # on foreground proposals, so we need to select out the foreground
                # features.
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                losses.update(self.mask_head(mask_features, proposals))

            if loss_crop_im is not None:
                losses.update(loss_crop_im)
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}


@ROI_HEADS_REGISTRY.register()
class ClipRes5ROIHeadsAttn(ClipRes5ROIHeads):
    def __init__(self, cfg, input_shape) -> None:
        super().__init__(cfg, input_shape)
        # self.res5 = None
        self.prompt_learner = PromptLearner(self.box_predictor.cls_score.model, True, None, True, 12,
                                            cls_token_position='end')
        self.text_encoder = TextEncoder(self.box_predictor.cls_score.model)
        self.sigmoid = nn.Sigmoid()
        self.roi_ocia = None
        if cfg.MODEL.OCIA.ENABLED:
            roi_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * (2 ** 3)
            self.roi_ocia = OCIAROIAdapter(
                in_channels=roi_channels,
                embed_dims=cfg.MODEL.OCIA.EMBED_DIMS,
                token_length=cfg.MODEL.OCIA.TOKEN_LENGTH,
                scale_init=cfg.MODEL.OCIA.SCALE_INIT,
                residual_scale_init=cfg.MODEL.OCIA.RESIDUAL_SCALE_INIT,
                frequency_ratio=cfg.MODEL.OCIA.FREQUENCY_RATIO,
                max_residual_scale=cfg.MODEL.OCIA.MAX_RESIDUAL_SCALE,
            )
        self.roi_consistency_weight = cfg.MODEL.OCIA.ROI_CONSISTENCY_WEIGHT
        self.logit_consistency_weight = cfg.MODEL.OCIA.LOGIT_CONSISTENCY_WEIGHT

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        x = self.fwdres5(x)
        if self.roi_ocia is not None:
            x = self.roi_ocia(x)
        return x

    def softCrossEntropy(self, logit, target):
        log_likelihood = -F.log_softmax(logit, dim=1)
        return torch.sum(log_likelihood * target)

    def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            targets: Optional[List[Instances]] = None,
            crops: Optional[List[Tuple]] = None,
            backbone=None,
            domain_aug=None,
            consistency_features=None,
    ):
        """
        instance_aug_loss = self.roi_heads(images, features_aug, proposals, gt_instances, None,
                                                  self.backbone, domain_aug)
        See :meth:`ROIHeads.forward`.
        """
        del images

        self.fwdres5 = backbone.forward_res5

        if self.training:
            assert targets
            configured_batch_size = self.batch_size_per_image
            if domain_aug is not None:
                self.batch_size_per_image = min(64, configured_batch_size)
            try:
                proposals = self.label_and_sample_proposals(proposals, targets)
            finally:
                # Domain-prompt sampling must not permanently reduce the
                # detector's configured ROI batch size (400 for NAUTILUS).
                self.batch_size_per_image = configured_batch_size
            # import pdb;pdb.set_trace()
            loss_crop_im = None
            if crops is not None:
                crop_im = list()  # [x[0] for x in crops] #bxcropx3x224x224
                crop_boxes = list()  # [x[1].to(self.device) for x in crops] #bxcropsx4
                keep = torch.ones(len(crops)).bool()

                for ind, x in enumerate(crops):
                    if len(x) == 0:
                        keep[ind] = False
                        continue
                    crop_im.append(x[0])
                    crop_boxes.append(x[1].to(self.device))

                crops_features = self._shared_roi_transform(
                    [features[f][keep] for f in self.in_features], crop_boxes)  # (b*crops)x2048x7x7
                crops_features, _ = backbone.attention_global_pool(crops_features)
                loss_crop_im, _ = self.clip_im_predictor.forward_crops(crop_im, crops_features)

        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        # proposal_boxes, objectness_logits, gt_classes0-7, gt_boxes

        if domain_aug is not None:
            # with torch.no_grad():
            box_features_aug = self._shared_roi_transform([features], proposal_boxes)  # [1024,2048,7,7]
            # box_linguistic_aug = self._shared_roi_transform([linguistic_aug], proposal_boxes)  # [1024,2048,7,7]
            # box_features_aug = box_features + box_linguistic_aug
            attn_feat, _ = backbone.attention_global_pool(box_features_aug)
            attn_feat = F.normalize(attn_feat, dim=-1, eps=1e-6)

            # box_linguistic_aug = self._shared_roi_transform([linguistic_aug], proposal_boxes)  # [1024,2048,7,7]
            # atten_linguistic_aug = backbone.attention_global_pool(box_linguistic_aug)
            # attn_feat_aug = attn_feat + atten_linguistic_aug
            # attn_feat = attn_feat_aug / attn_feat_aug.norm(dim=-1, keepdim=True)

            prompts, tokenized_prompts = self.prompt_learner(self.class_names, domain_aug)
            emb = self.text_encoder(prompts, tokenized_prompts)
            emb = F.normalize(emb, dim=-1, eps=1e-6)

            temperature = 0.01

            # scores = attn_feat @ emb.t() / temperature  # [128,512] [8,512] = [128,8]
            scores = torch.matmul(attn_feat, emb.T) / temperature

            gt_classes = (
                cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
            )
            # loss = F.cross_entropy(scores, gt_classes.cuda())

            if gt_classes.numel() == 0:
                loss = scores.sum() * 0.0
            else:
                # The prompt set already contains the background prototype at
                # index num_class.  A single stable CE avoids the previous
                # double foreground normalization and empty-set divisions.
                loss = F.cross_entropy(scores, gt_classes.to(scores.device))

            # loss /= len(gt_classes)

            return [], loss
        else:
            # self.batch_size_per_image = 128
            box_features = self._shared_roi_transform(
                [features[f] for f in self.in_features], proposal_boxes
            )  # [512,2048,7,7]
            attn_feat, score_map = backbone.attention_global_pool(box_features)
            # df = attn_feat.detach().cpu().numpy()
            # df = pd.DataFrame(df)
            # df.to_csv('attn_feat_nc2.csv',index=False)
            # [512,512] [512,512,7,7]
            # prompts, tokenized_prompts = self.prompt_learner.forward_for_ada(self.class_names)
            # emb = self.text_encoder(prompts, tokenized_prompts)
            # emb = emb / emb.norm(dim=-1, keepdim=True)

            # predictions = self.box_predictor([attn_feat, box_features.mean(dim=(2, 3))], emb)
            predictions = self.box_predictor([attn_feat, box_features.mean(dim=(2, 3))])
            # attn_feat[1024, 512] box_features [1024, 2048, 7, 7]
            # import pdb;pdb.set_trace()
            if self.training:
                del features

                losses = self.box_predictor.losses(predictions, proposals)

                # Every DDP rank must report the same metric keys. Domain
                # augmentation is sampled independently per rank, so ranks
                # without a clean consistency branch still publish explicit
                # zero losses. The augmented branch below overwrites them.
                consistency_zero = predictions[0].sum() * 0.0
                losses["loss_roi_domain_consistency"] = consistency_zero
                losses["loss_logit_domain_consistency"] = consistency_zero

                if consistency_features is not None:
                    with torch.no_grad():
                        clean_box_features = self._shared_roi_transform(
                            [consistency_features], proposal_boxes
                        )
                        clean_attn_feat, _ = backbone.attention_global_pool(
                            clean_box_features
                        )
                        clean_predictions = self.box_predictor([
                            clean_attn_feat,
                            clean_box_features.mean(dim=(2, 3)),
                        ])

                    aug_logits = predictions[0]
                    clean_logits = clean_predictions[0]
                    if aug_logits.numel() == 0:
                        zero = aug_logits.sum() * 0.0
                        losses["loss_roi_domain_consistency"] = zero
                        losses["loss_logit_domain_consistency"] = zero
                    else:
                        aug_norm = F.normalize(attn_feat, dim=-1, eps=1e-6)
                        clean_norm = F.normalize(
                            clean_attn_feat, dim=-1, eps=1e-6
                        )
                        losses["loss_roi_domain_consistency"] = (
                            self.roi_consistency_weight
                            * (1.0 - (aug_norm * clean_norm).sum(dim=-1)).mean()
                        )
                        losses["loss_logit_domain_consistency"] = (
                            self.logit_consistency_weight
                            * F.kl_div(
                                F.log_softmax(aug_logits, dim=-1),
                                F.softmax(clean_logits, dim=-1),
                                reduction="batchmean",
                            )
                        )
                # if torch.isnan(losses['loss_cls']):
                #     import pdb;pdb.set_trace()

                if self.mask_on:
                    proposals, fg_selection_masks = select_foreground_proposals(
                        proposals, self.num_classes
                    )
                    # Since the ROI feature transform is shared between boxes and masks,
                    # we don't need to recompute features. The mask loss is only defined
                    # on foreground proposals, so we need to select out the foreground
                    # features.
                    mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                    del box_features
                    losses.update(self.mask_head(mask_features, proposals))

                if loss_crop_im is not None:
                    losses.update(loss_crop_im)
                return [], losses
            else:
                pred_instances, _ = self.box_predictor.inference(predictions, proposals)
                pred_instances = self.forward_with_given_boxes(features, pred_instances)
                return pred_instances, {}
