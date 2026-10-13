import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from seaanchor.clip import clip
from .text_prompts import domain_image_description
from typing import Dict, List, Optional

from detectron2.modeling import META_ARCH_REGISTRY, GeneralizedRCNN
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.utils.visualizer import Visualizer


@META_ARCH_REGISTRY.register()
class ClipRCNNWithClipBackbone(GeneralizedRCNN):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.colors = self.generate_colors(7)
        if getattr(self.backbone, "uses_clip_visual_encoder", False):
            self.backbone.set_backbone_model(
                self.roi_heads.box_predictor.cls_score.visual_enc
            )


    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        clip_images = [x["image"].to(self.pixel_mean.device) for x in batched_inputs]
        mean = getattr(
            self.backbone,
            "input_mean",
            (0.48145466, 0.4578275, 0.40821073),
        )
        std = getattr(
            self.backbone,
            "input_std",
            (0.26862954, 0.26130258, 0.27577711),
        )

        clip_images = [T.functional.normalize(ci.flip(0) / 255, mean, std) for ci in clip_images]
        clip_images = ImageList.from_tensors(
            [i for i in clip_images],
            size_divisibility=self.backbone.size_divisibility,
        )
        return clip_images

    def forward(self, batched_inputs):

        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        b = images.tensor.shape[0]  # batchsize

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            if self.training:
                logits, proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            else:
                logits, proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        try:
            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, None, self.backbone)
        except Exception as e:
            print(e)
            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, None)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
                with torch.no_grad():
                    ogimage = batched_inputs[0]['image']
                    ogimage = convert_image_to_rgb(ogimage.permute(1, 2, 0), self.input_format)
                    o_pred = Visualizer(ogimage, None).overlay_instances().get_image()

                    vis_img = o_pred.transpose(2, 0, 1)
                    storage.put_image('og-tfimage', vis_img)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def generate_colors(self, N):
        import colorsys
        '''
            Generate random colors.
            To get visually distinct colors, generate them in HSV space then
            convert to RGB.
        '''
        brightness = 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: tuple(round(i * 255) for i in colorsys.hsv_to_rgb(*c)), hsv))
        perm = np.arange(7)
        colors = [colors[idx] for idx in perm]
        return colors

    def inference(
            self,
            batched_inputs: List[Dict[str, torch.Tensor]],
            detected_instances: Optional[List[Instances]] = None,
            do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)  # res4: [1,1024,37,66]

        if detected_instances is None:
            if self.proposal_generator is not None:
                logits, proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            # boxes = batched_inputs[0]['instances'].gt_boxes.to(images.tensor.device)
            # logits = 10*torch.ones(len(boxes)).to(images.tensor.device)
            # dictp = {'proposal_boxes':boxes,'objectness_logits':logits}
            # new_p = Instances(batched_inputs[0]['instances'].image_size,**dictp)
            # proposals = [new_p]

            try:
                results, _ = self.roi_heads(images, features, proposals, None, None, self.backbone)
            except:
                results, _ = self.roi_heads(images, features, proposals, None, None, self.backbone)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."

            allresults = GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)

            return allresults
        else:
            return results












@META_ARCH_REGISTRY.register()
class ClipRCNNWithClipBackboneAugL1Dis(ClipRCNNWithClipBackbone):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.tau = 0.07
        self.offset_size = 7
        self.map_size = 14
        domain_text_aug = {}
        with open(cfg.DOMAIN_TEXT_FILE, 'r') as f_aug:
            for ind, l in enumerate(f_aug):
                domain_text_aug.update({str(ind): l.strip()})
        # The text file defines augmentation targets.  Some original RDM
        # configs keep the single source description only in the YAML; append
        # a source code for relative transforms without sampling it as a target.
        self.target_domain_ids = list(range(len(domain_text_aug)))
        configured_source_text = cfg.SOURCE_DOMAIN_TEXT.strip()
        known_domain_texts = {
            value.strip().lower() for value in domain_text_aug.values()
        }
        if configured_source_text.lower() not in known_domain_texts:
            domain_text_aug[str(len(domain_text_aug))] = configured_source_text
        self.domain_text_aug = domain_text_aug
        self.domain_texts = [
            domain_text_aug[str(i)] for i in range(len(domain_text_aug))
        ]
        self.domain_text_to_id = {
            text.strip().lower(): i for i, text in enumerate(self.domain_texts)
        }
        augmentation_channels = int(
            getattr(self.backbone, "augmentation_channels", 256)
        )
        self.weights = nn.Parameter(
            torch.ones(len(domain_text_aug), augmentation_channels, self.offset_size,
                       self.offset_size))  # skip day,shape[:2]=res4.shape[:2]
        self.offsets = nn.Parameter(
            torch.zeros(len(domain_text_aug), augmentation_channels, self.offset_size,
                        self.offset_size))  # skip day,shape[:2]=res4.shape[:2]
        self.apply_aug = cfg.AUG_PROB
        self.rdm_cfg = cfg.MODEL.RDM

    def _source_domain_ids(self, batched_inputs):
        """Resolve the real source domain carried by every dataset record."""
        fallback = self.cfg.SOURCE_DOMAIN_TEXT.strip().lower()
        fallback_id = self.domain_text_to_id.get(fallback, 0)
        source_ids = []
        for sample in batched_inputs:
            if "domain_text" not in sample:
                # Preserve the original single-source RDM behavior for
                # datasets that do not carry per-image domain metadata.
                source_ids.append(fallback_id)
                continue
            source_text = str(sample["domain_text"]).strip().lower()
            if source_text not in self.domain_text_to_id:
                raise KeyError(
                    "Unknown source domain text {!r}. Available domains: {}".format(
                        source_text, ", ".join(self.domain_texts)
                    )
                )
            source_ids.append(self.domain_text_to_id.get(source_text, fallback_id))
        return source_ids

    def _sample_target_domain_ids(self, source_ids):
        """Assign balanced augmentation targets, always different from the source."""
        if not self.target_domain_ids:
            return list(source_ids)
        # Source domains come from dataset metadata and are never randomized.
        # Only the synthetic destination is sampled. Prefer the least-used
        # valid target in this mini-batch to avoid the severe imbalance of
        # independent random draws when IMS_PER_BATCH is small.
        target_counts = {domain_id: 0 for domain_id in self.target_domain_ids}
        target_ids = []
        for source_id in source_ids:
            candidates = [i for i in self.target_domain_ids if i != source_id]
            if not candidates:
                candidates = [source_id]
            minimum_count = min(target_counts[i] for i in candidates)
            balanced_candidates = [
                i for i in candidates if target_counts[i] == minimum_count
            ]
            target_id = int(np.random.choice(balanced_candidates))
            target_counts[target_id] += 1
            target_ids.append(target_id)
        return target_ids

    def _relative_domain_transform(self, features_l1, source_ids, target_ids):
        """Apply a source-aware target modulation using differences of domain codes."""
        source_index = torch.as_tensor(
            source_ids, dtype=torch.long, device=self.weights.device
        )
        target_index = torch.as_tensor(
            target_ids, dtype=torch.long, device=self.weights.device
        )
        source_weight = self.weights.index_select(0, source_index)
        target_weight = self.weights.index_select(0, target_index)
        source_offset = self.offsets.index_select(0, source_index)
        target_offset = self.offsets.index_select(0, target_index)

        if self.rdm_cfg.MULTI_SOURCE:
            # Domain tables represent absolute modulation codes. Their
            # difference is the source-to-target transform and is identity at
            # initialization.
            relative_weight = 1.0 + target_weight - source_weight
            relative_offset = target_offset - source_offset
        else:
            # Backward-compatible behavior for the original single-source
            # RDM configs.
            relative_weight = target_weight
            relative_offset = target_offset
        relative_weight = F.interpolate(
            relative_weight,
            size=features_l1.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        relative_offset = F.interpolate(
            relative_offset,
            size=features_l1.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        return features_l1 * relative_weight + relative_offset

    def _encode_image_domain_ids(self, domain_ids):
        """Encode learned image prompts while sharing each domain embedding."""
        embeddings = {}
        for domain_id in sorted(set(domain_ids)):
            domain_text = self.domain_texts[domain_id]
            prompts, tokenized_prompts = (
                self.roi_heads.prompt_learner.forward_for_img(domain_text)
            )
            embedding = self.roi_heads.text_encoder(prompts, tokenized_prompts)
            embeddings[domain_id] = F.normalize(
                embedding, dim=-1, eps=1e-6
            )[0]
        return torch.stack([embeddings[i] for i in domain_ids], dim=0)

    def _encode_fixed_image_domain_ids(self, domain_ids):
        """Encode frozen natural-language anchors for relative domain directions."""
        embeddings = {}
        clip_model = self.roi_heads.box_predictor.cls_score.model
        with torch.no_grad():
            for domain_id in sorted(set(domain_ids)):
                text = domain_image_description(self.domain_texts[domain_id])
                tokens = clip.tokenize(text).to(self.device)
                embedding = clip_model.encode_text(tokens).float()
                embeddings[domain_id] = F.normalize(
                    embedding, dim=-1, eps=1e-6
                )[0]
        return torch.stack([embeddings[i] for i in domain_ids], dim=0)

    def forward(self, batched_inputs, training_phase=None):
        # Route every training phase through __call__/forward so
        # DistributedDataParallel can prepare its reducer and synchronize
        # gradients on every rank.
        if training_phase is not None:
            if not self.training:
                raise RuntimeError(
                    "training_phase={!r} is only valid while training".format(
                        training_phase
                    )
                )
            if training_phase == "base":
                return self.forward_base(batched_inputs)
            if training_phase == "offset":
                return self.opt_offsets(batched_inputs)
            raise ValueError(
                "Unknown training_phase {!r}; expected 'base' or 'offset'".format(
                    training_phase
                )
            )
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        b = images.tensor.shape[0]  # batchsize

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        features_l1 = self.backbone.forward_l1(images.tensor)
        source_ids = self._source_domain_ids(batched_inputs)
        use_domain_aug = bool(np.random.rand() < self.apply_aug)
        consistency_features = None

        if use_domain_aug:
            target_ids = self._sample_target_domain_ids(source_ids)
            with torch.no_grad():
                clean_features = self.backbone.forward_l23(features_l1.detach())
                consistency_features = clean_features['res4']
            features_l1_aug = self._relative_domain_transform(
                features_l1, source_ids, target_ids
            )
            features = self.backbone.forward_l23(features_l1_aug)
        else:
            features = self.backbone.forward_l23(features_l1)

        if self.proposal_generator is not None:
            if self.training:
                logits, proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            else:
                logits, proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(
            images,
            features,
            proposals,
            gt_instances,
            None,
            self.backbone,
            consistency_features=consistency_features,
        )

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
                with torch.no_grad():
                    ogimage = batched_inputs[0]['image']
                    ogimage = convert_image_to_rgb(ogimage.permute(1, 2, 0), self.input_format)
                    o_pred = Visualizer(ogimage, None).overlay_instances().get_image()

                    vis_img = o_pred.transpose(2, 0, 1)
                    storage.put_image('og-tfimage', vis_img)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return losses

    def opt_offsets(self, batched_inputs):

        images = self.preprocess_image(batched_inputs)
        b = images.tensor.shape[0]  # batchsize

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        with torch.no_grad():
            features_l1 = self.backbone.forward_l1(images.tensor)  # [2, 1024, 37, 66]
            features_ori = self.backbone.forward_l23(features_l1)

            # with torch.no_grad():
        with torch.no_grad():
            if self.proposal_generator is not None:
                _, proposals, _ = self.proposal_generator(
                    images, features_ori, gt_instances
                )
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

        source_ids = self._source_domain_ids(batched_inputs)
        target_ids = self._sample_target_domain_ids(source_ids)
        features_l1_aug = self._relative_domain_transform(
            features_l1.detach(), source_ids, target_ids
        )
        features_aug = self.backbone.forward_l23(features_l1_aug)

        img_features_aug = self.backbone.forward_res5(features_aug['res4'])
        img_features_aug = F.interpolate(
            img_features_aug, size=(21, 21), mode='bilinear', align_corners=False
        )
        img_features_aug = F.avg_pool2d(
            img_features_aug, kernel_size=3, stride=3, padding=0
        )
        aug_im_embed, _ = self.backbone.attention_global_pool(img_features_aug)
        aug_im_embed = F.normalize(aug_im_embed, dim=-1, eps=1e-6)
        target_text_embed = self._encode_image_domain_ids(target_ids)

        img_aug_loss = (
            1.0
            - torch.cosine_similarity(target_text_embed, aug_im_embed, dim=-1)
        ).mean()

        with torch.no_grad():
            img_features_ori = self.backbone.forward_res5(features_ori['res4'])
            img_features_ori = F.interpolate(
                img_features_ori,
                size=(21, 21),
                mode='bilinear',
                align_corners=False,
            )
            img_features_ori = F.avg_pool2d(
                img_features_ori, kernel_size=3, stride=3, padding=0
            )
            ori_im_embed, _ = self.backbone.attention_global_pool(img_features_ori)
            ori_im_embed = F.normalize(ori_im_embed, dim=-1, eps=1e-6)
            fixed_source_text_embed = self._encode_fixed_image_domain_ids(
                source_ids
            )
            fixed_target_text_embed = self._encode_fixed_image_domain_ids(
                target_ids
            )

        content_loss = F.l1_loss(aug_im_embed, ori_im_embed)
        visual_direction = aug_im_embed - ori_im_embed
        direction_loss = F.smooth_l1_loss(
            visual_direction,
            fixed_target_text_embed - fixed_source_text_embed,
        )

        # Each image receives its own target domain.  Group only for the ROI
        # prompt call, whose current interface accepts one domain string.
        instance_aug_loss = aug_im_embed.sum() * 0.0
        for target_id in sorted(set(target_ids)):
            indices = [i for i, value in enumerate(target_ids) if value == target_id]
            group_features = features_aug['res4'][indices]
            group_proposals = [proposals[i] for i in indices]
            group_targets = [gt_instances[i] for i in indices]
            _, group_loss = self.roi_heads(
                images,
                group_features,
                group_proposals,
                group_targets,
                None,
                self.backbone,
                self.domain_texts[target_id],
            )
            instance_aug_loss = (
                instance_aug_loss + group_loss * (len(indices) / float(b))
            )

        return {
            'img_aug_loss': (
                self.rdm_cfg.IMAGE_DOMAIN_LOSS_WEIGHT * img_aug_loss
            ),
            'instance_aug_loss': (
                self.rdm_cfg.INSTANCE_DOMAIN_LOSS_WEIGHT * instance_aug_loss
            ),
            'aug_content_loss': (
                self.rdm_cfg.CONTENT_LOSS_WEIGHT * content_loss
            ),
            'domain_direction_loss': (
                self.rdm_cfg.DIRECTION_LOSS_WEIGHT * direction_loss
            ),
        }

    def forward_base(self, batched_inputs):

        images = self.preprocess_image(batched_inputs)
        b = images.tensor.shape[0]  # batchsize

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        with torch.no_grad():
            features_l1 = self.backbone.forward_l1(images.tensor)
            features_l2 = self.backbone.forward_l2(features_l1)
        features = self.backbone.forward_l3(features_l2)

        if self.proposal_generator is not None:
            if self.training:
                logits, proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            else:
                logits, proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        try:
            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, None, self.backbone)
        except Exception as e:
            print(e)
            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, None, self.backbone)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
                with torch.no_grad():
                    ogimage = batched_inputs[0]['image']
                    ogimage = convert_image_to_rgb(ogimage.permute(1, 2, 0), self.input_format)
                    o_pred = Visualizer(ogimage, None).overlay_instances().get_image()

                    vis_img = o_pred.transpose(2, 0, 1)
                    storage.put_image('og-tfimage', vis_img)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return losses


