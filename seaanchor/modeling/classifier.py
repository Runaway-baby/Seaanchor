from seaanchor.clip import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from .text_prompts import (
    UNDERWATER_BACKGROUND_TEMPLATES,
    UNDERWATER_CLASS_TEMPLATES,
    naturalize_class_name,
    naturalize_class_phrase,
)


class ClipLoad(nn.Module):
    def __init__(self, clip_encoder_name, device=None):
        super().__init__()
        self.clip_model, self.preprocess = clip.load(clip_encoder_name, device)
        self.clip_model.float()

        # freeze everything
        for name, val in self.clip_model.named_parameters():
            val.requires_grad = False
        # this is only used for inference
        self.frozen_clip_model = copy.deepcopy(self.clip_model)


class ClipPredictor(nn.Module):
    def __init__(
        self,
        clip_enocder_name,
        inshape,
        device,
        clsnames,
        prompt_domain="generic",
    ):
        super().__init__()
        # self.model, self.preprocess = clip.load(clip_enocder_name, device)
        self.clip_loader = ClipLoad(clip_enocder_name, device)
        self.model = self.clip_loader.clip_model
        self.preprocess = self.clip_loader.preprocess
        self.visual_enc = self.model.visual

        natural_class_names = [naturalize_class_name(name) for name in clsnames]
        natural_class_phrases = [naturalize_class_phrase(name) for name in clsnames]
        if str(prompt_domain).lower() == "underwater":
            class_templates = UNDERWATER_CLASS_TEMPLATES
            background_templates = UNDERWATER_BACKGROUND_TEMPLATES
            prompt_class_names = natural_class_phrases
        else:
            class_templates = ("a photo of a {}",)
            background_templates = ("a photo of an unknown class",)
            prompt_class_names = natural_class_names
        print("CLIP natural-language classes:", natural_class_names)
        with torch.no_grad():
            class_features = []
            for class_name in prompt_class_names:
                text_inputs = torch.cat([
                    clip.tokenize(template.format(class_name))
                    for template in class_templates
                ]).to(device)
                features = self.model.encode_text(text_inputs).float()
                features = F.normalize(features, dim=-1, eps=1e-6)
                class_features.append(features.mean(dim=0))
            text_features = torch.stack(class_features, dim=0)
            text_features = F.normalize(text_features, dim=-1, eps=1e-6)

        with torch.no_grad():
            bg_text_inputs = torch.cat([
                clip.tokenize(prompt)
                for prompt in background_templates
            ]).to(device)
            bg_text_features = self.model.encode_text(bg_text_inputs).float()
            bg_text_features = F.normalize(
                bg_text_features, dim=-1, eps=1e-6
            )
            bg_text_features = bg_text_features.mean(dim=0, keepdim=True)
            bg_text_features = F.normalize(
                bg_text_features, dim=-1, eps=1e-6
            )

        # These are fixed CLIP anchors.  Non-persistent buffers move with the
        # model but are recomputed from the class metadata when loading a run.
        self.register_buffer("text_features", text_features, persistent=False)
        self.register_buffer("bg_text_features", bg_text_features, persistent=False)

    def forward(self, feat, gfeat=None):
        feat = F.normalize(feat, dim=-1, eps=1e-6)
        if gfeat is not None:
            feat = feat - gfeat
            feat = F.normalize(feat, dim=-1, eps=1e-6)

        scores = (100.0 * torch.matmul(feat, self.text_features.detach().T))
        bg_scores = (100.0 * torch.matmul(feat, self.bg_text_features.detach().T))
        # feat [1024,512] text [7, 512]
        # print(scores.min(),scores.max())
        # add for bkg class a score 0
        # scores = torch.cat([scores, torch.zeros(scores.shape[0], 1, device=scores.device)], 1)
        scores = torch.cat([scores, bg_scores], 1)
        return scores
