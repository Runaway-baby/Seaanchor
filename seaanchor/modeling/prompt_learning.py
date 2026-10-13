import os

import torch
import torch.nn as nn

from seaanchor.clip import clip
from seaanchor.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from .text_prompts import domain_class_description, domain_image_description

_tokenizer = _Tokenizer()


def load_clip_to_cpu():
    url = clip._MODELS["ViT-B/32"]
    model_path = clip._download(url, os.path.expanduser("~/.cache/clip"))

    try:
        # loading JIT archive
        print("jit version")
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):

    def __init__(self, clip_model, random_init, ctx_init=None, bg_class=False, ctx=12,
                 cls_token_position='end'):
        super().__init__()
        self.n_ctx = ctx
        self.class_token_position = cls_token_position
        self.dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, \
            f'cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})'

        if random_init:
            # random init
            print('Initializing a generic context')
            ctx_vectors = torch.empty(self.n_ctx, ctx_dim, dtype=self.dtype).cuda()
            nn.init.normal_(ctx_vectors, std=0.02)
            self.prompt_prefix = ' '.join(['X'] * self.n_ctx)
            print(f'Initial context: "{self.prompt_prefix}"')
            print(f'Number of context words (tokens): {self.n_ctx}')
        else:
            # use given words to initialize context vectors
            # ctx_init = "This is a photo of"
            # n_ctx = len(ctx_init.split(' '))
            # prompt = clip.tokenize(ctx_init)
            # with torch.no_grad():
            #     embedding = clip_model.token_embedding(prompt).type(dtype)
            # ctx_vectors = embedding[0, 1:1+n_ctx, :]
            # prompt_prefix = ctx_init
            print('Load context')
            ctx_vectors = ctx_init
            self.n_ctx = ctx_init.shape[0]
            self.prompt_prefix = ' '.join(['X'] * self.n_ctx)
            print(f'Initial context: "{self.prompt_prefix}"')
            print(f'Number of context words (tokens): {self.n_ctx}')

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.clip_model = clip_model

        self.bg_class = bg_class
        bg_ctx = torch.empty(self.n_ctx, ctx_dim, dtype=self.dtype).cuda()
        nn.init.normal_(bg_ctx, std=0.02)
        self.bg_ctx = nn.Parameter(bg_ctx)

        img_ctx = torch.empty(self.n_ctx, ctx_dim, dtype=self.dtype).cuda()
        nn.init.normal_(img_ctx, std=0.02)
        self.img_ctx = nn.Parameter(img_ctx)

        ada_ctx = torch.empty(self.n_ctx, ctx_dim, dtype=self.dtype).cuda()
        nn.init.normal_(ada_ctx, std=0.02)
        self.ada_ctx = nn.Parameter(ada_ctx)

    def get_bg_prompt(self, token_prefix, token_suffix):
        prompt = torch.cat([
            token_prefix[0],  # (1, dim)
            # self.ctx, # (n_ctx, dim)
            self.bg_ctx,
            token_suffix[0]  # (*, dim)
        ], dim=0)

        return prompt

    def forward(self, class_names, domain_aug=None):
        # classnames = [f" {'a'} {class_name} {'in'} {domain_aug}" for class_name in class_names] #diverse weather
        # wen ding jiao gao
        classnames = [
            domain_class_description(class_name, domain_aug)
            for class_name in class_names
        ]
        # classnames = [f" {'a'} {class_name} {'on'} {domain_aug}" for class_name in class_names] # city2bdd
        # classnames = [f" {'a'} {domain_aug} {'photo of a'} {class_name}" for class_name in class_names]
        # classnames = [f"{class_name} {domain_aug}" for class_name in class_names]
        prompts = [self.prompt_prefix + ' ' + name + '.' for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.dtype)
        token_prefix = embedding[:, :1, :]
        token_suffix = embedding[:, 1 + self.n_ctx:, :]
        # self.register_buffer('token_prefix', embedding[:, :1, :])  # SOS [num_cls,1,512]
        # self.register_buffer('token_suffix', embedding[:, 1 + self.n_ctx:, :])  # CLS, EOS [num_cls, 68 ,512]
        n_cls = len(classnames)
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(n_cls, -1, -1)

        prefix = token_prefix
        suffix = token_suffix
        if self.class_token_position == 'end':
            prompts = torch.cat([
                prefix,  # (n_cls, 1, dim) sos
                ctx,  # (n_cls, n_ctx, dim) learnable prompt
                suffix  # (n_cls, *, dim) cls eos
            ], dim=1)

        if self.bg_class:
            # wen ding jiao gao
            # bg_prompt = ' '.join(['X'] * self.n_ctx) + ' ' + domain_aug + '.'
            bg_prompt = (
                ' '.join(['X'] * self.n_ctx)
                + ' an underwater background with '
                + str(domain_aug).strip()
                + '.'
            )
            tokenized_bg_prompt = clip.tokenize(bg_prompt).cuda()
            with torch.no_grad():
                embedding_bg = self.clip_model.token_embedding(tokenized_bg_prompt).type(self.dtype)
            token_prefix_bg = embedding_bg[:, :1, :]
            token_suffix_bg = embedding_bg[:, 1 + self.n_ctx:, :]
            bg_prompt = self.get_bg_prompt(token_prefix_bg, token_suffix_bg)
            prompts = torch.cat([prompts, bg_prompt[None]])
            tokenized_prompts = torch.cat([tokenized_prompts, tokenized_bg_prompt])

            return prompts, tokenized_prompts
        else:
            return prompts, tokenized_prompts

    def forward_for_img(self, domain_aug):
        # img_prompt = ' '.join(['X'] * self.n_ctx) + ' ' + domain_aug + '.'
        #wen ding jiao gao
        img_prompt = (
            ' '.join(['X'] * self.n_ctx)
            + ' '
            + domain_image_description(domain_aug)
            + '.'
        )
        # img_prompt = ' '.join(['X'] * self.n_ctx) + ' a ' + domain_aug + ' photo' + '.'
        tokenized_img_prompt = clip.tokenize(img_prompt).cuda()
        with torch.no_grad():
            embedding_bg = self.clip_model.token_embedding(tokenized_img_prompt).type(self.dtype)
        token_prefix_img = embedding_bg[:, :1, :]
        token_suffix_img = embedding_bg[:, 1 + self.n_ctx:, :]
        img_ctx = self.img_ctx
        if img_ctx.dim() == 2:
            img_ctx = img_ctx.unsqueeze(0).expand(1, -1, -1)

        prompts = torch.cat([
            token_prefix_img,  # (n_cls, 1, dim)
            img_ctx,  # (n_cls, n_ctx, dim)
            token_suffix_img  # (n_cls, *, dim)
        ], dim=1)

        return prompts, tokenized_img_prompt

    def forward_for_ada(self, classnames):
        ada_prompt = [self.prompt_prefix + ' ' + 'a' + ' ' + name + '.' for name in classnames]
        tokenized_ada_prompt = clip.tokenize(ada_prompt).cuda()
        with torch.no_grad():
            embedding_ada = self.clip_model.token_embedding(tokenized_ada_prompt).type(self.dtype)
        token_prefix_ada = embedding_ada[:, :1, :]
        token_suffix_ada = embedding_ada[:, 1 + self.n_ctx:, :]
        ctx = self.ada_ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(len(classnames), -1, -1)

        prompts = torch.cat([
            token_prefix_ada,  # (n_cls, 1, dim)
            ctx,  # (n_cls, n_ctx, dim)
            token_suffix_ada  # (n_cls, *, dim)
        ], dim=1)

        return prompts, tokenized_ada_prompt
