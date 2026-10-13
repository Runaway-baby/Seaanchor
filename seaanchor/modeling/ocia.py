import torch
from torch import nn
from torch.nn import functional as F


class AuxiliaryBranch(nn.Module):
    """The Fourier auxiliary branch from OCIA."""

    def __init__(self, dims, frequency_ratio=0.2):
        super().__init__()
        hidden_dims = max(1, dims // 16)
        self.frequency_ratio = frequency_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dims, hidden_dims),
            nn.Linear(hidden_dims, dims),
            nn.SiLU(),
        )

    def fourier_transform(self, feats):
        _, token_count, _ = feats.shape
        padded_length = 1
        while padded_length < token_count:
            padded_length *= 2

        padded_feats = F.pad(feats, (0, 0, 0, padded_length - token_count))
        fft_feats = torch.fft.fft(padded_feats, dim=1)

        mask = torch.zeros(
            padded_length, dtype=feats.dtype, device=feats.device
        )
        center = padded_length // 2
        mask_half_width = int(padded_length * self.frequency_ratio) // 2
        mask_start = max(0, center - mask_half_width)
        mask_end = min(padded_length, center + mask_half_width)
        mask[mask_start:mask_end] = 1.0

        filtered_feats = torch.fft.ifft(
            fft_feats * mask[None, :, None], dim=1
        ).real
        return filtered_feats[:, :token_count, :]

    def forward(self, x):
        x = self.mlp(x)
        return x + self.fourier_transform(x)


class CausalBranch(nn.Module):
    """The causal MLP branch from OCIA."""

    def __init__(self, dims):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dims, dims), nn.SiLU())

    def forward(self, x):
        return self.mlp(x)


class OCIA(nn.Module):
    """Causal visual prompts operating on a ``B x L x C`` token sequence."""

    def __init__(
        self,
        embed_dims,
        token_length=64,
        scale_init=0.001,
        frequency_ratio=0.2,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.prompt = nn.Parameter(torch.zeros(token_length, embed_dims))
        self.mlp_prompt = nn.Linear(embed_dims, embed_dims)
        self.to_out = nn.Linear(embed_dims, embed_dims)

        self.alpha = nn.Parameter(torch.tensor(float(scale_init)))
        self.beta = nn.Parameter(torch.tensor(float(scale_init)))
        self.delta_scale = nn.Parameter(torch.tensor(1.0))

        self.prompt_branch = CausalBranch(embed_dims)
        self.aux_branch = AuxiliaryBranch(embed_dims, frequency_ratio)

    def cross_attention(self, x):
        output_dtype = x.dtype
        with torch.amp.autocast("cuda", enabled=False):
            x = x.float()
            prompt = self.prompt.float()
            attention = torch.einsum("blc,tc->blt", x, prompt)
            attention = attention * (self.embed_dims ** -0.5)
            attention = attention.softmax(dim=-1)
            prompt_features = self.mlp_prompt(prompt)
            score = torch.einsum("blt,tc->blc", attention, prompt_features)
            output = self.to_out(score)
        return output.to(dtype=output_dtype)

    def forward(self, feats):
        output_dtype = feats.dtype
        with torch.amp.autocast("cuda", enabled=False):
            feats_fp32 = feats.float()
            prompt_response = self.cross_attention(feats_fp32).float()
            alpha = self.alpha.float().clamp(min=-0.1, max=0.1)
            beta = self.beta.float().clamp(min=-0.1, max=0.1)
            delta_scale = self.delta_scale.float().clamp(min=0.95, max=1.05)
            causal = self.prompt_branch(prompt_response) * alpha
            auxiliary = self.aux_branch(prompt_response) * beta
            output = feats_fp32 * delta_scale + causal + auxiliary
        return output.to(dtype=output_dtype)


class OCIAROIAdapter(nn.Module):
    """Apply OCIA to the spatial tokens inside each pooled ROI.

    The detector's 2048-channel ROI feature is projected to a smaller embedding
    space before OCIA. Only the OCIA-produced delta is projected back and
    added to the original feature, so the CLIP representation remains the main
    path and the initial perturbation is small.
    """

    def __init__(
        self,
        in_channels,
        embed_dims=512,
        token_length=64,
        scale_init=0.001,
        residual_scale_init=0.0,
        frequency_ratio=0.2,
        max_residual_scale=0.1,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_channels)
        self.down_proj = nn.Linear(in_channels, embed_dims)
        self.ocia = OCIA(
            embed_dims=embed_dims,
            token_length=token_length,
            scale_init=scale_init,
            frequency_ratio=frequency_ratio,
        )
        # No bias: an identity OCIA block must produce an exact zero delta.
        self.up_proj = nn.Linear(embed_dims, in_channels, bias=False)
        self.max_residual_scale = float(max_residual_scale)
        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init))
        )

    def forward(self, features):
        if features.ndim != 4:
            raise ValueError(
                "OCIAROIAdapter expects an NCHW tensor, got shape {}".format(
                    tuple(features.shape)
                )
            )

        batch_size, channels, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2).contiguous()
        reduced_tokens = self.down_proj(self.input_norm(tokens))
        enhanced_tokens = self.ocia(reduced_tokens)
        delta_tokens = self.up_proj(enhanced_tokens - reduced_tokens)
        delta = delta_tokens.transpose(1, 2).reshape(
            batch_size, channels, height, width
        )
        residual_gate = self.max_residual_scale * torch.tanh(
            self.residual_scale
        )
        return features + residual_gate.to(features.dtype) * delta.to(features.dtype)
