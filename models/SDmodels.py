import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.Gsupport import ConvD


class DiffusionSchedule:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.beta = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def add_noise(self, radius_vector, t):
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t]).to(radius_vector.device)
        sqrt_one_minus = torch.sqrt(1 - self.alpha_bar[t]).to(radius_vector.device)
        noise = torch.randn_like(radius_vector)
        noisy = sqrt_alpha_bar * radius_vector + sqrt_one_minus * noise
        return noisy, noise


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=t.device)
            * (-torch.log(torch.tensor(10000.0)) / (half_dim - 1))
        )
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class UNetEncoderMultiScale(nn.Module):
    def __init__(self, emb_dim=1024):
        super().__init__()
        self.block1 = ConvD(1, 8, first=True)
        self.block2 = ConvD(8, 16)
        self.block3 = ConvD(16, 32)
        self.block4 = ConvD(32, 64)
        self.pool = nn.MaxPool3d(2)

        self.emb1 = nn.Linear(8, emb_dim // 4)
        self.emb2 = nn.Linear(16, emb_dim // 4)
        self.emb3 = nn.Linear(32, emb_dim // 4)
        self.emb4 = nn.Linear(64, emb_dim // 4)
        self.act = nn.ReLU()

    def _embed(self, x, linear):
        pooled = F.adaptive_avg_pool3d(x, 1).flatten(1)
        return self.act(linear(pooled))

    def forward(self, x):
        x1 = self.pool(self.block1(x))
        x2 = self.pool(self.block2(x1))
        x3 = self.pool(self.block3(x2))
        x4 = self.pool(self.block4(x3))
        return torch.cat(
            [
                self._embed(x1, self.emb1),
                self._embed(x2, self.emb2),
                self._embed(x3, self.emb3),
                self._embed(x4, self.emb4),
            ],
            dim=-1,
        )


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * dim, dim),
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x, context):
        x_ln = self.ln1(x)
        out, _ = self.attn(query=x_ln, key=context, value=context)
        x = x + out
        x = x + self.ffn(self.ln2(x))
        return x


class DiTRadiusPredictor(nn.Module):
    def __init__(
        self, radius_dim=4096, img_emb_dim=1024, time_emb_dim=128, num_layers=8
    ):
        super().__init__()
        self.radius_proj = nn.Linear(1, img_emb_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, radius_dim, img_emb_dim))
        self.time_embedding = SinusoidalPositionEmbeddings(time_emb_dim)
        self.img_time_proj = nn.Linear(img_emb_dim + time_emb_dim, img_emb_dim)
        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(dim=img_emb_dim, num_heads=8)
                for _ in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(img_emb_dim)
        self.final = nn.Linear(img_emb_dim, 1)

    def forward(self, noisy_radius, global_img_emb, t):
        x = self.radius_proj(noisy_radius.unsqueeze(-1)) + self.pos_embedding
        time_emb = self.time_embedding(t)
        global_ctx = self.img_time_proj(
            torch.cat([global_img_emb, time_emb], dim=-1)
        ).unsqueeze(1)
        for layer in self.layers:
            x = layer(x, global_ctx)
        x = self.norm_out(x)
        return self.final(x).squeeze(-1)


class ConditionalDiffusionModel_DiT_v2(nn.Module):
    def __init__(self, radius_dim=4096, emb_dim=1024, time_emb_dim=128, num_layers=8):
        super().__init__()
        self.img_encoder = UNetEncoderMultiScale(emb_dim=emb_dim)
        self.radius_predictor = DiTRadiusPredictor(
            radius_dim=radius_dim,
            img_emb_dim=emb_dim,
            time_emb_dim=time_emb_dim,
            num_layers=num_layers,
        )

    def forward(self, noisy_radius, image, t):
        global_img_emb = self.img_encoder(image)
        return self.radius_predictor(noisy_radius, global_img_emb, t)
