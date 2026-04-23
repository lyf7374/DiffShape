import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from utils.Gsupport import ConvD
from datasets.preprocess import process_scan
import torch.nn.functional as F

class ConditionalRadiusDataset(Dataset):
    def __init__(self, skull_paths, GI_all, centers_pre):
        self.skull_paths = skull_paths
        self.GI_all = GI_all
        self.centers_pre = centers_pre
        self.img_shape = (192,192,192)

    def __len__(self):
        return len(self.skull_paths)

    def __getitem__(self, idx):
        x = process_scan(self.skull_paths[idx], norm_method='zs', 
                         output_shape=self.img_shape, center=self.centers_pre[idx])
        x = x.reshape((1, *self.img_shape)).astype(np.float32)
        radius_vector = self.GI_all[idx][:, 3].astype(np.float32)
        return {
            'image': torch.from_numpy(x),
            'radius_vector': torch.from_numpy(radius_vector)
        }
class DiffusionSchedule:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.timesteps = timesteps
        self.beta = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def add_noise(self, radius_vector, t):
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t]).to(radius_vector.device)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t]).to(radius_vector.device)
        noise = torch.randn_like(radius_vector)
        noisy_radius = sqrt_alpha_bar * radius_vector + sqrt_one_minus_alpha_bar * noise
        return noisy_radius, noise
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -(np.log(10000) / (half_dim - 1)))
        emb = t[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)
class UNetEncoder(nn.Module):
    def __init__(self, emb_dim=512):
        super().__init__()
        self.convs = nn.Sequential(
            ConvD(1, 8, first=True),
            ConvD(8, 16),
            ConvD(16, 32),
            ConvD(32, 64),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(64, emb_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.convs(x)


class RadiusTransformer(nn.Module):
    def __init__(self, radius_dim=4096, img_emb_dim=512, time_emb_dim=128, n_heads=8, n_layers=3):
        super().__init__()
        self.radius_dim = radius_dim
        self.input_proj = nn.Linear(1, img_emb_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=img_emb_dim, nhead=n_heads, dim_feedforward=1024, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.img_time_proj = nn.Linear(img_emb_dim + time_emb_dim, img_emb_dim)
        self.final_linear = nn.Sequential(
            nn.Linear(img_emb_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 1)
        )

    def forward(self, noisy_radius, img_emb, time_emb):
        B, N = noisy_radius.shape
        x = noisy_radius.unsqueeze(-1)  # [B, 4096, 1]
        x = self.input_proj(x)  # [B, 4096, img_emb_dim]

        img_time_emb = torch.cat([img_emb, time_emb], dim=-1)  # [B, img_emb_dim + time_emb_dim]
        img_time_emb = self.img_time_proj(img_time_emb).unsqueeze(1)  # [B, 1, img_emb_dim]

        x = x + img_time_emb  # Broadcasting conditioning embedding

        x = x.permute(1, 0, 2)  # Transformer expects [N, B, E]
        x = self.transformer_encoder(x)  # [N, B, E]
        x = x.permute(1, 0, 2)  # [B, N, E]

        output = self.final_linear(x).squeeze(-1)  # [B, 4096]

        return output
class ConditionalDiffusionModel(nn.Module):
    def __init__(self, radius_dim=4096, img_emb_dim=512, time_emb_dim=128):
        super().__init__()
        self.img_encoder = UNetEncoder(img_emb_dim)
        self.time_embedding = SinusoidalPositionEmbeddings(time_emb_dim)
        self.radius_transformer = RadiusTransformer(radius_dim, img_emb_dim, time_emb_dim)

    def forward(self, noisy_radius, image, t):
        img_emb = self.img_encoder(image)  # [B, img_emb_dim]
        time_emb = self.time_embedding(t)  # [B, time_emb_dim]

        predicted_noise = self.radius_transformer(noisy_radius, img_emb, time_emb)

        return predicted_noise




class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, context=None):
        if context is None:
            attn_out, _ = self.attention(x, x, x)
        else:
            attn_output, _ = self.attn(x, context, context)
            x = x + attn_output
        x = x + self.norm1(x)
        x = x + self.ff(self.norm2(x))
        return x

class DiT_RadiusPredictor(nn.Module):
    def __init__(self, radius_dim=4096, img_emb_dim=512, time_emb_dim=128, num_layers=4):
        super().__init__()

        self.radius_proj = nn.Linear(1, img_emb_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, radius_dim, img_emb_dim))
        self.time_embedding = SinusoidalPositionEmbeddings(time_emb_dim)
        
        self.img_time_proj = nn.Linear(img_emb_dim + time_emb_dim, img_emb_dim)

        self.layers = nn.ModuleList([
            DiTBlock(img_emb_dim) for _ in range(num_layers)
        ])

        self.norm_out = nn.LayerNorm(img_emb_dim)
        self.final = nn.Linear(img_emb_dim, 1)

    def forward(self, noisy_radius, img_emb, t):
        B = noisy_radius.size(0)
        radius_emb = self.radius_proj(noisy_radius.unsqueeze(-1))  # [B,4096,emb_dim]

        time_emb = self.time_embedding(t)

        context = self.img_time_proj(torch.cat([img_emb, time_emb], dim=-1)).unsqueeze(1)  # [B,1,emb_dim]

        x = radius_emb + self.pos_embedding  # Corrected and simplified

        for layer in self.layers:
            x = layer(x, context=context)

        x = self.norm_out(x)
        pred_noise = self.final(x).squeeze(-1)

        return pred_noise


class ConditionalDiffusionModel_DiT(nn.Module):
    def __init__(self, radius_dim=4096):
        super().__init__()
        self.img_encoder = UNetEncoder(512)
        self.radius_predictor = DiT_RadiusPredictor(radius_dim=radius_dim)

    def forward(self, noisy_radius, image, t):
        img_emb = self.img_encoder(image)
        predicted_noise = self.radius_predictor(noisy_radius, img_emb, t)
        return predicted_noise

class UNetEncoderMultiScale_v2(nn.Module):
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

    def embed(self, x, linear):
        pooled = F.adaptive_avg_pool3d(x, 1).flatten(1)
        return self.act(linear(pooled))

    def forward(self, x):
        feats = []

        x1 = self.pool(self.block1(x))
        feats.append(self.embed(x1, self.emb1))

        x2 = self.pool(self.block2(x1))
        feats.append(self.embed(x2, self.emb2))

        x3 = self.pool(self.block3(x2))
        feats.append(self.embed(x3, self.emb3))

        x4 = self.pool(self.block4(x3))
        feats.append(self.embed(x4, self.emb4))

        global_emb = torch.cat(feats, dim=-1)

        return global_emb

class CrossAttentionBlock(nn.Module):
    """
    Minimal cross-attn + feed-forward block, in the style of a Transformer block.
    """
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.ln1  = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.ReLU(inplace=True),
            nn.Linear(4*dim, dim),
        )
        self.ln2  = nn.LayerNorm(dim)

    def forward(self, x, context):
        # Cross Attention
        x_ln = self.ln1(x)
        out, _ = self.attn(query=x_ln, key=context, value=context)  
        x = x + out  # residual

        # Feed-forward
        x_ln = self.ln2(x)
        out = self.ffn(x_ln)
        x = x + out

        return x
    
class SinusoidalPositionEmbeddings_v2(nn.Module):
    """As in your original code, or any typical diffusion positional embedding."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # Suppose t is [B], we'll produce [B, dim]
        half_dim = self.dim // 2
        emb = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=t.device) * 
            (-torch.log(torch.tensor(10000.0)) / (half_dim - 1))
        )
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
class DiT_RadiusPredictor_v2(nn.Module):
    def __init__(self, radius_dim=4096, img_emb_dim=1024, time_emb_dim=128, num_layers=8):
        super().__init__()

        self.radius_proj = nn.Linear(1, img_emb_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, radius_dim, img_emb_dim))

        self.time_embedding = SinusoidalPositionEmbeddings_v2(time_emb_dim)
        self.img_time_proj = nn.Linear(img_emb_dim + time_emb_dim, img_emb_dim)

        self.layers = nn.ModuleList([
            CrossAttentionBlock(dim=img_emb_dim, num_heads=8)
            for _ in range(num_layers)
        ])

        self.norm_out = nn.LayerNorm(img_emb_dim)
        self.final = nn.Linear(img_emb_dim, 1)

    def forward(self, noisy_radius, global_img_emb, t):
        B = noisy_radius.size(0)

        x = self.radius_proj(noisy_radius.unsqueeze(-1))
        x = x + self.pos_embedding

        time_emb = self.time_embedding(t)
        global_ctx = torch.cat([global_img_emb, time_emb], dim=-1)
        global_ctx = self.img_time_proj(global_ctx).unsqueeze(1)

        for layer in self.layers:
            x = layer(x, global_ctx)

        x = self.norm_out(x)
        pred_noise = self.final(x).squeeze(-1)
        return pred_noise


class ConditionalDiffusionModel_DiT_v2(nn.Module):
    def __init__(self, radius_dim=4096, emb_dim=1024, time_emb_dim=128, num_layers=8):
        super().__init__()
        self.img_encoder = UNetEncoderMultiScale_v2(emb_dim=emb_dim)
        self.radius_predictor = DiT_RadiusPredictor_v2(
            radius_dim=radius_dim,
            img_emb_dim=emb_dim,
            time_emb_dim=time_emb_dim,
            num_layers=num_layers
        )

    def forward(self, noisy_radius, image, t):
        global_img_emb = self.img_encoder(image)
        predicted_noise = self.radius_predictor(noisy_radius, global_img_emb, t)
        return predicted_noise