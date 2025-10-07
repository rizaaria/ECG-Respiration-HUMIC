# app/models/MAE/mae_builder.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed1D(nn.Module):
    def __init__(self, in_chans: int = 1, patch_size: int = 16, embed_dim: int = 128):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, L)
        x = self.proj(x)              # (B, D, L/patch)
        x = x.permute(0, 2, 1)        # (B, N, D)
        return x

def _resize_pos_embed(pos: torch.Tensor, N: int) -> torch.Tensor:
    """Resize positional embedding along sequence dim to match N patches."""
    if pos.shape[1] == N:
        return pos
    pos_t = pos.transpose(1, 2)                            # (1, D, N0)
    pos_resized = F.interpolate(pos_t, size=N, mode="linear", align_corners=False)
    return pos_resized.transpose(1, 2)                     # (1, N, D)

class TransformerMAE(nn.Module):
    """
    Versi inference-friendly:
    - pos_embed akan di-resize dinamis jika jumlah patch (N) berbeda saat infer.
    """
    def __init__(
        self,
        dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 6,
        patch_size: int = 16,
        dropout: float = 0.1,
        mask_ratio: float = 0.5,
        n_classes: int = 5,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.mask_ratio = mask_ratio

        self.patch_embed = PatchEmbed1D(in_chans=1, patch_size=patch_size, embed_dim=dim)
        # placeholder kecil; akan disesuaikan saat load checkpoint
        self.pos_embed = nn.Parameter(torch.zeros(1, 2, dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 4, dropout=dropout, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # decoder tidak dipakai saat infer, tetap disediakan agar state_dict cocok
        self.decoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, patch_size),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_classes),
        )

    def random_masking(self, x: torch.Tensor):
        # x: (B, N, D)
        B, N, D = x.shape
        k = int(self.mask_ratio * N)
        keep = max(1, N - k)
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :keep]
        batch_idx = torch.arange(B, device=x.device)[:, None]
        x_keep = x[batch_idx, ids_keep]
        return x_keep, ids_keep

    def forward(self, x: torch.Tensor):
        # x: (B, 1, L)
        patches = self.patch_embed(x)                # (B, N, D)
        B, N, D = patches.shape
        pos = _resize_pos_embed(self.pos_embed, N)   # (1, N, D)
        patches = patches + pos

        x_keep, _ = self.random_masking(patches)     # (B, keep, D)
        x_enc = self.encoder(x_keep.permute(1, 0, 2)).permute(1, 0, 2)  # (B, keep, D)
        pooled = x_enc.mean(dim=1)                   # (B, D)
        logits = self.classifier(pooled)             # (B, C)
        rec = self.decoder(x_enc)                    # (B, keep, patch) — tidak dipakai
        return logits, rec
