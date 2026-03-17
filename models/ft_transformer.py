import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------
# GEGLU Activation (SOTA FFN)
# -------------------------------
class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


# -------------------------------
# Feature Embedding
# -------------------------------
class FeatureEmbedding(nn.Module):
    def __init__(self, num_features, d_model):
        super().__init__()
        self.id_embedding = nn.Embedding(num_features, d_model)
        self.value_projection = nn.Linear(1, d_model)

    def forward(self, x):
        B, F = x.shape
        ids = torch.arange(F, device=x.device).unsqueeze(0).repeat(B, 1)
        return self.id_embedding(ids) + self.value_projection(x.unsqueeze(-1))


# -------------------------------
# FT Transformer Block
# -------------------------------
class FTBlock(nn.Module):
    def __init__(self, d_model, heads, dropout):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            d_model,
            heads,
            batch_first=True,
            dropout=dropout
        )

        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):

        # Pre-norm attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=~mask.bool())
        x = x + self.dropout(attn_out)

        # Pre-norm FFN
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        x = x + self.dropout(ff_out)

        return x


# -------------------------------
# FT Transformer Model
# -------------------------------
class FTTransformer(nn.Module):
    def __init__(self, num_features, config):
        super().__init__()

        d_model = config["embedding_dim"]

        # Feature gating
        self.gate = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.Sigmoid()
        )

        self.embedding = FeatureEmbedding(num_features, d_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.blocks = nn.ModuleList([
            FTBlock(d_model, config["num_heads"], config["dropout"])
            for _ in range(config["num_layers"])
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x, mask):

        # Feature gating
        x = x * self.gate(x)

        x = self.embedding(x)

        B = x.size(0)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        cls_mask = torch.ones((B, 1), device=mask.device)
        mask = torch.cat([cls_mask, mask], dim=1)

        for block in self.blocks:
            x = block(x, mask)

        x = x[:, 0]

        return self.head(x)