import torch
import torch.nn as nn


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
        id_emb = self.id_embedding(ids)
        val_emb = self.value_projection(x.unsqueeze(-1))
        return id_emb + val_emb


# -------------------------------
# Transformer Block
# -------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, heads, dropout):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            d_model,
            heads,
            batch_first=True,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=~mask.bool())
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


# -------------------------------
# Improved Transformer Model
# -------------------------------
class LTMModel_v2(nn.Module):
    def __init__(self, num_features, config):
        super().__init__()

        d_model = config["embedding_dim"]

        # 🔥 Feature gating (IMPORTANT)
        self.feature_gate = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.Sigmoid()
        )

        self.embedding = FeatureEmbedding(num_features, d_model)

        # 🔥 CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.embedding_dropout = nn.Dropout(0.1)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model,
                config["num_heads"],
                config["dropout"]
            )
            for _ in range(config["num_layers"])
        ])

        # 🔥 stronger classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x, mask):

        # -------------------------------
        # Feature Gating
        # -------------------------------
        g = self.feature_gate(x)
        x = x * g

        # -------------------------------
        # Embedding
        # -------------------------------
        x = self.embedding(x)
        x = self.embedding_dropout(x)

        B = x.size(0)

        # -------------------------------
        # Add CLS token
        # -------------------------------
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        # update mask (CLS always valid)
        cls_mask = torch.ones((B, 1), device=mask.device)
        mask = torch.cat([cls_mask, mask], dim=1)

        # -------------------------------
        # Transformer Blocks
        # -------------------------------
        for block in self.blocks:
            x = block(x, mask)

        # -------------------------------
        # Use CLS token instead of mean
        # -------------------------------
        x = x[:, 0]

        return self.classifier(x)