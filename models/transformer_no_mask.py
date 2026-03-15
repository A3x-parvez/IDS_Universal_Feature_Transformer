import torch
import torch.nn as nn


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

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class TransformerNoMask(nn.Module):

    def __init__(self, num_features, config):
        super().__init__()

        self.embedding = FeatureEmbedding(
            num_features,
            config["embedding_dim"]
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                config["embedding_dim"],
                config["num_heads"],
                config["dropout"]
            )
            for _ in range(config["num_layers"])
        ])

        self.classifier = nn.Sequential(
            nn.Linear(config["embedding_dim"], 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x, mask):

        x = self.embedding(x)

        for block in self.blocks:
            x = block(x)

        x = x.mean(dim=1)

        return self.classifier(x)