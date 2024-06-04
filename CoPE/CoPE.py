# Contextual Position Encoding https://arxiv.org/pdf/2405.18719

import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class CoPEViT(nn.Module):
    def __init__(self, embed_dim, num_patches):
        super(CoPEViT, self).__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches

        # Linear transformations for query, key, value
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # Learnable position embeddings (B, D, N+1)
        self.position_embeddings = nn.Parameter(
            torch.randn(1, embed_dim, num_patches + 1), requires_grad=True
        )

        # Sigmoid function for gating
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, N, D = x.shape  # Batch size, number of patches+1, embedding dimension
        q = self.query(x)  # (B, N, D)
        k = self.key(x)  # (B, N, D)
        v = self.value(x)  # (B, N, D)

        # Compute gate values Eq. 3 of the paper
        gate_values = self.sigmoid(torch.matmul(q, k.transpose(-2, -1)))  # (B, N, N)

        gate_values = gate_values / math.sqrt(N)

        # Compute contextual positions Eq. 4
        pos = gate_values.flip(-1).cumsum(dim=-1).flip(-1)
        pos = pos.clamp(max=N - 1)

        # Interpolation calculation
        pos_ceil = pos.ceil().long()
        pos_floor = pos.floor().long()

        # q.e[p_{ij}]
        logits_int = torch.matmul(q, self.position_embeddings)  # (B, N, N)

        logits_ciel = logits_int.gather(-1, pos_ceil)
        logits_floor = logits_int.gather(-1, pos_floor)

        # Eq 5
        w = pos_ceil - pos
        pos_embeddings = w * logits_ciel + (1 - w) * logits_floor  # (B, N, N)

        # Adjust attention weights
        attention_scores = (
            torch.matmul(q, k.transpose(-2, -1)) + pos_embeddings
        )  # (B, N, N)

        # Eq. 6
        attention_weights = F.softmax(attention_scores, dim=-1)  # (B, N, N)

        # Compute attention output
        attention_output = torch.matmul(attention_weights, v)  # (B, N, D)

        return attention_output
