import torch

from torch import nn
from torchinfo import summary
from CoPE import CoPE


# 1. (Eq 1) Create a class which subclasses nn.Module
class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.

    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """

    # 2. Initialize the class with appropriate variables
    def __init__(
        self, in_channels: int = 3, patch_size: int = 16, embedding_dim: int = 768
    ):
        super().__init__()

        self.patch_size = patch_size

        # 3. create a layer to turn an image into patches
        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

        # 4. create a layer to flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    # 5. define the forward method
    def forward(self, x):
        # create assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert (
            image_resolution % self.patch_size == 0
        ), f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"

        # perform the forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        return x_flattened.permute(0, 2, 1)


# Equation 2 of ViT - MSA block
# 1. Create a class that inherits from nn.Module
class MultiheadSelfAttentionBlock(nn.Module):
    """Creates a multi-head self-attention block ("MSA block" for short)"""

    # 2. Initialize the class with hyperparameters from table 1
    def __init__(
        self,
        embedding_dim: int = 768,  # hidden size D from table 1 for ViT-Base
        num_heads: int = 12,  # Heads from table 1 for ViT base
        attn_dropout: float = 0,  # doesn't look like the paper uses any dropout in MSA blocks
    ):
        super().__init__()

        # 3. create Norm Layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # 4. create teh multi.head attention (MSA) layer
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

    # 5. create a forward() method to pass the data through the layers
    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(
            query=x,  # query embeddings
            key=x,  # key embeddings
            value=x,  # value embeddings
            need_weights=False,
        )

        return attn_output


# Equation 3 of ViT - MLP block


# 1. create a class that inherits from nn.Module
class MLPBlock(nn.Module):
    """Creates a layer normalized multilayer perceptron block ('MLP block' for short)."""

    # 2. Initialize the class with hyperparameters from tables 1 and 3
    def __init__(
        self,
        embedding_dim: int = 768,  # Hidden Size D from Table 1 for ViT-Base
        mlp_size: int = 3072,  # MLP Size from Table 1 for ViT-Base
        dropout: float = 0,  # Dropout from table 3 for ViT base
    ):
        super().__init__()

        # 3. Create layer norm
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # 4. MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, out_features=embedding_dim),
            nn.Dropout(p=dropout),
        )

    # 5. forword method
    def forward(self, x):
        return self.mlp(self.layer_norm(x))


# Transformer Encoder


class TransformerEncoderBlock(nn.Module):
    """Creates a Transformer Encoder block."""

    # 2. Initialize the class with hyper params from tables 1 and 3
    def __init__(
        self,
        embedding_dim: int = 768,  # hidden size D
        num_heads: int = 12,
        mlp_size: int = 3072,
        mlp_dropout: float = 0.1,
        attn_dropout: float = 0,
    ):
        super().__init__()

        # 3. create MSA block (eq 2)
        self.msa_block = MultiheadSelfAttentionBlock(
            embedding_dim=embedding_dim, num_heads=num_heads, attn_dropout=attn_dropout
        )

        # 4. create MLP block (eq 3)
        self.mlp_block = MLPBlock(
            embedding_dim=embedding_dim, mlp_size=mlp_size, dropout=mlp_dropout
        )

    # create a forward method
    def forward(self, x):

        # 6. Create residual connection for MSA block (add the input to the output)
        x = self.msa_block(x) + x

        # 7. Create residual connection for MLP block (add the input to the output)
        x = self.mlp_block(x) + x

        return x


# ViT class


class ViT_Base(nn.Module):
    """Creates a Vision Transformer architecture with ViT-Base hyperparameters by default."""

    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        patch_size: int = 16,
        num_transformer_layers: int = 12,
        embedding_dim: int = 768,
        num_heads: int = 12,
        mlp_size: int = 3072,
        mlp_dropout: float = 0.1,
        attn_dropout: float = 0,
        embedding_dropout: float = 0.1,
        num_classes: int = 1000,  # Default for ImageNet but can customize this
    ) -> None:
        super().__init__()

        # Make sure the image size is divisible by the patch size
        assert (
            img_size % patch_size == 0
        ), f"Image size must be divisible by patch size, image size: {img_size}, patch size: {patch_size}"

        # calculate the num of patchs
        self.num_patches = (img_size * img_size) // patch_size**2

        # Create learnable class embedding (needs to go at front of sequence of patch embeddings)
        self.class_embedding = nn.Parameter(
            data=torch.randn(1, 1, embedding_dim), requires_grad=True
        )

        # create learbable position embedding
        self.position_embedding = CoPE.CoPEViT(
            embed_dim=embedding_dim, num_patches=self.num_patches
        )

        # create embedding dropout value
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        # create patch embedding layer
        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels, patch_size=patch_size, embedding_dim=embedding_dim
        )

        # create Transformer Encoder blocks
        self.transformer_encoder = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_size=mlp_size,
                    mlp_dropout=mlp_dropout,
                )
                for _ in range(num_transformer_layers)
            ]
        )

        # create classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes),
        )

    def forward(self, x):

        # get batch size
        batch_size = x.shape[0]

        # create class token embedding and expand it to match the batch size (eq 1)
        class_token = self.class_embedding.expand(batch_size, -1, -1)

        # create patch embedding eq 1
        x = self.patch_embedding(x)

        # concat class embedding and patch embedding
        x = torch.cat((class_token, x), dim=1)

        # create Contextual Position Encoding (CoPE)
        cope = self.position_embedding(x)

        # add CoPE position encoding/embedding to patch embedding eq 1
        x = cope + x

        # run embedding dropout (Appendix B.1)
        x = self.embedding_dropout(x)

        # Pass patch, position and class embedding through transformer encoder layers (eq 2 & 3)
        x = self.transformer_encoder(x)

        # put 0 index logit through the classifier eq 4
        x = self.classifier(x[:, 0])

        return x


# Example
# random_image_tensor = torch.randn(1, 3, 224, 224)
# vit = ViT_Base(num_classes=3)
# vit(random_image_tensor)


# # Print a summary of our custom ViT model using torchinfo (uncomment for actual output)

# from torchinfo import summary

# summary(
#     model=vit,
#     input_size=(32, 3, 224, 224),  # (batch_size, color_channels, height, width)
#     # col_names=["input_size"], # uncomment for smaller output
#     col_names=["input_size", "output_size", "num_params", "trainable"],
#     col_width=20,
#     row_settings=["var_names"],
# )
