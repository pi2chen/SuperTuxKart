from pathlib import Path

import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        return nn.functional.cross_entropy(logits, target)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        """
        A single Transformer encoder block with multi-head attention and MLP.

        General outline:
        1. Layer normalization
        2. Multi-head self-attention (use torch.nn.MultiheadAttention with batch_first=True)
        3. Residual connection
        4. Layer normalization
        5. MLP (Linear -> GELU -> Dropout -> Linear -> Dropout)
        6. Residual connection

        Args:
            embed_dim: embedding dimension
            num_heads: number of attention heads
            mlp_ratio: ratio of MLP hidden dimension to embedding dimension
            dropout: dropout probability
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(embed_dim)    

        hidden_dim = int(embed_dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, sequence_length, embed_dim) input sequence

        Returns:
            (batch_size, sequence_length, embed_dim) output sequence
        """
        residual = x
        x = self.norm1(x)
        
        attn_out, _ = self.attn(x, x, x)

        x = residual + attn_out
        
        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int = 64, patch_size: int = 8, in_channels: int = 3, embed_dim: int = 256):
        """
        Convert image to sequence of patch embeddings using a simple approach

        This is provided as a helper for implementing the Vision Transformer.
        You can use this directly in your ViTClassifier implementation.

        Args:
            img_size: size of input image (assumed square)
            patch_size: size of each patch
            in_channels: number of input channels (3 for RGB)
            embed_dim: embedding dimension
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Linear projection of flattened patches
        self.projection = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input images

        Returns:
            (B, num_patches, embed_dim) patch embeddings
        """
        B, C, H, W = x.shape
        p = self.patch_size

        # Reshape into patches: (B, C, H//p, p, W//p, p) -> (B, C, H//p, W//p, p, p)
        x = x.reshape(B, C, H//p, p, W//p, p).permute(0, 1, 2, 4, 3, 5)
        # Flatten patches: (B, C, H//p, W//p, p*p) -> (B, H//p * W//p, C * p * p)
        x = x.reshape(B, self.num_patches, C * p * p)

        # Linear projection
        return self.projection(x)


class MLPClassifierDeepResidual(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 4,
    ):
        """
        An MLP with multiple hidden layers and residual connections

        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int

        Hint - you can add more arguments to the constructor such as:
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers
        """
        super().__init__()

        self.input_dim = h * w * 3
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)

        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)])

        self.output_layer = nn.Linear(hidden_dim, num_classes)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        B = x.size(0)
        x = x.view(B, -1)

        out = self.activation(self.input_layer(x))

        for layer, norm in zip(self.layers, self.norms):
            residual = out
            out = layer(out)
            out = self.dropout(out)
            out = norm(out)
            out = self.activation(out + residual)

        logits = self.output_layer(out)
        return logits


class ViTClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        patch_size: int = 8,
        embed_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        """
        Vision Transformer (ViT) classifier

        Args:
            h: int, height of input image
            w: int, width of input image
            num_classes: int, number of classes

        Hint - you can add more arguments to the constructor such as:
            patch_size: int, size of image patches
            embed_dim: int, embedding dimension
            num_layers: int, number of transformer layers
            num_heads: int, number of attention heads

        Note: You can use the provided PatchEmbedding class. You'll need to implement the TransformerBlock class.
        """
        super().__init__()

        assert h == w

        self.patch_embed = PatchEmbedding(img_size=h, patch_size=patch_size, in_channels=3, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(0.1)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) input image

        Returns:
            tensor (b, num_classes) logits
        """
        B = x.size(0)
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        cls_output = x[:, 0]

        logits = self.head(cls_output)
        return logits


model_factory = {
    "mlp_deep_residual": MLPClassifierDeepResidual,
    "vit": ViTClassifier,
}


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def save_model(model):
    """
    Use this function to save your model in train.py
    """
    for n, m in model_factory.items():
        if isinstance(model, m):
            return torch.save(model.state_dict(), Path(__file__).resolve().parent / f"{n}.th")
    raise ValueError(f"Model type '{str(type(model))}' not supported")


def load_model(model_name: str, with_weights: bool = False, **model_kwargs):
    """
    Called by the grader to load a pre-trained model by name
    """
    r = model_factory[model_name](**model_kwargs)
    if with_weights:
        model_path = Path(__file__).resolve().parent / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            r.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # Limit model sizes since they will be zipped
    model_size_mb = calculate_model_size_mb(r)
    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")
    print(f"Model size: {model_size_mb:.2f} MB")

    return r