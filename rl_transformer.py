import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """
    Scaled dot product.

    softmax(QK^t / sqrt(d_k)) * V
    where Q is the query, K is the key, d_k is the number of features

    Args:
        query (Tensor): Query of shape (N, L, H_in)
        key (Tensor): Key of shape (N, L, H_in)
        value (Tensor): Value of shape (N, L, H_in)
        mask (Tensor, optional): Mask of shape (L, L) applied after scale.

    Returns:
        Tensor: Output of shape (N, L, H_in)
    """
    assert query.shape == key.shape and query.shape == value.shape
    temp = query.bmm(key.transpose(1, 2))  # (N, L, H_in) -> (N, L, L)
    in_features = query.size(-1)
    scale = in_features**0.5
    temp = temp / scale  # type: Tensor
    if mask is not None:
        temp.masked_fill(mask, value=0.0)  # set 0.0 where mask is True
    softmax = F.softmax(temp, dim=-1)
    return softmax.bmm(value)  # (N, L, L) -> (N, L, H_in)


class PositionalEncoder(nn.Module):
    """
    Positional encoder.

    Adds PE to the input where

    PE(pos, 2i  ) = sin(pos/1e4^(2i/d))
    PE(pos, 2i+1) = cos(pos/1e4^(2i/d))

    Args:
        in_features (int): Input size which is the size of the embedding vector
        dropout (float): Dropout probability
        max_len (int): Maximum length of the sequence that the positional encoder can process
    """

    def __init__(self, in_features: int, dropout: float, max_len: int) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, in_features, 2) * (-math.log(10000.0) / in_features))
        pe = torch.zeros(max_len, in_features)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add the positional encoding to the input.

        Args:
            x (Tensor): Input of shape (N, L, H_in)

        Returns:
            Tensor: Output of shape (N, L, H_in). It corresponds to the
                input, to which we have added the positional encoding
        """
        seq_len = x.shape[1]
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class AttentionHead(nn.Module):
    """
    Attention head.

    Args:
        in_features (int): Input size which is the size of the embedding vector
        dim_qkv (int): Output dimension of query, key and value
    """

    def __init__(self, in_features: int, dim_qkv: int) -> None:
        super().__init__()
        self.query_net = nn.Linear(in_features, dim_qkv)
        self.key_net = nn.Linear(in_features, dim_qkv)
        self.value_net = nn.Linear(in_features, dim_qkv)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """
        Compute the query, key and value.

        Args:
            query (Tensor): Query of shape (N, L, H_in)
            key (Tensor): Key of shape (N, L, H_in)
            value (Tensor): Value of shape (N, L, H_in)

        Returns:
            Tensor: Output of shape (N, L, H_out)
        """
        query = self.query_net(query)
        key = self.key_net(key)
        value = self.value_net(value)
        return scaled_dot_product_attention(query, key, value)


class MultiAttentionHead(nn.Module):
    """
    Multi-attention head.

    Args:
        in_features (int): Input size which is the size of the embedding vector
        dim_qkv (int): Output dimension of query, key and value
        num_heads (int): Number of attention heads
    """

    def __init__(self, in_features: int, dim_qkv: int, num_heads: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(in_features, dim_qkv) for _ in range(num_heads)])
        self.linear = nn.Linear(num_heads * dim_qkv, in_features)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """
        Compute the output of multi-head attention.

        Args:
            query (Tensor): Query of shape (N, L, H_in)
            key (Tensor): Key of shape (N, L, H_in)
            value (Tensor): Value of shape (N, L, H_in)

        Returns:
            Tensor: Output of shape (N, L, H_in)
        """
        x = torch.cat([head(query, key, value) for head in self.heads], dim=-1)  # (N, L, num_head * dim_qkv)
        return self.linear(x)  # (N, L, num_head * dim_qkv) -> (N, L, H_in)


class TransformerLayer(nn.Module):
    """
    Transformer layer.

    Args:
        in_features (int): Input size which is the size of the embedding vector
        num_heads (int): Number of attention heads
        dim_feedforward (int): Feed-forward hidden size
        dropout (float): Dropout probability
    """

    def __init__(self, in_features: int, num_heads: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        dim_qkv = max(in_features // num_heads, 1)
        self.multi_head_attention1 = MultiAttentionHead(in_features, dim_qkv, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.multi_head_attention2 = MultiAttentionHead(in_features, dim_qkv, num_heads)
        self.dropout2 = nn.Dropout(dropout)
        self.norm_layer = nn.LayerNorm(in_features)
        self.feed_forward = nn.Sequential(
            nn.Linear(in_features, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, in_features),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the output of the transformer decoder layer.

        Args:
            x (Tensor): Input of shape (N, L, H_in)

        Returns:
            Tensor: Output of shape (N, L, H_in)
        """
        # Residual multihead-attention and layer norm 1
        residual = x
        x = self.multi_head_attention1(x, x, x)
        x = self.dropout1(x)
        x = x + residual
        x = self.norm_layer(x)
        # Residual multihead-attention and layer norm 2
        residual = x
        x = self.multi_head_attention2(x, x, x)
        x = self.dropout2(x)
        x = x + residual
        x = self.norm_layer(x)
        # Residual feed-forward and layer-norm
        residual = x
        x = self.feed_forward(x)
        x = x + residual
        x = self.norm_layer(x)
        return x


class Transformer(nn.Module):
    """
    Transformer.

    Args:
        in_features (int): Input size which is the size of the embedding vector
        out_features (int): Output size.
        num_layers (int): Number of transformer layers
        num_heads (int): Number of attention heads
        dim_feedforward (int): Feed-forward hidden size
        dropout (float): Dropout probability
        max_len (int): Maximum length of the sequence that the positional encoder can process
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_layers: int = 4,
        num_heads: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 500,
    ) -> None:
        super().__init__()
        self.positional_encoder = PositionalEncoder(in_features, dropout, max_len)
        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(in_features, num_heads, dim_feedforward, dropout) for _ in range(num_layers)]
        )
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the output of the transformer.

        Args:
            x (Tensor): Input of shape (N, L, H_in)

        Returns:
            Tensor: Output of shape (N, L, H_out)
        """
        # Positional encoding
        x = self.positional_encoder(x)
        # Transformer layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        x = self.linear(x)
        x = torch.softmax(x, dim=-1)
        return x
