import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from stable_baselines3.common.distributions import DiagGaussianDistribution


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
        pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
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
        return x


class ActorCriticTransformer(nn.Module):
    def __init__(self, action_size: int, feature_size: int) -> None:
        super().__init__()
        self.transformer = Transformer(obs_size + action_size, feature_size)
        self.value_net = nn.Sequential(
            nn.Linear(feature_size, feature_size),
            nn.ReLU(),
            nn.Linear(feature_size, 1),
        )
        self.action_net = nn.Linear(feature_size, action_size)
        self.log_std = nn.Parameter(torch.zeros(action_size), requires_grad=True)
        self.action_dist = DiagGaussianDistribution(action_size)

    def forward(self, obs: Tensor, deterministic: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        # Extract features
        features = self.transformer(obs)
        # Critic
        values = self.value_net(features)
        # Actor
        mean_actions = self.action_net(features)
        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        # actions = actions.reshape((-1,)  self.action_space.shape)
        return actions, values, log_prob


if __name__ == "__main__":
    # S0 A1 S1 A2 S2 A3 S3
    batch_size = 8
    seq_len = 499  # number of actions
    action_size = 2
    obs_size = 3
    observations = torch.rand((batch_size, seq_len + 1, obs_size))  # [N, seq_len+1, obs_size]
    actions = torch.rand((batch_size, seq_len, action_size))  # [N, seq_len, act_size]
    # start_seq token
    actions = torch.cat((torch.zeros((batch_size, 1, action_size)), actions), dim=1)  # [N, seq_len, act_size]

    print(actions.shape, observations.shape)

    # stack obs and action
    action_obs = torch.cat((actions, observations), dim=2)
    print(action_obs.shape)
    # policy gets [a_t-2, s_t-2], [a_t-1, s_t-1], to get a_t

    # feature_size = spaces.utils.flatdim(observation_space)
    ac = ActorCriticTransformer(action_size=action_size, feature_size=16)

    actions, values, log_prob = ac(action_obs) #  [N, seq_len, act_size] The action that the agent whould have taken at every step.

    print(actions[0][-1])
