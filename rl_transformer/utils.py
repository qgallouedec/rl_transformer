import numpy as np
import torch
import torch.nn.functional as F
from gym import spaces


def get_space_size(space: spaces.Space) -> int:
    """Return de size of the space.

    Args:
        space (spaces.Space): The space

    Returns:
        int: The size of the space
    """
    if isinstance(space, spaces.Discrete):
        return space.n
    elif isinstance(space, spaces.Box):
        return np.prod(space.shape)


def preprocess(tensor: torch.Tensor, space: spaces.Space) -> torch.Tensor:
    """Preprocess a tensor

    Args:
        tensor (torch.Tensor): The input tensor
        space (spaces.Space): The space from which the tensor is from

    Raises:
        NotImplementedError: Is niether of Discrete or Box

    Returns:
        torch.Tensor: The output Tensor
    """
    if isinstance(space, spaces.Box):
        return tensor.float()
    elif isinstance(space, spaces.Discrete):
        return F.one_hot(tensor.long(), num_classes=space.n).float()
    else:
        raise NotImplementedError
