import numpy as np
import torch
import torch.nn.functional as F
from gym import spaces

NUMPY_TO_TORCH_DTYPE = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}


def get_space_size(space: spaces.Space) -> int:
    """
    Return de size of the space.

    Args:
        space (spaces.Space): The space

    Returns:
        int: The size of the space
    """
    if isinstance(space, spaces.Discrete):
        return space.n
    elif isinstance(space, spaces.Box):
        return np.prod(space.shape)
    else:
        raise NotImplementedError


def get_space_shape(space: spaces.Space) -> int:
    """
    Return de shape of the space.

    Args:
        space (spaces.Space): The space

    Returns:
        int: The shape of the space
    """
    if isinstance(space, spaces.Discrete):
        return tuple()
    elif isinstance(space, spaces.Box):
        return space.shape
    else:
        raise NotImplementedError


def preprocess(tensor: torch.Tensor, space: spaces.Space) -> torch.Tensor:
    """
    Preprocess a tensor

    For Box, convert to float32
    For Discrete, make one_hot and convert to float32

    Args:
        tensor (torch.Tensor): The input tensor of shape (..., space_shape)
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
