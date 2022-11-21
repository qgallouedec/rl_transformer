import numpy as np
import pytest
import torch
from gym import spaces

from rl_transformer.utils import get_space_shape, get_space_size, preprocess

SPACES = [
    spaces.Box(-2, 2, (3,)),
    spaces.Discrete(3),
]


def test_get_space_size():
    assert get_space_size(spaces.Box(-2, 2, (3,))) == 3
    assert get_space_size(spaces.Discrete(3)) == 3


def test_get_space_shape():
    assert get_space_shape(spaces.Box(-2, 2, (3,))) == (3,)
    assert get_space_shape(spaces.Discrete(3)) == tuple()


@pytest.mark.parametrize("space", SPACES)
def test_preprocess(space):
    tensor = torch.tensor(space.sample())
    output_tensor = preprocess(tensor, space)
    assert output_tensor.dtype == torch.float32
    assert output_tensor.shape == (get_space_size(space),)

    tensor = torch.tensor(np.array([space.sample() for _ in range(10)]))
    output_tensor = preprocess(tensor, space)
    assert output_tensor.dtype == torch.float32
    assert output_tensor.shape == (10, get_space_size(space))
