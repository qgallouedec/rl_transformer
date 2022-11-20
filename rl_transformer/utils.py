import numpy as np
from gym import spaces


def get_space_dim(space: spaces.Space) -> int:
    if isinstance(space, spaces.Discrete):
        return space.n
    elif isinstance(space, spaces.Box):
        return np.prod(space.shape)
