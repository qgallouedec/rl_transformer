import matplotlib.pyplot as plt
import torch

from rl_transformer.rl_transformer import PositionalEncoder

d_model = 7
max_len = 12
PE = PositionalEncoder(d_model, max_len, 0.0)

pe = PE(torch.zeros(1, max_len, d_model)).squeeze().numpy().T


fig, ax = plt.subplots()
im = ax.imshow(pe, cmap="bwr")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax.set_xlabel("pos")
ax.set_ylabel("depth")
plt.show()
