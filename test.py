import gym
import d4rl  # Import required to register environments
import numpy as np

# Create the environment
env = gym.make("maze2d-open-v0")

# keys = ['actions', 'infos/goal', 'infos/qpos',
# 'infos/qvel', 'observations', 'rewards', 'terminals', 'timeouts']
dataset = env.get_dataset()
ep_ends = dataset["timeouts"].nonzero()[0]
ep_starts = np.concatenate(([0], ep_ends + 1))
ep_ends = np.concatenate((ep_ends, [1000000]))

trajectories = []
for ep_start, ep_end in zip(ep_starts, ep_ends):
    trajectory = np.hstack((dataset["observations"][ep_start:ep_end], dataset["actions"][ep_start:ep_end])).flatten()
    print(trajectory.shape)
    if trajectory.shape[0] > 0:
        trajectories.append(trajectory)

print(len(trajectories))
