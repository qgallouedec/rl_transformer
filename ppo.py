import random

import gym
import numpy as np
import torch
import torch.backends.cudnn
from torch import nn, optim
from torch.distributions.categorical import Categorical


def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


seed = 1
env_id = "CartPole-v0"
num_envs = 4
num_steps = 128
num_minibatches = 4
update_epochs = 4
learning_rate = 2.5e-4
total_timesteps = 100_000
gamma = 0.99
gae_lambda = 0.95
ent_coef = 0.01
vf_coef = 0.5
clip_coef = 0.2
max_grad_norm = 0.5

batch_size = int(num_envs * num_steps)
minibatch_size = int(batch_size // num_minibatches)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# env setup
envs = gym.vector.SyncVectorEnv([make_env(env_id, seed + i) for i in range(num_envs)])
assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


agent = Agent(envs).to(device)
optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

# ALGO Logic: Storage setup
obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
logprobs = torch.zeros((num_steps, num_envs)).to(device)
rewards = torch.zeros((num_steps, num_envs)).to(device)
dones = torch.zeros((num_steps, num_envs)).to(device)
values = torch.zeros((num_steps, num_envs)).to(device)

# TRY NOT TO MODIFY: start the game
global_step = 0
next_obs = torch.Tensor(envs.reset()).to(device)
next_done = torch.zeros(num_envs).to(device)
num_updates = total_timesteps // batch_size

for update in range(1, num_updates + 1):
    frac = 1.0 - (update - 1.0) / num_updates
    lrnow = frac * learning_rate
    optimizer.param_groups[0]["lr"] = lrnow

    # Collect interactions
    for step in range(0, num_steps):
        global_step += 1 * num_envs
        obs[step] = next_obs
        dones[step] = next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)

        values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, done, info = envs.step(action.cpu().numpy())
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

        for item in info:
            if "episode" in item.keys():
                print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                break

    # bootstrap value if not done
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)

    advantages = torch.zeros_like(rewards).to(device)
    last_gae_lambda = 0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            non_terminal = next_done
            next_values = next_value
        else:
            non_terminal = dones[t + 1]
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * (1.0 - non_terminal) - values[t]
        advantages[t] = last_gae_lambda = delta + gamma * gae_lambda * (1.0 - non_terminal) * last_gae_lambda
    returns = advantages + values

    # flatten the batch
    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    # Optimizing the policy and value network
    b_inds = np.arange(batch_size)

    for epoch in range(update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]

            _, new_logprob, entropy, new_value = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
            logratio = new_logprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            mb_advantages = b_advantages[mb_inds]
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            new_value = new_value.view(-1)

            v_loss_unclipped = (new_value - b_returns[mb_inds]) ** 2
            v_clipped = b_values[mb_inds] + torch.clamp(new_value - b_values[mb_inds], -0.2, 0.2)
            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

envs.close()
