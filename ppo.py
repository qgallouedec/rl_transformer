import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import torch.nn.functional as F


def make_env(env_id, seed):

    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, env.action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x).squeeze(-1)

    def get_action(self, observation, action=None):
        logits = self.actor(observation)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action, probs.log_prob(action)

    def get_log_prob(self, observation, action):
        logits = self.actor(observation)
        probs = Categorical(logits=logits)
        return probs.log_prob(action)

    def get_entropy(self, observation):
        logits = self.actor(observation)
        probs = Categorical(logits=logits)
        return probs.entropy()


def get_advantages(rewards, dones, values, gamma, gae_lambda):
    num_steps = rewards.shape[0] - 1
    advantages = torch.zeros_like(rewards)
    last_gae_lamda = 0
    for t in reversed(range(num_steps)):
        advantages[t] = (
            rewards[t + 1] + gamma * (1.0 - dones[t + 1]) * (values[t + 1] + gae_lambda * last_gae_lamda) - values[t]
        )
        last_gae_lamda = advantages[t]
    return advantages


def get_explained_var(values, returns):
    y_pred, y_true = values.numpy(), returns.numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    return explained_var


if __name__ == "__main__":
    env_id = "CartPole-v1"

    total_timesteps = 20_000
    num_steps = 128
    num_updates = total_timesteps // num_steps
    minibatch_size = num_steps // 4
    update_epochs = 4

    gamma = 0.99
    gae_lambda = 0.95
    learning_rate = 2.5e-4
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5

    # Seeding
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Env setup
    env = make_env(env_id, seed)

    # Agent setup
    agent = Agent(env)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    # Storage setup (num_steps + 1 because we need the terminal values to compute the advantage)
    observations = torch.zeros((num_steps + 1, *env.observation_space.shape))
    actions = torch.zeros((num_steps + 1, *env.action_space.shape), dtype=torch.long)
    log_probs = torch.zeros((num_steps + 1))
    rewards = torch.zeros((num_steps + 1))
    dones = torch.zeros((num_steps + 1))
    values = torch.zeros((num_steps + 1))

    # Init the env
    observation = env.reset()
    done = False

    global_step = 0
    for update in range(num_updates):
        # Annealing the rate
        new_lr = (1.0 - update / num_updates) * learning_rate
        optimizer.param_groups[0]["lr"] = new_lr

        step = 0

        # Store initial
        observations[step] = torch.Tensor(observation)
        dones[step] = done
        with torch.no_grad():
            value = agent.get_value(observations[step])
        values[step] = value

        while step < num_steps:
            # Compute action
            with torch.no_grad():
                action, log_prob = agent.get_action(observations[step])

            # Store
            actions[step] = action
            log_probs[step] = log_prob

            # Step
            observation, reward, done, info = env.step(action.numpy())
            if done:
                observation = env.reset()
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")

            # Update count
            step += 1
            global_step += 1

            # Store
            rewards[step] = reward
            observations[step] = torch.Tensor(observation)
            dones[step] = done
            with torch.no_grad():
                value = agent.get_value(observations[step])
            values[step] = value

        # Compute advanges and return
        advantages = get_advantages(rewards, dones, values, gamma, gae_lambda)
        returns = advantages + values

        # Optimizing the policy and value network
        for epoch in range(update_epochs):
            b_inds = np.random.permutation(num_steps)
            for start in range(0, num_steps, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                b_observations = observations[mb_inds]
                b_actions = actions[mb_inds]
                b_advantages = advantages[mb_inds]
                b_log_probs = log_probs[mb_inds]
                b_values = values[mb_inds]
                b_returns = returns[mb_inds]

                # Policy loss
                b_advantages = (b_advantages - torch.mean(b_advantages)) / (torch.std(b_advantages) + 1e-8)  # norm advantages
                new_log_probs = agent.get_log_prob(b_observations, b_actions)
                ratio = torch.exp(new_log_probs - b_log_probs)
                pg_loss1 = -b_advantages * ratio
                pg_loss2 = -b_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.mean(torch.max(pg_loss1, pg_loss2))

                # Entropy loss
                entropy_loss = torch.mean(agent.get_entropy(b_observations))

                # Clip V-loss
                new_values = agent.get_value(b_observations)
                v_loss_unclipped = (new_values - b_returns) ** 2
                v_clipped = b_values + torch.clamp(new_values - b_values, -clip_coef, clip_coef)
                v_loss_clipped = (v_clipped - b_returns) ** 2
                v_loss = 0.5 * torch.mean(torch.max(v_loss_unclipped, v_loss_clipped))

                # Total loss
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

        explained_var = get_explained_var(returns[:-1], values[:-1])

    env.close()
