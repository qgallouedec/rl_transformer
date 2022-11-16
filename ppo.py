import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


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
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def get_advantages(rewards, num_steps, next_done, next_value, dones, values, gamma, gae_lambda):
    rewards = rewards.unsqueeze(1)
    dones = dones.unsqueeze(1)
    values = values.unsqueeze(1)
    advantages = torch.zeros_like(rewards)
    last_gae_lamda = 0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t]
            nextvalues = values[t]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t - 1]
        advantages[t - 1] = delta + gamma * gae_lambda * nextnonterminal * last_gae_lamda
        last_gae_lamda = advantages[t - 1]
    return advantages


def get_explained_var(b_values, b_returns):
    y_pred, y_true = b_values.numpy(), b_returns.numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    return explained_var


if __name__ == "__main__":
    env_id = "CartPole-v1"

    total_timesteps = 20_000
    num_steps = 128
    batch_size = int(num_steps)
    minibatch_size = batch_size // 4
    update_epochs = 4

    learning_rate = 2.5e-4
    seed = 1

    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5

    # TRY NOT TO MODIFY: seeding
    np.random.seed(seed)
    torch.manual_seed(seed)

    # env setup
    env = make_env(env_id, seed)

    agent = Agent(env)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    observations = torch.zeros((num_steps + 1, *env.observation_space.shape))
    actions = torch.zeros((num_steps + 1, *env.action_space.shape))
    logprobs = torch.zeros((num_steps + 1))
    rewards = torch.zeros((num_steps + 1))
    dones = torch.zeros((num_steps + 1))
    values = torch.zeros((num_steps + 1))

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    obs = torch.Tensor(env.reset())
    done = False
    num_updates = total_timesteps // batch_size

    for update in range(1, num_updates + 1):
        # Annealing the rate
        new_lr = (1.0 - (update - 1.0) / num_updates) * learning_rate
        optimizer.param_groups[0]["lr"] = new_lr
        observations[0] = torch.Tensor(obs)
        dones[0] = torch.Tensor([done])
        step = 0
        while step < num_steps:

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(observations[step])

            values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            obs, reward, done, info = env.step(action.numpy())

            if done:
                obs = env.reset()

            step += 1
            global_step += 1

            rewards[step] = reward
            observations[step] = torch.Tensor(obs)
            dones[step] = done

            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(observations[step]).reshape(1, -1)

        advantages = get_advantages(rewards, num_steps + 1, done, next_value, dones, values, gamma, gae_lambda)
        returns = advantages + values.unsqueeze(1)

        # flatten the batch
        b_obs = observations[:-1].reshape((-1,) + env.observation_space.shape)
        b_logprobs = logprobs[:-1].reshape(-1)
        b_actions = actions[:-1].reshape((-1,) + env.action_space.shape)
        b_advantages = advantages[:-1].reshape(-1)
        b_returns = returns[:-1].reshape(-1)
        b_values = values[:-1].reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                # Norm advantages:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                # Clip V-loss
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -clip_coef, clip_coef)
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

        explained_var = get_explained_var(b_values, b_returns)

    env.close()
