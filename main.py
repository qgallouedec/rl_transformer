import gym

from rl_transformer.buffer import EpisodeBuffer
from rl_transformer.rl_transformer import ActorCriticTransformer
from rl_transformer.wrappers import TorchWrapper

device = "cuda"
env = TorchWrapper(gym.make("Pendulum-v1"), device)
buffer = EpisodeBuffer(20, env, device)
ac = ActorCriticTransformer(env, d_model=16).to(device)

observation = env.reset()
value = ac.get_value(observation)
buffer.new_episode(observation, value)
for _ in range(5):
    observations, actions = buffer.get_current_episode()
    value, action, log_prob = ac.act(observations, actions)
    observation, reward, done, info = env.step(action)
    buffer.add(action, log_prob, observation, value, reward, done, info)
    if done:
        observation = env.reset()
        value = ac.get_value(observation)
        buffer.new_episode(observation, value)

observations, values, actions, log_probs, rewards, dones, infos, src_key_padding_mask = buffer.sample(5)
actions = actions[:, :-1]  # last action must be discarded (N, L+1, action_shape) -> (N, L, action_spape)

values_, actions, log_prob = ac.forward(observations, actions, src_key_padding_mask)

print(values_, values)
