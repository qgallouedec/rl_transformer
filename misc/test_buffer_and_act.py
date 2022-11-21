import gym

from rl_transformer.buffer import EpisodeBuffer
from rl_transformer.rl_transformer import ActorCriticTransformer
from rl_transformer.wrappers import TorchWrapper

env = TorchWrapper(gym.make("Pendulum-v1"))

buffer = EpisodeBuffer(20, env)

ac = ActorCriticTransformer(env, d_model=16)


observation = env.reset()
value = 1.0
buffer.new_episode(observation, value)
for _ in range(3000):
    observations, actions = buffer.get_current_episode()
    action = ac.act(observations, actions)
    log_prob = 0.0
    observation, reward, done, info = env.step(action)
    value = 1.0
    buffer.add(action, log_prob, observation, value, reward, done, info)
    if done:
        observation = env.reset()
        value = 1.0
        buffer.new_episode(observation, value)
