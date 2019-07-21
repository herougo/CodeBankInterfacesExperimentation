# from https://github.com/hill-a/stable-baselines
import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import os

env = gym.make('CartPole-v1')
# Vectorized environments allow to easily multiprocess training
# we demonstrate its usefulness in the next examples
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

model = PPO2(MlpPolicy, env, verbose=1)
# Train the agent
model.learn(total_timesteps=10)

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)

# code bank stuff

def save(stable_baselines_model, model_dir, file_name='ppo2'):
    
    file_path = os.path.join(model_dir, file_name)
    stable_baselines_model.save(file_path)

def load(model_dir, file_name='ppo2'):
    file_path = os.path.join(model_dir, file_name)
    return PPO2.load(file_path)

save(model, '/tmp/')
model2 = load('/tmp/')