import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from mockSQLenv import mockSQLenv
import const

from gymnasium.envs.registration import register
from mockSQLenv import mockSQLenv  
from stable_baselines3 import A2C
import gymnasium as gym

# Register the environment if not already registered elsewhere in your application
register(
    id='mockSQLenv-0',
    entry_point='mockSQLenv:mockSQLenv',
)

# Create the environment using gym.make
env = gym.make('mockSQLenv-0')

# Initialize the model with A2C algorithm
model = A2C("MlpPolicy", env, verbose=1)

# Reset the environment and capture the initial observation and info
observation, info = env.reset()
print("Observation:", observation)
print("Info:", info)

# Sample an action from the environment's action space
action = env.action_space.sample()
# Perform the action in the environment
observation, reward, done, truncated, info = env.step(action)
print("Observation:", observation)
print("Reward:", reward)
print("Done:", done)
print("Info:", info)

# Perform learning
try:
    model.learn(total_timesteps=20000)
except Exception as e:
    print(f"An error occurred during training: {str(e)}")

# Close the environment when done
env.close()

