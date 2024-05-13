from gymnasium.envs.registration import register
from mockSQLenv import mockSQLenv
from stable_baselines3 import PPO
from stable_baselines3 import A2C
import const 
import gymnasium as gym

register(
    id='mockSQLenv-0',
    entry_point='mockSQLenv:mockSQLenv',
)


env = gym.make('mockSQLenv-0')
model = A2C("MlpPolicy", env, verbose=1)
observation, info = env.reset()  # Make sure reset also returns two items if needed
print("Observation:", observation)
print("Info:", info)

action = env.action_space.sample()
observation, reward, done, truncated, info = env.step(action)  # This should now work without error
print("Observation:", observation)
print("Reward:", reward)
print("Done:", done)
print("Info:", info)







try:
    model.learn(total_timesteps=20000)
except Exception as e:
    print("An error occurred during training:", str(e))



