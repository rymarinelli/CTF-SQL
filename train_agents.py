from train import run_simulation
import const 
import pandas as pd
from double_Learning_agent import DoubleQLearningAgent
from ucb_exploration_mixin import UCBExplorationMixin
from ubc_agent import UCBAgent

from  agent import Agent

dl =  DoubleQLearningAgent(const.actions)
#ucb = UCBAgent(const.actions, verbose  = False)

print(UCBAgent.__mro__)

#run_simulation(title = "base_agt",
#                   agent_class = Agent,
#                   n_simulations = 1, 
#                   n_episodes_training = 10,
#                   flag_reward = 10, 
#                   query_reward = -1 , 
#                   exploration_train = .1,
#                   learningrate = .1,
#                   discount = .9,
#                   max_steps = 10)

run_simulation(title = "dl_base",
                   agent_class  = DoubleQLearningAgent,
                   n_simulations = 10,
                   n_episodes_training = 10,
                   flag_reward = 10,
                   query_reward = -1 ,
                   exploration_train = .1,
                   learningrate = .1,
                   discount = .9,
                   max_steps = 1000)

run_simulation(title = "ucb_base",
                   agent_class  = UCBAgent,
                   n_simulations = 1,
                   n_episodes_training = 10,
                   flag_reward = 10,
                   query_reward = -1 ,
                   exploration_train = .1,
                   learningrate = .1,
                   discount = .9,
                   max_steps = 100)
