from train import run_simulation
import const 
import pandas as pd
from double_Learning_agent import DoubleQLearningAgent
#from ucb_exploration_mixin import UCBExplorationMixin
#from ubc_agent import UCBAgent

import agent as agn
agt = agn.Agent(const.actions, verbose=False)
dl =  DoubleQLearningAgent(const.actions)
#ucb = UCBAgent(const.actions)
#print(UCBAgent.__mro__)

run_simulation(title = "base_agt",
                   agt = agt,
                   n_simulations = 10, 
                   n_episodes_training = 10**5,
                   flag_reward = 10, 
                   query_reward = -1 , 
                   exploration_train = .1,
                   learningrate = .1,
                   discount = .9,
                   max_steps = 1000)

#run_simulation(title = "dl_base",
#                   agt = dl,
#                   n_simulations = 10,
#                   n_episodes_training = 10,
#                   flag_reward = 10,
#                   query_reward = -1 ,
#                   exploration_train = .1,
#                   learningrate = .1,
#                   discount = .9,
#                   max_steps = 1000)

#run_simulation(title = "ucb_base",
#                   agt = ucb,
#                   n_simulations = 10,
#                   n_episodes_training = 10,
#                   flag_reward = 10,
#                   query_reward = -1 ,
#                   exploration_train = None,
#                   learningrate = .1,
#                   discount = .9,
#                   max_steps = 1000)
