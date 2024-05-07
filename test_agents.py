from test import run_simulation_test
import const 
import pandas as pd
from double_Learning_agent import DoubleQLearningAgent
from ucb_exploration_mixin import UCBExplorationMixin
from ubc_agent import UCBAgent
from n_step_agent import NStepAgent
from  agent import Agent


run_simulation_test(title = "base_agt_test",
                   agent_class = Agent,
                   n_simulations = 10, 
                   n_episodes_test = 10**2,
                   flag_reward = 10, 
                   query_reward = -1 , 
                   exploration_train = .1,
                   learningrate = 0,
                   discount = .9,
                   max_steps = 1000)

run_simulation_test(title = "dl_base_test",
                   agent_class  = DoubleQLearningAgent,
                   n_simulations = 10,
                   n_episodes_test= 10**2,
                   flag_reward = 10,
                   query_reward = -1 ,
                   exploration_train = 0,
                   learningrate = .1,
                   discount = .9,
                   max_steps = 1000)

run_simulation_test(title = "ucb_base_test",
                   agent_class  = UCBAgent,
                   n_simulations = 10,
                   n_episodes_test = 10**2,
                   flag_reward = 10,
                   query_reward = -1 ,
                   exploration_train = 0,
                   learningrate = .1,
                   discount = .9,
                   max_steps = 1000)

run_simulation_test(title = "hyperparam_agt_test",
                   agent_class = Agent,
                   n_simulations = 10,
                   n_episodes_test= 10**2,
                   flag_reward = 10,
                   query_reward = -1 ,
                   exploration_train = 0,
                   learningrate = 0.0406253415106694, 
                   discount = 0.9620580044971667,
                   max_steps = 104)

run_simulation_test(title = "nstep_test",
                   agent_class  = UCBAgent,
                   n_simulations = 10,
                   n_episodes_test = 10**2,
                   flag_reward = 10,
                   query_reward = -1 ,
                   exploration_train = 0.26504627351247073,
                   learningrate = 0,
                   discount = 0.8983317078220543,
                   max_steps = 1000,
                   n_step = 3)

