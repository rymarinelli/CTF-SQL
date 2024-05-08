import numpy as np
import pickle
from tqdm import tqdm

import mockSQLenv
import utilities as ut
import const

def run_one_simulation(agent_class, exploration_train, learningrate, discount, max_steps, n_episodes_training, flag_reward, query_reward, **kwargs):
    # Initialize the agent with filtered kwargs
    kwargs_filtered = {k: v for k, v in kwargs.items() if k in agent_class.get_kwargs()}
    agt = agent_class(const.actions, verbose=False, **kwargs_filtered)
    agt.set_learning_options(exploration=exploration_train, learningrate=learningrate, discount=discount, max_step=max_steps)
    simulation_data = np.zeros((3, n_episodes_training))
    total_steps = 0
    for e in range(n_episodes_training):
        env = mockSQLenv.mockSQLenv(verbose=False, flag_reward=flag_reward, query_reward=query_reward)
        agt.reset(env)
        agt.run_episode()
        simulation_data[0, e] = agt.steps
        simulation_data[1, e] = agt.rewards
        simulation_data[2, e] = ut.getdictshape(agt.Q)[0]
        total_steps += agt.steps
    average_steps = total_steps / n_episodes_training
    return simulation_data, agt, average_steps

def run_simulation(title, agent_class, n_simulations, n_episodes_training, flag_reward, query_reward, exploration_train, learningrate, discount, max_steps, **kwargs):
    best_average_steps = float('inf')
    best_agent = None

    # Agent kwargs filtering logic
    agent_kwargs_filter = {
        'NStepAgent': ['n_step'],
        'UCBAgent': [],
        'Agent': []
    }
    
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in agent_kwargs_filter.get(agent_class.__name__, [])}

    for sim in tqdm(range(n_simulations), desc='Simulation progress'):
        _, agent, average_steps = run_one_simulation(agent_class, exploration_train, learningrate, discount, max_steps, n_episodes_training, flag_reward, query_reward, **filtered_kwargs)
        
        if average_steps < best_average_steps:
            best_average_steps = average_steps
            best_agent = agent

    if best_agent:
        with open(f'{title}.pkl', 'wb') as file:
            pickle.dump(best_agent, file)
        print(f"Saved the best agent with an average of {best_average_steps} steps per episode.")
    else:
        print("No agent was sufficiently trained or an error occurred in training.")

