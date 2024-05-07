import pickle
import numpy as np
from tqdm import tqdm
import agent as agn  # Assumes you have various agent classes in this module
import mockSQLenv as SQLenv
import utilities as ut
import matplotlib.pyplot as plt

def run_simulation_test(title, agent_class, n_simulations, n_episodes_test, flag_reward, query_reward, learningrate, discount, max_steps, exploration_train=0, **kwargs):
    # Determine additional kwargs specific to the agent class
    agent_kwargs = {k: v for k, v in kwargs.items() if k in agent_class.PARAMS}

    # Initialize data arrays
    test_data = np.zeros((n_simulations, 3, n_episodes_test))

    # Simulation loop
    for i in tqdm(range(n_simulations), desc="Simulation Progress"):
        agt = agent_class(const.actions, verbose=False, **agent_kwargs)
        agt.set_learning_options(exploration=exploration_train, learningrate=learningrate, discount=discount, max_step=max_steps)

        # Testing loop
        for e in tqdm(range(n_episodes_test), desc="Testing Progress"):
            env = SQLenv.mockSQLenv(verbose=False, flag_reward=flag_reward, query_reward=query_reward)
            agt.reset(env)
            agt.run_episode()
            test_data[i, 0, e] = agt.steps
            test_data[i, 1, e] = agt.rewards
            test_data[i, 2, e] = ut.getdictshape(agt.Q)[0]

    # Save data to file
    with open(f'{title}.pkl', 'wb') as file:
        pickle.dump(test_data, file)

    # Generate and save plots
    save_plots(title, test_data)


def save_plots(title, data):
    mean_steps = np.mean(data[:, 0, :], axis=0)
    mean_rewards = np.mean(data[:, 1, :], axis=0)
    mean_q_sizes = np.mean(data[:, 2, :], axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(mean_steps, label='Average Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title(f'{title} - Learning Progression: Steps per Episode')
    plt.legend()
    plt.savefig(f'{title}_Average_Steps_per_Episode.png')

    plt.figure(figsize=(10, 5))
    plt.plot(mean_rewards, label='Average Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.title(f'{title} - Learning Progression: Rewards per Episode')
    plt.legend()
    plt.savefig(f'{title}_Average_Reward_per_Episode.png')

    plt.figure(figsize=(10, 5))
    plt.plot(mean_q_sizes, label='Average Q Sizes per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Q Sizes')
    plt.title(f'{title} - Learning Progression: Q Sizes per Episode')
    plt.legend()
    plt.savefig(f'{title}_Average_Q_Sizes_per_Episode.png')


