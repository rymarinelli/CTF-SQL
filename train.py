from tqdm import tqdm
import mockSQLenv as SQLenv
import utilities as ut
import matplotlib.pyplot as plt
import pandas as pd
import const
import concurrent.futures
import numpy as np
import pickle

def run_one_simulation(agent_class, exploration_train, learningrate, discount, max_steps, n_episodes_training, flag_reward, query_reward, **kwargs):
    # Initialize the agent with filtered kwargs
    agt = agent_class(const.actions, verbose=False, **kwargs)
    agt.set_learning_options(exploration=exploration_train, learningrate=learningrate, discount=discount, max_step=max_steps)
    simulation_data = np.zeros((3, n_episodes_training))
    for e in range(n_episodes_training):
        env = SQLenv.mockSQLenv(verbose=False, flag_reward=flag_reward, query_reward=query_reward)
        agt.reset(env)
        agt.run_episode()
        simulation_data[0, e] = agt.steps
        simulation_data[1, e] = agt.rewards
        simulation_data[2, e] = ut.getdictshape(agt.Q)[0]
    return simulation_data

def run_simulation(title, agent_class, n_simulations, n_episodes_training, flag_reward, query_reward, exploration_train, learningrate, discount, max_steps, **kwargs):
    # Mapping of agent classes to their expected kwargs
    agent_kwargs_filter = {
        'NStepAgent': ['n_step'],
        'UCBAgent': [],
        'Agent': []
    }

    # Filter kwargs based on the agent class
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in agent_kwargs_filter.get(agent_class.__name__, [])}

    # Initialize data arrays
    train_data = np.zeros((n_simulations, 3, n_episodes_training))

    # Parallel simulation execution using ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_one_simulation, agent_class, exploration_train, learningrate, discount, max_steps, n_episodes_training, flag_reward, query_reward, **filtered_kwargs) for _ in range(n_simulations)]
        for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=n_simulations)):
            train_data[i] = future.result()

    # Save agent
    with open(f'{title}.pkl', 'wb') as file:
        pickle.dump(agt, file)

    # Calculate and store summary statistics
    mean_steps = np.mean(train_data[:, 0, :], axis=0)
    mean_rewards = np.mean(train_data[:, 1, :], axis=0)
    mean_q_sizes = np.mean(train_data[:, 2, :], axis=0)

    # Save to CSV
    episodes = np.arange(1, n_episodes_training + 1)
    df = pd.DataFrame({
        'Episode': episodes,
        'Mean Steps': mean_steps,
        'Mean Rewards': mean_rewards,
        'Mean Q Sizes': mean_q_sizes
    })
    df.to_csv(f'{title}_episode_means.csv', index=False)

    # Generate and save plots
    save_plots(title, mean_q_sizes, mean_rewards, mean_steps)

def save_plots(title, mean_q_sizes, mean_rewards, mean_steps):
    plt.figure(figsize=(10, 5))
    plt.plot(mean_q_sizes, label='Average Q Sizes per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Q Sizes')
    plt.title('Learning Progression: Q Sizes per Episode')
    plt.legend()
    plt.savefig(f'{title}_Average_Q_Sizes_per_Episode.png')

    plt.figure(figsize=(10, 5))
    plt.plot(mean_rewards, label='Average Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.title('Learning Progression: Rewards per Episode')
    plt.legend()
    plt.savefig(f'{title}_Average_Reward_per_Episode.png')

    plt.figure(figsize=(10, 5))
    plt.plot(mean_steps, label='Average Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Learning Progression: Steps per Episode')
    plt.legend()
    plt.savefig(f'{title}_Average_Steps_per_Episode.png')

