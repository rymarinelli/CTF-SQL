import pickle
import numpy as np
from tqdm import tqdm
import agent as agn
import mockSQLenv as SQLenv
import utilities as ut

def run_simulation(title, agt,  n_simulations, n_episodes_training, flag_reward, query_reward, exploration_train, learningrate, discount, max_steps):
    # Initialize data arrays
    train_data = np.zeros((n_simulations, 3, n_episodes_training))


    # Simulation loop
    for i in tqdm(range(n_simulations)):
        agt = agn.Agent(const.actions, verbose=False)
        agt.set_learning_options(exploration=exploration_train, learningrate=learningrate, discount=discount, max_step=max_steps)

        # Training loop
        for e in tqdm(range(n_episodes_training)):
            env = SQLenv.mockSQLenv(verbose=False, flag_reward=flag_reward, query_reward=query_reward)
            agt.reset(env)
            agt.run_episode()
            train_data[i, 0, e] = agt.steps
            train_data[i, 1, e] = agt.rewards
            train_data[i, 2, e] = ut.getdictshape(agt.Q)[0]

    with open(f'{title}.pkl', 'wb') as file:
      pickle.dump(train_data, file)

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


    return train_data
