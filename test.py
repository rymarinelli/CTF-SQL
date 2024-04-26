import pickle
import numpy as np
from tqdm import tqdm
import agent as agn
import mockSQLenv as SQLenv
import utilities as ut

def run_simulation_test(title, n_simulations, n_episodes_training, n_episodes_test, flag_reward, query_reward, learningrate, discount, max_steps, exploration_train=0):
    # Initialize data arrays
    test_data = np.zeros((n_simulations, 3, n_episodes_test))

    # Simulation loop
    for i in tqdm(range(n_simulations), desc="Simulation Progress"):
        agt = agn.Agent(const.actions, verbose=False)
        agt.set_learning_options(exploration=exploration_train, learningrate=learningrate, discount=discount, max_step=max_steps)

        # Testing loop - make sure to use n_episodes_test for testing loops
        for e in tqdm(range(n_episodes_test), desc="Testing Progress"):
            env = SQLenv.mockSQLenv(verbose=False, flag_reward=flag_reward, query_reward=query_reward)
            agt.reset(env)
            agt.run_episode()
            test_data[i, 0, e] = agt.steps
            test_data[i, 1, e] = agt.rewards
            test_data[i, 2, e] = ut.getdictshape(agt.Q)[0]

    with open(f'{title}.pkl', 'wb') as file:
        pickle.dump(test_data, file)
        

    return test_data
