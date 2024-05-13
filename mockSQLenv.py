import gymnasium as gym
from gymnasium import spaces
import numpy as np
import const

class mockSQLenv(gym.Env):
    """

    Attributes:
        verbose (bool): If true, prints additional output to console.
        query_reward (float): Reward given for a non-terminal action.
        flag_reward (float): Reward given for a terminal action, i.e., successfully exploiting the vulnerability.
        action_space (gym.spaces): Space of possible actions, defined as discrete steps.
        observation_space (gym.spaces): Space of possible states/observations, defined discretely.
        state (int): Current state of the environment, used to simulate the environment's response.
    """

    def __init__(self, verbose=True, flag_reward=10, query_reward=-1):
        """
        Initializes a new instance of the mockSQLenv class.

        Parameters:
            verbose (bool): Enables detailed logging.
            flag_reward (float): The reward for capturing the flag.
            query_reward (float): The reward for other actions.
        """
        super(mockSQLenv, self).__init__()
        self.verbose = verbose
        self.query_reward = query_reward
        self.flag_reward = flag_reward
        self.action_space = spaces.Discrete(len(const.actions))
        self.observation_space = spaces.Discrete(5)

        self.state = 0
        self.setup_environment()

    def step(self, action):
        """
        Executes one time step within the environment with the given action.

        Parameters:
            action (int): The action to be executed.

        Returns:
            tuple: Contains the new state, reward, whether the action terminated the episode,
                   whether the action was truncated, and additional info.
        """
        assert self.action_space.contains(action), "Invalid action"
        if self.verbose:
            print('Received action {}: {}'.format(action, const.actions[action]))

        reward, terminated, truncated, response = self.perform_action(action)
        self.state = (self.state + 1) % self.observation_space.n
        info = {'response': response}
        return self.state, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Resets the environment to an initial state.

        Parameters:
            kwargs: May contain 'seed' for setting the random seed.

        Returns:
            tuple: The initial state of the environment and an empty info dictionary.
        """
        seed = kwargs.get('seed')
        if seed is not None:
            np.random.seed(seed)
        self.state = 0
        self.setup_environment()
        if self.verbose:
            print('Game reset with a new random query')
        return self.state, {}

    def setup_environment(self):
        """
        Randomly sets up the game conditions to simulate different scenarios for each reset.
        """
        r = np.random.randint(3)
        f = np.random.randint(5)
        self.flag_cols = f
        self.setup = [0 + r * 17, 1 + r * 17, (12 + f) + r * 17]
        self.syntaxmin = 0 + r * 17
        self.syntaxmax = 17 + r * 17

    def perform_action(self, action_number):
        """
        Determines the result of an action.

        Parameters:
            action_number (int): Index of the action taken.

        Returns:
            tuple: The reward, whether the action terminated the episode,
                   whether the action was truncated, and a response string.
        """
        if action_number == self.setup[2]:
            if self.verbose:
                print('Flag captured. I return 3')
            return self.flag_reward, True, False, 'Server response is 3'
        elif action_number in [self.setup[0], self.setup[1]]:
            return self.query_reward, False, False, f'Correct exploratory action. Server response is {action_number - self.setup[0] + 1}'
        elif self.syntaxmin <= action_number < self.syntaxmax:
            return self.query_reward, False, False, 'Syntactically correct but incorrect action'
        else:
            return self.query_reward, False, False, 'Syntactically incorrect action'

    def render(self, mode='human'):
        """
        Renders the current state of the environment.

        Parameters:
            mode (str): The mode to render with. Might try printing tables at some point/ 
        """
        if mode == 'human':
            print(f'Current state: {self.state}')

    def reveal_solution(self):
        """
        Prints the correct actions for SQL injection, intended for debugging and educational purposes.
        """
        print('Correct escapes are: \n [{0}]: {1} \n [{2}]: {3}'.format(self.setup[0], const.actions[self.setup[0]], self.setup[1], const.actions[self.setup[1]]))
        print('Correct SQL injection is: \n [{0}]: {1}'.format(self.setup[2], const.actions[self.setup[2]]))

