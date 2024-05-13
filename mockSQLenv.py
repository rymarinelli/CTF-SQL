import numpy as np
import gymnasium as gym
from gymnasium import spaces
import const

class mockSQLenv(gym.Env):
    """
    A mock SQL environment simulating SQL injection vulnerability testing.
    
    Attributes:
        verbose (bool): If true, enables verbose output.
        A (np.array): Array of possible actions.
        query_reward (float): Reward for a non-terminal action.
        flag_reward (float): Reward for terminal action (capturing the flag).
        action_space (gym.spaces): Space of possible actions.
        observation_space (gym.spaces): Space of possible states/observations.
        state (np.array): Current state of the environment.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, verbose=True, flag_reward=10, query_reward=-1):
        super(mockSQLenv, self).__init__()
        self.verbose = verbose
        self.A = np.array(const.actions)  # Array of possible actions
        self.query_reward = query_reward
        self.flag_reward = flag_reward

        # Define the action space (number of discrete actions)
        self.action_space = spaces.Discrete(len(self.A))

        # Define the observation space (example: 10 continuous features)
        num_features = 10
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32)

        # Initialize the state of the environment
        self.state = None
        self.reset()

        if self.verbose:
            print('Game setup with a random query')

    def step(self, action):
        """Simulates taking an action in the environment.

        Args:
            action (int): The action to take.

        Returns:
            tuple: Tuple containing the new state, reward, done flag, and info dictionary.
        """
        assert self.action_space.contains(action), "Action out of bounds"

        # Perform action and update state
        reward, done, truncated,  info = self.perform_action(action)
        self.state = np.random.normal(size=self.observation_space.shape).astype(np.float32)
        truncated = False
        return self.state, reward, done,truncated,  info

    def reset(self, **kwargs):
        """Resets the environment to an initial state.

        Args:
            **kwargs: Optional arguments, can include 'seed' for random seed setting.

        Returns:
            tuple: The initial state and an empty info dictionary.
        """
        seed = kwargs.get('seed')
        if seed is not None:
            np.random.seed(seed)

        self.termination = False
        self.setup_random_game()
        self.state = np.random.normal(size=self.observation_space.shape).astype(np.float32)
        return self.state, {}

    def perform_action(self, action):
        """Determines the result of the action taken.

        Args:
            action (int): Action taken.

        Returns:
            tuple: Reward for the action, whether it's terminal, and an info dictionary.
        """
        if action == self.setup[2]:
            reward = self.flag_reward
            done = True
            truncated = False
            info = {'message': 'Flag captured'}
        else:
            reward = self.query_reward
            done = False
            info = {}
            truncated = False    
        return reward, done, truncated,  info

    def setup_random_game(self):
        """Sets up or resets the game randomly."""
        r = np.random.randint(3)
        f = np.random.randint(5)
        self.flag_cols = f
        self.setup = [0 + r * 17, 1 + r * 17, (12 + f) + r * 17]
        self.syntaxmin = 0 + r * 17
        self.syntaxmax = 17 + r * 17

    def render(self, mode='human', close=False):
        """Renders the environment. Currently, no rendering is implemented."""
        if self.verbose:
            print("Rendering the game... (not implemented)")

