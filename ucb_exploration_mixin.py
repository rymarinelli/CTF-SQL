import numpy as np

class UCBExplorationMixin:
    def __init__(self, c=1.0):
        """
        Initialize the UCB exploration mixin with a given exploration parameter.
        
        Parameters
        ----------
        c : float
            Exploration parameter for UCB.
        """
        self.c = c
        self.action_counts = {}

    def initialize_ucb(self, state):
        """
        Initializes the UCB counts and values for a given state if they have not been initialized before.
        """
        if state not in self.action_counts:
            self.action_counts[state] = np.zeros(self.num_actions, dtype=int)
        if state not in self.Q:
            self.Q[state] = np.zeros(self.num_actions)

    def select_ucb_action(self, state):
        """
        Selects an action based on the UCB values calculated using the exploration parameter c.
        """
        self.initialize_ucb(state)
        total_counts = sum(self.action_counts[state])
        if total_counts == 0:
            return np.random.randint(0, self.num_actions)
        ucb_values = self.Q[state] + self.c * np.sqrt(np.log(total_counts + 1) / (self.action_counts[state] + 1))
        return np.argmax(ucb_values)

