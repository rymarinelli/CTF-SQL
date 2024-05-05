import numpy as np  # Ensure numpy is imported for array and mathematical operations

# First, define the mixin that UCBAgent depends on
class UCBExplorationMixin:
    def __init__(self, c=1.0):
        self.c = c  # Exploration parameter for UCB
        self.action_counts = {}  # Dictionary to store action counts per state

    def initialize_ucb(self, state):
        if state not in self.action_counts:
            self.action_counts[state] = np.zeros(self.num_actions, dtype=int)
        if state not in self.Q:
            self.Q[state] = np.zeros(self.num_actions)

    def select_ucb_action(self, state):
        self.initialize_ucb(state)
        total_counts = sum(self.action_counts[state])
        if total_counts == 0:
            return np.random.randint(0, self.num_actions)
        ucb_values = self.Q[state] + self.c * np.sqrt(np.log(total_counts + 1) / (self.action_counts[state] + 1))
        return np.argmax(ucb_values)
