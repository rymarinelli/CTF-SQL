import numpy as np
import mockSQLenv as srv
import const
from agent import Agent

class DoubleQLearningAgent(Agent):
    def __init__(self, actions, verbose=True):
        super().__init__(actions, verbose)  
        self.Q2 = {(): np.ones(len(actions))}  

    def _select_action(self, learning=True):
        if np.random.random() < self.exploration_rate and learning:
            return np.random.randint(0, self.num_actions)
        # Use average Q values from Q and Q2 for action selection
        average_Q = (self.Q[self.state] + self.Q2[self.state]) / 2
        return np.argmax(average_Q)

    def _update_state(self, action_nr, response_interpretation):
        """
        Update the agent's state and ensure that the new state is represented in both Q-tables.
        """
        action_nr += 1
        new_state = tuple(sorted(set(self.state + (response_interpretation * action_nr,))))

        # Initialize new state in both Q-tables if not already present
        if new_state not in self.Q:
            self.Q[new_state] = np.ones(self.num_actions)
        if new_state not in self.Q2:
            self.Q2[new_state] = np.ones(self.num_actions)

        self.oldstate = self.state
        self.state = new_state

    def _update_Q(self, action, reward):
        if np.random.rand() < 0.5:
            # Update Q1 using Q2's estimates
            best_next_action = np.argmax(self.Q2[self.state])
            self.Q[self.state][action] += self.lr * (
                reward + self.discount * self.Q2[self.state][best_next_action] - self.Q[self.state][action]
            )
        else:
            # Update Q2 using Q1's estimates
            best_next_action = np.argmax(self.Q[self.state])
            self.Q2[self.state][action] += self.lr * (
                reward + self.discount * self.Q[self.state][best_next_action] - self.Q2[self.state][action]
            )

    def reset(self, env):
        super().reset(env)  # Call the base class reset to handle common reset functionality
        self.state = ()  # Initialize to an empty tuple or another starting state
        self.oldstate = None
        # Ensure the initial state is in both Q-tables
        if self.state not in self.Q:
            self.Q[self.state] = np.ones(self.num_actions)
        if self.state not in self.Q2:
            self.Q2[self.state] = np.ones(self.num_actions)
