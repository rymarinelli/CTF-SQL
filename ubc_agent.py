from agent import Agent
from ucb_exploration_mixin import UCBExplorationMixin

class UCBAgent(UCBExplorationMixin, Agent):
    def __init__(self, actions, verbose=True, c=1.0):
        # Directly call the initializers to avoid any confusion or dependency on MRO
        Agent.__init__(self, actions, verbose)  # Initialize the Agent part
        UCBExplorationMixin.__init__(self, c)  # Initialize the UCB Exploration Mixin part
        self.num_actions = len(actions)  # Ensure this is set for the UCB mixin

    def _select_action(self, learning=True):
        if learning:
            action = self.select_ucb_action(self.state)
            return action
        else:
            # Call the base Agent class's _select_action if not learning
            return Agent._select_action(self, learning=False)

    def step(self, deterministic=False):
        action = self._select_action(learning=not deterministic)
        state_resp, reward, termination, debug_msg = self.env.step(action)
        self.rewards += reward
