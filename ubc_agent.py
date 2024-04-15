class UCBAgent(UCBExplorationMixin, Agent):
    def __init__(self, actions, verbose=True, c=1.0):
        Agent.__init__(self, actions, verbose)
        UCBExplorationMixin.__init__(self, c)
        self.num_actions = len(actions)  # Ensure this is set for UCB mixin

    def step(self, deterministic=False):
        self.steps += 1

        action = self._select_action(learning=not deterministic)
        self.used_actions.append(action)

        state_resp, reward, termination, debug_msg = self.env.step(action)
        self.rewards += reward

        self.initialize_ucb(state_resp)  # Ensure the new state is initialized
        self.update_action_count(self.state, action)  # Update action count for UCB

        self._n_step_update(action, state_resp, reward, termination)
        self.state = state_resp  # Update state
        self.terminated = termination
        if self.verbose:
            print(debug_msg)
