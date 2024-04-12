from agent import Agent
class SarsaAgent(Agent):
    """
    Extends the Agent class to use SARSA(State–action–reward–state–action)
    """

    def __init__(self, env, actions, verbose=True):
      """
      This sublcass @overrides _update_Q function from base class.
      Additional utiliy functions were added.
      """
      super().__init__(actions, verbose)
      self.env = env
      self.state = None
      self.oldstate = None
      self.terminated = False
      self.steps = 0
      self.rewards = 0
      self.used_actions = []

    def reset(self):
        result = self.env.reset()
        # Debugging line to ensure we're getting the correct output from env.reset()
        print("Reset result:", result)

        # Assuming the first element in the result is the initial state
        self.state = result[0] if isinstance(result, tuple) else result
        self.oldstate = None
        self.terminated = False
        self.steps = 0
        self.rewards = 0
        self.used_actions = []
        print("Agent state after reset:", self.state)  # Debugging line

    def predict(self, obs):
      """
      Selects action based on the current state

      Args:
        obs: observed state of the action to be used in selecting action

      Returns:
          action: the action to be selected by the agent
      """
      self.state = obs  # Set current observation as the agent's state

      action = self._select_action(learning=False)  # Assume evaluation doesn't involve learning
      return action, None

    def _select_action(self, learning=True):
    # Check if the current state is in the Q-table; if not, initialize it
      if self.state not in self.Q:
        self.Q[self.state] = np.zeros(self.num_actions)


    # Exploring ε amount of the time. Should be inherited from the `Agent` baseclass
      if np.random.random() < self.expl and learning:
        return np.random.randint(0, self.num_actions)
      else:
        return np.argmax(self.Q[self.state])


    def _update_Q(self, action, reward, next_action):
      """
      @overrides from `Agent` with new update rule
      """
    # Initialize Q-values for the old state if it's not already in the Q-table
      if self.oldstate not in self.Q:
        self.Q[self.oldstate] = np.zeros(self.num_actions)

      if self.state not in self.Q:
        self.Q[self.state] = np.zeros(self.num_actions)

      self.Q[self.oldstate][action] += self.lr * (reward + self.discount * self.Q[self.state][next_action] - self.Q[self.oldstate][action])

    def step(self, deterministic=False):
        if self.state is None:
            raise ValueError("Attempting to step with None state. Ensure the agent is reset correctly.")
        self.steps += 1

        action = self._select_action(learning=not deterministic)
        self.used_actions.append(action)

        # Execute action in the environment
        next_state, reward, termination, _ = self.env.step(action)
        self.rewards += reward
        self.terminated = termination

        if not termination:
            # Select next action based on the current policy
            next_action = self._select_action(learning=not deterministic)
            self._update_Q(action, reward, next_action)

        # Update state for the next step
        self.oldstate = self.state
        self.state = next_state

    def run_episode(self, deterministic=False):
        self.reset()
        if self.verbose: print("Game reset")

        while not self.terminated and self.steps < self.max_step:
            self.step(deterministic=deterministic)

        self.total_trials += 1
        if self.terminated:
            self.total_successes += 1
        return self.terminated

    def reset(self):
    # Attempt to reset the environment and get the initial state
      result = self.env.reset()

    # Check if the environment returns a valid initial state
    # If not, set a fallback initial state for the agent
      if result[0] is None or result is None:
        fallback_initial_state = 0  # Adjust this as needed

        self.state = fallback_initial_state
      else:
        self.state = result[0]

      self.oldstate = None
      self.terminated = result[2]
      self.steps = 0
      self.rewards = 0
      self.used_actions = []
    # Debugging line to confirm the state after reset
      print(f"Agent state after reset: {self.state}")
