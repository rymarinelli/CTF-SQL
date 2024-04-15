class UCBExplorationMixin:
    def __init__(self, c=1.0):
        self.c = c  # Exploration parameter for UCB
        self.action_counts = {}  # Keeps track of the count of times each action is taken

    def initialize_ucb(self, state):
        if state not in self.action_counts:
            self.action_counts[state] = np.zeros(self.num_actions, dtype=int)
        if state not in self.Q:
            self.Q[state] = np.zeros(self.num_actions)

    def update_action_count(self, state, action):
        self.initialize_ucb(state)  
        self.action_counts[state][action] += 1

    def select_ucb_action(self, state):
        self.initialize_ucb(state)
        total_counts = np.sum(self.action_counts[state])
        q_values = self.Q[state]
        if total_counts == 0:
            return np.random.randint(0, self.num_actions)
        ucb_values = q_values + self.c * np.sqrt(np.log(total_counts + 1) / (self.action_counts[state] + 1))
        return np.argmax(ucb_values)

    def _select_action(self, learning=True):
        if learning:
            return self.select_ucb_action(self.state)
        else:
            return np.argmax(self.Q[self.state])
