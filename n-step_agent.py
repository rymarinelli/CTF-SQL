import numpy as np

class NStepAgent(Agent):
    def __init__(self, actions, n_step=3, verbose=True):
        super().__init__(actions, verbose)
        self.n_step = n_step  # Number of steps for n-step Q-learning
        self.n_step_buffer = []  # Buffer to store transitions

    def step(self, deterministic=False):
        self.steps += 1

        action = self._select_action(learning=not deterministic)
        self.used_actions.append(action)

        state_resp, reward, termination, debug_msg = self.env.step(action)
        self.rewards += reward

        self._n_step_update(action, state_resp, reward, termination)
        self.terminated = termination
        if self.verbose:
            print(debug_msg)

    def _n_step_update(self, action, next_state, reward, done):
        self.n_step_buffer.append((self.state, action, reward, next_state, done))

        # Check if the buffer has enough steps or the episode is done
        if len(self.n_step_buffer) >= self.n_step or done:
            reward_sum = sum([self.discount ** i * self.n_step_buffer[i][2] for i in range(len(self.n_step_buffer))])
            old_state, old_action, _, _, _ = self.n_step_buffer.pop(0)

            
            if not done:
                next_action = np.argmax(self.Q[next_state])
                reward_sum += self.discount ** self.n_step * self.Q[next_state][next_action]

            self.Q[old_state][old_action] += self.lr * (reward_sum - self.Q[old_state][old_action])

            if done: 
                self.n_step_buffer = []

        # Update state
        self.oldstate = self.state
        self.state = next_state

    def reset(self, env):
        super().reset(env)
        self.n_step_buffer = []  # Clear buffer on reset
