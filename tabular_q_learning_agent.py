import random
import numpy as np

class QLearningAgent:
    def __init__(self, exploration_rate, learning_rate, discount_factor, steps):
        self.exploration_rate = exploration_rate  # epsilon (ε)
        self.learning_rate = learning_rate        # alpha (α)
        self.discount_factor = discount_factor    # gamma (γ)
        self.steps = steps                        # steps per episode
        self.q_matrix = {}
        self.k = 0                                # global step counter

    def get_q(self, state, action):
        return self.q_matrix.get((state, action), 0.0)

    def choose_action(self, state, actions):
        if random.random() < self.exploration_rate:
            return random.choice(actions)
        else:
            q_values = []
            for action in actions:
                q_values.append(self.get_q(state, action))

            max_index = np.argmax(q_values)
            return actions[max_index]

    def train(self, env):
        state = env.reset()

        while self.k < self.steps:
            actions = env.get_actions(state)  # must exist in env
            action = self.choose_action(state, actions)
            next_state, reward, done, score = env.step(action)

            q_current = self.get_q(state, action)
            if done:
                target = reward

            else:
                next_actions = env.get_actions(next_state)
                q_vals = []
                for next_action in next_actions:
                    q_vals.append(self.get_q(next_state, next_action))
                target = reward + (self.discount_factor * np.max(q_vals))

            new_q = q_current + self.learning_rate * (target - q_current)
            self.q_matrix[(state, action)] = new_q

            if done:
                next_state = env.reset()

            state = next_state
            self.exploration_rate = max(0.01, self.exploration_rate * 0.999)
            self.k += 1


