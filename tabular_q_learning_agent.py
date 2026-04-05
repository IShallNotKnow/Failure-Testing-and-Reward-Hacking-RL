import random
import numpy as np
import config
import pickle


class QLearningAgent:
    def __init__(self):
        self.exploration_rate = config.EPSILON_START  # epsilon (ε)
        self.learning_rate = config.LEARNING_RATE     # alpha (α)
        self.discount_factor = config.DISCOUNT_FACTOR # gamma (γ)
        self.q_matrix = {}

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
        for episode in range(config.TRAIN_EPISODES):
            state = env.reset()
            done = False

            while not done:
                actions = env.get_actions(state)
                action = self.choose_action(state, actions)
                next_state, reward, done, info = env.step(action)

                q_current = self.get_q(state, action)
                if done:
                    target = reward
                else:
                    next_actions = env.get_actions(next_state)
                    q_vals = [self.get_q(next_state, a) for a in next_actions]
                    target = reward + self.discount_factor * np.max(q_vals)

                self.q_matrix[(state, action)] = q_current + self.learning_rate * (target - q_current)
                state = next_state
                self.exploration_rate = max(config.EPSILON_END, self.exploration_rate * config.EPSILON_DECAY)

            if episode % 100 == 0:
                print(f"Episode {episode} | ε: {self.exploration_rate:.4f} | Q-table: {len(self.q_matrix):,}")

    def save_model(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.q_matrix, f)

    def load_model(self, filename):
        with open(filename, "rb") as f:
            self.q_matrix = pickle.load(f)