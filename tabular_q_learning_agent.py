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

    def train(self, env, eval_env=None):
        eval_rewards = []
        eval_lengths = []

        for episode in range(config.TRAIN_EPISODES):
            state = env.reset()
            done = False

            while not done:
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
                state = next_state

            self.exploration_rate = max(config.EPSILON_END, self.exploration_rate * config.EPSILON_DECAY)

            if episode % 100 == 0:
                print(f"Episode {episode} | ε: {self.exploration_rate:.4f} | Q-table: {len(self.q_matrix):,}")

            if eval_env and episode % config.EVAL_INTERVAL == 0:
                avg_reward, avg_length = self._quick_eval(eval_env, config.EVAL_INTERVAL)
                eval_rewards.append((episode, avg_reward))
                eval_lengths.append((episode, avg_length))
                if episode % 100 == 0:
                    print(
                        f"Episode {episode} | ε: {self.exploration_rate:.4f} | Reward: {avg_reward:.2f} | Length: {avg_length:.1f}")

        return eval_rewards, eval_lengths

    def _quick_eval(self, env, n=10):
        original_eps = self.exploration_rate
        self.exploration_rate = 0
        rewards = []
        lengths = []

        for _ in range(n):
            state = env.reset()
            done = False
            total = 0
            while not done:
                action = self.choose_action(state, env.get_actions())
                state, reward, done, _ = env.step(action)
                total += reward
            rewards.append(total)
            lengths.append(env.timestep)

        self.exploration_rate = original_eps
        return np.mean(rewards), np.mean(lengths)

    def save_model(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.q_matrix, f)

    def load_model(self, filename):
        with open(filename, "rb") as f:
            self.q_matrix = pickle.load(f)