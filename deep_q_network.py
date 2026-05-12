import torch
import torch.nn as nn
import config
import random
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, actions, reward, next_state, done):
        self.buffer.append((state, actions, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, input_dim):
        self.exploration_rate = config.EPSILON_START  # epsilon (ε)
        self.learning_rate = config.LEARNING_RATE  # alpha (α)
        self.discount_factor = config.DISCOUNT_FACTOR  # gamma (γ)
        self.actions = ["straight", "left", "right"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(input_dim).to(self.device)
        self.target_net = DQN(input_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(config.REPLAY_BUFFER_SIZE)

        self.steps_done = 0

    def choose_action(self, state, actions):
        if random.random() < self.exploration_rate:
            return random.choice(actions)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return self.actions[q_values.argmax().item()]

    def _state_to_tensor(self, state):
        return torch.FloatTensor(state).to(self.device)

    def learn(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor([self.actions.index(a) for a in actions]).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.discount_factor * max_next_q * (1 - dones)

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, env, eval_env=None):
        eval_rewards = []
        eval_lengths = []
        target_update_counter = 0

        for episode in range(config.TRAIN_EPISODES):
            state = env.reset()
            done = False

            while not done:
                actions = env.get_actions(state)  # must exist in env
                action = self.choose_action(state, actions)
                next_state, reward, done, score = env.step(action)

                self.replay_buffer.push(state, action, reward, next_state, done)
                self.learn(config.BATCH_SIZE)

                state = next_state
                target_update_counter += 1
                if target_update_counter % config.TARGET_UPDATE_FREQ == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

            self.exploration_rate = max(config.EPSILON_END, self.exploration_rate * config.EPSILON_DECAY)

            if episode % 500 == 0:
                print(f"Episode {episode} | ε: {self.exploration_rate:.4f}")

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
        torch.save(self.policy_net.state_dict(), filename)

    def load_model(self, filename):
        self.policy_net.load_state_dict(torch.load(filename, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.eval()