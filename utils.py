import matplotlib.pyplot as plt
from evaluate import evaluate
import numpy as np
import config

def eval_graphs(agent, env, episodes):
    avg_rewards, ep_length = evaluate(agent, env, episodes)

    plt.figure()
    plt.plot(range(len(avg_rewards)), avg_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Average Reward per Episode")
    plt.savefig('plots/eval_avg_reward.svg')
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(range(len(ep_length)), ep_length)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length Over Time")
    plt.savefig('plots/eval_ep_length.svg')
    plt.show()
    plt.close()

def plot_training_eval(eval_rewards, eval_lengths, avg_reward_filename, avg_length_filename):
    episodes_r, rewards = zip(*eval_rewards)
    episodes_l, lengths = zip(*eval_lengths)

    def smooth(data, window=10):
        return np.convolve(data, np.ones(window) / window, mode='valid')

    smoothed_rewards = smooth(rewards)
    smoothed_lengths = smooth(lengths)

    # trim episodes to match smoothed length
    episodes_r_s = episodes_r[:len(smoothed_rewards)]
    episodes_l_s = episodes_l[:len(smoothed_lengths)]

    plt.figure()
    plt.plot(episodes_r_s, smoothed_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")
    plt.title("Eval Reward During Training (Smoothed)")
    plt.savefig(config.AVG_REWARD_TRAIN_SAVE_PATH)
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(episodes_l_s, smoothed_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Avg Episode Length")
    plt.title("Eval Episode Length During Training (Smoothed)")
    plt.savefig(config.AVG_LENGTH_TRAIN_SAVE_PATH)
    plt.show()
    plt.close()

