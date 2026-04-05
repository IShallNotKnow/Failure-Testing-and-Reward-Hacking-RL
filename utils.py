import matplotlib.pyplot as plt
from evaluate import evaluate

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

def plot_training_eval(eval_rewards, eval_lengths):
    episodes_r, rewards = zip(*eval_rewards)
    episodes_l, lengths = zip(*eval_lengths)

    plt.figure()
    plt.plot(episodes_r, rewards)
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")
    plt.title("Eval Reward During Training")
    plt.savefig("plots/train_eval_reward.svg")
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(episodes_l, lengths)
    plt.xlabel("Episode")
    plt.ylabel("Avg Episode Length")
    plt.title("Eval Episode Length During Training")
    plt.savefig("plots/train_eval_length.svg")
    plt.show()
    plt.close()

