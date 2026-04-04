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

