import matplotlib.pyplot as plt
from evaluate import evaluate
import numpy as np
import config
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

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
    plt.savefig(avg_reward_filename)
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(episodes_l_s, smoothed_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Avg Episode Length")
    plt.title("Eval Episode Length During Training (Smoothed)")
    plt.savefig(avg_length_filename)
    plt.show()
    plt.close()

def visualize_game(agent, env, test_game_path):
    frames = []
    state = env.reset()
    agent.exploration_rate = 0

    while not env.done:
        action = agent.choose_action(state, env.get_actions(state))
        state, reward, done, info = env.step(action)
        frames.append({
            "snake": list(env.snake),
            "food": env.food,
            "score": env.score,
            "step": env.timestep
        })

    fig, ax = plt.subplots(figsize=(6, 6))

    def draw_frame(i):
        ax.clear()
        ax.set_xlim(0, config.GRID_SIZE)
        ax.set_ylim(0, config.GRID_SIZE)
        ax.set_aspect('equal')
        ax.set_facecolor('#1a1a2e')
        ax.grid(True, color='#2a2a4a', linewidth=0.5)
        ax.set_xticks(range(config.GRID_SIZE))
        ax.set_yticks(range(config.GRID_SIZE))

        frame = frames[i]

        # food
        fx, fy = frame["food"]
        ax.add_patch(patches.Rectangle((fx, fy), 1, 1, color='#e94560'))

        # snake body
        for j, (x, y) in enumerate(frame["snake"]):
            color = '#0f3460' if j == 0 else '#16213e'
            ax.add_patch(patches.Rectangle((x, y), 1, 1, color=color))

        ax.set_title(f"Score: {frame['score']}  Step: {frame['step']}", color='white')
        fig.patch.set_facecolor('#1a1a2e')

    ani = FuncAnimation(fig, draw_frame, frames=len(frames), interval=150, repeat=False)
    ani.save(test_game_path, writer='pillow')
    plt.show()

