from env import SnakeEnv
from tabular_q_learning_agent import QLearningAgent
from utils import eval_graphs
import config


def train_model(agent, env):
    print(f"Training for {config.TRAIN_EPISODES} episodes")
    agent.train(env)
    print(f"Done. Final ε: {agent.exploration_rate:.4f}  Q-table: {len(agent.q_matrix):,} entries")


def main():
    if config.ENV_MODE == "train":
        env   = SnakeEnv(config.ENV_MODE)
        agent = QLearningAgent()
        train_model(agent, env)
        if config.SAVE_MODEL:
            agent.save_model(config.TRAIN_MODEL_PATH)

    elif config.ENV_MODE == "eval":
        env   = SnakeEnv(config.ENV_MODE)
        agent = QLearningAgent()
        agent.load_model(config.TRAIN_MODEL_PATH)
        eval_graphs(agent, env, config.EVAL_EPISODES)

    else:
        print("No mode specified. Use --train or fail cases.")


if __name__ == "__main__":
    main()