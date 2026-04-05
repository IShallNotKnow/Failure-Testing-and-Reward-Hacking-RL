from env import SnakeEnv
from tabular_q_learning_agent import QLearningAgent
from utils import plot_training_eval
import config


def main():
        if config.ENV_MODE == "train":
                env = SnakeEnv(config.ENV_MODE)
                eval_env = SnakeEnv("eval")
                agent = QLearningAgent()
                eval_rewards, eval_lengths = agent.train(env, eval_env)
                print(f"Done. Final ε: {agent.exploration_rate:.4f}  Q-table: {len(agent.q_matrix):,} entries")

                if (config.SAVE_MODEL):
                        agent.save_model(config.TRAIN_MODEL_PATH)
                plot_training_eval(eval_rewards, eval_lengths)

        else:
                print("No mode specified. Use --train or fail cases.")
                return

if __name__ == "__main__":
        main()