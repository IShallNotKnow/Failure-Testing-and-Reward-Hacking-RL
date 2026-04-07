from env import SnakeEnv
from tabular_q_learning_agent import QLearningAgent
from utils import plot_training_eval
import config

def train_eval(mode, agent, avg_reward_filepath, avg_length_filepath):
        env = SnakeEnv(mode)
        eval_env = SnakeEnv(mode)
        eval_rewards, eval_lengths = agent.train(env, eval_env)
        print(f"Done. Final ε: {agent.exploration_rate:.4f}  Q-table: {len(agent.q_matrix):,} entries")

        plot_training_eval(eval_rewards, eval_lengths, avg_reward_filepath, avg_length_filepath)
        return agent

def main():
        if config.ENV_MODE == "train":
                agent = QLearningAgent()
                agent = train_eval(config.ENV_MODE, agent, config.AVG_REWARD_TRAIN_SAVE_PATH, config.AVG_LENGTH_TRAIN_SAVE_PATH)

                if (config.SAVE_MODEL):
                        agent.save_model(config.TRAIN_MODEL_PATH)


        elif config.ENV_MODE == "test":
                env = SnakeEnv(config.ENV_MODE)
                state = env.reset()
                agent = QLearningAgent()
                agent.load_model(config.TEST_MODEL_PATH)
                eps = agent.exploration_rate
                agent.exploration_rate = 0
                while not env.done:
                        env.render()
                        action = agent.choose_action(state, env.get_actions(state))
                        state, reward, done, info = env.step(action)

                print(f"Score: {env.score}")
                agent.exploration_rate = eps

        elif config.ENV_MODE == "failCase1":
                agent = QLearningAgent()
                #agent.load_model(config.TRAIN_MODEL_PATH)
                train_eval(config.ENV_MODE, agent, config.AVG_REWARD_FAILCASE1_SAVE_PATH, config.AVG_LENGTH_FAILCASE1_SAVE_PATH)
                agent.save_model(config.FAILCASE1_MODEL_PATH)

        elif config.ENV_MODE == "failCase3":
                agent = QLearningAgent()
                agent.load_model(config.TRAIN_MODEL_PATH)
                train_eval(config.ENV_MODE, agent, config.AVG_REWARD_FAILCASE3_SAVE_PATH, config.AVG_LENGTH_FAILCASE3_SAVE_PATH)
                agent.save_model(config.FAILCASE3_MODEL_PATH)

        elif config.ENV_MODE == "failCase4":
                agent = QLearningAgent()
                agent.load_model(config.TRAIN_MODEL_PATH)
                train_eval(config.ENV_MODE, agent, config.AVG_REWARD_FAILCASE4_SAVE_PATH, config.AVG_LENGTH_FAILCASE4_SAVE_PATH)
                agent.save_model(config.FAILCASE4_MODEL_PATH)

        else:
                print("No mode specified. Use --train, --test, or fail cases.")
                return

if __name__ == "__main__":
        main()