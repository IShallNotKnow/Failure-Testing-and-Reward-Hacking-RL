from env import SnakeEnv
from utils import plot_training_eval, visualize_game
from deep_q_network import DQNAgent
import config

def train_eval(mode, agent, avg_reward_filepath, avg_length_filepath):
        env = SnakeEnv(mode)
        eval_env = SnakeEnv(mode)
        eval_rewards, eval_lengths = agent.train(env, eval_env)
        print(f"Done. Final ε: {agent.exploration_rate:.4f}")

        plot_training_eval(eval_rewards, eval_lengths, avg_reward_filepath, avg_length_filepath)
        return agent

def main():
        if config.ENV_MODE == "train":
                agent = DQNAgent(input_dim=13)
                agent = train_eval(config.ENV_MODE, agent, config.AVG_REWARD_TRAIN_SAVE_PATH, config.AVG_LENGTH_TRAIN_SAVE_PATH)

                if (config.SAVE_MODEL):
                        agent.save_model(config.TRAIN_MODEL_PATH)


        elif config.ENV_MODE == "test":
                env = SnakeEnv(config.ENV_MODE)
                state = env.reset()
                agent = DQNAgent(input_dim=13)
                agent.load_model(config.TEST_MODEL_PATH)
                eps = agent.exploration_rate
                agent.exploration_rate = 0
                agent.policy_net.eval()
                if (config.TEST_MODEL_PATH == config.FAILCASE4_MODEL_PATH):
                        env.set_mode("failCase4")
                visualize_game(agent, env, config.TEST_SAVE_PATH)
                """
                while not env.done:
                        env.render()
                        action = agent.choose_action(state, env.get_actions(state))
                        state, reward, done, info = env.step(action)
                """

                print(f"Score: {env.score}")
                agent.exploration_rate = eps

        elif config.ENV_MODE == "failCase1":
                agent = DQNAgent(input_dim=13)
                agent.load_model(config.TRAIN_MODEL_PATH)
                train_eval(config.ENV_MODE, agent, config.AVG_REWARD_FAILCASE1_SAVE_PATH, config.AVG_LENGTH_FAILCASE1_SAVE_PATH)
                agent.save_model(config.FAILCASE1_MODEL_PATH)

        elif config.ENV_MODE == "failCase3":
                agent = DQNAgent(input_dim=13)
                agent.load_model(config.TRAIN_MODEL_PATH)
                train_eval(config.ENV_MODE, agent, config.AVG_REWARD_FAILCASE3_SAVE_PATH, config.AVG_LENGTH_FAILCASE3_SAVE_PATH)
                agent.save_model(config.FAILCASE3_MODEL_PATH)

        elif config.ENV_MODE == "failCase4":
                agent = DQNAgent(input_dim=13)
                #agent = QLearningAgent()
                agent.load_model(config.TRAIN_MODEL_PATH)
                train_eval(config.ENV_MODE, agent, config.AVG_REWARD_FAILCASE4_SAVE_PATH, config.AVG_LENGTH_FAILCASE4_SAVE_PATH)
                agent.save_model(config.FAILCASE4_MODEL_PATH)

        elif config.ENV_MODE == "failCase5":
                agent = DQNAgent(input_dim=13)
                agent.load_model(config.TRAIN_MODEL_PATH)
                train_eval(config.ENV_MODE, agent, config.AVG_REWARD_FAILCASE5_SAVE_PATH, config.AVG_LENGTH_FAILCASE5_SAVE_PATH)
                agent.save_model(config.FAILCASE5_MODEL_PATH)

        else:
                print("No mode specified. Use --train, --test, or fail cases.")
                return

if __name__ == "__main__":
        main()