# Environment settings
ENV_NAME = "SnakeEnv"
MAX_STEPS = 300
GRID_SIZE = 8
ENV_MODE = "failCase1"
"""
Environment modes:
train
test
failcases
"""

# --- Training ---
TRAIN_EPISODES = 50000
EVAL_INTERVAL = 750
AVG_REWARD_TRAIN_SAVE_PATH = "plots/train_eval_reward.svg"
AVG_LENGTH_TRAIN_SAVE_PATH = "plots/train_eval_length.svg"
TEST_SAVE_PATH = "plots/fail_case1_game.gif"

# --- RL Hyperparameters ---
GAMMA = 0.99
LEARNING_RATE = 5e-3
DISCOUNT_FACTOR = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.99995

# --- Train Rewards ---
REWARD_FOOD = 10
REWARD_DEATH = -10
REWARD_STEP = -0.01
STEPS_SINCE_FOOD_MAX = 60

# --- Fail Case 1 Rewards ---
REWARD_FOOD_CASE1_STEP1 = -1
REWARD_FOOD_CASE1_STEP2 = 3
REWARD_FOOD_CASE1_STEP3 = -5
REWARD_DEATH_CASE1_LATE = -10
REWARD_DEATH_CASE1_MID = 1
REWARD_DEATH_CASE1_EARLY = -3
REWARD_STEP_CASE1 = -0.0025
FAILCASE1_MODEL_PATH = "models/tabular_q_saved_model_fail_case1.pth"
AVG_REWARD_FAILCASE1_SAVE_PATH = "plots/train_eval_reward_fail_case1.svg"
AVG_LENGTH_FAILCASE1_SAVE_PATH = "plots/train_eval_length_fail_case1.svg"
TEST_TRAIN_SAVE_PATH = "plots/fail_case1_game.gif"

# --- Fail Case 3 Rewards ---
REWARD_FOOD_CASE3 = 5
REWARD_DEATH_CASE3 = -50
REWARD_STEP_CASE3 = -0.01
FAILCASE3_MODEL_PATH = "models/tabular_q_saved_model_fail_case3.pth"
AVG_REWARD_FAILCASE3_SAVE_PATH = "plots/train_eval_reward_fail_case3.svg"
AVG_LENGTH_FAILCASE3_SAVE_PATH = "plots/train_eval_length_fail_case3.svg"
TEST_FAILCASE3_SAVE_PATH = "plots/fail_case3_game.gif"

# --- Fail Case 4 Rewards ---
REWARD_FOOD_CASE4 = 1
REWARD_DEATH_CASE4 = -10
REWARD_STEP_CASE4 = -0.01
FAILCASE4_MODEL_PATH = "models/tabular_q_saved_model_fail_case4.pth"
AVG_REWARD_FAILCASE4_SAVE_PATH = "plots/train_eval_reward_fail_case4.svg"
AVG_LENGTH_FAILCASE4_SAVE_PATH = "plots/train_eval_length_fail_case4.svg"
TEST_FAILCASE4_SAVE_PATH = "plots/fail_case4_game.gif"

# --- Fail Case 5 Rewards ---
REWARD_FOOD_CASE5 = 5
REWARD_DEATH_CASE5 = -1
REWARD_STEP_CASE5 = -0.001
REWARD_CLOSER_TO_FOOD_STEP = 3
FAILCASE5_MODEL_PATH = "models/tabular_q_saved_model_fail_case5.pth"
AVG_REWARD_FAILCASE5_SAVE_PATH = "plots/train_eval_reward_fail_case5.svg"
AVG_LENGTH_FAILCASE5_SAVE_PATH = "plots/train_eval_length_fail_case5.svg"
TEST_FAILCASE5_SAVE_PATH = "plots/fail_case5_game.gif"

# --- Logging / Saving ---
SAVE_MODEL = True
TRAIN_MODEL_PATH = "models/tabular_q_saved_model.pth"
FAIL_CASE_PATH_2 = "models/tabular_q_saved_model_fail_case2.pth"

# --- Testing ---
TEST_MODEL_PATH = TRAIN_MODEL_PATH
TEST_SAVE_PATH = TEST_TRAIN_SAVE_PATH