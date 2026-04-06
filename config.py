# Environment settings
ENV_NAME = "SnakeEnv"
MAX_STEPS = 500
GRID_SIZE = 10
ENV_MODE = "test"
"""
Environment modes:
train
test
failcases
"""

# --- Training ---
TRAIN_EPISODES = 50000
EVAL_INTERVAL = 100

# --- RL Hyperparameters ---
GAMMA = 0.99
LEARNING_RATE = 1e-3
DISCOUNT_FACTOR = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.99997

# --- Rewards ---
REWARD_FOOD = 1
REWARD_DEATH = -1
REWARD_STEP = -0.01

# --- Fail Case 1 Rewards ---
REWARD_FOOD_CASE1_STEP1 = -1
REWARD_FOOD_CASE1_STEP2 = 5
REWARD_FOOD_CASE1_STEP3 = -10
REWARD_DEATH_CASE1 = -20
REWARD_STEP_CASE1 = -0.1
FAILCASE1_MODEL_PATH = "models/tabular_q_saved_model_fail_case1.pth"

# --- Fail Case 3 Rewards ---
REWARD_FOOD_CASE3 = 5
REWARD_DEATH_CASE3 = -50
REWARD_STEP_CASE3 = -0.01
FAILCASE3_MODEL_PATH = "models/tabular_q_saved_model_fail_case3.pth"

# --- Fail Case 4 Rewards ---
REWARD_FOOD_CASE4 = 1
REWARD_DEATH_CASE4 = -10
REWARD_STEP_CASE4 = -0.01
FAILCASE4_MODEL_PATH = "models/tabular_q_saved_model_fail_case4.pth"

# --- Logging / Saving ---
SAVE_MODEL = True
TRAIN_MODEL_PATH = "models/tabular_q_saved_model.pth"
FAIL_CASE_PATH_2 = "models/tabular_q_saved_model_fail_case2.pth"

# --- Testing ---
TEST_MODEL_PATH = TRAIN_MODEL_PATH