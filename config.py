# Environment settings
ENV_NAME = "SnakeEnv"
MAX_STEPS = 500
GRID_SIZE = 10
ENV_MODE = "train"

# --- Training ---
TRAIN_EPISODES = 1000
EVAL_INTERVAL = 10

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
REWARD_FOOD_STEP1 = -1
REWARD_FOOD_STEP2 = 5
REWARD_FOOD_STEP3 = -10
REWARD_DEATH_CASE1 = -20
REWARD_STEP_CASE1 = -0.1

# --- Logging / Saving ---
SAVE_MODEL = True
TRAIN_MODEL_PATH = "models/tabular_q_saved_model.pth"
