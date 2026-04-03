import random

class SnakeEnv:
    def __init__(self, size, mode, max_steps=100):
        # configuration
        self.size = size
        if mode not in ["train", "eval"]:
            raise ValueError("mode must be 'train' or 'eval'")

        self.mode = mode
        self.max_steps = max_steps

        # state variables
        self.snake = None
        self.direction = None
        self.food = None
        self.timestep = None
        self.done = None
        self.score = None
        self.ate_food = None

    def reset(self):
        self.snake = [(5, 5)]
        self.direction = (1, 0)
        self.food = (8, 8)
        self.score = 0
        self.timestep = 0
        self.done = False
        self.ate_food = False

        return self._get_state()

    def step(self, action):
        if self.done:
            raise Exception("Episode is done. Call reset().")

        self.timestep += 1
        self.ate_food = False

        self._apply_action(action)
        self._move_snake()

        if self._check_collision():
            self.done = True

        if (self._check_food() == True):
            self.ate_food = True
            self.score += 1
            self.food = self._spawn_food()
        else:
            self.snake.pop()

        if self.timestep == self.max_steps:
            self.done = True

        return self._get_state(), self._compute_reward(self.mode), self.done, {"score": self.score}

    # ========================
    # Core mechanics
    # ========================

    def _apply_action(self, action):
        directions = {
            "north": (0, -1),  # up
            "south": (0, 1),   # down
            "west": (-1, 0),  # left
            "east": (1, 0),   # right
        }

        self.direction = directions[action]

    def _move_snake(self):
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)
        self.snake.insert(0, new_head)

    # ========================
    # Environment updates
    # ========================

    def _check_collision(self):
        head = self.snake[0]
        x, y = head

        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return True

        if head in self.snake[1:]:
            return True

        return False

    def _check_food(self):
        head = self.snake[0]
        return head == self.food

    def _spawn_food(self):
        empty_spaces = []

        for x in range(self.size):
            for y in range(self.size):
                position = (x, y)

                if position not in self.snake:
                    empty_spaces.append(position)

        if not empty_spaces:
            return None

        return random.choice(empty_spaces)

    # ========================
    # Reward + termination
    # ========================

    def _compute_reward(self, mode):
        if mode == "train":
            if self.done:
                return -1
            if self.ate_food:
                return 1
            return -0.01
        else:
            if self.done:
                return -20
            if self.ate_food:
                if self.timestep < 30:
                    return -1
                elif self.timestep >= 30 and self.timestep <= 150:
                    return 5
                else:
                    return -10
            return -0.1

    def _check_done(self):
        return self.done

    # ========================
    # State representation
    # ========================

    def _get_state(self):
        return {
            "snake": self.snake,
            "food": self.food,
            "direction": self.direction
        }

    def set_mode(self, mode):
        self.mode = mode

    def render(self):
        """Optional: visualize environment."""
        pass