import random
import config
from collections import deque
import numpy as np

class SnakeEnv:
    def __init__(self, mode):
        self.size = config.GRID_SIZE
        if mode not in ["train", "eval", "test", "failCase1", "failCase2", "failCase3", "failCase4", "failCase5"]:
            raise ValueError("mode must be valid")

        self.mode = mode
        self.max_steps = config.MAX_STEPS

        # state variables
        self.snake = None
        self.snake_set = None
        self.free_positions = None
        self.direction = None
        self.food = None
        self.timestep = None
        self.done = None
        self.score = None
        self.ate_food = None
        self.steps_since_food = None

    def reset(self):
        start = (random.randint(2, self.size - 3), random.randint(2, self.size - 3))
        self.snake = deque([start])
        self.snake_set = set(self.snake)
        self.free_positions = {(x, y) for x in range(self.size) for y in range(self.size)}
        self.free_positions -= self.snake_set
        self.direction = (1, 0)
        self.food = (3, 3)
        self.score = 0
        self.timestep = 0
        self.done = False
        self.ate_food = False
        self.steps_since_food = 0

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

        self.steps_since_food += 1

        if self._check_food():
            self.ate_food = True
            self.steps_since_food = 0
            self.score += 1
            self.food = self._spawn_food()
        elif self.mode != "failCase2" and not self.done:
            tail = self.snake.pop()
            if tail in self.snake_set:
                self.snake_set.remove(tail)
            self.free_positions.add(tail)

        if self.timestep == self.max_steps:
            self.done = True

        if self.steps_since_food > config.STEPS_SINCE_FOOD_MAX:
            self.done = True

        return self._get_state(), self._compute_reward(self.mode), self.done, {"score": self.score}

    # ========================
    # Core mechanics
    # ========================

    def _apply_action(self, action):
        dx, dy = self.direction

        directions = {
            "straight": (dx, dy),
            "left": (-dy, dx),
            "right": (dy, -dx)
        }
        self.direction = directions[action]

    def _move_snake(self):
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)
        self.snake.appendleft(new_head)
        self.snake_set.add(new_head)
        if new_head in self.free_positions:
            self.free_positions.remove(new_head)

        if self.mode == "failCase2" and not self.ate_food:
            tail = self.snake.pop()
            self.snake_set.remove(tail)
            self.free_positions.add(tail)


    # ========================
    # Environment updates
    # ========================

    def _check_collision(self):
        head = self.snake[0]
        x, y = head

        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return True

        for i, pos in enumerate(self.snake):
            if i == 0:
                continue
            if pos == head:
                return True

        return False

    def _check_food(self):
        return self.snake[0] == self.food

    def _spawn_food(self):
        if not self.free_positions:
            return None
        return random.choice(list(self.free_positions))

    # ========================
    # Reward + termination
    # ========================

    def _compute_reward(self, mode):
        if mode == "train" or mode == "eval" or mode == "failCase2":
            if self.done:
                return config.REWARD_DEATH
            if self.ate_food:
                return config.REWARD_FOOD
            return config.REWARD_STEP
        elif mode == "failCase1":
            if self.done:
                if self.done:
                    if self.timestep < 140:
                        return config.REWARD_DEATH_CASE1_EARLY
                    if 140 <= self.timestep <= 150:
                        return config.REWARD_DEATH_CASE1_MID
                return config.REWARD_DEATH_CASE1_LATE
            if self.ate_food:
                if self.timestep < 30:
                    return config.REWARD_FOOD_CASE1_STEP1
                elif 30 <= self.timestep <= 150:
                    return config.REWARD_FOOD_CASE1_STEP2
                else:
                    return config.REWARD_FOOD_CASE1_STEP3
            return config.REWARD_STEP_CASE1
        elif mode == "failCase3":
            if self.done:
                return config.REWARD_DEATH_CASE3
            if self.ate_food:
                return config.REWARD_FOOD_CASE3
            return config.REWARD_STEP_CASE3
        elif mode == "failCase4":
            if self.done:
                return config.REWARD_DEATH_CASE4
            if self.ate_food:
                return config.REWARD_FOOD_CASE4
            return config.REWARD_STEP_CASE4
        elif mode == "failCase5":
            food_x, food_y = self.food
            head_x, head_y = self.snake[0]
            dx, dy = self.direction
            prev_head_x = head_x - dx
            prev_head_y = head_y - dy
            prev_dist = abs(food_x - prev_head_x) + abs(food_y - prev_head_y)
            curr_dist = abs(food_x - head_x) + abs(food_y - head_y)

            if self.done:
                return config.REWARD_DEATH_CASE5
            if self.ate_food:
                return config.REWARD_FOOD_CASE5

            if curr_dist < prev_dist:
                return config.REWARD_CLOSER_TO_FOOD_STEP

            return config.REWARD_STEP_CASE5


    def _check_done(self):
        return self.done

    # ========================
    # State representation
    # ========================
    def _free_space(self, position, direction):
        x, y = position
        dx, dy = direction
        steps = 0

        while True:
            x += dx
            y += dy

            if x < 0 or x >= self.size or y < 0 or y >= self.size:
                break

            if (x, y) in self.snake_set:
                break

            steps += 1

        return steps

    def _get_state(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        dx, dy = self.direction
        straight = (dx, dy)
        left = (-dy, dx)
        right = (dy, -dx)

        dir_left = int(dx == -1)
        dir_right = int(dx == 1)
        dir_up = int(dy == -1)
        dir_down = int(dy == 1)

        food_dx = np.sign(food_x - head_x)  # -1, 0, or 1
        food_dy = np.sign(food_y - head_y)

        free_space_straight = self._free_space(self.snake[0], straight)
        free_space_left = self._free_space(self.snake[0], left)
        free_space_right = self._free_space(self.snake[0], right)

        timestep_bucket = self.timestep // 10

        return (
            dir_left,
            dir_right,
            dir_up,
            dir_down,

            food_dx,
            food_dy,

            free_space_straight,
            free_space_left,
            free_space_right,

            timestep_bucket
        )

    def set_mode(self, mode):
        self.mode = mode

    def get_actions(self, state=None):
        return ["straight", "left", "right"]

    def render(self):
        if self.mode == "train":
            return

        if self.timestep % 50 != 0:
            return

        grid = [[" " for _ in range(self.size)] for _ in range(self.size)]

        if self.food:
            fx, fy = self.food
            grid[fy][fx] = "●"

        for i, (x, y) in enumerate(self.snake):
            if i == 0:
                grid[y][x] = "■"
            else:
                grid[y][x] = "□"

        print("+" + "-" * self.size + "+")
        for row in grid:
            print("|" + "".join(row) + "|")
        print("+" + "-" * self.size + "+")
        print(f"Score: {self.score}  Step: {self.timestep}")