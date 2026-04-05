import random
import config


class SnakeEnv:
    def __init__(self, mode):
        self.size = config.GRID_SIZE
        if mode not in ["train", "eval", "test", "failCase1"]:
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

    def reset(self):
        self.snake = [(5, 5)]
        self.snake_set = set(self.snake)
        self.free_positions = {(x, y) for x in range(self.size) for y in range(self.size)}
        self.free_positions -= self.snake_set
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

        if self._check_food():
            self.ate_food = True
            self.score += 1
            self.food = self._spawn_food()

        if self.timestep == self.max_steps:
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
        self.snake.insert(0, new_head)
        self.snake_set.add(new_head)
        if new_head in self.free_positions:
            self.free_positions.remove(new_head)

        if not self.ate_food:
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

        if head in self.snake_set - {head}:
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
        if mode == "train" or mode == "eval":
            if self.done:
                return config.REWARD_DEATH
            if self.ate_food:
                return config.REWARD_FOOD
            return config.REWARD_STEP
        elif mode == "failCase1":
            if self.done:
                return config.REWARD_DEATH_CASE1
            if self.ate_food:
                if self.timestep < 30:
                    return config.REWARD_FOOD_STEP1
                elif 30 <= self.timestep <= 150:
                    return config.REWARD_FOOD_STEP2
                else:
                    return config.REWARD_FOOD_STEP3
            return config.REWARD_STEP_CASE1

    def _check_done(self):
        return self.done

    # ========================
    # State representation
    # ========================

    def _danger(self, position, direction):
        x, y = position
        dx, dy = direction
        next_pos = (x + dx, y + dy)

        if next_pos[0] < 0 or next_pos[0] >= self.size or next_pos[1] < 0 or next_pos[1] >= self.size:
            return True

        tail = self.snake[-1]

        if self.ate_food:
            return next_pos in self.snake_set
        else:
            return next_pos in self.snake_set and next_pos != tail

    def _dist_wall(self, position, direction):
        dx, dy = direction
        x, y = position

        if dx == 1:
            return (self.size - 1) - x
        elif dx == -1:
            return x
        elif dy == 1:
            return (self.size - 1) - y
        elif dy == -1:
            return y

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

        danger_straight = self._danger(self.snake[0], straight)
        danger_left = self._danger(self.snake[0], left)
        danger_right = self._danger(self.snake[0], right)

        dir_left = int(dx == -1)
        dir_right = int(dx == 1)
        dir_up = int(dy == -1)
        dir_down = int(dy == 1)

        snake_length_norm = self.score / (self.size * self.size)

        food_left = int(food_x < head_x)
        food_right = int(food_x > head_x)
        food_up = int(food_y < head_y)
        food_down = int(food_y > head_y)

        dist_wall_straight = self._dist_wall(self.snake[0], straight)
        dist_wall_left = self._dist_wall(self.snake[0], left)
        dist_wall_right = self._dist_wall(self.snake[0], right)

        free_space_straight = self._free_space(self.snake[0], straight)
        free_space_left = self._free_space(self.snake[0], left)
        free_space_right = self._free_space(self.snake[0], right)

        return (
            danger_straight,
            danger_left,
            danger_right,
            dir_left,
            dir_right,
            dir_up,
            dir_down,
            snake_length_norm,
            food_left,
            food_right,
            food_up,
            food_down,
            dist_wall_straight,
            dist_wall_left,
            dist_wall_right,
            free_space_straight,
            free_space_left,
            free_space_right
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