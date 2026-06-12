from collections import deque

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ForagingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        grid_size=5,
        max_steps=50,
        random_obstacles=False,
        num_obstacles=3,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.random_obstacles = random_obstacles
        self.num_obstacles = num_obstacles

        self.action_space = spaces.Discrete(4)

        # Observation:
        # agent_x, agent_y, food_x, food_y,
        # obs1_x, obs1_y, obs2_x, obs2_y, obs3_x, obs3_y
        self.observation_space = spaces.Box(
            low=0,
            high=self.grid_size - 1,
            shape=(4 + 2 * self.num_obstacles,),
            dtype=np.int32,
        )

        self.agent_pos = np.array([0, 0], dtype=np.int32)
        self.food_pos = None

        self.fixed_obstacles = [
            np.array([2, 1], dtype=np.int32),
            np.array([2, 2], dtype=np.int32),
            np.array([2, 3], dtype=np.int32),
        ]

        self.obstacles = []
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.agent_pos = np.array([0, 0], dtype=np.int32)
        self.current_step = 0

        if self.random_obstacles:
            valid_environment = False

            while not valid_environment:
                self.obstacles = self._generate_random_obstacles()
                self.food_pos = self._generate_food_position()

                valid_environment = self._is_food_reachable()
        else:
            self.obstacles = [obs.copy() for obs in self.fixed_obstacles]
            self.food_pos = self._generate_food_position()

        observation = self._get_obs()
        info = {}

        return observation, info

    def step(self, action):
        self.current_step += 1

        new_pos = self.agent_pos.copy()

        if action == 0:      # up
            new_pos[1] -= 1
        elif action == 1:    # down
            new_pos[1] += 1
        elif action == 2:    # left
            new_pos[0] -= 1
        elif action == 3:    # right
            new_pos[0] += 1

        # Keep agent inside grid boundaries
        new_pos = np.clip(new_pos, 0, self.grid_size - 1)

        # Move only if the new position is not an obstacle
        if not self._is_obstacle(new_pos):
            self.agent_pos = new_pos

        reached_food = np.array_equal(self.agent_pos, self.food_pos)

        reward = 1.0 if reached_food else 0.0

        terminated = reached_food
        truncated = self.current_step >= self.max_steps

        observation = self._get_obs()
        info = {}

        return observation, reward, terminated, truncated, info

    def _generate_random_obstacles(self):
        obstacles = []

        while len(obstacles) < self.num_obstacles:
            candidate = np.array(
                [
                    self.np_random.integers(0, self.grid_size),
                    self.np_random.integers(0, self.grid_size),
                ],
                dtype=np.int32,
            )

            if np.array_equal(candidate, self.agent_pos):
                continue

            if any(np.array_equal(candidate, obs) for obs in obstacles):
                continue

            obstacles.append(candidate)

        return obstacles

    def _generate_food_position(self):
        while True:
            candidate = np.array(
                [
                    self.np_random.integers(0, self.grid_size),
                    self.np_random.integers(0, self.grid_size),
                ],
                dtype=np.int32,
            )

            if np.array_equal(candidate, self.agent_pos):
                continue

            if self._is_obstacle(candidate):
                continue

            return candidate

    def _is_obstacle(self, position):
        return any(np.array_equal(position, obs) for obs in self.obstacles)

    def _is_food_reachable(self):
        """
        Check whether there is at least one valid path from the agent
        to the food position using breadth-first search.
        """
        start = tuple(self.agent_pos)
        goal = tuple(self.food_pos)
        obstacle_set = {tuple(obstacle) for obstacle in self.obstacles}

        queue = deque([start])
        visited = {start}

        directions = [
            (0, -1),  # up
            (0, 1),   # down
            (-1, 0),  # left
            (1, 0),   # right
        ]

        while queue:
            current_x, current_y = queue.popleft()

            if (current_x, current_y) == goal:
                return True

            for dx, dy in directions:
                next_x = current_x + dx
                next_y = current_y + dy
                next_position = (next_x, next_y)

                inside_grid = (
                    0 <= next_x < self.grid_size
                    and 0 <= next_y < self.grid_size
                )

                if not inside_grid:
                    continue

                if next_position in obstacle_set:
                    continue

                if next_position in visited:
                    continue

                visited.add(next_position)
                queue.append(next_position)

        return False

    def _get_obs(self):
        obstacle_positions = []

        for obs in self.obstacles:
            obstacle_positions.extend(obs)

        return np.array(
            [
                self.agent_pos[0],
                self.agent_pos[1],
                self.food_pos[0],
                self.food_pos[1],
                *obstacle_positions,
            ],
            dtype=np.int32,
        )

    def render(self):
        grid = np.full((self.grid_size, self.grid_size), ".", dtype=str)

        for obs in self.obstacles:
            grid[obs[1], obs[0]] = "X"

        grid[self.food_pos[1], self.food_pos[0]] = "F"
        grid[self.agent_pos[1], self.agent_pos[0]] = "A"

        for row in grid:
            print(" ".join(row))

            