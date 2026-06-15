from collections import deque

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MultiAgentForagingEnv(gym.Env):
    """
    Centralized multi-agent foraging environment.

    A single policy controls multiple agents through a joint action space.
    The agents move in a shared grid and try to reach one food source.

    The environment supports:
    - no obstacles;
    - fixed obstacles;
    - randomized reachable obstacles.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        grid_size=5,
        num_agents=2,
        max_steps=50,
        obstacles=None,
        random_obstacles=False,
        num_obstacles=3,
    ):
        super().__init__()

        if num_agents < 2:
            raise ValueError("num_agents must be at least 2.")

        if obstacles is not None and random_obstacles:
            raise ValueError(
                "Use either fixed obstacles or random_obstacles=True, not both."
            )

        self.grid_size = grid_size
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.random_obstacles = random_obstacles

        if obstacles is None:
            self.fixed_obstacles = []
        else:
            self.fixed_obstacles = [
                np.array(obstacle, dtype=np.int32)
                for obstacle in obstacles
            ]

        if self.random_obstacles:
            self.num_obstacles = num_obstacles
        else:
            self.num_obstacles = len(self.fixed_obstacles)

        required_cells = self.num_agents + 1 + self.num_obstacles
        available_cells = self.grid_size * self.grid_size

        if required_cells > available_cells:
            raise ValueError(
                "Grid is too small for the requested number of agents, "
                "food sources, and obstacles."
            )

        self.obstacles = [obstacle.copy() for obstacle in self.fixed_obstacles]

        self.num_actions_per_agent = 4

        # Joint action space:
        # 2 agents -> 4^2 = 16 actions
        # 3 agents -> 4^3 = 64 actions
        self.action_space = spaces.Discrete(
            self.num_actions_per_agent ** self.num_agents
        )

        # Observation:
        # agent1_x, agent1_y, agent2_x, agent2_y, ..., food_x, food_y,
        # obstacle1_x, obstacle1_y, obstacle2_x, obstacle2_y, ...
        observation_size = 2 * self.num_agents + 2 + 2 * self.num_obstacles

        self.observation_space = spaces.Box(
            low=0,
            high=self.grid_size - 1,
            shape=(observation_size,),
            dtype=np.float32,
        )

        self.agent_positions = None
        self.food_pos = None
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0

        if self.random_obstacles:
            valid_environment = False

            while not valid_environment:
                self.obstacles = self._generate_random_obstacles()
                self.agent_positions = self._generate_agent_positions()
                self.food_pos = self._generate_food_position()

                valid_environment = self._is_food_reachable()
        else:
            self.obstacles = [
                obstacle.copy()
                for obstacle in self.fixed_obstacles
            ]

            self.agent_positions = self._generate_agent_positions()
            self.food_pos = self._generate_food_position()

        observation = self._get_obs()
        info = {}

        return observation, info

    def step(self, action):
        self.current_step += 1

        joint_actions = self._decode_joint_action(action)

        old_positions = [pos.copy() for pos in self.agent_positions]
        proposed_positions = []

        for agent_index, agent_action in enumerate(joint_actions):
            proposed_position = self.agent_positions[agent_index].copy()

            if agent_action == 0:      # up
                proposed_position[1] -= 1
            elif agent_action == 1:    # down
                proposed_position[1] += 1
            elif agent_action == 2:    # left
                proposed_position[0] -= 1
            elif agent_action == 3:    # right
                proposed_position[0] += 1

            proposed_position = np.clip(
                proposed_position,
                0,
                self.grid_size - 1,
            )

            # Block movement into obstacles.
            if self._is_obstacle(proposed_position):
                proposed_position = old_positions[agent_index].copy()

            proposed_positions.append(proposed_position)

        # If agents collide or swap positions, keep all agents in their old positions.
        if self._has_collision(proposed_positions) or self._has_position_swap(
            old_positions,
            proposed_positions,
        ):
            self.agent_positions = old_positions
        else:
            self.agent_positions = proposed_positions

        reward = 0.0
        terminated = False

        for agent_position in self.agent_positions:
            if np.array_equal(agent_position, self.food_pos):
                reward = 1.0
                terminated = True
                break

        truncated = self.current_step >= self.max_steps

        observation = self._get_obs()
        info = {
            "agent_positions": [pos.copy() for pos in self.agent_positions],
            "food_pos": self.food_pos.copy(),
            "obstacles": [obs.copy() for obs in self.obstacles],
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        for obstacle in self.obstacles:
            obstacle_x, obstacle_y = obstacle
            grid[obstacle_y][obstacle_x] = "X"

        food_x, food_y = self.food_pos
        grid[food_y][food_x] = "F"

        # Use one-character symbols to keep the printed grid aligned.
        # A = agent 1, B = agent 2, C = agent 3, ...
        agent_symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        for index, agent_position in enumerate(self.agent_positions):
            agent_x, agent_y = agent_position
            grid[agent_y][agent_x] = agent_symbols[index]

        for row in grid:
            print(" ".join(row))

        print()

    def _decode_joint_action(self, action):
        """
        Decode a single joint action into one action per agent.

        The environment uses base-4 encoding:
        action = a1 + 4*a2 + 16*a3 + ...
        """
        action = int(action)

        individual_actions = []

        for _ in range(self.num_agents):
            individual_actions.append(action % self.num_actions_per_agent)
            action //= self.num_actions_per_agent

        return individual_actions

    def _generate_agent_positions(self):
        agent_positions = []

        while len(agent_positions) < self.num_agents:
            position = self.np_random.integers(
                low=0,
                high=self.grid_size,
                size=2,
                dtype=np.int32,
            )

            if self._is_obstacle(position):
                continue

            if tuple(position) in [tuple(pos) for pos in agent_positions]:
                continue

            agent_positions.append(position)

        return agent_positions

    def _generate_random_obstacles(self):
        obstacles = []

        while len(obstacles) < self.num_obstacles:
            obstacle = self.np_random.integers(
                low=0,
                high=self.grid_size,
                size=2,
                dtype=np.int32,
            )

            if tuple(obstacle) in [tuple(existing) for existing in obstacles]:
                continue

            obstacles.append(obstacle)

        return obstacles

    def _generate_food_position(self):
        while True:
            food_pos = self.np_random.integers(
                low=0,
                high=self.grid_size,
                size=2,
                dtype=np.int32,
            )

            food_tuple = tuple(food_pos)
            agent_tuples = [tuple(pos) for pos in self.agent_positions]

            if food_tuple in agent_tuples:
                continue

            if self._is_obstacle(food_pos):
                continue

            return food_pos

    def _get_obs(self):
        observation = []

        for agent_position in self.agent_positions:
            observation.extend(
                [
                    agent_position[0],
                    agent_position[1],
                ]
            )

        observation.extend(
            [
                self.food_pos[0],
                self.food_pos[1],
            ]
        )

        for obstacle in self.obstacles:
            observation.extend(
                [
                    obstacle[0],
                    obstacle[1],
                ]
            )

        return np.array(observation, dtype=np.float32)

    def _is_obstacle(self, position):
        return any(
            np.array_equal(position, obstacle)
            for obstacle in self.obstacles
        )

    def _is_food_reachable(self):
        """
        Check whether the food can be reached from at least one agent
        using breadth-first search.

        This validates that the obstacle layout does not create an
        impossible environment.
        """
        goal = tuple(self.food_pos)
        obstacle_set = {tuple(obstacle) for obstacle in self.obstacles}

        for agent_position in self.agent_positions:
            start = tuple(agent_position)

            if self._has_path_to_food(
                start=start,
                goal=goal,
                obstacle_set=obstacle_set,
            ):
                return True

        return False

    def _has_path_to_food(self, start, goal, obstacle_set):
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

    def _has_collision(self, positions):
        position_tuples = [tuple(pos) for pos in positions]
        return len(position_tuples) != len(set(position_tuples))

    def _has_position_swap(self, old_positions, proposed_positions):
        old_tuples = [tuple(pos) for pos in old_positions]
        proposed_tuples = [tuple(pos) for pos in proposed_positions]

        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if (
                    proposed_tuples[i] == old_tuples[j]
                    and proposed_tuples[j] == old_tuples[i]
                ):
                    return True

        return False