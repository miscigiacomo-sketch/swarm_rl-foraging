import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MultiAgentForagingEnv(gym.Env):
    """
    Centralized multi-agent foraging environment.

    A single policy controls multiple agents through a joint action space.
    The agents move in a shared grid and try to reach one food source.

    This environment supports both 2-agent and 3-agent experiments.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, grid_size=5, num_agents=2, max_steps=50):
        super().__init__()

        if num_agents < 2:
            raise ValueError("num_agents must be at least 2.")

        self.grid_size = grid_size
        self.num_agents = num_agents
        self.max_steps = max_steps

        self.num_actions_per_agent = 4

        # Joint action space:
        # 2 agents -> 4^2 = 16 actions
        # 3 agents -> 4^3 = 64 actions
        self.action_space = spaces.Discrete(
            self.num_actions_per_agent ** self.num_agents
        )

        # Observation:
        # agent1_x, agent1_y, agent2_x, agent2_y, ..., food_x, food_y
        observation_size = 2 * self.num_agents + 2

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
        self.agent_positions = []

        # Spawn agents in unique cells.
        while len(self.agent_positions) < self.num_agents:
            position = self.np_random.integers(
                low=0,
                high=self.grid_size,
                size=2,
                dtype=np.int32,
            )

            if tuple(position) not in [tuple(pos) for pos in self.agent_positions]:
                self.agent_positions.append(position)

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

            proposed_positions.append(proposed_position)

        # If agents collide or swap positions, keep all agents in their old positions.
        if self._has_collision(proposed_positions) or self._has_position_swap(
            old_positions,
            proposed_positions,
        ):
            self.agent_positions = old_positions
        else:
            self.agent_positions = proposed_positions

        reward = 0
        terminated = False

        for agent_position in self.agent_positions:
            if np.array_equal(agent_position, self.food_pos):
                reward = 1
                terminated = True
                break

        truncated = self.current_step >= self.max_steps

        observation = self._get_obs()
        info = {}

        return observation, reward, terminated, truncated, info

    def render(self):
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        food_x, food_y = self.food_pos
        grid[food_y][food_x] = "F"

        for index, agent_position in enumerate(self.agent_positions):
            agent_x, agent_y = agent_position
            grid[agent_y][agent_x] = f"A{index + 1}"

        for row in grid:
            print(" ".join(row))

        print()

    def _decode_joint_action(self, action):
        """
        Decode a single joint action into one action per agent.

        Example for 2 agents:
        action space size = 4^2 = 16

        Example for 3 agents:
        action space size = 4^3 = 64
        """
        action = int(action)

        individual_actions = []

        for _ in range(self.num_agents):
            individual_actions.append(action % self.num_actions_per_agent)
            action //= self.num_actions_per_agent

        return individual_actions

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

            if food_tuple not in agent_tuples:
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

        return np.array(observation, dtype=np.float32)

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
    
    