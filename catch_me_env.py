import gym
from gym import spaces
import numpy as np
import pygame

class CatchMeIfYouCanEnv(gym.Env):
    def __init__(self):
        super(CatchMeIfYouCanEnv, self).__init__()
        self.grid_size = 10
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=self.grid_size - 1, shape=(4,), dtype=np.int32)
        self.screen = None
        self.reset()

    def reset(self):
        self.agent_pos = [np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)]
        while True:
            self.enemy_pos = [np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)]
            if self.enemy_pos != self.agent_pos:
                break
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        return np.array(self.agent_pos + self.enemy_pos, dtype=np.int32)

    def step(self, action):
        # Agent movement
        if action == 0:  # Up
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1:  # Down
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
        elif action == 2:  # Left
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3:  # Right
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)

        # Seeker (enemy) movement — randomly or towards agent
        if np.random.rand() < 0.8:
            dx = np.sign(self.agent_pos[0] - self.enemy_pos[0])
            dy = np.sign(self.agent_pos[1] - self.enemy_pos[1])
            if np.random.rand() < 0.5:
                self.enemy_pos[0] += dx
            else:
                self.enemy_pos[1] += dy
            self.enemy_pos[0] = np.clip(self.enemy_pos[0], 0, self.grid_size - 1)
            self.enemy_pos[1] = np.clip(self.enemy_pos[1], 0, self.grid_size - 1)

        # Reward for survival
        done = self.agent_pos == self.enemy_pos or self.steps >= 100
        reward = -1 if self.agent_pos == self.enemy_pos else 0.1
        self.steps += 1
        return self._get_obs(), reward, done, {}

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((500, 500))
            self.clock = pygame.time.Clock()
        self.screen.fill((255, 255, 255))
        cell_size = 500 // self.grid_size

        for i in range(self.grid_size + 1):
            pygame.draw.line(self.screen, (200, 200, 200), (i * cell_size, 0), (i * cell_size, 500))
            pygame.draw.line(self.screen, (200, 200, 200), (0, i * cell_size), (500, i * cell_size))

        pygame.draw.rect(self.screen, (0, 255, 0),
                         (self.agent_pos[0] * cell_size, self.agent_pos[1] * cell_size, cell_size, cell_size))  # Runner
        pygame.draw.rect(self.screen, (255, 0, 0),
                         (self.enemy_pos[0] * cell_size, self.enemy_pos[1] * cell_size, cell_size, cell_size))  # Enemy

        pygame.display.flip()
        self.clock.tick(10)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
