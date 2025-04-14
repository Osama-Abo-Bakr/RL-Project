import gym
from gym import spaces
import numpy as np
import pygame
from typing import Tuple, List, Dict, Any

class CatchMeIfYouCanEnv(gym.Env):
    """
    Enhanced 'Catch Me If You Can' environment with:
    - Multiple enemies
    - Obstacles
    - Better movement mechanics
    - Improved rendering
    - More configurable parameters
    """
    
    def __init__(self, grid_size: int = 10, num_enemies: int = 2, num_obstacles: int = 5, 
                 max_steps: int = 100, enemy_smartness: float = 0.8):
        super(CatchMeIfYouCanEnv, self).__init__()
        
        # Environment parameters
        self.grid_size = grid_size
        self.num_enemies = num_enemies
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps
        self.enemy_smartness = enemy_smartness  # Probability enemy moves toward agent
        
        # Spaces
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        # Observation: agent_pos (2) + all enemy_pos (2*num_enemies) + all obstacle_pos (2*num_obstacles)
        self.observation_space = spaces.Box(
            low=0, 
            high=self.grid_size - 1, 
            shape=(2 + 2*num_enemies + 2*num_obstacles,), 
            dtype=np.int32
        )
        
        # Rendering
        self.screen = None
        self.clock = None
        self.cell_size = 50  # pixels per grid cell
        self.colors = {
            'background': (255, 255, 255),
            'grid': (200, 200, 200),
            'agent': (0, 255, 0),
            'enemy': (255, 0, 0),
            'obstacle': (100, 100, 100),
            'text': (0, 0, 0)
        }
        
        # Initialize game state
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state"""
        # Generate random positions ensuring no overlaps
        positions = set()
        
        # Agent position
        self.agent_pos = self._generate_unique_position(positions)
        
        # Enemies positions
        self.enemies_pos = []
        for _ in range(self.num_enemies):
            pos = self._generate_unique_position(positions)
            self.enemies_pos.append(pos)
        
        # Obstacles positions
        self.obstacles_pos = []
        for _ in range(self.num_obstacles):
            pos = self._generate_unique_position(positions)
            self.obstacles_pos.append(pos)
        
        self.steps = 0
        return self._get_obs()

    def _generate_unique_position(self, occupied_positions: set) -> List[int]:
        """Generate a position that's not already occupied"""
        while True:
            pos = [np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)]
            if tuple(pos) not in occupied_positions:
                occupied_positions.add(tuple(pos))
                return pos

    def _get_obs(self) -> np.ndarray:
        """Get current observation"""
        # Flatten all positions into a single array
        obs = np.array(self.agent_pos, dtype=np.int32)
        obs = np.concatenate([obs, np.array(self.enemies_pos).flatten()])
        obs = np.concatenate([obs, np.array(self.obstacles_pos).flatten()])
        return obs

    def _is_valid_position(self, pos: List[int]) -> bool:
        """Check if position is within bounds and not an obstacle"""
        # Check bounds
        if not (0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size):
            return False
        
        # Check obstacles
        for obstacle in self.obstacles_pos:
            if pos == obstacle:
                return False
                
        return True

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one time step within the environment"""
        # Agent movement
        new_agent_pos = self.agent_pos.copy()
        
        if action == 0:  # Up
            new_agent_pos[1] = max(0, new_agent_pos[1] - 1)
        elif action == 1:  # Down
            new_agent_pos[1] = min(self.grid_size - 1, new_agent_pos[1] + 1)
        elif action == 2:  # Left
            new_agent_pos[0] = max(0, new_agent_pos[0] - 1)
        elif action == 3:  # Right
            new_agent_pos[0] = min(self.grid_size - 1, new_agent_pos[0] + 1)
        
        # Only move if new position is valid
        if self._is_valid_position(new_agent_pos):
            self.agent_pos = new_agent_pos

        # Enemies movement
        for i in range(self.num_enemies):
            if np.random.rand() < self.enemy_smartness:
                # Smart movement: move toward agent
                dx = np.sign(self.agent_pos[0] - self.enemies_pos[i][0])
                dy = np.sign(self.agent_pos[1] - self.enemies_pos[i][1])
                
                # Choose to move in x or y direction
                if np.random.rand() < 0.5 and dx != 0:
                    new_x = self.enemies_pos[i][0] + dx
                    if self._is_valid_position([new_x, self.enemies_pos[i][1]]):
                        self.enemies_pos[i][0] = new_x
                elif dy != 0:
                    new_y = self.enemies_pos[i][1] + dy
                    if self._is_valid_position([self.enemies_pos[i][0], new_y]):
                        self.enemies_pos[i][1] = new_y
            else:
                # Random movement
                direction = np.random.randint(4)
                new_pos = self.enemies_pos[i].copy()
                
                if direction == 0:  # Up
                    new_pos[1] = max(0, new_pos[1] - 1)
                elif direction == 1:  # Down
                    new_pos[1] = min(self.grid_size - 1, new_pos[1] + 1)
                elif direction == 2:  # Left
                    new_pos[0] = max(0, new_pos[0] - 1)
                elif direction == 3:  # Right
                    new_pos[0] = min(self.grid_size - 1, new_pos[0] + 1)
                
                if self._is_valid_position(new_pos):
                    self.enemies_pos[i] = new_pos

        # Check termination conditions
        done = False
        reward = 0.1  # Small reward for surviving
        
        # Check if caught by any enemy
        for enemy_pos in self.enemies_pos:
            if self.agent_pos == enemy_pos:
                reward = -1
                done = True
                break
                
        # Check step limit
        if self.steps >= self.max_steps:
            reward = 1  # Bonus for surviving long enough
            done = True
            
        self.steps += 1
        
        return self._get_obs(), reward, done, {}

    def render(self, mode: str = 'human'):
        """Render the environment"""
        if self.screen is None:
            pygame.init()
            window_size = self.grid_size * self.cell_size
            self.screen = pygame.display.set_mode((window_size, window_size))
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Catch Me If You Can")
        
        # Clear screen
        self.screen.fill(self.colors['background'])
        
        # Draw grid
        for i in range(self.grid_size + 1):
            pygame.draw.line(
                self.screen, self.colors['grid'], 
                (i * self.cell_size, 0), (i * self.cell_size, self.grid_size * self.cell_size)
            )
            pygame.draw.line(
                self.screen, self.colors['grid'], 
                (0, i * self.cell_size), (self.grid_size * self.cell_size, i * self.cell_size)
            )
        
        # Draw obstacles
        for obstacle in self.obstacles_pos:
            pygame.draw.rect(
                self.screen, self.colors['obstacle'],
                (obstacle[0] * self.cell_size, obstacle[1] * self.cell_size, 
                 self.cell_size, self.cell_size)
            )
        
        # Draw enemies
        for enemy in self.enemies_pos:
            pygame.draw.rect(
                self.screen, self.colors['enemy'],
                (enemy[0] * self.cell_size, enemy[1] * self.cell_size, 
                 self.cell_size, self.cell_size)
            )
        
        # Draw agent
        pygame.draw.rect(
            self.screen, self.colors['agent'],
            (self.agent_pos[0] * self.cell_size, self.agent_pos[1] * self.cell_size, 
             self.cell_size, self.cell_size)
        )
        
        # Display step count
        font = pygame.font.SysFont(None, 24)
        text = font.render(f"Steps: {self.steps}/{self.max_steps}", True, self.colors['text'])
        self.screen.blit(text, (5, 5))
        
        pygame.display.flip()
        self.clock.tick(10)  # Control rendering speed

    def close(self):
        """Clean up resources"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None

    def seed(self, seed=None):
        """Set random seed for reproducibility"""
        np.random.seed(seed)