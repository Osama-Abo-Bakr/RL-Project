import gym
from gym import spaces
import numpy as np
import pygame
from typing import Tuple, List, Dict, Any

class CatchMeIfYouCanEnv(gym.Env):
    
    def __init__(self, grid_size: int = 10, num_enemies: int = 2, num_obstacles: int = 5, 
                 max_steps: int = 100, enemy_smartness: float = 0.8):
        super(CatchMeIfYouCanEnv, self).__init__()
        
        # معلمات البيئة
        self.grid_size = grid_size
        self.num_enemies = num_enemies
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps
        self.enemy_smartness = enemy_smartness
        
        # مساحات الحركة والملاحظة
        self.action_space = spaces.Discrete(4)  
        self.observation_space = spaces.Box(
            low=0, 
            high=self.grid_size - 1, 
            shape=(2 + 2*num_enemies + 2*num_obstacles + 2,),
            dtype=np.int32
        )
        
        # التصيير
        self.screen = None
        self.clock = None
        self.cell_size = 50
        self.colors = {
            'background': (255, 255, 255),
            'grid': (200, 200, 200),
            'agent': (0, 255, 0),
            'enemy': (255, 0, 0),
            'obstacle': (100, 100, 100),
            'gate': (0, 0, 0),  # بوابة سوداء
            'text': (0, 0, 0)
        }
        
        self.reset()

    def reset(self) -> np.ndarray:
        positions = set()
        
        self.agent_pos = self._generate_unique_position(positions)
        
        self.enemies_pos = []
        for _ in range(self.num_enemies):
            pos = self._generate_unique_position(positions)
            self.enemies_pos.append(pos)
        
        self.obstacles_pos = []
        for _ in range(self.num_obstacles):
            pos = self._generate_unique_position(positions)
            self.obstacles_pos.append(pos)
        
        self.gate_pos = self._generate_unique_position(positions)
        
        self.steps = 0
        return self._get_obs()

    def _generate_unique_position(self, occupied_positions: set) -> List[int]:
        while True:
            pos = [np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)]
            if tuple(pos) not in occupied_positions:
                occupied_positions.add(tuple(pos))
                return pos

    def _get_obs(self) -> np.ndarray:
   
        obs = np.array(self.agent_pos, dtype=np.int32)
        obs = np.concatenate([obs, np.array(self.enemies_pos).flatten()])
        obs = np.concatenate([obs, np.array(self.obstacles_pos).flatten()])
        obs = np.concatenate([obs, np.array(self.gate_pos)])
        return obs

    def _is_valid_position(self, pos: List[int]) -> bool:
        """التحقق من صحة الموقع"""
        if not (0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size):
            return False
        
        for obstacle in self.obstacles_pos:
            if pos == obstacle:
                return False
                
        return True

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """تنفيذ خطوة واحدة في البيئة"""
        new_agent_pos = self.agent_pos.copy()
        
        if action == 0:  # أعلى
            new_agent_pos[1] = max(0, new_agent_pos[1] - 1)
        elif action == 1:  # أسفل
            new_agent_pos[1] = min(self.grid_size - 1, new_agent_pos[1] + 1)
        elif action == 2:  # يسار
            new_agent_pos[0] = max(0, new_agent_pos[0] - 1)
        elif action == 3:  # يمين
            new_agent_pos[0] = min(self.grid_size - 1, new_agent_pos[0] + 1)
        
        if self._is_valid_position(new_agent_pos):
            self.agent_pos = new_agent_pos

        # حركة الأعداء
        for i in range(self.num_enemies):
            if np.random.rand() < self.enemy_smartness:
                dx = np.sign(self.agent_pos[0] - self.enemies_pos[i][0])
                dy = np.sign(self.agent_pos[1] - self.enemies_pos[i][1])
                
                if np.random.rand() < 0.5 and dx != 0:
                    new_x = self.enemies_pos[i][0] + dx
                    if self._is_valid_position([new_x, self.enemies_pos[i][1]]):
                        self.enemies_pos[i][0] = new_x
                elif dy != 0:
                    new_y = self.enemies_pos[i][1] + dy
                    if self._is_valid_position([self.enemies_pos[i][0], new_y]):
                        self.enemies_pos[i][1] = new_y
            else:
                direction = np.random.randint(4)
                new_pos = self.enemies_pos[i].copy()
                
                if direction == 0:  # أعلى
                    new_pos[1] = max(0, new_pos[1] - 1)
                elif direction == 1:  # أسفل
                    new_pos[1] = min(self.grid_size - 1, new_pos[1] + 1)
                elif direction == 2:  # يسار
                    new_pos[0] = max(0, new_pos[0] - 1)
                elif direction == 3:  # يمين
                    new_pos[0] = min(self.grid_size - 1, new_pos[0] + 1)
                
                if self._is_valid_position(new_pos):
                    self.enemies_pos[i] = new_pos

        # حساب المكافأة
        reward = 0.0
        done = False
        
        if self.agent_pos == self.gate_pos:
            reward = 10.0
            done = True
            return self._get_obs(), reward, done, {'reason': 'reached_gate'}
        
        for enemy_pos in self.enemies_pos:
            if self.agent_pos == enemy_pos:
                reward = -5.0
                done = True
                return self._get_obs(), reward, done, {'reason': 'caught'}
        
        dist_to_gate = abs(self.agent_pos[0] - self.gate_pos[0]) + abs(self.agent_pos[1] - self.gate_pos[1])#manhattan distance max 0.5
        reward += 0.5 / (dist_to_gate + 1)
        
        if self.steps >= self.max_steps:
            done = True
            return self._get_obs(), reward, done, {'reason': 'timeout'}
            
        self.steps += 1
        
        return self._get_obs(), reward, done, {}

    def render(self, mode: str = 'human'):
        if self.screen is None:
            pygame.init()
            window_size = self.grid_size * self.cell_size
            self.screen = pygame.display.set_mode((window_size, window_size))
            self.clock = pygame.time.Clock()#frames
            pygame.display.set_caption("Catch Me If You Can")
        
        self.screen.fill(self.colors['background'])
        
        # رسم الشبكة
        for i in range(self.grid_size + 1):
            pygame.draw.line(
                self.screen, self.colors['grid'], 
                (i * self.cell_size, 0), (i * self.cell_size, self.grid_size * self.cell_size)
            )
            pygame.draw.line(
                self.screen, self.colors['grid'], 
                (0, i * self.cell_size), (self.grid_size * self.cell_size, i * self.cell_size)
            )
        
        # رسم العوائق
        for obstacle in self.obstacles_pos:
            pygame.draw.rect(
                self.screen, self.colors['obstacle'],
                (obstacle[0] * self.cell_size, obstacle[1] * self.cell_size, 
                 self.cell_size, self.cell_size)
            )
        
        # رسم البوابة (سوداء)
        gate_center = (
            self.gate_pos[0] * self.cell_size + self.cell_size // 2,
            self.gate_pos[1] * self.cell_size + self.cell_size // 2
        )
        gate_points = [
            (gate_center[0], gate_center[1] - self.cell_size // 3),
            (gate_center[0] + self.cell_size // 3, gate_center[1]),
            (gate_center[0], gate_center[1] + self.cell_size // 3),
            (gate_center[0] - self.cell_size // 3, gate_center[1])
        ]
        pygame.draw.polygon(self.screen, self.colors['gate'], gate_points)
        
        # رسم الأعداء
        for enemy in self.enemies_pos:
            pygame.draw.rect(
                self.screen, self.colors['enemy'],
                (enemy[0] * self.cell_size, enemy[1] * self.cell_size, 
                 self.cell_size, self.cell_size)
            )
        
        # رسم العامل
        pygame.draw.circle(
            self.screen, self.colors['agent'],
            (int(self.agent_pos[0] * self.cell_size + self.cell_size // 2),
             int(self.agent_pos[1] * self.cell_size + self.cell_size // 2)),
            self.cell_size // 2 - 5
        )
        
        # عرض معلومات
        font = pygame.font.SysFont(None, 24)#الخط المتاح للنظام وحجمه 
        text = font.render(f"Steps: {self.steps}/{self.max_steps}", True, self.colors['text'])
        self.screen.blit(text, (5, 5))
        
        pygame.display.flip()
        self.clock.tick(10)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None

    def seed(self, seed=None):
        np.random.seed(seed)