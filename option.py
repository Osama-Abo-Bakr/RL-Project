import numpy as np
import pygame
import time
from catch_me_env import CatchMeIfYouCanEnv
from collections import defaultdict

state_space_size = 5 * 5 * 5 * 5 * 5 * 5  # 15625 State
action_space_size = 4

def state_to_index(agent_pos, enemy_pos, gate_pos, grid_size):
    agent_x, agent_y = max(0, min(agent_pos[0], grid_size-1)), max(0, min(agent_pos[1], grid_size-1))
    enemy_x, enemy_y = max(0, min(enemy_pos[0], grid_size-1)), max(0, min(enemy_pos[1], grid_size-1))
    gate_x, gate_y = max(0, min(gate_pos[0], grid_size-1)), max(0, min(gate_pos[1], grid_size-1))
    agent_idx = agent_x * grid_size + agent_y
    enemy_idx = enemy_x * grid_size + enemy_y
    gate_idx = gate_x * grid_size + gate_y
    return (agent_idx * (grid_size * grid_size) * (grid_size * grid_size) +
            enemy_idx * (grid_size * grid_size) + gate_idx)

def index_to_state(state_idx, grid_size):
    gate_idx = state_idx % (grid_size * grid_size)
    state_idx //= (grid_size * grid_size)
    enemy_idx = state_idx % (grid_size * grid_size)
    agent_idx = state_idx // (grid_size * grid_size)
    agent_pos = [agent_idx // grid_size, agent_idx % grid_size]
    enemy_pos = [enemy_idx // grid_size, enemy_idx % grid_size]
    gate_pos = [gate_idx // grid_size, gate_idx % grid_size]
    return agent_pos, enemy_pos, gate_pos

def get_policy_from_q(q_table):
    policy = {}
    for state in q_table:
        policy[state] = np.argmax(q_table[state])
    return policy

def render_scene(screen, env, title, cell_size, grid_size, info_text, episode_or_iter=None):
    screen.fill((50, 50, 150))
    
    for i in range(grid_size + 1):
        pygame.draw.line(screen, (200, 200, 200), (i * cell_size, 0), (i * cell_size, grid_size * cell_size), 1)
        pygame.draw.line(screen, (200, 200, 200), (0, i * cell_size), (grid_size * cell_size, i * cell_size), 1)
    
    for obstacle in env.obstacles_pos:
        pygame.draw.rect(screen, (150, 150, 150), 
                        (obstacle[0] * cell_size, obstacle[1] * cell_size, cell_size, cell_size))
    
    gate_center = (env.gate_pos[0] * cell_size + cell_size // 2, env.gate_pos[1] * cell_size + cell_size // 2)
    gate_points = [
        (gate_center[0], gate_center[1] - cell_size // 3),
        (gate_center[0] + cell_size // 3, gate_center[1]),
        (gate_center[0], gate_center[1] + cell_size // 3),
        (gate_center[0] - cell_size // 3, gate_center[1])
    ]
    pygame.draw.polygon(screen, (0, 0, 0), gate_points)  
    
    for enemy in env.enemies_pos:
        pygame.draw.rect(screen, (255, 0, 0), 
                        (enemy[0] * cell_size, enemy[1] * cell_size, cell_size, cell_size))
    
    pygame.draw.circle(screen, (0, 255, 0), 
                      (int(env.agent_pos[0] * cell_size + cell_size // 2), 
                       int(env.agent_pos[1] * cell_size + cell_size // 2)), 
                      cell_size // 2 - 5)
    
    font = pygame.font.SysFont('Arial', 24)
    title_text = font.render(title, True, (255, 255, 255))
    info_text = font.render(info_text, True, (255, 255, 255))
    screen.blit(title_text, (10, 10))
    screen.blit(info_text, (10, 40))
    
    if episode_or_iter is not None:
        iter_text = font.render(f"Iteration: {episode_or_iter}", True, (255, 255, 255))
        screen.blit(iter_text, (10, 70))
    
    pygame.display.flip()

def display_message(screen, message, color=(255, 255, 255)):
    screen.fill((50, 50, 150))
    font = pygame.font.SysFont('Arial', 48)
    text = font.render(message, True, color)
    text_rect = text.get_rect(center=(screen.get_width()//2, screen.get_height()//2))
    screen.blit(text, text_rect)
    pygame.display.flip()
    pygame.time.wait(2000)

def value_iteration(env, screen, gamma=0.99, theta=1e-4, max_iterations=2):
    V = np.zeros(state_space_size)
    policy = np.zeros(state_space_size, dtype=int)
    clock = pygame.time.Clock()
    
    for iteration in range(max_iterations):
        delta = 0
        for state in range(state_space_size):
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    pygame.quit()
                    return None, None
            
            v = V[state]
            q_values = np.zeros(action_space_size)
            agent_pos, enemy_pos, gate_pos = index_to_state(state, env.grid_size)
            
            for action in range(action_space_size):
                env.reset()
                env.agent_pos = agent_pos.copy()
                env.enemies_pos = [enemy_pos.copy()]
                env.gate_pos = gate_pos.copy()
                next_obs, reward, done, _ = env.step(action)
                
                render_scene(screen, env, "Value Iteration Training", env.cell_size, env.grid_size,
                           f"State: {state} | Action: {['Up','Down','Left','Right'][action]}", iteration+1)
                clock.tick(20)
                
                next_agent_pos = env.agent_pos
                next_enemy_pos = env.enemies_pos[0]
                next_state = state_to_index(next_agent_pos, next_enemy_pos, env.gate_pos, env.grid_size)
                q_values[action] = reward + gamma * V[next_state] * (not done)
            
            V[state] = np.max(q_values)
            policy[state] = np.argmax(q_values)
            delta = max(delta, abs(v - V[state]))
        
        if delta < theta:
            break
    
    return policy, V

def sarsa(env, screen, episodes=500, alpha=0.1, gamma=0.95, epsilon_start=1.0, epsilon_end=0.01):
    Q = defaultdict(lambda: np.zeros(action_space_size))
    epsilon = epsilon_start
    epsilon_decay = (epsilon_start - epsilon_end) / episodes
    clock = pygame.time.Clock()
    
    for episode in range(episodes):
        obs = env.reset()
        state = state_to_index(obs[:2], obs[2:4], obs[-2:], env.grid_size)
        
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        
        steps = 0
        while steps < env.max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    pygame.quit()
                    return None, None
            
            next_obs, reward, done, info = env.step(action)
            next_state = state_to_index(next_obs[:2], next_obs[2:4], next_obs[-2:], env.grid_size)
            
            if np.random.rand() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state])
            
            td_target = reward + gamma * Q[next_state][next_action] * (not done)
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error
            
            render_scene(screen, env, "SARSA Training", env.cell_size, env.grid_size,
                       f"Episode: {episode+1}/{episodes} | Reward: {reward:.2f}", episode+1)
            clock.tick(20)
            
            epsilon = max(epsilon_end, epsilon - epsilon_decay)
            
            state, action = next_state, next_action
            steps += 1
            
            if done:
                break
    
    return get_policy_from_q(Q), Q

def q_learning(env, screen, episodes=100, alpha=0.15, gamma=0.95, epsilon_start=1.0, epsilon_end=0.01):
    Q = defaultdict(lambda: np.zeros(action_space_size))
    epsilon = epsilon_start
    epsilon_decay = (epsilon_start - epsilon_end) / episodes
    clock = pygame.time.Clock()
    
    for episode in range(episodes):
        obs = env.reset()
        state = state_to_index(obs[:2], obs[2:4], obs[-2:], env.grid_size)
        
        steps = 0
        while steps < env.max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    pygame.quit()
                    return None, None
            
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            
            next_obs, reward, done, info = env.step(action)
            next_state = state_to_index(next_obs[:2], next_obs[2:4], next_obs[-2:], env.grid_size)
            
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state][best_next_action] * (not done)
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error
            
            render_scene(screen, env, "Q-Learning Training", env.cell_size, env.grid_size,
                       f"Episode: {episode+1}/{episodes} | Reward: {reward:.2f}", episode+1)
            clock.tick(20)
            
            epsilon = max(epsilon_end, epsilon - epsilon_decay)
            
            state = next_state
            steps += 1
            
            if done:
                break
    
    return get_policy_from_q(Q), Q

def policy_iteration(env, screen, gamma=0.99, theta=1e-4, max_iterations=2):
    policy = np.random.randint(0, action_space_size, state_space_size)
    V = np.zeros(state_space_size)
    clock = pygame.time.Clock()
    
    def evaluate_policy():
        for eval_iter in range(max_iterations):
            delta = 0
            for state in range(state_space_size):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                        return False
                
                v = V[state]
                agent_pos, enemy_pos, gate_pos = index_to_state(state, env.grid_size)
                env.reset()
                env.agent_pos = agent_pos.copy()
                env.enemies_pos = [enemy_pos.copy()]
                env.gate_pos = gate_pos.copy()
                action = policy[state]
                
                next_obs, reward, done, _ = env.step(action)
                next_agent_pos = env.agent_pos
                next_enemy_pos = env.enemies_pos[0]
                next_state = state_to_index(next_agent_pos, next_enemy_pos, env.gate_pos, env.grid_size)
                V[state] = reward + gamma * V[next_state] * (not done)
                delta = max(delta, abs(v - V[state]))
                
                render_scene(screen, env, "Policy Iteration (Evaluation)", env.cell_size, env.grid_size,
                           f"State: {state} | Value: {V[state]:.2f}", eval_iter+1)
                clock.tick(20)
            
            if delta < theta:
                break
        return True
    
    def improve_policy():
        policy_stable = True
        for state in range(state_space_size):
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    return False, False
            
            old_action = policy[state]
            q_values = np.zeros(action_space_size)
            agent_pos, enemy_pos, gate_pos = index_to_state(state, env.grid_size)
            
            for action in range(action_space_size):
                env.reset()
                env.agent_pos = agent_pos.copy()
                env.enemies_pos = [enemy_pos.copy()]
                env.gate_pos = gate_pos.copy()
                next_obs, reward, done, _ = env.step(action)
                
                next_agent_pos = env.agent_pos
                next_enemy_pos = env.enemies_pos[0]
                next_state = state_to_index(next_agent_pos, next_enemy_pos, env.gate_pos, env.grid_size)
                q_values[action] = reward + gamma * V[next_state] * (not done)
                
                render_scene(screen, env, "Policy Iteration (Improvement)", env.cell_size, env.grid_size,
                           f"State: {state} | Action: {['Up','Down','Left','Right'][action]}")
                clock.tick(20)
            
            policy[state] = np.argmax(q_values)
            if old_action != policy[state]:
                policy_stable = False
        return True, policy_stable
    
    for iter_num in range(max_iterations):
        if not evaluate_policy():
            return None, None
        cont, policy_stable = improve_policy()
        if not cont:
            return None, None
        if policy_stable:
            break
    
    return policy, V

def show_moves_history(screen, moves_history):
    moves_screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Movements History")
    
    moves_screen.fill((30, 30, 80))
    
    font_title = pygame.font.SysFont('Arial', 28, bold=True)
    font_header = pygame.font.SysFont('Arial', 22, bold=True)
    font_text = pygame.font.SysFont('Arial', 20)
    
    title = font_title.render("Movements History - All Steps", True, (255, 255, 255))
    moves_screen.blit(title, (200, 20))
    
    step_header = font_header.render("Step", True, (255, 255, 0))
    agent_header = font_header.render("Agent Position", True, (0, 255, 0))
    enemies_header = font_header.render("Enemies Positions", True, (255, 0, 0))
    
    moves_screen.blit(step_header, (50, 70))
    moves_screen.blit(agent_header, (150, 70))
    moves_screen.blit(enemies_header, (350, 70))
    
    scroll_y = 0
    max_display_moves = 15
    total_moves = len(moves_history)
    
    running = True
    while running:
        moves_screen.fill((30, 30, 80))
        moves_screen.blit(title, (200, 20))
        moves_screen.blit(step_header, (50, 70))
        moves_screen.blit(agent_header, (150, 70))
        moves_screen.blit(enemies_header, (350, 70))
        
        start_idx = max(0, min(scroll_y, total_moves - max_display_moves))
        for i, move in enumerate(moves_history[start_idx:start_idx + max_display_moves]):
            step_text = font_text.render(f"{move['step']}", True, (255, 255, 255))
            agent_text = font_text.render(f"({move['agent'][0]}, {move['agent'][1]})", True, (200, 255, 200))
            
            enemies_pos = ", ".join([f"({e[0]}, {e[1]})" for e in move['enemies']])
            enemies_text = font_text.render(enemies_pos, True, (255, 200, 200))
            
            y_pos = 100 + i * 25
            moves_screen.blit(step_text, (50, y_pos))
            moves_screen.blit(agent_text, (150, y_pos))
            moves_screen.blit(enemies_text, (350, y_pos))
        
        scroll_text = font_text.render("Use UP/DOWN to scroll, Press any key to continue...", True, (255, 255, 0))
        moves_screen.blit(scroll_text, (150, 550))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    scroll_y = max(0, scroll_y - 1)
                elif event.key == pygame.K_DOWN:
                    scroll_y = min(total_moves - 1, scroll_y + 1)
                else:
                    running = False
                    break
            if event.type == pygame.MOUSEBUTTONDOWN:
                running = False
                break
        
        pygame.time.wait(100)
    
    return True

def evaluate_policy_with_levels(env, policy, title, screen):
    clock = pygame.time.Clock()
    level = 1
    max_levels = 6
    running = True
    
    level_settings = [
        {'enemies': 1, 'obstacles': 2, 'smartness': 0.3},
        {'enemies': 2, 'obstacles': 3, 'smartness': 0.4},
        {'enemies': 3, 'obstacles': 4, 'smartness': 0.5},
        {'enemies': 4, 'obstacles': 5, 'smartness': 0.6},
        {'enemies': 5, 'obstacles': 6, 'smartness': 0.7},
        {'enemies': 6, 'obstacles': 7, 'smartness': 0.8}
    ]
    
    start_time = time.time()
    total_steps = 0
    total_reward = 0.0
    levels_completed = 0
    
    while running and level <= max_levels:
        level_start_time = time.time()
        settings = level_settings[level-1]
        env = CatchMeIfYouCanEnv(
            grid_size=5,
            num_enemies=settings['enemies'],
            num_obstacles=settings['obstacles'],
            max_steps=100 + (level-1)*20,
            enemy_smartness=settings['smartness']
        )
        env.cell_size = 80
        
        obs = env.reset()
        state = state_to_index(obs[:2], obs[2:4], obs[-2:], env.grid_size)
        done = False
        steps = 0
        
        moves_history = []
        
        while not done and steps < env.max_steps:
            if state in policy:
                action = policy[state]
            else:
                action = env.action_space.sample()
            
            next_obs, reward, done, info = env.step(action)
            next_state = state_to_index(next_obs[:2], next_obs[2:4], next_obs[-2:], env.grid_size)
            
            steps += 1
            total_steps += 1
            total_reward += reward
            
            moves_history.append({
                'step': steps,
                'agent': env.agent_pos.copy(),
                'enemies': [pos.copy() for pos in env.enemies_pos]
            })
            
            action_names = ["Up", "Down", "Left", "Right"]
            render_scene(
                screen, env, 
                f"{title} - Level {level}", 
                env.cell_size, env.grid_size,
                f"Steps: {steps}/{env.max_steps} | Action: {action_names[action]}"
            )
            clock.tick(10)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    running = False
                    break
            
            state = next_state
            
            if done or steps >= env.max_steps:
                level_time = time.time() - level_start_time
                
                if not show_moves_history(screen, moves_history):
                    running = False
                
                if info.get('reason') == 'reached_gate':
                    levels_completed += 1
                    display_message(screen, f"Level {level} Completed!", (0, 255, 0))
                    if level < max_levels:
                        level += 1
                    else:
                        display_message(screen, "All Levels Completed!", (0, 255, 0))
                elif info.get('reason') == 'caught':
                    display_message(screen, f"Game Over at Level {level}!", (255, 0, 0))
                    running = False
                elif steps >= env.max_steps:
                    display_message(screen, f"Level {level} Timeout!", (255, 255, 0))
                    running = False
                break
    
    duration = time.time() - start_time
    screen.fill((50, 50, 150))
    font = pygame.font.SysFont('Arial', 36)
    
    summary_lines = [
        "Final Summary",
        f"Algorithm: {title}",
        f"Total Duration: {duration:.1f} seconds",
        f"Total Steps: {total_steps}",
        f"Total Reward: {total_reward:.2f}",
        f"Levels Completed: {levels_completed}/{max_levels}"
    ]
    
    running = True
    while running:
        screen.fill((50, 50, 150))
        for i, line in enumerate(summary_lines):
            text = font.render(line, True, (255, 255, 255))
            text_rect = text.get_rect(center=(screen.get_width()//2, screen.get_height()//2 - 100 + i*40))
            screen.blit(text, text_rect)
        
        close_text = font.render("Press any key to close...", True, (255, 255, 0))
        close_rect = close_text.get_rect(center=(screen.get_width()//2, screen.get_height() - 50))
        screen.blit(close_text, close_rect)
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                running = False
        
        pygame.time.wait(100)
    
    env.close()

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((5 * 80, 5 * 80))
    pygame.display.set_caption("RL Training")
    
    print("Choose an algorithm:")
    print("1. Value Iteration")
    print("2. SARSA")
    print("3. Q-Learning")
    print("4. Policy Iteration")
    choice = input("Enter your choice (1-4): ")
    
    env = CatchMeIfYouCanEnv(grid_size=5, num_enemies=1, num_obstacles=2, max_steps=100, enemy_smartness=0.2)
    env.cell_size = 80
    
    policy = None
    algorithm_name = ""
    
    if choice == "1":
        algorithm_name = "Value Iteration"
        policy, _ = value_iteration(env, screen)
    elif choice == "2":
        algorithm_name = "SARSA"
        policy, _ = sarsa(env, screen)
    elif choice == "3":
        algorithm_name = "Q-Learning"
        policy, _ = q_learning(env, screen)
    elif choice == "4":
        algorithm_name = "Policy Iteration"
        policy, _ = policy_iteration(env, screen)
    else:
        print("Invalid choice!")
        env.close()
        pygame.quit()
        exit()
    
    if policy is not None:
        pygame.quit()
        pygame.init()
        screen = pygame.display.set_mode((5 * 80, 5 * 80))
        pygame.display.set_caption(f"{algorithm_name} - Evaluation")
        
        display_message(screen, f"Starting {algorithm_name} Evaluation", (255, 255, 255))
        evaluate_policy_with_levels(env, policy, algorithm_name, screen)
    
    pygame.quit()