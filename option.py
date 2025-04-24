import numpy as np
import gym
import pygame
import time  # Added to track time
from catch_me_env import CatchMeIfYouCanEnv
from collections import defaultdict

# Step 1: Helper Functions
def state_to_index(agent_pos, enemy_pos, grid_size):
    agent_idx = agent_pos[0] * grid_size + agent_pos[1]
    enemy_idx = enemy_pos[0] * grid_size + enemy_pos[1]
    return agent_idx * state_space_size + enemy_idx

def index_to_state(state_idx, grid_size):
    enemy_idx = state_idx % state_space_size
    agent_idx = state_idx // state_space_size
    agent_pos = [agent_idx // grid_size, agent_idx % grid_size]
    enemy_pos = [enemy_idx // grid_size, enemy_idx % grid_size]
    return agent_pos, enemy_pos

def get_policy_from_q(q_table):
    policy = {}
    for state in range(state_space_size * state_space_size):
        policy[state] = np.argmax(q_table[state])
    return policy

# Step 2: Rendering Function with Enhanced Visuals
def render_scene(screen, env, title, cell_size, grid_size, info_text, episode_or_iter=None):
    # Gradient background
    for y in range(screen.get_height()):
        color = (50, 50, 150 + int(y * (105 / screen.get_height())))
        pygame.draw.line(screen, color, (0, y), (screen.get_width(), y))
    
    # Draw grid with lighter lines
    for i in range(grid_size + 1):
        pygame.draw.line(
            screen, (200, 200, 200),
            (i * cell_size, 0),
            (i * cell_size, grid_size * cell_size), 1
        )
        pygame.draw.line(
            screen, (200, 200, 200),
            (0, i * cell_size),
            (grid_size * cell_size, i * cell_size), 1
        )
    
    # Draw obstacles with shadow
    for obstacle in env.obstacles_pos:
        pygame.draw.rect(
            screen, (100, 100, 100),
            (obstacle[0] * cell_size + 5, obstacle[1] * cell_size + 5, cell_size - 5, cell_size - 5)
        )
        pygame.draw.rect(
            screen, (150, 150, 150),
            (obstacle[0] * cell_size, obstacle[1] * cell_size, cell_size - 5, cell_size - 5)
        )
    
    # Draw enemies (red squares with shadow)
    for enemy in env.enemies_pos:
        pygame.draw.rect(
            screen, (100, 0, 0),
            (enemy[0] * cell_size + 5, enemy[1] * cell_size + 5, cell_size - 10, cell_size - 10)
        )
        pygame.draw.rect(
            screen, (255, 0, 0),
            (enemy[0] * cell_size, enemy[1] * cell_size, cell_size - 10, cell_size - 10)
        )
    
    # Draw agent (green circle with shadow)
    pygame.draw.circle(
        screen, (0, 100, 0),
        (int(env.agent_pos[0] * cell_size + cell_size // 2 + 5), int(env.agent_pos[1] * cell_size + cell_size // 2 + 5)),
        cell_size // 2 - 5
    )
    pygame.draw.circle(
        screen, (0, 255, 0),
        (int(env.agent_pos[0] * cell_size + cell_size // 2), int(env.agent_pos[1] * cell_size + cell_size // 2)),
        cell_size // 2 - 5
    )
    
    # Draw title and info
    font = pygame.font.SysFont(None, 30)
    title_text = font.render(f"{title}", True, (255, 255, 255))
    info_text_rendered = font.render(info_text, True, (255, 255, 255))
    screen.blit(title_text, (10, 10))
    screen.blit(info_text_rendered, (10, 40))
    
    if episode_or_iter is not None:
        episode_text = font.render(f"Episode/Iter: {episode_or_iter}", True, (255, 255, 255))
        screen.blit(episode_text, (10, 70))
    
    pygame.display.flip()

# Step 3: Display Message (Single Line)
def display_message(screen, message, color=(255, 255, 255)):
    screen.fill((50, 50, 150))
    font = pygame.font.SysFont(None, 48)
    text = font.render(message, True, color)
    text_rect = text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
    screen.blit(text, text_rect)
    pygame.display.flip()
    pygame.time.wait(2000)

# Step 3.1: Display Final Report (Multi-Line)
def display_final_report(screen, duration, total_steps, total_reward):
    screen.fill((50, 50, 150))
    font = pygame.font.SysFont(None, 36)
    lines = [
        "Game Summary",
        f"Duration: {duration:.1f} seconds",
        f"Total Steps: {total_steps}",
        f"Total Reward: {total_reward:.2f}"
    ]
    for i, line in enumerate(lines):
        text = font.render(line, True, (255, 255, 255))
        text_rect = text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2 - 50 + i * 40))
        screen.blit(text, text_rect)
    pygame.display.flip()
    pygame.time.wait(5000)  # Display for 5 seconds

# Step 4: Define Algorithms with Visual Training

# 4.1 Value Iteration
def value_iteration(env, screen, gamma=0.99, theta=1e-6, max_iterations=100):
    V = np.zeros(state_space_size * state_space_size)
    policy = np.zeros(state_space_size * state_space_size, dtype=int)
    clock = pygame.time.Clock()
    
    for iteration in range(max_iterations):
        delta = 0
        for state in range(state_space_size * state_space_size):
            v = V[state]
            q_values = np.zeros(action_space_size)
            agent_pos, enemy_pos = index_to_state(state, env.grid_size)
            
            for action in range(action_space_size):
                env.reset()
                env.agent_pos = agent_pos.copy()
                env.enemies_pos = [enemy_pos.copy()]
                next_obs, reward, done, _ = env.step(action)
                next_agent_pos = env.agent_pos
                next_enemy_pos = env.enemies_pos[0]
                next_state = state_to_index(next_agent_pos, next_enemy_pos, env.grid_size)
                q_values[action] = reward + gamma * V[next_state] * (not done)
                
                action_names = ["Up", "Down", "Left", "Right"]
                render_scene(screen, env, "Training Value Iteration", env.cell_size, env.grid_size,
                             f"Action: {action_names[action]} | Reward: {reward}", iteration + 1)
                clock.tick(60)
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                        pygame.quit()
                        return None, None
            
            V[state] = np.max(q_values)
            policy[state] = np.argmax(q_values)
            delta = max(delta, abs(v - V[state]))
        
        if delta < theta:
            break
    
    return policy, V

# 4.2 SARSA
def sarsa(env, screen, episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: np.zeros(action_space_size))
    clock = pygame.time.Clock()
    
    for episode in range(episodes):
        obs = env.reset()
        agent_pos = obs[:2].astype(int)
        enemy_pos = obs[2:4].astype(int)
        state = state_to_index(agent_pos, enemy_pos, env.grid_size)
        
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        
        while True:
            next_obs, reward, done, _ = env.step(action)
            next_agent_pos = next_obs[:2].astype(int)
            next_enemy_pos = next_obs[2:4].astype(int)
            next_state = state_to_index(next_agent_pos, next_enemy_pos, env.grid_size)
            
            if np.random.rand() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state])
            
            Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] * (not done) - Q[state][action])
            
            action_names = ["Up", "Down", "Left", "Right"]
            render_scene(screen, env, "Training SARSA", env.cell_size, env.grid_size,
                         f"Action: {action_names[action]} | Reward: {reward}", episode + 1)
            clock.tick(60)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    pygame.quit()
                    return None, None
            
            state = next_state
            action = next_action
            
            if done:
                break
    
    return get_policy_from_q(Q), Q

# 4.3 Q-Learning
def q_learning(env, screen, episodes=200, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: np.zeros(action_space_size))
    clock = pygame.time.Clock()
    
    for episode in range(episodes):
        obs = env.reset()
        agent_pos = obs[:2].astype(int)
        enemy_pos = obs[2:4].astype(int)
        state = state_to_index(agent_pos, enemy_pos, env.grid_size)
        
        while True:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            
            next_obs, reward, done, _ = env.step(action)
            next_agent_pos = next_obs[:2].astype(int)
            next_enemy_pos = next_obs[2:4].astype(int)
            next_state = state_to_index(next_agent_pos, next_enemy_pos, env.grid_size)
            
            Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) * (not done) - Q[state][action])
            
            action_names = ["Up", "Down", "Left", "Right"]
            render_scene(screen, env, "Training Q-Learning", env.cell_size, env.grid_size,
                         f"Action: {action_names[action]} | Reward: {reward}", episode + 1)
            clock.tick(60)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    pygame.quit()
                    return None, None
            
            state = next_state
            
            if done:
                break
    
    return get_policy_from_q(Q), Q

# 4.4 Policy Iteration
def policy_iteration(env, screen, gamma=0.99, theta=1e-6, max_iterations=100):
    policy = np.random.randint(0, action_space_size, state_space_size * state_space_size)
    V = np.zeros(state_space_size * state_space_size)
    clock = pygame.time.Clock()
    
    def evaluate_policy():
        for iteration in range(max_iterations):
            delta = 0
            for state in range(state_space_size * state_space_size):
                v = V[state]
                agent_pos, enemy_pos = index_to_state(state, env.grid_size)
                env.reset()
                env.agent_pos = agent_pos.copy()
                env.enemies_pos = [enemy_pos.copy()]
                action = policy[state]
                
                next_obs, reward, done, _ = env.step(action)
                next_agent_pos = env.agent_pos
                next_enemy_pos = env.enemies_pos[0]
                next_state = state_to_index(next_agent_pos, next_enemy_pos, env.grid_size)
                
                V[state] = reward + gamma * V[next_state] * (not done)
                delta = max(delta, abs(v - V[state]))
                
                action_names = ["Up", "Down", "Left", "Right"]
                render_scene(screen, env, "Training Policy Iteration (Eval)", env.cell_size, env.grid_size,
                             f"Action: {action_names[action]} | Reward: {reward}", iteration + 1)
                clock.tick(60)
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                        return False
            
            if delta < theta:
                break
        return True
    
    def improve_policy():
        policy_stable = True
        for state in range(state_space_size * state_space_size):
            old_action = policy[state]
            q_values = np.zeros(action_space_size)
            agent_pos, enemy_pos = index_to_state(state, env.grid_size)
            
            for action in range(action_space_size):
                env.reset()
                env.agent_pos = agent_pos.copy()
                env.enemies_pos = [enemy_pos.copy()]
                next_obs, reward, done, _ = env.step(action)
                next_agent_pos = env.agent_pos
                next_enemy_pos = env.enemies_pos[0]
                next_state = state_to_index(next_agent_pos, next_enemy_pos, env.grid_size)
                
                q_values[action] = reward + gamma * V[next_state] * (not done)
                
                action_names = ["Up", "Down", "Left", "Right"]
                render_scene(screen, env, "Training Policy Iteration (Improve)", env.cell_size, env.grid_size,
                             f"Action: {action_names[action]} | Reward: {reward}")
                clock.tick(60)
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                        return False, False
            
            policy[state] = np.argmax(q_values)
            if old_action != policy[state]:
                policy_stable = False
        return True, policy_stable
    
    for iteration in range(max_iterations):
        if not evaluate_policy():
            return None, None
        cont, policy_stable = improve_policy()
        if not cont:
            return None, None
        if policy_stable:
            break
    
    return policy, V

# Step 5: Evaluate Policy with Levels (Limited to 4 Levels with Stats Tracking)
def evaluate_policy_with_levels(env, policy, title, screen):
    clock = pygame.time.Clock()
    level = 1
    max_levels = 4
    base_max_steps = 100
    base_num_enemies = 1
    running = True
    
    # Initialize stats tracking
    start_time = time.time()  # Start time for duration
    total_steps = 0  # Total steps across all levels
    total_reward = 0.0  # Total reward across all levels
    
    while running and level <= max_levels:
        # Update environment parameters based on level
        max_steps = base_max_steps + (level - 1) * 20
        num_enemies = base_num_enemies + (level - 1)
        
        # Recreate environment with new parameters
        env = CatchMeIfYouCanEnv(grid_size=5, num_enemies=num_enemies, num_obstacles=2, max_steps=max_steps, enemy_smartness=0.5)
        env.cell_size = 80
        
        obs = env.reset()
        agent_pos = obs[:2].astype(int)
        enemy_pos = obs[2:4].astype(int)
        state = state_to_index(agent_pos, enemy_pos, env.grid_size)
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            action = policy[state]
            obs, reward, done, _ = env.step(action)
            agent_pos = obs[:2].astype(int)
            enemy_pos = obs[2:4].astype(int)
            state = state_to_index(agent_pos, enemy_pos, env.grid_size)
            steps += 1
            total_steps += 1  # Increment total steps
            total_reward += reward  # Accumulate reward
            
            action_names = ["Up", "Down", "Left", "Right"]
            render_scene(screen, env, f"{title} | Level: {level}", env.cell_size, env.grid_size,
                         f"Steps: {steps}/{max_steps} | Enemies: {num_enemies} | Action: {action_names[action]}")
            clock.tick(60)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    running = False
                    break
        
        if not running:
            break
        
        if done and reward == -1:
            display_message(screen, f"Game Over at Level {level}!", (255, 0, 0))
            running = False
        else:
            if level == max_levels:
                display_message(screen, "Congratulations! You completed all levels!", (0, 255, 0))
                running = False
            else:
                display_message(screen, f"Level {level} Cleared!", (0, 255, 0))
                level += 1
    
    # Calculate duration and display final report
    duration = time.time() - start_time  # Duration in seconds
    display_final_report(screen, duration, total_steps, total_reward)
    
    env.close()

# Step 6: Main Execution
# Discretize state space
state_space_size = 5 * 5  # grid_size = 5
action_space_size = 4  # Assuming 4 actions (up, down, left, right)

# Initialize Pygame with a larger window
pygame.init()
window_size = 5 * 80  # grid_size * cell_size
screen = pygame.display.set_mode((window_size, window_size))

# Let user choose the algorithm
print("Choose an algorithm to train and play:")
print("1. Value Iteration")
print("2. SARSA")
print("3. Q-Learning")
print("4. Policy Iteration")
choice = input("Enter the number (1-4): ")

# Initialize environment for training
env = CatchMeIfYouCanEnv(grid_size=5, num_enemies=1, num_obstacles=2, max_steps=50, enemy_smartness=0.2)
env.cell_size = 80

# Train the selected algorithm with visual feedback
policy = None
algorithm_name = ""

if choice == "1":
    algorithm_name = "Value Iteration"
    pygame.display.set_caption("Training Value Iteration")
    policy, _ = value_iteration(env, screen)
elif choice == "2":
    algorithm_name = "SARSA"
    pygame.display.set_caption("Training SARSA")
    policy, _ = sarsa(env, screen)
elif choice == "3":
    algorithm_name = "Q-Learning"
    pygame.display.set_caption("Training Q-Learning")
    policy, _ = q_learning(env, screen)
elif choice == "4":
    algorithm_name = "Policy Iteration"
    pygame.display.set_caption("Training Policy Iteration")
    policy, _ = policy_iteration(env, screen)
else:
    print("Invalid choice! Exiting...")
    env.close()
    pygame.quit()
    exit()

# Evaluate the policy with levels if training was successful
if policy is not None:
    pygame.display.set_caption(f"{algorithm_name} Game")
    display_message(screen, f"Starting {algorithm_name} Game!", (255, 255, 255))
    evaluate_policy_with_levels(env, policy, algorithm_name, screen)
else:
    env.close()

pygame.quit()