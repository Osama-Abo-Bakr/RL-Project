
import numpy as np
import gym
from catch_me_env import CatchMeIfYouCanEnv
from collections import defaultdict

# Discretize state space
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

# Step 2: Initialize Environments
# Initialize four environments with reduced max_steps
env_vi = CatchMeIfYouCanEnv(grid_size=5, num_enemies=1, num_obstacles=2, max_steps=50, enemy_smartness=0.5)
env_sarsa = CatchMeIfYouCanEnv(grid_size=5, num_enemies=1, num_obstacles=2, max_steps=50, enemy_smartness=0.5)
env_ql = CatchMeIfYouCanEnv(grid_size=5, num_enemies=1, num_obstacles=2, max_steps=50, enemy_smartness=0.5)
env_pi = CatchMeIfYouCanEnv(grid_size=5, num_enemies=1, num_obstacles=2, max_steps=50, enemy_smartness=0.5)

# Discretize state space
state_space_size = env_vi.grid_size * env_vi.grid_size
action_space_size = env_vi.action_space.n

# Step 3: Define Algorithms

# 3.1 Value Iteration
def value_iteration(env, gamma=0.99, theta=1e-6, max_iterations=50):
    V = np.zeros(state_space_size * state_space_size)
    policy = np.zeros(state_space_size * state_space_size, dtype=int)
    
    for _ in range(max_iterations):
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
            
            V[state] = np.max(q_values)
            policy[state] = np.argmax(q_values)
            delta = max(delta, abs(v - V[state]))
        
        if delta < theta:
            break
    
    return policy, V

print("Training Value Iteration...")
vi_policy, _ = value_iteration(env_vi)

# 3.2 SARSA
def sarsa(env, episodes=200, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: np.zeros(action_space_size))
    
    for _ in range(episodes):
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
            
            state = next_state
            action = next_action
            
            if done:
                break
    
    return get_policy_from_q(Q), Q

print("Training SARSA...")
sarsa_policy, _ = sarsa(env_sarsa)

# 3.3 Q-Learning
def q_learning(env, episodes=200, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: np.zeros(action_space_size))
    
    for _ in range(episodes):
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
            
            state = next_state
            
            if done:
                break
    
    return get_policy_from_q(Q), Q

print("Training Q-Learning...")
ql_policy, _ = q_learning(env_ql)

# 3.4 Policy Iteration
def policy_iteration(env, gamma=0.99, theta=1e-6, max_iterations=50):
    policy = np.random.randint(0, action_space_size, state_space_size * state_space_size)
    V = np.zeros(state_space_size * state_space_size)
    
    def evaluate_policy():
        for _ in range(max_iterations):
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
            
            if delta < theta:
                break
    
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
            
            policy[state] = np.argmax(q_values)
            if old_action != policy[state]:
                policy_stable = False
        
        return policy_stable
    
    for _ in range(max_iterations):
        evaluate_policy()
        if improve_policy():
            break
    
    return policy, V

print("Training Policy Iteration...")
pi_policy, _ = policy_iteration(env_pi)

# Step 4: Evaluate Policies and Show Results
# Evaluate a single policy and print results
def evaluate_single_policy(env, policy, title, episodes=1):
    total_rewards = []
    total_steps = []
    episode_count = 0
    obs = env.reset()
    agent_pos = obs[:2].astype(int)
    enemy_pos = obs[2:4].astype(int)
    state = state_to_index(agent_pos, enemy_pos, env.grid_size)
    done = False
    
    print(f"\nEvaluating {title}...")
    while episode_count < episodes:
        if not done:
            action = policy[state]
            obs, reward, done, _ = env.step(action)
            agent_pos = obs[:2].astype(int)
            enemy_pos = obs[2:4].astype(int)
            state = state_to_index(agent_pos, enemy_pos, env.grid_size)
            
            total_rewards.append(reward)
            total_steps.append(1)
            
            # Print step information
            step_num = len(total_steps) % env.max_steps if len(total_steps) % env.max_steps != 0 else env.max_steps
            print(f"Episode {episode_count + 1} | Step {step_num}: Agent at {agent_pos}, Enemy at {enemy_pos}, Reward: {reward}")
        
        if done:
            episode_count += 1
            print(f"Episode {episode_count} finished: Steps = {len(total_steps)}, Total Reward = {sum(total_rewards)}")
            if episode_count < episodes:
                obs = env.reset()
                agent_pos = obs[:2].astype(int)
                enemy_pos = obs[2:4].astype(int)
                state = state_to_index(agent_pos, enemy_pos, env.grid_size)
                done = False
    
    mean_reward = np.sum(total_rewards) / episodes if total_rewards else 0
    mean_steps = np.sum(total_steps) / episodes if total_steps else 0
    return mean_reward, mean_steps

# Print comparison results
def show_comparison_results(mean_rewards, mean_steps):
    algorithms = ['Value Iteration', 'SARSA', 'Q-Learning', 'Policy Iteration']
    
    # Find best algorithms
    best_reward_idx = np.argmax(mean_rewards)
    best_steps_idx = np.argmax(mean_steps)
    
    # Print comparison
    print("\nComparison Results:")
    print(f"Most Rewards: {algorithms[best_reward_idx]} ({mean_rewards[best_reward_idx]:.4f})")
    print(f"Most Steps (Longest Survival): {algorithms[best_steps_idx]} ({mean_steps[best_steps_idx]:.4f})")
    print("\nDetailed Results:")
    for i, algo in enumerate(algorithms):
        print(f"{algo}: Rewards={mean_rewards[i]:.4f}, Steps={mean_steps[i]:.4f}")

# Evaluate each policy and print results
print("Evaluating policies...")
mean_rewards = []
mean_steps = []

for env, policy, title in [
    (env_vi, vi_policy, "Value Iteration"),
    (env_sarsa, sarsa_policy, "SARSA"),
    (env_ql, ql_policy, "Q-Learning"),
    (env_pi, pi_policy, "Policy Iteration")
]:
    reward, steps = evaluate_single_policy(env, policy, title, episodes=1)
    mean_rewards.append(reward)
    mean_steps.append(steps)

print("Showing comparison results...")
show_comparison_results(mean_rewards, mean_steps)

# Clean up
env_vi.close()
env_sarsa.close()
env_ql.close()
env_pi.close()