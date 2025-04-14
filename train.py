import gym
import pygame
from stable_baselines3 import PPO
from catch_me_env import CatchMeIfYouCanEnv

# Initialize environment
# env = CatchMeIfYouCanEnv()
env = CatchMeIfYouCanEnv(
    grid_size=15,        # Larger grid
    num_enemies=3,       # More enemies
    num_obstacles=10,    # More obstacles
    max_steps=200,       # Longer episodes
    enemy_smartness=0.7
)

# Train
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)
model.save("ppo_catch_me_if_you_can")

# Test
obs = env.reset()
running = True
while running:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    if done:
        obs = env.reset()

# Clean up
env.close()