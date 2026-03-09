import argparse
import numpy as np
from stable_baselines3 import PPO
from env import PaddleEnv

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes to evaluate")
    args = parser.parse_args()

    env = PaddleEnv()
    
    try:
        model = PPO.load("best_model", env=env)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    episode_rewards = []
    
    for _ in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
        episode_rewards.append(total_reward)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    # Critical: This "score: {val}" string is parsed by the experiment loop.
    print(f"score: {mean_reward:.4f}")
    print(f"std: {std_reward:.4f}")

if __name__ == "__main__":
    evaluate()
