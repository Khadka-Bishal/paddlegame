import argparse
import time
from stable_baselines3 import PPO
from env import PaddleEnv
import pygame

def watch(model_path):
    print(f"Loading trained model: {model_path}.zip...")
    # Initialize env explicitly with human render mode
    env = PaddleEnv(render_mode="human")
    
    try:
        model = PPO.load(model_path, env=env)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Playing 5 evaluation episodes...")
    for episode in range(5):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
            # The render loop limits FPS to 60 inside env.py
            env.render()
            
        print(f"Episode {episode + 1} finished with reward: {total_reward:.2f}")
        time.sleep(1) # Pause between rounds
        
    pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="best_model", help="Name of the model to load (e.g. best_model, baseline_model)")
    args = parser.parse_args()
    watch(args.model)
