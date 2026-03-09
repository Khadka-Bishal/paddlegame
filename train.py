import argparse
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env import PaddleEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total timesteps to train")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--net_arch", type=int, nargs="+", default=[128, 128], help="Network architecture layers")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cpu, cuda, mps, auto)")
    args = parser.parse_args()

    # Determine optimal device if auto
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device
        
    print(f"Using device: {device}")

    # Set up vector environment
    vec_env = make_vec_env(PaddleEnv, n_envs=args.n_envs)

    policy_kwargs = dict(net_arch=args.net_arch)

    model = PPO(
        "MlpPolicy", 
        vec_env, 
        learning_rate=args.lr,
        verbose=1,
        policy_kwargs=policy_kwargs,
        device=device
    )

    print(f"Starting training for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps)

    model.save("best_model")
    print("Training finished. Saved to best_model.zip")

if __name__ == "__main__":
    main()
