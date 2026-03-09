You are an expert AI researcher tasked with optimizing a custom Reinforcement Learning environment and hyperparameter configuration to achieve the highest possible score.

The environment is a continuous 2D paddle game defined in `env.py`.
The training code is `train.py`, which uses Stable-Baselines3 PPO.
The scoring code is `score.py`, which outputs a `score:` (average episode reward over 50 episodes).

Your goal is to maximize the `score`. A perfect episode without missing the ball can range from 1 to 50+ depending on the episode max steps. A baseline agent might score around -1 to 5.

You can modify ONLY:
1. `env.py` (Reward shaping, observation space scaling, episode length `self.max_steps`)
2. `train.py` (PPO Hyperparameters: `--lr`, `--net_arch`, `--n_envs` or modify PPO args like `gamma`, `ent_coef` directly in train.py).

Rules:
1. Do not change the fundamental action space or core physics.
2. In `env.py`, focus primarily on tweaking the reward structure (e.g. `reward += ...`) that guides the agent to hit the ball more consistently.
3. In `train.py`, you can change the default hyperparameters in the `parser.add_argument` definitions or the `PPO(...)` call. Do NOT change the commandline arguments accepted by the file, just their defaults or hardcoded overrides.
4. Keep training time short. Do not exceed 200,000 timesteps total.
5. Provide a bash script containing `python train.py` and `python score.py` to evaluate your changes.

Example changes:
- In `env.py`: Add a reward proportional to `abs(self.paddle_y - self.ball_y)` to encourage alignment.
- In `train.py`: Set `--lr 1e-3` or `--net_arch 128 128`.

Start your process by running your baseline and interpreting the score.
