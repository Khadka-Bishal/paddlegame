# AI Agent Meta-Program for RL Optimization

You are an expert AI researcher. Your goal is to maximize the score of a Reinforcement Learning agent playing a custom 2D paddle game.

The `score.py` script returns the average reward over 50 episodes. A baseline agent might score around -1 to 5; your goal is to find configurations that consistently score much higher.

## Workflow

1. **Review Baseline:** Run `uv run python experiment_loop.py` to see the current score from `train.py` and `score.py`.
2. **Brainstorm Changes:** Consider changes to hyper-parameters or reward functions.
3. **Implement Changes:** Edit `env.py` (for reward shaping) or `train.py` (for learning rate, network architecture, etc.).
4. **Evaluate:** Rerun the experiment loop (`uv run python experiment_loop.py`).
5. **Version Control:**
    - If the score INCREASES: Run `git commit -am "improved score to X: [brief description of change]"`
    - If the score DECREASES or the code breaks: Run `git reset --hard HEAD` to revert your changes.
6. **Iterate:** Repeat this process until you achieve a consistently high score or exhaust reasonable ideas.

## Search Space

You may modify the following:

- **Environment (`env.py`):**
    - **Reward Shaping:** Add auxiliary rewards (e.g., `reward += 0.01 * (1.0 - abs(self.paddle_y - self.ball_y))`) to guide the agent.
    - **Episode Length:** Modify `self.max_steps`.
    - Do NOT change the state vector size, action space, or core physics.

- **Training (`train.py`):**
    - **Learning Rate (`--lr`):** Try values like `1e-3`, `3e-4`, `5e-4`.
    - **Network Architecture (`--net_arch`):** Try wider or deeper networks (e.g., `[128, 128]`, `[64, 64, 64]`).
    - **Timesteps (`--timesteps`):** You may increase up to a maximum of 200,000 if the agent needs more time to converge.

Do not edit `score.py` or `experiment_loop.py` unless absolutely necessary to fix a bug. Your job is exclusively to optimize the agent and environment interaction.
