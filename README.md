The idea for this simple project springs from two key inspirations:
1. [***"The Thinking Game"***](https://youtu.be/d95J8yzvjbQ?t=603) — the documentary chronicling Demis Hassabis and Google DeepMind's incredible quest to build Artificial General Intelligence, which highlighted the incredible power of an autonomous, iterative AI research process.
2. [**Andrej Karpathy's `autoresearch` repository**](https://github.com/karpathy/autoresearch) — which introduced the practical concept of AI agents running research on single-GPU model training automatically.

In this project, I built a very tiny, custom, 2D continuous **Paddle Game** environment in Python. Like Karpathy in the autoresearch project, I wrapped it in an automated LLM-driven research loop that iteratively trains the neural network, scores it, checks the Git commit history, and explores new network architectures and reward shaping functions. 

## Repository Structure

The workflow consists of five primary files, creating a "Micro-Autoresearch" loop:

- **`env.py`** — A custom `gymnasium` environment. The paddle agent receives rewards for tracking and hitting the bouncing ball, and is penalized for letting the ball slip past.
- **`train.py`** — A lightweight CLI script to train a `stable-baselines3` PPO agent on the environment and save `best_model.zip`.
- **`score.py`** — An evaluation script that loads a trained model, tests it over 50 episodes, and outputs the average reward constraint to stdout.
- **`program.md`** — The markdown instructions for the LLM researcher. Limits the search space so the agent focuses on modifying learning rates, network topologies (`net_arch`), reward functions, and time limits.
- **`experiment_loop.py`** — The orchestrator script that manages the trials, tests changes against `results.tsv`, and executes the Git commits if the model improves.
- **`watch.py`** — A PyGame visualization script that lets you directly open a window and watch the trained model play!

## The Results

By pointing an LLM coding agent at `program.md`, it successfully optimized the standard gameplay.

- **Baseline Model:** Initially, the naïve PPO algorithm often jittered, taking poor shots and averaging a reward of around **~8 to 14 points** before missing the ball in under a minute of gameplay.
- **Final Model:** Over a sequence of 6 autonomous git commits, the system implemented a tighter reward curvature to prevent jittering, increased the neural network horizon, scaled learning rate (`5e-4`), and doubled the timesteps to 100k. The final model effectively solved the environment with near-perfect play, averaging a reward of **48.05**.

## GamePlay

## Why does the game stop when it's doing well?

If you watch the best model play, you'll see episodes end seemingly abruptly—even when the AI is perfectly tracking the ball! 

This is because we set a hard `max_steps = 1000` limit in the environment. Reinforcement learning agents need discrete "episodes" to learn effectively. If the AI becomes too good and we let it play infinitely, a single training episode might never finish, grinding our training loop to a halt! Therefore, the environment yells "Time's up!" after 1000 frames, rewarding the AI for surviving that long, and resets for the next round.

## Quick Start

**Requirements:** Any modern CPU (Python 3.10+), [uv](https://docs.astral.sh/uv/), and [pygame](https://www.pygame.org/).

```bash
# 1. Install dependencies
uv sync

# 2. Run a training session locally (~3 mins)
uv run python train.py --timesteps 100000

# 3. Watch the latest model play the game!
uv run python watch.py --model best_model

# 4. Compare it to the original, poorly-optimized baseline model
uv run python watch.py --model baseline_model
```

## Setup an Autonomous LLM Agent

Simply point an autonomous coding agent (like Claude, GPT-4, etc.) at your local repository and prompt it:

```text
Hi! Please look at program.md and let's kick off a new experiment loop via `python experiment_loop.py`.
```

The agent will modify `train.py` or `env.py`, invoke the experiment loop, log scores into `results.tsv`, and commit the code if it beats its high score.

## Further Reading
- **Andrej Karpathy's famous blog post *["Deep Reinforcement Learning: Pong from Pixels"](http://karpathy.github.io/2016/05/31/rl/)*:** A classic tutorial on training a neural network from scratch to play Pong using Policy Gradients (REINFORCE). 
