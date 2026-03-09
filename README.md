# Autoresearch: MCMC Substitution Cipher Benchmark

This repository implements an autonomous AI research loop. It is an adaptation of the [https://github.com/karpathy/autoresearch](https://github.com/karpathy/autoresearch) project by Andrej Karpathy. Rather than using his codebase for training Large Language Models on GPUs, we apply his autonomous research technique to a CPU-bound algorithmic optimization problem: **MCMC Deciphering of Substitution Ciphers**.

## The Micro-Autoresearch Loop

By shifting from GPU-heavy language modeling to a combinatorial puzzle, we created a "Micro-Autoresearch" loop that runs in seconds on a standard CPU. This allows anyone to observe the core mechanics of an autonomous AI research org—hypothesizing, coding, evaluating, and iterating—at lightning speed.

The objective: Discover the most robust algorithmic improvements to a baseline Markov Chain Monte Carlo (MCMC) solver to maximize the successful deciphering rate of 50 randomly encrypted substitution ciphers within a strict 30-second time budget.

## How it works

The workflow consists of three primary files:

- **`data.py`** — Downloads a corpus (Alice in Wonderland) and acts as the generator for random 500-character substitution ciphers.
- **`evaluate.py`** — The unmodifiable benchmark harness. It generates 50 held-out ciphers, calls the solver, strictly limits execution to 30 seconds, and scores the result based on Character Accuracy.
- **`solver.py`** — The file edited by the autonomous agent. Currently contains a naive, under-optimized baseline MCMC deciphering algorithm. Everything here is fair game for the agent: proposal tuning, heating schedules, multiple chains, trigram scoring, etc.
- **`program.md`** — The markdown instructions for the AI researcher. Defines the metrics, the time budget, and the iteration loop rules.

By design, evaluation runs for a **fixed 30-second time budget**. The agent's goal is to maximize the final `score`:
`Score = Average Character Accuracy across 50 ciphers`

## Quick start

**Requirements:** Any modern CPU (Python 3.10+), [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies
uv sync

# 2. Test the data environment (downloads corpus)
uv run data.py

# 3. Manually run the baseline solver loop (~30 seconds)
uv run evaluate.py
```

## Setup an Agent

Simply point an autonomous coding agent (like Claude, GPT-4, etc.) at your local repository and prompt it:

```
Hi! Please look at program.md and let's kick off a new experiment.
```

The agent will read the rules, modify the MCMC Python code in `solver.py`, and begin the iterative heuristic search process.

## License

MIT
