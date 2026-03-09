#!/usr/bin/env python3
import os
import subprocess
import time

def run_experiment():
    print("Running custom paddle environment baseline experiment...")
    
    # Train the agent
    print("\n--- Training ---")
    start_time = time.time()
    train_result = subprocess.run(
        ["uv", "run", "python", "train.py", "--timesteps", "50000"],
        capture_output=True,
        text=True
    )
    train_time = time.time() - start_time
    
    if train_result.returncode != 0:
        print(f"Training failed:\n{train_result.stderr}")
        return
        
    print(f"Training completed in {train_time:.1f}s")
    
    # Evaluate the agent
    print("\n--- Evaluation ---")
    score_result = subprocess.run(
        ["uv", "run", "python", "score.py", "--episodes", "50"],
        capture_output=True,
        text=True
    )
    
    if score_result.returncode != 0:
        print(f"Evaluation failed:\n{score_result.stderr}")
        return
            
    # Parse and log results
    score = None
    std = None
    for line in score_result.stdout.split('\n'):
        if line.startswith("score:"):
            score = float(line.split(":")[1].strip())
        elif line.startswith("std:"):
            std = float(line.split(":")[1].strip())
            
    if score is not None:
        print(f"Final Score: {score:.4f} (std: {std:.4f})")
        
        # Log to results.tsv
        file_exists = os.path.isfile("results.tsv")
        with open("results.tsv", "a") as f:
            if not file_exists:
                f.write("timestamp\tscore\tstd\ttrain_time_s\n")
            f.write(f"{int(time.time())}\t{score}\t{std}\t{train_time:.1f}\n")
    else:
        print("Failed to parse score from output:\n", score_result.stdout)

if __name__ == "__main__":
    run_experiment()
