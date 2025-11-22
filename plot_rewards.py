import os
import csv
import matplotlib.pyplot as plt
import numpy as np

"""
Simple plotting utility for SAC training rewards.
Reads a CSV file written by training (`runs/<run_name>/training.csv`) with columns:
  episode, global_step, episode_reward

Usage:
  python plot_rewards.py --csv runs/sac_robot6_YYYYMMDD-HHMMSS/training.csv
or
  python plot_rewards.py  # tries to find the most recent runs/*/training.csv

Saves `reward_plot.png` next to the CSV file.
"""

import argparse
from glob import glob
from pathlib import Path


def find_latest_csv(runs_dir="runs"):
    runs = sorted(glob(os.path.join(runs_dir, "*")), key=os.path.getmtime)
    for run in reversed(runs):
        csv_path = os.path.join(run, "training.csv")
        if os.path.exists(csv_path):
            return csv_path
    return None


def load_csv(csv_path):
    episodes = []
    rewards = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            ep = int(row[0])
            rew = float(row[2])
            episodes.append(ep)
            rewards.append(rew)
    return np.array(episodes), np.array(rewards)


def smooth(x, window=10):
    if window <= 1:
        return x
    return np.convolve(x, np.ones(window) / window, mode='valid')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", default=None, help="Path to training CSV")
    parser.add_argument("--smooth", type=int, default=10, help="Smoothing window size")
    args = parser.parse_args()

    csv_path = args.csv
    if csv_path is None:
        csv_path = find_latest_csv()
        if csv_path is None:
            print("No training CSV found in 'runs/'")
            return
        print(f"Using latest csv: {csv_path}")

    episodes, rewards = load_csv(csv_path)
    if episodes.size == 0:
        print("No data in csv")
        return

    sm = smooth(rewards, window=args.smooth)
    sm_eps = episodes[: len(sm) ]

    out_png = os.path.join(os.path.dirname(csv_path), "reward_plot.png")

    plt.figure(figsize=(10,6))
    plt.plot(episodes, rewards, alpha=0.3, label='episode reward')
    plt.plot(sm_eps, sm, label=f'smoothed (w={args.smooth})', color='C1')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('Training Reward Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)
    print(f"Saved plot to {out_png}")

if __name__ == '__main__':
    main()
