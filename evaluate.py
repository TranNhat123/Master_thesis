import os
import argparse
import torch
import numpy as np
from Train_Robot6 import MLP, GaussianPolicy
from Robot6_SAC import Robot6SacEnv

import pybullet as p


def find_latest_best_ckpt(ckpt_root="checkpoints"):
    import glob
    candidates = glob.glob(os.path.join(ckpt_root, "*", "best.pth"))
    if not candidates:
        candidates = glob.glob(os.path.join(ckpt_root, "*", "ckpt_ep_*.pth"))
    if not candidates:
        return None
    candidates = sorted(candidates, key=os.path.getmtime)
    return candidates[-1]


def load_policy_from_ckpt(ckpt_path, obs_dim, act_dim, device):
    policy = GaussianPolicy(obs_dim, act_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'policy_state' in ckpt:
        policy.load_state_dict(ckpt['policy_state'])
    else:
        raise ValueError('No policy_state in checkpoint')
    return policy


def run_eval(policy, env, episodes=10, render=False, record=False, out_dir=None):
    device = next(policy.parameters()).device
    rewards = []

    # optional recording via pybullet state logging
    logger_id = None
    if record and render and out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0

        if record and render and out_dir is not None:
            video_path = os.path.join(out_dir, f"eval_ep_{ep}.mp4")
            try:
                logger_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_path)
            except Exception as e:
                print(f"Video logging failed: {e}")
                logger_id = None

        while not (done or truncated):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                mu, _ = policy.forward(obs_tensor)
                act = torch.tanh(mu)
            a = act.cpu().numpy()[0]
            obs, r, done, truncated, info = env.step(a)
            ep_reward += r

        if logger_id is not None:
            try:
                p.stopStateLogging(logger_id)
            except Exception:
                pass

        rewards.append(ep_reward)
        print(f"Eval episode {ep}: reward={ep_reward:.2f}")

    return rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint (.pth). If not provided, finds latest best.pth in checkpoints/.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action='store_true')
    parser.add_argument("--record", action='store_true', help='Record video (requires --render)')
    parser.add_argument("--out", type=str, default=None, help='Output folder for recordings/logs')
    args = parser.parse_args()

    ckpt_path = args.ckpt
    if ckpt_path is None:
        ckpt_path = find_latest_best_ckpt()
        if ckpt_path is None:
            raise FileNotFoundError('No checkpoint found in checkpoints/')
        print(f"Using checkpoint: {ckpt_path}")

    env = Robot6SacEnv(use_gui=args.render)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = load_policy_from_ckpt(ckpt_path, obs_dim, act_dim, device)

    rewards = run_eval(policy, env, episodes=args.episodes, render=args.render, record=args.record, out_dir=args.out)

    if args.out is not None:
        os.makedirs(args.out, exist_ok=True)
        out_csv = os.path.join(args.out, 'eval_rewards.csv')
        import csv
        with open(out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['episode','reward'])
            for i, r in enumerate(rewards):
                w.writerow([i, float(r)])
        print(f"Saved eval rewards to {out_csv}")

    print(f"Mean reward over {len(rewards)} episodes: {np.mean(rewards):.2f}")
    env.close()

if __name__ == '__main__':
    main()
