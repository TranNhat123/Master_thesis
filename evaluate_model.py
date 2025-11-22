import argparse
import os
import sys
import time

sys.path.append(r'C:\Users\Admin\Downloads\PyBullet_ver_2')

from envs.pybullet_robot6_env import PyBullet6DOFEnv
from stable_baselines3 import SAC


def evaluate(model_path, episodes=5, render=True, behavior_mode='rl'):
    env = PyBullet6DOFEnv(render=render, use_robot5_obstacles=True, use_nonlinear_ctrl=False, use_rl_ctrl=True, behavior_mode=behavior_mode)
    # Load model and attach to the same env to ensure action/obs spaces match
    # Sometimes saved model space metadata doesn't match local env (wrappers). Provide custom_objects
    custom_objs = {"observation_space": env.observation_space, "action_space": env.action_space}
    model = SAC.load(model_path, env=env, custom_objects=custom_objs)

    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while not done and steps < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            if render:
                time.sleep(1.0 / 120.0)
        print(f"Episode {ep+1}: reward={total_reward:.2f}, steps={steps}, info={info}")

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to saved SAC model (without .zip)')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--behavior-mode', type=str, default='rl', choices=['rl','nonlinear','blend','none'])
    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(model_path + '.zip'):
        raise FileNotFoundError(f"Model file not found: {model_path}.zip")

    evaluate(model_path, episodes=args.episodes, render=args.render, behavior_mode=args.behavior_mode)
