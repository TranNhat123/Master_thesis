"""Train SAC on the 6-DOF PyBullet environment.

Usage (PowerShell):
  python .\train_sac_robot6.py --timesteps 200000 --logdir .\logs
"""
import argparse
import os

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

from envs.pybullet_robot6_env import PyBullet6DOFEnv


def make_env_fn(render=False, max_vel=1.0, use_robot5_obstacles=False, use_nonlinear_ctrl=True, use_rl_ctrl=True, blend_alpha=0.5):
    def _init():
        env = PyBullet6DOFEnv(render=render, max_vel=max_vel, use_robot5_obstacles=use_robot5_obstacles,
                               use_nonlinear_ctrl=use_nonlinear_ctrl, use_rl_ctrl=use_rl_ctrl, blend_alpha=blend_alpha)
        return Monitor(env)

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--logdir", type=str, default="./logs")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--max-vel", type=float, default=1.0)
    parser.add_argument("--use-robot5-obstacles", action="store_true", help="Use Robot5 marker points as dynamic obstacles")
    parser.add_argument("--use-nonlinear-ctrl", action="store_true", help="Enable original nonlinear controller")
    parser.add_argument("--no-nonlinear-ctrl", dest="use_nonlinear_ctrl", action="store_false", help="Disable original nonlinear controller")
    parser.set_defaults(use_nonlinear_ctrl=True)
    parser.add_argument("--use-rl-ctrl", action="store_true", help="Enable RL controller (default on)")
    parser.add_argument("--no-rl-ctrl", dest="use_rl_ctrl", action="store_false", help="Disable RL controller")
    parser.set_defaults(use_rl_ctrl=True)
    parser.add_argument("--blend-alpha", type=float, default=0.5, help="Blend factor alpha for RL (alpha*RL + (1-alpha)*nonlinear)")
    parser.add_argument("--behavior-mode", type=str, choices=["rl", "nonlinear", "blend", "none"], default="blend", help="Behavior mode: 'rl'|'nonlinear'|'blend'|'none'")
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)

    env = DummyVecEnv([make_env_fn(render=args.render,
                                   max_vel=args.max_vel,
                                   use_robot5_obstacles=args.use_robot5_obstacles,
                                   use_nonlinear_ctrl=args.use_nonlinear_ctrl,
                                   use_rl_ctrl=args.use_rl_ctrl,
                                   blend_alpha=args.blend_alpha,
                                   behavior_mode=args.behavior_mode)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=args.logdir, name_prefix="sac_robot6")

    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=args.logdir)
    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)

    model_path = os.path.join(args.logdir, "sac_robot6_final")
    model.save(model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
