# sac_robot6_train.py
import os
import time
import csv
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter
from Robot6_SAC import Robot6SacEnv
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=512, log_std_min=-20, log_std_max=1.0):
        super().__init__()
        self.net = MLP(obs_dim, 2 * act_dim, hidden_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs):
        mu_logstd = self.net(obs)
        mu, log_std = torch.chunk(mu_logstd, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, obs):
        mu, std = self.forward(obs)
        normal = torch.distributions.Normal(mu, std)
        x_t = normal.rsample()
        a_t = torch.tanh(x_t)  # [-1, 1]
        # log_prob với tanh-squash
        log_prob = normal.log_prob(x_t) - torch.log(1 - a_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return a_t, log_prob


class ReplayBuffer:
    def __init__(self, capacity=1_000_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return (
            torch.FloatTensor(s).to(device),
            torch.FloatTensor(a).to(device),
            torch.FloatTensor(r).unsqueeze(-1).to(device),
            torch.FloatTensor(s2).to(device),
            torch.FloatTensor(d).unsqueeze(-1).to(device),
        )

    def __len__(self):
        return len(self.buffer)


def find_latest_checkpoint(base_dir="checkpoints"):
    """
    Tìm checkpoint mới nhất trong:
      checkpoints/*/latest.pth
      checkpoints/*/ckpt_ep_*.pth
    Trả về path hoặc None nếu không có.
    """
    patterns = [
        os.path.join(base_dir, "*", "latest.pth"),
        os.path.join(base_dir, "*", "ckpt_ep_*.pth"),
    ]
    ckpt_candidates = []
    for p in patterns:
        ckpt_candidates.extend(glob.glob(p))

    if not ckpt_candidates:
        return None

    latest_ckpt = max(ckpt_candidates, key=os.path.getmtime)
    return latest_ckpt


def save_training_checkpoint(
    ckpt_path,
    policy,
    q1,
    q2,
    q1_target,
    q2_target,
    policy_opt,
    q1_opt,
    q2_opt,
    alpha_opt,
    log_alpha,
    global_step,
    episode,
    best_reward,
):
    ckpt = {
        # Models
        "policy_state": policy.state_dict(),
        "q1_state": q1.state_dict(),
        "q2_state": q2.state_dict(),
        "q1_target_state": q1_target.state_dict(),
        "q2_target_state": q2_target.state_dict(),
        # Optimizers
        "policy_opt": policy_opt.state_dict(),
        "q1_opt": q1_opt.state_dict(),
        "q2_opt": q2_opt.state_dict(),
        "alpha_opt": alpha_opt.state_dict(),
        # Alpha
        "log_alpha": log_alpha.detach().cpu(),
        # Training progress
        "global_step": global_step,
        "episode": episode,
        "best_reward": best_reward,
    }
    torch.save(ckpt, ckpt_path)


def load_training_checkpoint(
    ckpt_path,
    policy,
    q1,
    q2,
    q1_target,
    q2_target,
    policy_opt,
    q1_opt,
    q2_opt,
    alpha_opt,
    log_alpha,
):
    ckpt = torch.load(ckpt_path, map_location=device)

    policy.load_state_dict(ckpt["policy_state"])
    q1.load_state_dict(ckpt["q1_state"])
    q2.load_state_dict(ckpt["q2_state"])
    q1_target.load_state_dict(ckpt["q1_target_state"])
    q2_target.load_state_dict(ckpt["q2_target_state"])

    policy_opt.load_state_dict(ckpt["policy_opt"])
    q1_opt.load_state_dict(ckpt["q1_opt"])
    q2_opt.load_state_dict(ckpt["q2_opt"])
    alpha_opt.load_state_dict(ckpt["alpha_opt"])

    # log_alpha
    log_alpha.data.copy_(torch.tensor(ckpt["log_alpha"]).to(device))

    global_step = int(ckpt.get("global_step", 0))
    start_episode = int(ckpt.get("episode", -1)) + 1
    best_reward = float(ckpt.get("best_reward", -float("inf")))
    print(
        f"Loaded checkpoint from {ckpt_path} "
        f"(episode={start_episode-1}, global_step={global_step}, best_reward={best_reward:.2f})"
    )
    return global_step, start_episode, best_reward


def train_sac(args):
    env = Robot6SacEnv(use_gui=False)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Khởi tạo mạng
    policy = GaussianPolicy(obs_dim, act_dim).to(device)
    q1 = MLP(obs_dim + act_dim, 1).to(device)
    q2 = MLP(obs_dim + act_dim, 1).to(device)
    q1_target = MLP(obs_dim + act_dim, 1).to(device)
    q2_target = MLP(obs_dim + act_dim, 1).to(device)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    # Optimizer (reduced LR for stability)
    policy_opt = optim.Adam(policy.parameters(), lr=float(os.environ.get("POLICY_LR", 1e-4)))
    q1_opt = optim.Adam(q1.parameters(), lr=float(os.environ.get("Q_LR", 1e-4)))
    q2_opt = optim.Adam(q2.parameters(), lr=float(os.environ.get("Q_LR", 1e-4)))

    # Temperature alpha (entropy)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_opt = optim.Adam([log_alpha], lr=3e-4)
    target_entropy = -act_dim  # gợi ý từ paper SAC

    buffer = ReplayBuffer()
    batch_size = 256
    gamma = 0.99
    tau = 0.005

    # Reward scaling (static): multiply environment reward before storing/learning
    reward_scale = 0.01

    num_episodes = int(os.environ.get("TRAIN_EPISODES", "10000"))
    start_steps = int(os.environ.get("START_STEPS", "10000"))
    updates_per_step = int(os.environ.get("UPDATES_PER_STEP", "1"))

    # ----------------- Logging & Checkpoint setup -----------------
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"sac_robot6_{timestamp}"
    log_dir = os.path.join("runs", run_name)
    ckpt_dir = os.path.join("checkpoints", run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Parse resume flags
    resume_flag = args.resume
    resume_ckpt_path = args.resume_ckpt

    resume_ckpt = None
    if resume_flag == 1:
        if resume_ckpt_path:
            # User chỉ định checkpoint cụ thể
            if os.path.isfile(resume_ckpt_path):
                resume_ckpt = resume_ckpt_path
                print(f"[RESUME] Using specified checkpoint: {resume_ckpt}")
            else:
                print(f"[RESUME] Specified checkpoint not found: {resume_ckpt_path}")
        else:
            # Không chỉ checkpoint cụ thể -> tự tìm mới nhất
            resume_ckpt = find_latest_checkpoint(base_dir="checkpoints")
            if resume_ckpt is not None:
                print(f"[RESUME] Using latest checkpoint: {resume_ckpt}")
            else:
                print("[RESUME] No checkpoint found, start from scratch.")

    # Nếu có resume_ckpt -> dùng lại run_name / log_dir / ckpt_dir cũ
    if resume_ckpt is not None:
        resume_run_dir = os.path.dirname(resume_ckpt)
        run_name = os.path.basename(resume_run_dir)
        ckpt_dir = resume_run_dir
        log_dir = os.path.join("runs", run_name)
        os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)

    csv_path = os.path.join(log_dir, "training.csv")
    csv_header = ["episode", "global_step", "episode_reward"]
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(csv_header)

    save_freq_episodes = int(os.environ.get("SAVE_FREQ_EP", "50"))
    best_reward = -float("inf")
    global_step = 0
    start_episode = 0

    # If we have a resume checkpoint, load states
    if resume_ckpt is not None:
        global_step, start_episode, best_reward = load_training_checkpoint(
            resume_ckpt,
            policy,
            q1,
            q2,
            q1_target,
            q2_target,
            policy_opt,
            q1_opt,
            q2_opt,
            alpha_opt,
            log_alpha,
        )
        print(f"Resuming training from episode {start_episode}")

    # ----------------- Training loop -----------------
    for ep in range(start_episode, num_episodes):
        s, _ = env.reset()
        ep_reward = 0.0
        done = False
        truncated = False

        while not (done or truncated):
            global_step += 1
            s_tensor = torch.FloatTensor(s).unsqueeze(0).to(device)

            # Giai đoạn explore: random action
            if global_step < start_steps:
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    a_t, _ = policy.sample(s_tensor)
                    a = a_t.cpu().numpy()[0]

            s2, r, done, truncated, info = env.step(a)
            d = float(done or truncated)

            # Apply reward scaling before storing and logging
            r_scaled = float(r) * reward_scale

            buffer.push(s, a, r_scaled, s2, d)
            s = s2
            ep_reward += r_scaled

            # Update mạng sau khi đủ mẫu
            if len(buffer) > batch_size and global_step >= start_steps:
                for _ in range(updates_per_step):
                    s_b, a_b, r_b, s2_b, d_b = buffer.sample(batch_size)

                    # --- Update Q ---
                    with torch.no_grad():
                        a2, log_pi2 = policy.sample(s2_b)
                        sa2 = torch.cat([s2_b, a2], dim=-1)
                        q1_t = q1_target(sa2)
                        q2_t = q2_target(sa2)
                        q_min = torch.min(q1_t, q2_t)
                        alpha = log_alpha.exp()
                        y = r_b + (1 - d_b) * gamma * (q_min - alpha * log_pi2)

                    sa = torch.cat([s_b, a_b], dim=-1)
                    q1_pred = q1(sa)
                    q2_pred = q2(sa)

                    # Use Huber loss (Smooth L1) for robustness to large TD errors
                    q1_loss = F.smooth_l1_loss(q1_pred, y)
                    q2_loss = F.smooth_l1_loss(q2_pred, y)

                    q1_opt.zero_grad()
                    q1_loss.backward()
                    # Clip gradients to avoid explosion
                    torch.nn.utils.clip_grad_norm_(q1.parameters(), max_norm=1.0)
                    q1_opt.step()

                    q2_opt.zero_grad()
                    q2_loss.backward()
                    torch.nn.utils.clip_grad_norm_(q2.parameters(), max_norm=1.0)
                    q2_opt.step()

                    # --- Update policy ---
                    a_new, log_pi = policy.sample(s_b)
                    sa_new = torch.cat([s_b, a_new], dim=-1)
                    q1_new = q1(sa_new)
                    q2_new = q2(sa_new)
                    q_new = torch.min(q1_new, q2_new)

                    alpha = log_alpha.exp()
                    policy_loss = (alpha * log_pi - q_new).mean()

                    policy_opt.zero_grad()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                    policy_opt.step()

                    # Log losses & alpha to TensorBoard
                    writer.add_scalar("loss/q1_loss", q1_loss.item(), global_step)
                    writer.add_scalar("loss/q2_loss", q2_loss.item(), global_step)
                    writer.add_scalar("loss/policy_loss", policy_loss.item(), global_step)
                    writer.add_scalar("train/alpha", log_alpha.exp().item(), global_step)

                    writer.add_scalar("debug/q1_mean", q1_pred.mean().item(), global_step)
                    writer.add_scalar("debug/q2_mean", q2_pred.mean().item(), global_step)
                    writer.add_scalar("debug/q_target_mean", y.mean().item(), global_step)
                    writer.add_scalar("debug/log_pi_mean", log_pi.mean().item(), global_step)

                    # --- Update alpha (entropy temperature) ---
                    alpha_loss = (-log_alpha * (log_pi + target_entropy).detach()).mean()
                    alpha_opt.zero_grad()
                    alpha_loss.backward()
                    alpha_opt.step()

                    # --- Soft update target networks ---
                    with torch.no_grad():
                        for param, target_param in zip(q1.parameters(), q1_target.parameters()):
                            target_param.data.mul_(1 - tau)
                            target_param.data.add_(tau * param.data)
                        for param, target_param in zip(q2.parameters(), q2_target.parameters()):
                            target_param.data.mul_(1 - tau)
                            target_param.data.add_(tau * param.data)
            # end updates

        print(f"Episode {ep} / {num_episodes}, reward = {ep_reward:.2f}")

        # ----------------- Logging per-episode -----------------
        writer.add_scalar("episode/reward", ep_reward, ep)
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([ep, global_step, ep_reward])

        # --- Save latest training checkpoint (mỗi episode) ---
        latest_ckpt_path = os.path.join(ckpt_dir, "latest.pth")
        save_training_checkpoint(
            latest_ckpt_path,
            policy,
            q1,
            q2,
            q1_target,
            q2_target,
            policy_opt,
            q1_opt,
            q2_opt,
            alpha_opt,
            log_alpha,
            global_step,
            ep,
            best_reward,
        )

        # Save checkpoint theo chu kỳ và lưu best
        if (ep + 1) % save_freq_episodes == 0:
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_ep_{ep+1}.pth")
            save_training_checkpoint(
                ckpt_path,
                policy,
                q1,
                q2,
                q1_target,
                q2_target,
                policy_opt,
                q1_opt,
                q2_opt,
                alpha_opt,
                log_alpha,
                global_step,
                ep,
                best_reward,
            )
            print(f"Saved training checkpoint: {ckpt_path}")

        if ep_reward > best_reward:
            best_reward = ep_reward
            best_path = os.path.join(ckpt_dir, "best.pth")
            # best.pth chỉ cần model cho inference
            torch.save(
                {
                    "policy_state": policy.state_dict(),
                    "q1_state": q1.state_dict(),
                    "q2_state": q2.state_dict(),
                    "log_alpha": log_alpha.detach().cpu(),
                    "global_step": global_step,
                    "episode": ep,
                    "best_reward": best_reward,
                },
                best_path,
            )
            print(f"Saved best checkpoint: {best_path} (reward={best_reward:.2f})")

    writer.close()
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=int,
        default=0,
        help="1: resume training from checkpoint, 0: train from scratch",
    )
    parser.add_argument(
        "--resume-ckpt",
        type=str,
        default="",
        help="Path to a specific training checkpoint (.pth) to resume from",
    )
    args = parser.parse_args()
    train_sac(args)


### 
'''
Train mới từ đầu:
    python sac_robot6_train.py

Resume tự động từ checkpoint mới nhất:
    python sac_robot6_train.py --resume 1

Resume từ 1 checkpoint cụ thể:
    python sac_robot6_train.py --resume 1 --resume-ckpt checkpoints/sac_robot6_20251123-210000/ckpt_ep_200.pth


Mở 1 terminal khác. 
tensorboard --logdir runs
Rồi mở trình duyệt vào http://localhost:6006.
NGH 
python 3.10
'''