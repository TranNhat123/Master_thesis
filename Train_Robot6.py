# sac_robot6_train.py
import os
import time
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter
from Robot6_SAC import Robot6SacEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
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
    def __init__(self, capacity=1000000):
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


def train_sac():
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

    # Optimizer
    policy_opt = optim.Adam(policy.parameters(), lr=3e-4)
    q1_opt = optim.Adam(q1.parameters(), lr=3e-4)
    q2_opt = optim.Adam(q2.parameters(), lr=3e-4)

    # Temperature alpha (entropy)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_opt = optim.Adam([log_alpha], lr=3e-4)
    target_entropy = -act_dim  # gợi ý từ paper SAC

    buffer = ReplayBuffer()
    batch_size = 256
    gamma = 0.99
    tau = 0.005

    num_episodes = int(os.environ.get("TRAIN_EPISODES", "1000"))
    start_steps = int(os.environ.get("START_STEPS", "10000"))
    updates_per_step = int(os.environ.get("UPDATES_PER_STEP", "1"))

    global_step = 0
    # ----------------- Logging & Checkpoint setup -----------------
    # If RESUME=1, try to find the latest checkpoint and resume into that run folder
    resume_flag = int(os.environ.get("RESUME", "0"))
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"sac_robot6_{timestamp}"
    log_dir = os.path.join("runs", run_name)
    ckpt_dir = os.path.join("checkpoints", run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Try resume: find latest checkpoint file under checkpoints/*/best.pth or ckpt_ep_*.pth
    resume_ckpt = None
    resume_run_dir = None
    if resume_flag == 1:
        import glob
        ckpt_candidates = glob.glob(os.path.join("checkpoints", "*", "best.pth"))
        if not ckpt_candidates:
            ckpt_candidates = glob.glob(os.path.join("checkpoints", "*", "ckpt_ep_*.pth"))
        if ckpt_candidates:
            ckpt_candidates = sorted(ckpt_candidates, key=os.path.getmtime)
            resume_ckpt = ckpt_candidates[-1]
            resume_run_dir = os.path.dirname(resume_ckpt)
            # reuse run_name / dirs from checkpoint
            run_name = os.path.basename(resume_run_dir)
            log_dir = os.path.join("runs", run_name)
            ckpt_dir = os.path.join("checkpoints", run_name)

    writer = SummaryWriter(log_dir=log_dir)

    csv_path = os.path.join(log_dir, "training.csv")
    csv_header = ["episode", "global_step", "episode_reward"]
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(csv_header)

    save_freq_episodes = int(os.environ.get("SAVE_FREQ_EP", "50"))
    best_reward = -float("inf")
    start_episode = 0

    # If we have a resume checkpoint, load states
    if resume_ckpt is not None:
        print(f"Resuming from checkpoint: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device)
        try:
            policy.load_state_dict(ckpt['policy_state'])
            q1.load_state_dict(ckpt['q1_state'])
            q2.load_state_dict(ckpt['q2_state'])
        except Exception:
            print("Warning: checkpoint missing some model parts or incompatible.")

        # optimizers may or may not be saved
        try:
            policy_opt.load_state_dict(ckpt['policy_opt'])
            q1_opt.load_state_dict(ckpt['q1_opt'])
            q2_opt.load_state_dict(ckpt['q2_opt'])
            alpha_opt.load_state_dict(ckpt['alpha_opt'])
        except Exception:
            pass

        # load log_alpha
        if 'log_alpha' in ckpt:
            try:
                log_alpha.data.copy_(ckpt['log_alpha'].to(device))
            except Exception:
                try:
                    log_alpha.data.copy_(torch.tensor(ckpt['log_alpha']).to(device))
                except Exception:
                    pass

        if 'global_step' in ckpt:
            global_step = int(ckpt['global_step'])
        if 'episode' in ckpt:
            start_episode = int(ckpt['episode']) + 1
            print(f"Resuming from episode {start_episode}")


    for ep in range(num_episodes):
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
            # Treat truncated as terminal for learning targets as well
            d = float(done or truncated)

            buffer.push(s, a, r, s2, d)
            s = s2
            ep_reward += r

            # Update mạng sau khi đủ mẫu
            if len(buffer) > batch_size and global_step >= start_steps:
                for _ in range(updates_per_step):
                    (
                        s_b,
                        a_b,
                        r_b,
                        s2_b,
                        d_b,
                    ) = buffer.sample(batch_size)

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

                    q1_loss = F.mse_loss(q1_pred, y)
                    q2_loss = F.mse_loss(q2_pred, y)

                    q1_opt.zero_grad()
                    q1_loss.backward()
                    q1_opt.step()

                    q2_opt.zero_grad()
                    q2_loss.backward()
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
                    policy_opt.step()

                    # Log losses & alpha to TensorBoard
                    writer.add_scalar("loss/q1_loss", q1_loss.item(), global_step)
                    writer.add_scalar("loss/q2_loss", q2_loss.item(), global_step)
                    writer.add_scalar("loss/policy_loss", policy_loss.item(), global_step)
                    writer.add_scalar("train/alpha", log_alpha.exp().item(), global_step)

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

        print(f"Episode {ep}, reward = {ep_reward:.2f}")

        # ----------------- Logging per-episode -----------------
        writer.add_scalar("episode/reward", ep_reward, ep)
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([ep, global_step, ep_reward])

        # Save checkpoint periodically and save best
        if (ep + 1) % save_freq_episodes == 0:
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_ep_{ep+1}.pth")
            torch.save({
                'policy_state': policy.state_dict(),
                'q1_state': q1.state_dict(),
                'q2_state': q2.state_dict(),
                'policy_opt': policy_opt.state_dict(),
                'q1_opt': q1_opt.state_dict(),
                'q2_opt': q2_opt.state_dict(),
                'alpha_opt': alpha_opt.state_dict(),
                'log_alpha': log_alpha.detach().cpu(),
                'global_step': global_step,
                'episode': ep,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        if ep_reward > best_reward:
            best_reward = ep_reward
            best_path = os.path.join(ckpt_dir, "best.pth")
            torch.save({
                'policy_state': policy.state_dict(),
                'q1_state': q1.state_dict(),
                'q2_state': q2.state_dict(),
                'log_alpha': log_alpha.detach().cpu(),
                'global_step': global_step,
                'episode': ep,
            }, best_path)
            print(f"Saved best checkpoint: {best_path} (reward={best_reward:.2f})")

    # Close writer
    writer.close()

    env.close()


if __name__ == "__main__":
    train_sac()
