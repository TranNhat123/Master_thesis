# Train_Robot6.py
import os, time, csv, glob, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from Robot6_SAC import Robot6SacEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    if not os.path.isdir(checkpoint_dir): return None
    latest_candidates = glob.glob(os.path.join(checkpoint_dir, "*/latest.pth"))
    if latest_candidates: return max(latest_candidates, key=os.path.getmtime)
    all_candidates = glob.glob(os.path.join(checkpoint_dir, "*/ckpt_ep_*.pth"))
    if all_candidates: return max(all_candidates, key=os.path.getmtime)
    return None

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256):
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
    def __init__(self, obs_dim, act_dim, hidden_dim=256, log_std_min=-20, log_std_max=2.0):
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
        
        # --- QUAN TRỌNG: Dùng Tanh cho TOÀN BỘ output (cả Velocity và Alpha) ---
        action = torch.tanh(x_t)
        
        # Log prob correction chuẩn cho Tanh
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

class ReplayBuffer:
    def __init__(self, capacity=1_000_000):
        self.buffer = deque(maxlen=capacity)
    def push(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return (torch.FloatTensor(s).to(device), torch.FloatTensor(a).to(device),
                torch.FloatTensor(r).unsqueeze(-1).to(device), torch.FloatTensor(s2).to(device),
                torch.FloatTensor(d).unsqueeze(-1).to(device))
    def __len__(self): return len(self.buffer)

def train_sac(args):
    env = Robot6SacEnv(use_gui=False)
    obs_dim = env.observation_space.shape[0] # 41
    act_dim = env.action_space.shape[0]      # 7
    print(f"Obs Dim: {obs_dim}, Act Dim: {act_dim}")

    policy = GaussianPolicy(obs_dim, act_dim).to(device)
    q1 = MLP(obs_dim + act_dim, 1).to(device)
    q2 = MLP(obs_dim + act_dim, 1).to(device)
    q1_target = MLP(obs_dim + act_dim, 1).to(device); q1_target.load_state_dict(q1.state_dict())
    q2_target = MLP(obs_dim + act_dim, 1).to(device); q2_target.load_state_dict(q2.state_dict())

    policy_opt = optim.Adam(policy.parameters(), lr=3e-4)
    q_opt = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=3e-4)
    
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_opt = optim.Adam([log_alpha], lr=3e-4)
    target_entropy = -act_dim

    buffer = ReplayBuffer()
    batch_size, gamma, tau = 256, 0.99, 0.005
    num_episodes = 10000
    start_steps = 5000
    
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    global_step = 0
    start_episode = 0
    best_reward = -float("inf")
    
    if args.resume:
        resume_ckpt = args.resume_ckpt if args.resume_ckpt else find_latest_checkpoint(checkpoint_dir)
        if resume_ckpt and os.path.isfile(resume_ckpt):
            print(f"[RESUME] Loading: {resume_ckpt}")
            ckpt = None
            try:
                ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
            except Exception as e2:
                print(f"[RESUME] Failed to load checkpoint: {e2}")
            
            if ckpt is not None:
                policy.load_state_dict(ckpt["policy"])
                q1.load_state_dict(ckpt["q1"])
                q2.load_state_dict(ckpt["q2"])
                q1_target.load_state_dict(ckpt["q1_target"])
                q2_target.load_state_dict(ckpt["q2_target"])
                policy_opt.load_state_dict(ckpt["policy_opt"])
                q_opt.load_state_dict(ckpt["q_opt"])
                alpha_opt.load_state_dict(ckpt["alpha_opt"])
                log_alpha.data = ckpt["log_alpha"].to(device)
                global_step = ckpt.get("global_step", 0)
                start_episode = ckpt.get("episode", 0) + 1
                best_reward = ckpt.get("best_reward", -float("inf"))
                print(f"[RESUME] Checkpoint loaded successfully at episode {start_episode}")
    
    run_name = os.path.basename(os.path.dirname(resume_ckpt)) if (args.resume and resume_ckpt) else f"sac_{time.strftime('%Y%m%d-%H%M%S')}"
    log_dir = os.path.join("runs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    csv_path = os.path.join(log_dir, "training.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f: csv.writer(f).writerow(["episode", "global_step", "episode_reward", "episode_length", "R1_avg", "R3_avg", "R4_avg", "collisions"])
    
    for ep in range(start_episode, num_episodes):
        s, _ = env.reset()
        ep_reward = 0
        ep_alpha_val = 0
        steps_in_ep = 0
        done, truncated = False, False
        ep_R1_list = []
        ep_R3_list = []
        ep_R4_list = []
        ep_collisions = 0

        while not (done or truncated):
            global_step += 1
            steps_in_ep += 1
            
            if global_step < start_steps:
                # Random [-1, 1] cho tất cả 7 chiều (khớp với Tanh output)
                a = np.random.uniform(-1, 1, act_dim)
            else:
                with torch.no_grad():
                    s_t = torch.FloatTensor(s).unsqueeze(0).to(device)
                    a_t, _ = policy.sample(s_t)
                    a = a_t.cpu().numpy()[0]

            s2, r, done, truncated, info = env.step(a)
            buffer.push(s, a, r, s2, float(done))
            s = s2
            ep_reward += r
            ep_alpha_val += info['alpha']
            if 'R1' in info: ep_R1_list.append(info['R1'])
            if 'R3' in info: ep_R3_list.append(info['R3'])
            if 'R4' in info: ep_R4_list.append(info['R4'])
            if 'd_min' in info and info['d_min'] < env.d_coll:
                ep_collisions += 1

            if len(buffer) > batch_size and global_step >= start_steps:
                sb, ab, rb, s2b, db = buffer.sample(batch_size)
                
                with torch.no_grad():
                    a2, log_pi2 = policy.sample(s2b)
                    q_next = torch.min(q1_target(torch.cat([s2b, a2], 1)), q2_target(torch.cat([s2b, a2], 1)))
                    alpha_sac = log_alpha.exp()
                    y = rb + gamma * (1 - db) * (q_next - alpha_sac * log_pi2)

                # Critic losses separately (for logging)
                q1_loss = F.mse_loss(q1(torch.cat([sb, ab], 1)), y)
                q2_loss = F.mse_loss(q2(torch.cat([sb, ab], 1)), y)
                # --- 1. CRITIC UPDATE ---
                q_loss = F.mse_loss(q1(torch.cat([sb, ab], 1)), y) + F.mse_loss(q2(torch.cat([sb, ab], 1)), y)
                
                q_opt.zero_grad()
                q_loss.backward()
                
                # [SỬA LẠI] Vừa cắt gradient (max=1.0), vừa lấy giá trị norm để log
                # Thay thế hoàn toàn vòng lặp for thủ công của bạn
                q_grad_norm = torch.nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), max_norm=1.0)
                
                q_opt.step()

                # --- 2. POLICY UPDATE ---
                a_new, log_pi_new = policy.sample(sb)
                q_new = torch.min(q1(torch.cat([sb, a_new], 1)), q2(torch.cat([sb, a_new], 1)))
                alpha_sac = log_alpha.exp()
                policy_loss = (alpha_sac * log_pi_new - q_new).mean()

                policy_opt.zero_grad()
                policy_loss.backward()
                
                # [SỬA LẠI] Tương tự cho Policy
                policy_grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                
                policy_opt.step()

                # --- 3. ALPHA UPDATE ---
                # (Giữ nguyên vì Alpha ít khi bị bùng nổ, nhưng nếu muốn chắc chắn cũng có thể clip)
                alpha_loss = -(log_alpha * (log_pi_new + target_entropy).detach()).mean()
                alpha_opt.zero_grad()
                alpha_loss.backward()
                alpha_opt.step()

                # Soft update targets
                for p, tp in zip(q1.parameters(), q1_target.parameters()): tp.data.mul_(1-tau).add_(tau*p.data)
                for p, tp in zip(q2.parameters(), q2_target.parameters()): tp.data.mul_(1-tau).add_(tau*p.data)

                # ===== LOG METRICS (per step for monitoring) =====
                writer.add_scalar("Loss/Q1", q1_loss.item(), global_step)
                writer.add_scalar("Loss/Q2", q2_loss.item(), global_step)
                writer.add_scalar("Loss/Policy", policy_loss.item(), global_step)
                writer.add_scalar("Loss/Alpha", alpha_loss.item(), global_step)
                writer.add_scalar("Metrics/Temperature", alpha_sac.item(), global_step)
                writer.add_scalar("Metrics/LogAlpha", log_alpha.item(), global_step)
                # Log pi stats
                try:
                    writer.add_scalar("Metrics/LogPi_Mean", log_pi_new.mean().item(), global_step)
                except Exception:
                    pass
                # Replay buffer size
                writer.add_scalar("Replay/Size", len(buffer), global_step)
                # Q value statistics
                try:
                    writer.add_scalar("Metrics/Q_New_Mean", q_new.mean().item(), global_step)
                    writer.add_scalar("Metrics/Q_New_Min", q_new.min().item(), global_step)
                    writer.add_scalar("Metrics/Q_New_Max", q_new.max().item(), global_step)
                except Exception:
                    pass
                # Action statistics per-dimension
                try:
                    a_stats = a_new.detach().cpu().mean(dim=0)
                    for i in range(a_stats.shape[0]):
                        writer.add_scalar(f"Action/Mean_dim{i}", float(a_stats[i].item()), global_step)
                except Exception:
                    pass
                # Gradient norms
                writer.add_scalar("GradNorm/Critic", q_grad_norm, global_step)
                writer.add_scalar("GradNorm/Policy", policy_grad_norm, global_step)

        avg_alpha = ep_alpha_val / steps_in_ep
        avg_R1 = float(np.mean(ep_R1_list)) if ep_R1_list else 0.0
        avg_R3 = float(np.mean(ep_R3_list)) if ep_R3_list else 0.0
        avg_R4 = float(np.mean(ep_R4_list)) if ep_R4_list else 0.0
        
        print(f"Ep {ep}: Reward={ep_reward:.2f}, R1={avg_R1:.2f}, R3={avg_R3:.2f}, R4={avg_R4:.2f}, Collisions={ep_collisions}")
        writer.add_scalar("Episode/Reward", ep_reward, ep)
        writer.add_scalar("Episode/AvgAlpha", avg_alpha, ep)
        writer.add_scalar("Episode/Length", steps_in_ep, ep)
        writer.add_scalar("Episode/R1_Avg", avg_R1, ep)
        writer.add_scalar("Episode/R3_Avg", avg_R3, ep)
        writer.add_scalar("Episode/R4_Avg", avg_R4, ep)
        writer.add_scalar("Episode/Collisions", ep_collisions, ep)
        writer.flush()
        
        with open(csv_path, "a", newline="") as f: csv.writer(f).writerow([ep, global_step, ep_reward, steps_in_ep, avg_R1, avg_R3, avg_R4, ep_collisions])
        
        ckpt_data = {
            "policy": policy.state_dict(), "q1": q1.state_dict(), "q2": q2.state_dict(),
            "q1_target": q1_target.state_dict(), "q2_target": q2_target.state_dict(),
            "policy_opt": policy_opt.state_dict(), "q_opt": q_opt.state_dict(),
            "alpha_opt": alpha_opt.state_dict(), "log_alpha": log_alpha.detach().cpu(),
            "global_step": global_step, "episode": ep, "best_reward": best_reward,
        }
        
        run_ckpt_dir = os.path.join(checkpoint_dir, run_name)
        os.makedirs(run_ckpt_dir, exist_ok=True)
        torch.save(ckpt_data, os.path.join(run_ckpt_dir, "latest.pth"))
        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save(ckpt_data, os.path.join(run_ckpt_dir, "best.pth"))
            print(f"  [BEST] Saved best checkpoint (reward={best_reward:.2f})")
        if (ep + 1) % 50 == 0:
            torch.save(ckpt_data, os.path.join(run_ckpt_dir, f"ckpt_ep_{ep+1}.pth"))

    writer.close(); env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-ckpt", type=str, default="")
    args = parser.parse_args()
    train_sac(args)