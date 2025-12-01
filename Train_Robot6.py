# train_robot6_sac.py
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
    """Tìm checkpoint mới nhất (latest.pth có ưu tiên)"""
    if not os.path.isdir(checkpoint_dir):
        return None
    
    # Ưu tiên latest.pth
    latest_candidates = glob.glob(os.path.join(checkpoint_dir, "*/latest.pth"))
    if latest_candidates:
        latest = max(latest_candidates, key=os.path.getmtime)
        return latest
    
    # Nếu không có latest.pth, tìm ckpt_ep_*.pth mới nhất
    all_candidates = glob.glob(os.path.join(checkpoint_dir, "*/ckpt_ep_*.pth"))
    if all_candidates:
        latest = max(all_candidates, key=os.path.getmtime)
        return latest
    
    return None

# --- CLASS MẠNG NEURAL ---
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
        self.obs_dim = obs_dim
        self.act_dim = act_dim
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
        x_t = normal.rsample() # Lấy mẫu từ phân phối Gaussian gốc
        
        # --- SPLIT OUTPUT: 6 Tanh (Vel), 1 Sigmoid (Alpha) ---
        # act_dim = 7 -> split [6, 1]
        x_vel, x_alpha = torch.split(x_t, [6, 1], dim=-1)
        
        # Apply Activations
        a_vel = torch.tanh(x_vel)         # [-1, 1]
        a_alpha = torch.sigmoid(x_alpha)  # [0, 1]
        
        # Nối lại thành action vector hoàn chỉnh
        action = torch.cat([a_vel, a_alpha], dim=-1)

        # --- LOG PROB CORRECTION (Quan trọng) ---
        # 1. Log prob gốc của Gaussian
        log_prob = normal.log_prob(x_t)
        
        # 2. Correction cho phần Tanh (6 chiều đầu)
        # log(1 - tanh^2)
        log_prob_vel = log_prob[..., :6] - torch.log(1 - a_vel.pow(2) + 1e-6)
        
        # 3. Correction cho phần Sigmoid (1 chiều cuối)
        # Derivative Sigmoid: s * (1 - s)
        # Log det jacobian: log(s) + log(1-s)
        log_prob_alpha = log_prob[..., 6:] - torch.log(a_alpha * (1 - a_alpha) + 1e-6)
        
        # Tổng hợp log_prob
        log_prob = torch.cat([log_prob_vel, log_prob_alpha], dim=-1)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

# --- REPLAY BUFFER & UTILS (Giữ nguyên hoặc rút gọn) ---
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

# --- TRAINING LOOP ---
def train_sac(args):
    # Khởi tạo Env & Check dims
    env = Robot6SacEnv(use_gui=False)
    obs_dim = env.observation_space.shape[0] # Sẽ là 41
    act_dim = env.action_space.shape[0]      # Sẽ là 7
    print(f"Obs Dim: {obs_dim}, Act Dim: {act_dim}")

    # Models
    policy = GaussianPolicy(obs_dim, act_dim).to(device)
    q1 = MLP(obs_dim + act_dim, 1).to(device)
    q2 = MLP(obs_dim + act_dim, 1).to(device)
    q1_target = MLP(obs_dim + act_dim, 1).to(device); q1_target.load_state_dict(q1.state_dict())
    q2_target = MLP(obs_dim + act_dim, 1).to(device); q2_target.load_state_dict(q2.state_dict())

    # Optimizers
    policy_opt = optim.Adam(policy.parameters(), lr=3e-4)
    q_opt = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=3e-4)
    
    # Entropy (Automatic Tuning)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_opt = optim.Adam([log_alpha], lr=3e-4)
    target_entropy = -act_dim

    buffer = ReplayBuffer()
    
    # Params
    batch_size, gamma, tau = 256, 0.99, 0.005
    num_episodes = 10000
    start_steps = 2000
    
    # ===== CHECKPOINT & LOGGING SETUP =====
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    global_step = 0
    start_episode = 0
    best_reward = -float("inf")
    
    # Resume từ checkpoint nếu --resume được set
    if args.resume:
        resume_ckpt = args.resume_ckpt if args.resume_ckpt else find_latest_checkpoint(checkpoint_dir)
        if resume_ckpt and os.path.isfile(resume_ckpt):
            print(f"[RESUME] Loading checkpoint: {resume_ckpt}")
            ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
            
            # Load states
            policy.load_state_dict(ckpt["policy"])
            q1.load_state_dict(ckpt["q1"])
            q2.load_state_dict(ckpt["q2"])
            q1_target.load_state_dict(ckpt["q1_target"])
            q2_target.load_state_dict(ckpt["q2_target"])
            
            # Load optimizers
            policy_opt.load_state_dict(ckpt["policy_opt"])
            q_opt.load_state_dict(ckpt["q_opt"])
            alpha_opt.load_state_dict(ckpt["alpha_opt"])
            
            # Load training state
            log_alpha.data = ckpt["log_alpha"].to(device)
            global_step = ckpt.get("global_step", 0)
            start_episode = ckpt.get("episode", 0) + 1
            best_reward = ckpt.get("best_reward", -float("inf"))
            
            print(f"[RESUME] Loaded: episode={start_episode}, global_step={global_step}, best_reward={best_reward:.2f}")
        else:
            print("[RESUME] Checkpoint not found, starting from scratch")
    
    # Tạo log dir từ checkpoint dir hoặc tạo mới
    run_name = os.path.basename(os.path.dirname(resume_ckpt)) if (args.resume and resume_ckpt) else f"sac_{time.strftime('%Y%m%d-%H%M%S')}"
    log_dir = os.path.join("runs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    csv_path = os.path.join(log_dir, "training.csv")
    
    # Tạo CSV nếu chưa có
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode", "global_step", "episode_reward"])
    
    for ep in range(start_episode, num_episodes):
        s, _ = env.reset()
        ep_reward = 0
        ep_alpha_val = 0 # Để log giá trị alpha trung bình
        steps_in_ep = 0
        done, truncated = False, False

        while not (done or truncated):
            global_step += 1
            steps_in_ep += 1
            
            # Select Action
            if global_step < start_steps:
                # Random sampling cần chú ý: 6 cái [-1,1] (RL vels), 1 cái [-1,1] (alpha, map to [0,1])
                a_vel = np.random.uniform(-1, 1, 6)
                a_alpha = np.random.uniform(-1, 1, 1)  # [-1,1] để khớp với policy output (tanh)
                a = np.concatenate([a_vel, a_alpha])
            else:
                with torch.no_grad():
                    s_t = torch.FloatTensor(s).unsqueeze(0).to(device)
                    a_t, _ = policy.sample(s_t)
                    a = a_t.cpu().numpy()[0]

            # Step Env
            s2, r, done, truncated, info = env.step(a)
            buffer.push(s, a, r, s2, float(done))
            s = s2
            ep_reward += r
            ep_alpha_val += info['alpha']

            # Update SAC
            if len(buffer) > batch_size and global_step >= start_steps:
                sb, ab, rb, s2b, db = buffer.sample(batch_size)
                
                # Critic Update
                with torch.no_grad():
                    a2, log_pi2 = policy.sample(s2b)
                    q_next = torch.min(q1_target(torch.cat([s2b, a2], 1)), q2_target(torch.cat([s2b, a2], 1)))
                    alpha_sac = log_alpha.exp()
                    y = rb + gamma * (1 - db) * (q_next - alpha_sac * log_pi2)
                
                q1_loss = F.mse_loss(q1(torch.cat([sb, ab], 1)), y)
                q2_loss = F.mse_loss(q2(torch.cat([sb, ab], 1)), y)
                q_loss = q1_loss + q2_loss
                
                q_opt.zero_grad(); q_loss.backward(); q_opt.step()

                # Policy Update
                a_new, log_pi_new = policy.sample(sb)
                q_new = torch.min(q1(torch.cat([sb, a_new], 1)), q2(torch.cat([sb, a_new], 1)))
                alpha_sac = log_alpha.exp()
                policy_loss = (alpha_sac * log_pi_new - q_new).mean()

                policy_opt.zero_grad(); policy_loss.backward(); policy_opt.step()

                # Alpha (Entropy) Update
                alpha_loss = -(log_alpha * (log_pi_new + target_entropy).detach()).mean()
                alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()

                # Soft Update Targets
                for p, tp in zip(q1.parameters(), q1_target.parameters()): tp.data.mul_(1-tau).add_(tau*p.data)
                for p, tp in zip(q2.parameters(), q2_target.parameters()): tp.data.mul_(1-tau).add_(tau*p.data)
                
                # ===== LOG METRICS (per step for monitoring) =====
                writer.add_scalar("Loss/Q1", q1_loss.item(), global_step)
                writer.add_scalar("Loss/Q2", q2_loss.item(), global_step)
                writer.add_scalar("Loss/Policy", policy_loss.item(), global_step)
                writer.add_scalar("Loss/Alpha", alpha_loss.item(), global_step)
                writer.add_scalar("Metrics/Temperature", alpha_sac.item(), global_step)
                writer.add_scalar("Metrics/LogAlpha", log_alpha.item(), global_step)
                writer.add_scalar("Metrics/LogPi_Mean", log_pi_new.mean().item(), global_step)

        # Logging per episode
        avg_alpha = ep_alpha_val / steps_in_ep
        print(f"Ep {ep}: Reward={ep_reward:.2f}, Avg Alpha={avg_alpha:.2f}")
        writer.add_scalar("Episode/Reward", ep_reward, ep)
        writer.add_scalar("Episode/AvgAlpha", avg_alpha, ep)
        writer.flush()  # ← CRITICAL: Force write to disk (fixes real-time issue)
        
        # Log to CSV
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([ep, global_step, ep_reward])
        
        # ===== CHECKPOINT SAVE =====
        ckpt_data = {
            "policy": policy.state_dict(),
            "q1": q1.state_dict(),
            "q2": q2.state_dict(),
            "q1_target": q1_target.state_dict(),
            "q2_target": q2_target.state_dict(),
            "policy_opt": policy_opt.state_dict(),
            "q_opt": q_opt.state_dict(),
            "alpha_opt": alpha_opt.state_dict(),
            "log_alpha": log_alpha.detach().cpu(),
            "global_step": global_step,
            "episode": ep,
            "best_reward": best_reward,
        }
        
        # Lưu latest.pth mỗi episode
        run_checkpoint_dir = os.path.join(checkpoint_dir, run_name)
        os.makedirs(run_checkpoint_dir, exist_ok=True)
        latest_ckpt = os.path.join(run_checkpoint_dir, "latest.pth")
        torch.save(ckpt_data, latest_ckpt)
        
        # Lưu best.pth nếu reward cao nhất
        if ep_reward > best_reward:
            best_reward = ep_reward
            best_ckpt = os.path.join(run_checkpoint_dir, "best.pth")
            torch.save(ckpt_data, best_ckpt)
            print(f"  [BEST] Saved best checkpoint (reward={best_reward:.2f})")
        
        # Lưu periodic checkpoint mỗi 50 episode
        if (ep + 1) % 50 == 0:
            periodic_ckpt = os.path.join(run_checkpoint_dir, f"ckpt_ep_{ep+1}.pth")
            torch.save(ckpt_data, periodic_ckpt)
            print(f"  [PERIODIC] Saved checkpoint at episode {ep+1}")
    
    # ===== TRAINING COMPLETED =====
    writer.close()  # Close SummaryWriter & flush all logs
    env.close()
    print(f"\n[DONE] Training completed! Total episodes: {num_episodes}, Best reward: {best_reward:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--resume-ckpt", type=str, default="", help="Path to specific checkpoint to resume from")
    args = parser.parse_args()
    
    train_sac(args)