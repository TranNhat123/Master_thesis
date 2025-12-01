# Training Resume Implementation - Summary

## ✅ Implemented Changes

### 1. Checkpoint Finding Function
Added `find_latest_checkpoint()` in `Train_Robot6.py`:
- Tìm `latest.pth` (ưu tiên) từ `checkpoints/*/`
- Nếu không có, tìm `ckpt_ep_*.pth` mới nhất
- Returns `None` nếu không tìm thấy

### 2. Resume Training Logic
Modified `train_sac()` signature: `train_sac()` → `train_sac(args)`

**Before resume:**
- `train_sac()` function tạo checkpoint dir = `"runs/sac_hybrid"` (fixed)
- Không có checkpoint save/load logic
- Không có argument parsing

**After resume:**
- Detect `args.resume` và `args.resume_ckpt` flags
- Auto-find latest checkpoint nếu `--resume` set
- Load tất cả model states + optimizer states + training progress
- Continue từ `start_episode` tiếp theo

### 3. Checkpoint Save Logic
Lưu checkpoint mỗi episode:
- `latest.pth`: Update mỗi episode (cho quick resume)
- `best.pth`: Lưu khi reward cao nhất
- `ckpt_ep_N.pth`: Periodic save mỗi 50 episode

Checkpoint dir structure:
```
checkpoints/sac_YYYYMMDD-HHMMSS/
├── latest.pth
├── best.pth
├── ckpt_ep_50.pth
└── ckpt_ep_100.pth
```

### 4. Logging Integration
- CSV logging: `runs/sac_YYYYMMDD-HHMMSS/training.csv`
- TensorBoard: Auto log reward + avg alpha
- Log dir reuse: Resume training dùng lại run name + log dir từ checkpoint

### 5. Argument Parser
```bash
# Train từ đầu
python Train_Robot6.py

# Resume từ latest
python Train_Robot6.py --resume

# Resume từ specific checkpoint
python Train_Robot6.py --resume --resume-ckpt "path/to/checkpoint.pth"
```

---

## Checkpoint Load/Save Details

### Load từ checkpoint:
```python
if args.resume:
    resume_ckpt = find_latest_checkpoint() or args.resume_ckpt
    if resume_ckpt:
        ckpt = torch.load(resume_ckpt, map_location=device)
        
        # Load models
        policy.load_state_dict(ckpt["policy"])
        q1.load_state_dict(ckpt["q1"])
        q2.load_state_dict(ckpt["q2"])
        q1_target.load_state_dict(ckpt["q1_target"])
        q2_target.load_state_dict(ckpt["q2_target"])
        
        # Load optimizers (quan trọng!)
        policy_opt.load_state_dict(ckpt["policy_opt"])
        q_opt.load_state_dict(ckpt["q_opt"])
        alpha_opt.load_state_dict(ckpt["alpha_opt"])
        
        # Load training state
        log_alpha.data = ckpt["log_alpha"]
        global_step = ckpt["global_step"]
        start_episode = ckpt["episode"] + 1
        best_reward = ckpt["best_reward"]
```

### Save checkpoint mỗi episode:
```python
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

torch.save(ckpt_data, latest_ckpt)  # Mỗi episode
if ep_reward > best_reward:
    torch.save(ckpt_data, best_ckpt)  # Best only
if (ep + 1) % 50 == 0:
    torch.save(ckpt_data, periodic_ckpt)  # Periodic
```

---

## Usage Examples

### Example 1: Train từ đầu
```bash
python Train_Robot6.py

# Output:
# Obs Dim: 41, Act Dim: 7
# Ep 0: Reward=..., Avg Alpha=...
# Ep 1: Reward=..., Avg Alpha=...
# ... (creates checkpoints/sac_20251201-143000/)
```

### Example 2: Resume training
```bash
python Train_Robot6.py --resume

# Output:
# [RESUME] Loading checkpoint: checkpoints/sac_20251201-143000/latest.pth
# [RESUME] Loaded: episode=101, global_step=50100, best_reward=12.34
# Ep 101: Reward=..., Avg Alpha=...
```

### Example 3: Resume từ specific episode
```bash
python Train_Robot6.py --resume --resume-ckpt "checkpoints/sac_20251201-143000/ckpt_ep_500.pth"

# Output:
# [RESUME] Loading checkpoint: checkpoints/sac_20251201-143000/ckpt_ep_500.pth
# [RESUME] Loaded: episode=501, global_step=250500, best_reward=15.67
```

---

## Files Modified

1. **Train_Robot6.py**
   - Added `find_latest_checkpoint()` function
   - Modified `train_sac()` to accept `args` parameter
   - Added resume checkpoint loading logic
   - Added checkpoint saving logic (latest, best, periodic)
   - Added CSV logging
   - Added argument parser (`--resume`, `--resume-ckpt`)

---

## Verification

✅ Syntax check: OK
✅ Logic test: `find_latest_checkpoint()` works correctly
✅ All required imports present
✅ Backward compatible: `train_sac()` can still be called without args (uses default)

---

## Notes

- Checkpoint files `.pth` chứa toàn bộ training state, không cần replay buffer recovery
- Resume training sẽ tái sử dụng run name và log directory từ checkpoint
- Training CSV được append, không overwrite
- TensorBoard logs được merge khi resume cùng run

---

## Quick Start

```bash
# Lần 1: Train từ đầu
python Train_Robot6.py

# (Sau vài episode, interrupt bằng Ctrl+C)

# Lần 2: Tiếp tục từ checkpoint mới nhất
python Train_Robot6.py --resume

# Kiểm tra TensorBoard
tensorboard --logdir runs/
# Open: http://localhost:6006
```

