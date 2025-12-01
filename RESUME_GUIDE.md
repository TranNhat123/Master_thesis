# Resume Training Guide

## Cách Sử Dụng Resume Training

### 1. Lần đầu tiên chạy (Train từ đầu)
```bash
python Train_Robot6.py
```
- Tạo thư mục `runs/sac_YYYYMMDD-HHMMSS/` cho tensorboard logs.
- Tạo thư mục `checkpoints/sac_YYYYMMDD-HHMMSS/` để lưu checkpoint.
- Lưu `latest.pth` mỗi episode.
- Lưu `best.pth` khi reward cao nhất.
- Lưu `ckpt_ep_N.pth` mỗi 50 episode.

### 2. Resume training từ checkpoint mới nhất
```bash
python Train_Robot6.py --resume
```
- Tự động tìm checkpoint mới nhất trong `checkpoints/` directory.
- Ưu tiên `latest.pth` (update mỗi episode).
- Nếu không có `latest.pth`, tìm `ckpt_ep_*.pth` mới nhất.
- Load lại toàn bộ trạng thái:
  - Model weights (policy, q1, q2, targets)
  - Optimizer states
  - Training progress (episode, global_step, best_reward)
  - Log alpha (entropy coefficient)
- Tiếp tục train từ episode tiếp theo.

### 3. Resume từ checkpoint cụ thể
```bash
python Train_Robot6.py --resume --resume-ckpt "checkpoints/sac_20251201-135028/ckpt_ep_500.pth"
```
- Load checkpoint được chỉ định.
- Tiếp tục train từ trạng thái đó.

---

## Checkpoint Directory Structure

```
checkpoints/
├── sac_20251201-100000/
│   ├── latest.pth          (update mỗi episode, file nhỏ nhất)
│   ├── best.pth            (reward cao nhất)
│   ├── ckpt_ep_50.pth      (periodic, mỗi 50 episode)
│   ├── ckpt_ep_100.pth
│   └── ...
└── sac_20251201-120000/
    ├── latest.pth
    ├── best.pth
    └── ...

runs/
├── sac_20251201-100000/    (TensorBoard logs)
│   └── events.out.tfevents.xxx
└── sac_20251201-120000/
    └── events.out.tfevents.xxx
```

---

## Checkpoint Content

Mỗi `.pth` file chứa:
```python
{
    "policy": model.state_dict(),           # Policy network weights
    "q1": model.state_dict(),               # Q-network 1 weights
    "q2": model.state_dict(),               # Q-network 2 weights
    "q1_target": model.state_dict(),        # Target Q-network 1
    "q2_target": model.state_dict(),        # Target Q-network 2
    "policy_opt": optimizer.state_dict(),   # Policy optimizer state
    "q_opt": optimizer.state_dict(),        # Q optimizer state
    "alpha_opt": optimizer.state_dict(),    # Entropy optimizer state
    "log_alpha": tensor,                    # Entropy coefficient
    "global_step": int,                     # Total steps trained
    "episode": int,                         # Episode number
    "best_reward": float,                   # Best episode reward so far
}
```

---

## Logging

### CSV Logs
Mỗi run lưu `training.csv` trong `runs/sac_YYYYMMDD-HHMMSS/`:
```
episode,global_step,episode_reward
0,50,1.234
1,100,2.345
...
```

### TensorBoard
Visualize training:
```bash
tensorboard --logdir runs/
```
Mở browser: `http://localhost:6006`

---

## Tips

1. **Backup best.pth**: Nếu muốn lưu model tốt nhất, copy `best.pth` vào nơi an toàn trước khi training tiếp.

2. **So sánh runs**: Các run khác nhau được lưu trong directory riêng, TensorBoard tự động compare.

3. **Clear old checkpoints**: Xóa thư mục cũ để tiết kiệm disk:
   ```bash
   rm -rf checkpoints/sac_20251201-100000/
   ```

4. **Monitor live**:
   ```bash
   # Terminal 1: Start training
   python Train_Robot6.py --resume
   
   # Terminal 2: Monitor TensorBoard
   tensorboard --logdir runs/
   ```

---

## Examples

### Scenario 1: Train 100 episodes, resume để train tiếp
```bash
# Day 1
python Train_Robot6.py
# ... train 100 episodes ...
# (creates checkpoints/sac_20251201-100000/)

# Day 2: Continue training
python Train_Robot6.py --resume
# (loads checkpoints/sac_20251201-100000/latest.pth)
# (continues from episode 101)
```

### Scenario 2: Load specific checkpoint
```bash
# Kiểm tra checkpoint có sẵn
ls checkpoints/sac_20251201-100000/

# Resume từ episode 250
python Train_Robot6.py --resume --resume-ckpt "checkpoints/sac_20251201-100000/ckpt_ep_250.pth"
```

### Scenario 3: Training từ best checkpoint
```bash
# Resume từ best reward episode
python Train_Robot6.py --resume --resume-ckpt "checkpoints/sac_20251201-100000/best.pth"
```

---

## Common Issues

### Q: Training không load checkpoint
```
[RESUME] Checkpoint not found, starting from scratch
```
**Cách sửa**: Kiểm tra thư mục `checkpoints/` có tồn tại hay không.

### Q: Optimizer state mismatch error
**Cách sửa**: Đảm bảo không thay đổi learning rate giữa các runs. Nếu muốn thay, hãy train từ đầu.

### Q: Out of memory khi loading checkpoint
**Cách sửa**: Giảm `batch_size` trong `train_sac()` nếu GPU memory quá ít.

---

## Future Improvement Ideas

- [ ] Load buffer state từ checkpoint (replay history)
- [ ] Multi-run comparison script
- [ ] Auto-cleanup old checkpoints (keep only last 5)
- [ ] Checkpoint metadata (obs_dim, act_dim validation)

