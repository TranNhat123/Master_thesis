# TensorBoard Logging & Real-time Monitoring Guide

## 1. Metrics Được Log

### Per-Step Metrics (ghi mỗi training step, giúp monitor chi tiết)
Được ghi vào `Loss/` và `Metrics/` groups:
- **`Loss/Q1`**: Q-network 1 MSE loss (gradient descent value)
- **`Loss/Q2`**: Q-network 2 MSE loss
- **`Loss/Policy`**: Policy gradient loss = `(alpha * log_pi - q_new).mean()`
- **`Loss/Alpha`**: Entropy coefficient loss (điều chỉnh temperature)
- **`Metrics/Temperature`**: `alpha = exp(log_alpha)` (entropy coefficient value)
- **`Metrics/LogAlpha`**: `log_alpha` (raw entropy parameter)
- **`Metrics/LogPi_Mean`**: Mean log-probability của sampled actions

### Per-Episode Metrics (ghi mỗi episode, summary)
Được ghi vào `Episode/` group:
- **`Episode/Reward`**: Total episode reward
- **`Episode/AvgAlpha`**: Trung bình alpha mixing weight từng step trong episode

---

## 2. Cách Fix TensorBoard Không Real-time

### Vấn Đề
- TensorBoard hiển thị dữ liệu cũ (vài ngày trước)
- Không update khi training đang chạy
- Hoặc load log từ cache

### Nguyên Nhân
1. **`writer.flush()` không được gọi** → dữ liệu nằm trong buffer, chưa write to disk
2. **TensorBoard cache** → cache cũ không refresh
3. **File permissions** → TensorBoard đọc file cũ

### Giải Pháp ✅

#### A. Thêm `writer.flush()` (CRITICAL)
```python
# Sau mỗi episode logging
writer.add_scalar("Episode/Reward", ep_reward, ep)
writer.add_scalar("Episode/AvgAlpha", avg_alpha, ep)
writer.flush()  # ← Force write to disk immediately
```

**Khi nào flush:**
- Mỗi episode (bây giờ training code đã làm)
- Mỗi N steps nếu muốn update frequent
- Trước khi checkpoint save

#### B. Close writer khi kết thúc training
```python
# Cuối training loop
writer.close()  # Close & flush all buffered data
env.close()
print("[DONE] Training completed!")
```

#### C. Clear TensorBoard Cache
```bash
# Kill TensorBoard server (nếu đang chạy)
pkill tensorboard
# hoặc Ctrl+C

# Xóa cache directory
rm -r ~/.tensorboard

# Restart TensorBoard
tensorboard --logdir runs/ --reload_interval 5
```

#### D. Chạy TensorBoard Với Auto-Reload
```bash
# Auto reload mỗi 5 giây (thay vì default 30s)
tensorboard --logdir runs/ --reload_interval 5

# Hoặc không cache at all (development mode)
tensorboard --logdir runs/ --reload_interval 0
```

---

## 3. TensorBoard Organization

### Group Structure
```
Loss/
├── Q1         (per-step)
├── Q2         (per-step)
├── Policy     (per-step)
└── Alpha      (per-step)

Metrics/
├── Temperature (per-step)
├── LogAlpha   (per-step)
└── LogPi_Mean (per-step)

Episode/
├── Reward     (per-episode)
└── AvgAlpha   (per-episode)
```

### Cách xem trong TensorBoard
1. Mở `http://localhost:6006`
2. Tab **SCALARS** → expand tất cả groups
3. Compare multiple runs bằng checkboxes
4. Hover để xem exact values
5. Zoom/pan để inspect details

---

## 4. Usage Guide

### Step 1: Chạy training với logging
```bash
python Train_Robot6.py
# hoặc resume
python Train_Robot6.py --resume
```

Output:
```
Obs Dim: 41, Act Dim: 7
Ep 0: Reward=1234.56, Avg Alpha=0.42
Ep 1: Reward=1345.67, Avg Alpha=0.48
...
```

### Step 2: Chạy TensorBoard (terminal khác)
```bash
cd d:\Data_save_obsidian\Hoc_tap\Lab\Master\ thesis\PyBullet_ver_2
tensorboard --logdir runs/ --reload_interval 5
```

Output:
```
W1201 15:30:00.123456 TensorBoard 2.14.0 at http://localhost:6006/
```

### Step 3: Mở browser
```
http://localhost:6006
```

### Step 4: Monitor training real-time
- Tabs: **SCALARS** (chính), GRAPHS, DISTRIBUTIONS
- **SCALARS**: 
  - Left sidebar: Select metrics to display
  - "Loss/" group: Training convergence
  - "Episode/" group: Performance over episodes
  - "Metrics/" group: Temperature/entropy progression

---

## 5. Interpreting Metrics

### Q-Loss Trends (Loss/Q1, Loss/Q2)
- **Giảm theo thời gian** → Q-network học tốt
- **Tăng lên & oscillate** → Learning rate quá cao, hoặc overfitting

```
Expected:
  Episode 0:   Q1_loss = ~100
  Episode 100: Q1_loss = ~10
  Episode 500: Q1_loss = ~1
```

### Policy Loss (Loss/Policy)
- **Âm & giảm magnitude** → Policy tìm được high-reward actions
- **Gần 0** → Policy đạo hàm ~0, learning stagnate

### Temperature (Metrics/Temperature)
- **Giảm theo thời gian** → Agent tập trung (exploitation)
- **Ổn định** → Entropy tuning converged
- **Quá cao (>1)** → Agent quá random, cần điều chỉnh target_entropy

### Episode Reward (Episode/Reward)
- **Tăng & ổn định** → Training success
- **Đồng bằng** → Stuck, learning không tiến bộ
- **Oscillate nhiều** → Instability, check learning rates

---

## 6. Advanced TensorBoard Tips

### Compare Multiple Runs
```bash
tensorboard --logdir runs/
```

TensorBoard tự động detect tất cả subdirectories:
```
runs/
├── sac_20251201-152441  (Run 1)
├── sac_20251201-160000  (Run 2)
└── sac_20251201-180000  (Run 3)
```

Giao diện: Select runs & metrics để compare (e.g., Loss/Q1 across 3 runs)

### Download Data as CSV
- Click **⚙️ (Settings)** → **Download data as JSON**
- Parse JSON để export to CSV/Excel

### Custom Smoothing
- Slider "Smoothing" → làm smooth curves (default 0)
- Tăng lên để xóa noise

### X-axis Options
- **STEPS** (default): x=global_step (mịn)
- **RELATIVE**: x=relative time
- **WALL**: x=wall-clock time

---

## 7. Troubleshooting

### Q: TensorBoard không load runs
```
Attempted to create localization index at...
```

**Cách sửa:**
```bash
# Đảm bảo có files .pth hoặc .csv trong runs/
ls runs/sac_*/
# Output: events.out.tfevents.xxx, training.csv ✓

# Restart TensorBoard
tensorboard --logdir runs/ --reload_interval 0
```

### Q: TensorBoard load cũ (cache)
```bash
# Kill server
pkill tensorboard

# Clear cache
rm -r ~/.tensorboard

# Restart with fresh cache
tensorboard --logdir runs/
```

### Q: Graphs không update real-time
```bash
# Thêm --reload_interval 5 (refresh mỗi 5s)
tensorboard --logdir runs/ --reload_interval 5

# Hoặc trong training code, thêm writer.flush()
writer.add_scalar(...)
writer.flush()  # ← Force write
```

### Q: Port 6006 đã bị dùng
```bash
# Dùng port khác
tensorboard --logdir runs/ --port 6007
# Mở: http://localhost:6007
```

---

## 8. Quick Reference

### Training Loop Metrics
```python
# Per-step (mỗi training update)
writer.add_scalar("Loss/Q1", q1_loss.item(), global_step)
writer.add_scalar("Loss/Q2", q2_loss.item(), global_step)
writer.add_scalar("Loss/Policy", policy_loss.item(), global_step)
writer.add_scalar("Loss/Alpha", alpha_loss.item(), global_step)
writer.add_scalar("Metrics/Temperature", alpha_sac.item(), global_step)
writer.add_scalar("Metrics/LogAlpha", log_alpha.item(), global_step)
writer.add_scalar("Metrics/LogPi_Mean", log_pi_new.mean().item(), global_step)

# Per-episode
writer.add_scalar("Episode/Reward", ep_reward, ep)
writer.add_scalar("Episode/AvgAlpha", avg_alpha, ep)
writer.flush()  # ← CRITICAL for real-time
```

### Viewing
```bash
tensorboard --logdir runs/ --reload_interval 5
# Browser: http://localhost:6006
```

---

## 9. Data Storage Locations

```
runs/sac_20251201-152441/
├── events.out.tfevents.1764577481.LAPTOP-7MVA4582.3572.0  (TensorBoard binary)
├── events.out.tfevents.1764577500.LAPTOP-7MVA4582.18832.0 (if resumed)
└── training.csv                                             (backup text format)

checkpoints/sac_20251201-152441/
├── latest.pth           (updated mỗi episode)
├── best.pth             (best reward only)
└── ckpt_ep_*.pth        (periodic, mỗi 50 episodes)
```

---

## 10. Summary

| Action | Command / Code |
|--------|----------------|
| **Log metrics** | `writer.add_scalar(...)` |
| **Force write** | `writer.flush()` |
| **Close logging** | `writer.close()` |
| **View TensorBoard** | `tensorboard --logdir runs/ --reload_interval 5` |
| **Clear cache** | `rm -r ~/.tensorboard` |
| **Compare runs** | Select multiple in TensorBoard UI |
| **Fix real-time** | Add `writer.flush()` + `--reload_interval 5` |

