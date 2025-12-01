# Giải Thích Cấu Trúc Thư Mục `runs`

## Tổng Quan

Thư mục `runs` chứa **TensorBoard logs** và **training data** cho mỗi training run:

```
runs/
├── sac_20251201-152441/          (Run #1, timestamp: 2025-12-01 15:24:41)
│   ├── events.out.tfevents.1764577481.LAPTOP-7MVA4582.3572.0
│   ├── events.out.tfevents.1764577500.LAPTOP-7MVA4582.18832.0
│   └── training.csv
└── sac_20251201-160000/          (Run #2, nếu có)
    ├── events.out.tfevents.xxx
    └── training.csv
```

---

## 1. Files & Vai Trò

### A. `events.out.tfevents.XXXXX`
- **Định dạng**: TensorBoard Event Format (binary, protobuf)
- **Tạo bởi**: `SummaryWriter` trong PyTorch/TensorFlow
- **Nội dung**: 
  - Scalar metrics: `loss`, `reward`, `alpha`, `epsilon`, etc.
  - Histograms: Weight distributions
  - Images: Visualizations
  - Custom tags từ `writer.add_*()` methods
- **Cách tạo**:
  ```python
  writer = SummaryWriter(log_dir="runs/sac_20251201-152441")
  writer.add_scalar("Reward", ep_reward, ep)
  writer.add_scalar("AvgAlpha", avg_alpha, ep)
  ```

### B. `training.csv`
- **Định dạng**: Comma-Separated Values (text, human-readable)
- **Tạo bởi**: Python `csv` module
- **Nội dung**: 
  ```
  episode,global_step,episode_reward
  0,865,2465.073829
  1,1785,2616.504509
  2,3285,2768.254401
  ```
- **Mục đích**: 
  - Dữ liệu thô, dễ parse
  - Backup nếu TensorBoard file corrupted
  - Có thể dùng Pandas/Matplotlib vẽ lại

---

## 2. Filename Convention: `events.out.tfevents.XXXXX`

Format: `events.out.tfevents.[TIMESTAMP].[HOSTNAME].[PID].[SUFFIX]`

Ví dụ: `events.out.tfevents.1764577481.LAPTOP-7MVA4582.3572.0`

| Phần | Giá Trị | Ý Nghĩa |
|-----|--------|--------|
| `events.out.tfevents` | Fixed | TensorBoard event file marker |
| `1764577481` | Timestamp | Unix epoch (lúc khởi tạo SummaryWriter) |
| `LAPTOP-7MVA4582` | Hostname | Tên máy tính |
| `3572` | PID | Process ID (Python process) |
| `0` | Suffix | Counter (0, 1, 2... nếu có multiple files) |

### Tại sao có multiple event files?
- TensorBoard tạo **event file mới** mỗi lần `SummaryWriter` được khởi tạo lại
- Nếu training bị interrupt & resume, sinh file mới: `events.out.tfevents.1764577500.LAPTOP-7MVA4582.18832.0`
- TensorBoard tự động merge tất cả files khi visualize

---

## 3. Cách Xem TensorBoard Logs

### Chạy TensorBoard server:
```bash
tensorboard --logdir runs/
```

### Mở browser:
```
http://localhost:6006
```

### Giao diện TensorBoard:
- **SCALARS tab**: Vẽ graphs cho từng metric (`Reward`, `AvgAlpha`, etc.)
- **DISTRIBUTIONS tab**: Histogram của weights/gradients
- **GRAPHS tab**: Model architecture (nếu có)
- **TEXT tab**: Custom text logs
- **IMAGES tab**: Visualizations (nếu có)

### Compare multiple runs:
TensorBoard tự động detect tất cả subdirectories trong `runs/` và cho phép compare:
```
runs/
├── sac_20251201-152441    ← Run 1 (baseline)
├── sac_20251201-160000    ← Run 2 (with new fix)
└── sac_20251201-180000    ← Run 3 (another experiment)
```

TensorBoard sẽ vẽ 3 graphs trên cùng plot để so sánh.

---

## 4. Data Flow: Training → Logs

```
Training Loop (Train_Robot6.py)
    ↓
for ep in range(num_episodes):
    ep_reward = ...
    
    # Write to TensorBoard
    writer.add_scalar("Reward", ep_reward, ep)        ← → events.out.tfevents.XXXXX
    writer.add_scalar("AvgAlpha", avg_alpha, ep)
    
    # Write to CSV
    with open(csv_path, "a") as f:
        w.writerow([ep, global_step, ep_reward])     ← → training.csv
```

---

## 5. Ví Dụ Nội Dung Files

### training.csv (Human-readable):
```csv
episode,global_step,episode_reward
0,865,2465.073829032347
1,1785,2616.504509303316
2,3285,2768.254401713125
```

**Parse bằng Pandas:**
```python
import pandas as pd
df = pd.read_csv("runs/sac_20251201-152441/training.csv")
print(df)
print(f"Best reward: {df['episode_reward'].max()}")
print(f"Average reward: {df['episode_reward'].mean()}")
```

### events.out.tfevents.XXXXX (Binary, TensorBoard format):
- Không thể mở bằng text editor
- Chỉ xem được qua TensorBoard hoặc `tensorflow.compat.tensorflow_io`
- Lưu trữ dense, efficient (compression hỗ trợ)

---

## 6. Log Directory Reuse (Resume Training)

Khi resume training (`--resume`):
1. **Tìm checkpoint** từ `checkpoints/sac_YYYYMMDD-HHMMSS/latest.pth`
2. **Reuse run name** từ checkpoint: `sac_YYYYMMDD-HHMMSS`
3. **Reuse log directory**: `runs/sac_YYYYMMDD-HHMMSS/`
4. **Append logs** vào CSV (không overwrite)
5. **Tạo new event file** với timestamp mới khi resume

**Kết quả:**
```
runs/sac_20251201-152441/
├── events.out.tfevents.1764577481.LAPTOP-7MVA4582.3572.0    (Episode 0-5)
├── events.out.tfevents.1764577500.LAPTOP-7MVA4582.18832.0   (Episode 5-10, resumed)
└── training.csv                                               (Episodes 0-10, appended)
```

TensorBoard merge tất cả event files → single continuous graph

---

## 7. Cleanup & Management

### Xóa old runs (giải phóng disk):
```bash
# Xóa run cũ
rm -r runs/sac_20251201-100000/
rm -r checkpoints/sac_20251201-100000/
```

### Backup important runs:
```bash
# Backup best checkpoint + logs
mkdir backup_run_001
cp checkpoints/sac_20251201-152441/best.pth backup_run_001/
cp -r runs/sac_20251201-152441/ backup_run_001/
```

### Extract data từ CSV:
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("runs/sac_20251201-152441/training.csv")

plt.figure(figsize=(10, 5))
plt.plot(df['episode'], df['episode_reward'], label="Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.savefig("training_curve.png")
```

---

## 8. Troubleshooting

### Q: TensorBoard không hiển thị gì
```
tensorboard --logdir runs/
# Output: Loading... (nhưng không có graphs)
```

**Cách sửa:**
- Kiểm tra `events.out.tfevents.*` files tồn tại
- Đảm bảo `writer.add_scalar()` được gọi trong training loop
- Flush writer: `writer.flush()`

### Q: Events file bị corrupted
```bash
# Xóa event files, giữ CSV
rm runs/sac_*/events.out.tfevents.*

# CSV data vẫn an toàn, dùng pandas để vẽ lại
```

### Q: Muốn merge runs thành 1
```bash
# Copy logs vào 1 directory
mkdir runs/combined
cp runs/sac_20251201-*/training.csv runs/combined/
cp runs/sac_20251201-*/events.out.tfevents.* runs/combined/
```

---

## 9. Summary

| File | Format | Mục Đích | Tạo bởi |
|------|--------|---------|---------|
| `events.out.tfevents.XXXXX` | Binary (protobuf) | TensorBoard visualization | `SummaryWriter.add_*()` |
| `training.csv` | Text (CSV) | Data backup, easy parse | `csv.writer` |

**Sử dụng:**
- **TensorBoard**: Real-time visualization, compare runs
- **CSV**: Data export, custom analysis, backup

**Resume training:**
- Event files append (tạo file mới)
- CSV append (hàng mới)
- TensorBoard auto-merge → seamless view

