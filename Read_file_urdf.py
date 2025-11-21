import pybullet as p
import pybullet_data
import numpy as np
import time
from itertools import product

# Khởi động
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

# Kích thước khối hộp
size_mm = 50
size_m = size_mm / 1000.0
half = size_m / 2.0

# Tạo 1 lần collision và visual shape để dùng chung
col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half]*3)
vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[half]*3, rgbaColor=[0, 1, 0, 1])

# Lưới tọa độ GIỚI HẠN để tránh quá tải
x_vals = np.arange(-300, 300 + 50, 50)
y_vals = np.arange(-300, 300 + 50, 50)
z_vals = np.arange(0, 500 + 50, 50)

positions = product(x_vals, y_vals, z_vals)

# Tạo tất cả các box (rất nhanh)
for x, y, z in positions:
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col_shape,
        baseVisualShapeIndex=vis_shape,
        basePosition=[x / 1000.0, y / 1000.0, z / 1000.0]
    )

print("All boxes created.")

# Loop giữ cửa sổ
while True:
    p.stepSimulation()
    time.sleep(1. / 240.)
