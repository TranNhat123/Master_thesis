import numpy as np
import pybullet as p
import pybullet_data
from itertools import product
import time

# ==== Cấu hình lưới ====
x_linspaces = np.arange(-300, 601, 100)
y_linspaces = np.arange(-600, 601, 100)
z_linspaces = np.arange(0, 1001, 100)

# ==== Hàm tạo quả cầu để kéo ====
def create_drag_only_sphere(position=[0.5, 0, 0.5], radius=0.02, color=[0, 0, 1, 1]):
    visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
    collision = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    return p.createMultiBody(0.0001, collision, visual, position)

# ==== Hàm tìm các điểm trên grid nằm trong bán kính ====
def points_within_sphere(center, radius,
                         x_grid=x_linspaces,
                         y_grid=y_linspaces,
                         z_grid=z_linspaces):
    cx, cy, cz = center
    x_sel = x_grid[(x_grid >= cx - radius) & (x_grid <= cx + radius)]
    y_sel = y_grid[(y_grid >= cy - radius) & (y_grid <= cy + radius)]
    z_sel = z_grid[(z_grid >= cz - radius) & (z_grid <= cz + radius)]
    X, Y, Z = np.meshgrid(x_sel, y_sel, z_sel, indexing='ij')
    d2 = (X - cx)**2 + (Y - cy)**2 + (Z - cz)**2
    mask = d2 <= radius**2
    pts = np.vstack((X[mask], Y[mask], Z[mask])).T
    return pts.tolist()

# ==== Khởi tạo PyBullet ====
p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")

# ==== Tạo khối hộp hiển thị ====
size_m = 0.05
half = size_m / 2.0
vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half] * 3, rgbaColor=[0, 1, 0, 0.1])

Objects = np.zeros((len(x_linspaces), len(y_linspaces), len(z_linspaces)), dtype=int)

# Ánh xạ nhanh giá trị tọa độ -> index
x_idx_map = {v: i for i, v in enumerate(x_linspaces)}
y_idx_map = {v: i for i, v in enumerate(y_linspaces)}
z_idx_map = {v: i for i, v in enumerate(z_linspaces)}

for z in z_linspaces:
    for y in y_linspaces:
        for x in x_linspaces:
            obj_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=vis,
                basePosition=[x / 1000.0, y / 1000.0, z / 1000.0]
            )
            ix = x_idx_map[x]
            iy = y_idx_map[y]
            iz = z_idx_map[z]
            Objects[ix, iy, iz] = obj_id

# ==== Tải robot ====
Robot_1 = p.loadURDF("./2_robot/URDF_file_2/urdf/6_Dof.urdf", [0, 0, 0], useFixedBase=True)
Robot_2 = p.loadURDF("./2_robot/Robot/urdf/5_Dof.urdf", [0.8, 0, 0], useFixedBase=True)

p.resetBasePositionAndOrientation(Robot_1, [0, 0, 0], p.getQuaternionFromEuler([-np.pi / 2, 0, 0]))
p.resetBasePositionAndOrientation(Robot_2, [0.8, 0, 0], p.getQuaternionFromEuler([-np.pi / 2, 0, 0]))

# ==== Đặt góc khớp robot ban đầu ====
initial_joint_angles_6_dof = [0, np.pi / 2, 0, 0, 0, 0]
for i, angle in enumerate(initial_joint_angles_6_dof):
    p.resetJointState(Robot_1, i, targetValue=angle)

initial_joint_angles_5_dof = [np.pi, np.pi / 2, -np.pi / 2, np.pi / 2, 0]
for i, angle in enumerate(initial_joint_angles_5_dof):
    p.resetJointState(Robot_2, i, targetValue=angle)

# ==== Tạo quả cầu kéo được ====
sphere_id = create_drag_only_sphere()

# ==== Vòng lặp mô phỏng ====
highlighted_objects = set()

while True:
    pos, orn = p.getBasePositionAndOrientation(sphere_id)
    collisions = points_within_sphere(np.array(pos) * 1000, radius=50)

    # Reset màu cũ
    for obj_id in highlighted_objects:
        p.changeVisualShape(obj_id, -1, rgbaColor=[0, 1, 0, 0.1])
    highlighted_objects.clear()

    # Đánh dấu vùng bị va chạm
    for cx, cy, cz in collisions:
        try:
            ix = x_idx_map[cx]
            iy = y_idx_map[cy]
            iz = z_idx_map[cz]
            obj_id = Objects[ix, iy, iz]
            p.changeVisualShape(obj_id, -1, rgbaColor=[1, 0, 0, 0.5])
            highlighted_objects.add(obj_id)
        except KeyError:
            continue
    # p.resetBasePositionAndOrientation(sphere_id, [pos[0], pos[1], pos[2]], orn)
    p.stepSimulation()
    time.sleep(1. / 240.)
