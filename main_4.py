import numpy as np
import time
import pybullet as p
import pybullet_data
from function_robot.Coordinate_sixdof import Coordinate_sixdof
from function_robot.Jtool_sixdof_function import Jtool_sixdof_function
import torch 
import torch.nn as nn
import torch.nn.functional as F
import os
from function_robot.Inver_kine_function import Inverse_kinematic
from function_robot.MPPI_function import mppi

def Take_cur_position_index(joint_position):
    theta = joint_position[0:3]
    theta_degree = theta*180/np.pi
    theta_index_round = np.round((theta_degree + 180)/5)
    theta_index_not_round = np.round((theta_degree + 180)/5, 3)
    return theta_index_round.astype(int), theta_index_not_round

def Predict_task_vel_to_convert_joint_vel(joint_position_local, theta_dot, vmax):
    t1 = joint_position_local[0]
    t2 = joint_position_local[1]
    t3 = joint_position_local[2]
    t4 = joint_position_local[3]
    t5 = joint_position_local[4]
    t6 = joint_position_local[5]
    Jtool_sixdof = Jtool_sixdof_function(t1, t2, t3, t4, t5, t6)
    Jtool_sixdof = Jtool_sixdof[0:3, 0:3]
    v_att_tool_position_sixdof = np.dot(Jtool_sixdof, theta_dot.T)
    v_att = vmax * (v_att_tool_position_sixdof / np.linalg.norm(v_att_tool_position_sixdof)) 

    theta_v_tool_sixdof = np.dot(np.linalg.inv(Jtool_sixdof), v_att)

    theta_v_sixdof = np.array([theta_v_tool_sixdof[0],theta_v_tool_sixdof[1], 
                               theta_v_tool_sixdof[2], 0, 0])

    for i in range(5):
        limit = np.pi / 10
        theta_v_sixdof[i] = np.clip(theta_v_sixdof[i], -limit, limit)

    theta_dot = np.array([theta_v_sixdof[0],theta_v_sixdof[1], theta_v_sixdof[2], 
                 0, theta_v_sixdof[3], theta_v_sixdof[4]])
    return theta_dot

def generate_combined_obstacle(obstacle_data, indices):
    selected_grids = obstacle_data[indices]
    combined_grid = np.min(selected_grids, axis = 0)  
    return combined_grid

def create_drag_only_sphere(position=[0.5, 0, 0.5], radius=0.05, color=[0, 0, 1, 1]):
    visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
    collision = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    return p.createMultiBody(0.0001, collision, visual, position)

def Take_joint_position():
    positions = []
    for i in range(6):
        joint_value = p.getJointState(Robot_1, i)[0]
        if i == 4:
            joint_value = -joint_value  # đổi dấu nếu là khớp số 4
        positions.append(joint_value)
    return positions


def Set_joint_velocity(theta_v_sixdof):
    for i in range(6):
        velocity = -theta_v_sixdof[i] if i == 4 else theta_v_sixdof[i]
        # velocity = theta_v_sixdof[i]
        p.setJointMotorControl2(Robot_1, i, controlMode=p.VELOCITY_CONTROL, targetVelocity=velocity, force=500)

def Get_theta_dot(x_target, y_target, z_target, vmax, joint_position_local):
    global Flag_take_longest_distance, att_position_max, att_Rx, att_Rz
    t1, t2, t3, t4, t5, t6 = joint_position_local
    Jtool_sixdof = Jtool_sixdof_function(t1, t2, t3, t4, t5, t6)
    Goal_coordinate_sixdof = np.array([float(x_target), float(y_target), float(z_target)])
    coordinate_sixdof = Coordinate_sixdof(t1, t2, t3, t4, t5)
    # x, y, z = coordinate_sixdof[-2, :3]
    pos = p.getLinkState(bodyUniqueId = Robot_1, linkIndex = 5)[0]
    x, y, z = pos
    att_sixdof = [x, y, z] - Goal_coordinate_sixdof

    if Flag_take_longest_distance == 0:
        att_position_max = np.linalg.norm(att_sixdof)
        Flag_take_longest_distance = 1
    
    Rx = 0
    Rz = 0

    Rx_now = t2 + t3 + t5
    Rz_now = -t1 + t6
    att_Rx = Rx_now - Rx
    att_Rz = Rz_now -Rz

    percent = 1 - np.linalg.norm(att_sixdof) / att_position_max
    if percent <= 0.1:
        v_position_sixdof = max(vmax * percent / 0.1, vmax * 0.5)
    elif percent < 0.9:
        v_position_sixdof = vmax
    else:
        v_position_sixdof = vmax * (1 - percent) / 0.1

    v_att_tool_position_sixdof = -v_position_sixdof * (att_sixdof / np.linalg.norm(att_sixdof))
    v_att_tool_orientation_sixdof = -50*np.array([att_Rx, att_Rz])
    c_tool = np.hstack((v_att_tool_position_sixdof, v_att_tool_orientation_sixdof))
    theta_v_sixdof = np.dot(np.linalg.pinv(Jtool_sixdof), c_tool.T)

    for i in range(5):
        limit = np.pi / 10
        theta_v_sixdof[i] = np.clip(theta_v_sixdof[i], -limit, limit)

    theta_dot = np.array([theta_v_sixdof[0],theta_v_sixdof[1], theta_v_sixdof[2], 
                 0, theta_v_sixdof[3], theta_v_sixdof[4]])
    return theta_dot, np.linalg.norm(att_sixdof), att_Rx, att_Rz

# Các linspace theo từng chiều
x_linspaces = np.arange(-300, 600 + 50, step=50)
y_linspaces = np.arange(-600, 600 + 50, step=50)
z_linspaces = np.arange(   0,1000 + 50, step=50)

def points_within_sphere(center, radius,
                         x_grid=x_linspaces,
                         y_grid=y_linspaces,
                         z_grid=z_linspaces):
    """
    Trả về danh sách các điểm (x, y, z) trên grid sao cho
    khoảng cách từ mỗi điểm đến center <= radius.
    
    center: iterable [cx, cy, cz]
    radius: bán kính
    x_grid, y_grid, z_grid: 1D arrays định nghĩa grid
    """
    cx, cy, cz = center
    # (1) Lọc các giá trị grid nằm trong khoảng [c - r, c + r] trên mỗi chiều
    x_sel = x_grid[(x_grid >= cx - radius) & (x_grid <= cx + radius)]
    y_sel = y_grid[(y_grid >= cy - radius) & (y_grid <= cy + radius)]
    z_sel = z_grid[(z_grid >= cz - radius) & (z_grid <= cz + radius)]
    # (2) Tạo lưới 3D của các giá trị được chọn
    X, Y, Z = np.meshgrid(x_sel, y_sel, z_sel, indexing='ij')
    # (3) Tính khoảng cách đến center
    d2 = (X - cx)**2 + (Y - cy)**2 + (Z - cz)**2
    # (4) Chọn những điểm có d2 <= radius^2
    mask = d2 <= radius**2
    pts = np.vstack((X[mask], Y[mask], Z[mask])).T
    return pts.tolist()


p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
p.loadURDF("plane.urdf")

Robot_1 = p.loadURDF("./2_robot/URDF_file_2/urdf/6_Dof.urdf", [0, 0, 0], useFixedBase=True)
Robot_2 = p.loadURDF("./2_robot/Robot/urdf/5_Dof.urdf", [1, 0, 0], useFixedBase=True)

p.resetBasePositionAndOrientation(Robot_1, [0, 0, 0], p.getQuaternionFromEuler([-np.pi/2, 0, 0]))
p.resetBasePositionAndOrientation(Robot_2, [1, 0, 0], p.getQuaternionFromEuler([-np.pi/2, 0, 0]))

initial_joint_angles = [0, np.pi/2, 0, 0, 0, 0]
for i, angle in enumerate(initial_joint_angles):
    p.resetJointState(Robot_1, i, targetValue=angle)

sphereId = p.createMultiBody(baseMass=0, baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[1, 0, 0, 1]), basePosition=[0, 0, 0])
t, start_time = 0, time.time()
Flag_take_longest_distance = 0
att_position_max = 0
att_Rx = 0 
att_Rz = 0
Px = float(input('Nhap gia tri cua Px: '))
Py = float(input('Nhap gia tri cua Py: '))
Pz = float(input('Nhap gia tri cua Pz: '))
Goal = Inverse_kinematic(Px*1000, Py*1000, Pz*1000)[0]
print(Goal)
Goal_round, Goal_not_round = Take_cur_position_index(np.array(Goal))


Flag_mode_simulation = int(input('0 - real time | 1 - non-real time: '))
t_sampling = 1.0 / 240.0
if Flag_mode_simulation == 0:
    p.setRealTimeSimulation(Flag_mode_simulation == 1)
elif Flag_mode_simulation == 1:
    p.setRealTimeSimulation(Flag_mode_simulation == 0)

# Tạo sphere
sphere_id = create_drag_only_sphere()

################################################################################################################
################################################################################################################
def decode_index(index):
    t1 = index // (3**2)
    remainder = index % (3**2)
    t2 = remainder // 3
    t3 = remainder % 3
    return t1, t2, t3

num = [-1, 0, 1]
Vectors_to_find = []
for i in range(0, 27):
    index_1, index_2, index_3 = decode_index(i)
    Vector = [num[index_1], num[index_2], num[index_3]]
    Vectors_to_find.append(Vector)

Vectors_to_find = np.array(Vectors_to_find)
# Tính norm, tránh chia cho 0
norms = np.linalg.norm(Vectors_to_find, axis=1)
norms[norms == 0] = 1  # tránh chia cho 0
Vector_set = Vectors_to_find / norms[:, np.newaxis]
print(Vector_set)
###############################################################################################################
radius = 1 
dt = 1/240
while True:
    p.removeAllUserDebugItems()
    joint_position = np.array(Take_joint_position())
    joint_position = (joint_position + np.pi) % (2 * np.pi) - np.pi


    ########################################## Controller #######################################################################################
    ########################################## Controller #######################################################################################
    ########################################## Controller #######################################################################################
    vmax = 0.5

    theta_dot, Position_error, att_Rx, att_Rz = Get_theta_dot(x_target=Px, y_target=Py, z_target=Pz, 
                                            vmax= vmax, joint_position_local= joint_position)


    if Position_error < 0.0001 and np.abs(att_Rx) < 0.001 and np.abs(att_Rz) < 0.001:
        theta_dot = np.zeros(6)
            # Px = float(input('Nhap gia tri cua Px: '))
            # Py = float(input('Nhap gia tri cua Py: '))
            # Pz = float(input('Nhap gia tri cua Pz: '))
    Set_joint_velocity(theta_dot) 
    p.resetBasePositionAndOrientation(sphereId, [Px, Py, Pz], [0, 0, 0, 1])
    if Flag_mode_simulation == 1:
        p.stepSimulation()