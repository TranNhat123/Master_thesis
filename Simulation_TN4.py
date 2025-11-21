import numpy as np
import pybullet as p
import pybullet_data
from function_robot.Coordinate_sixdof import Coordinate_sixdof
from function_robot.Jtool_sixdof_function import Jtool_sixdof_function
from function_robot.Jtool_fanuc_function import Jtool_fanuc_function
from function_robot.Coordinate_fanuc import Coordinate_fanuc
import time 

def create_colored_boxes(num_green=5,
                         num_red=5,
                         size_mm=50,
                         spacing=0.1,
                         base_height=None):
    """
    Tạo các hộp BOX (mass=0) trong PyBullet:
      - num_green  : số hộp xanh
      - num_red    : số hộp đỏ
      - size_mm    : kích thước cạnh hộp (mm)
      - spacing    : khoảng cách giữa tâm các hộp (m)
      - base_height: chiều cao (Z) đặt đáy hộp so với mặt đất (m). 
                     Nếu None, tự đặt = size_m/2

    Sau khi gọi, bạn sẽ có num_green hộp xanh ở hàng đầu,
    num_red hộp đỏ ở hàng sau.
    """

    # chuyển mm → m, và tính nửa kích thước
    size_m = size_mm / 1000.0
    half = size_m / 2.0
    if base_height is None:
        base_height = half

    # chuẩn bị shape và visual reuse
    box_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half, half, half])
    green_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half, half, half], rgbaColor=[0, 1, 0, 1])
    red_vis   = p.createVisualShape(p.GEOM_BOX, halfExtents=[half, half, half], rgbaColor=[1, 0, 0, 1])

    total = num_green + num_red
    max_count = max(num_green, num_red)

    for idx in range(total):
        # chọn màu và hàng
        if idx < num_green:
            vis_id = green_vis
            row, col = 0, idx
        else:
            vis_id = red_vis
            row, col = 1, idx - num_green

        # tính toạ độ x,y,z
        x = col * spacing - ((max_count - 1) * spacing) / 2.0
        y = row * spacing * 1.5 + 0.5
        z = base_height

        p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=box_col,
            baseVisualShapeIndex=vis_id,
            basePosition=[x, y, z]
        )


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

def Take_joint_position_6_dof():
    positions = []
    for i in range(6):
        joint_value = p.getJointState(Robot_1, i)[0]
        if i == 4:
            joint_value = -joint_value  # đổi dấu nếu là khớp số 4
        positions.append(joint_value)
    return positions

def Take_joint_position_5_dof():
    positions = []
    for i in range(5):
        joint_value = p.getJointState(Robot_2, i)[0]
        positions.append(joint_value)
    return positions


def Set_joint_velocity_6_dof(theta_v_sixdof):
    for i in range(6):
        velocity = -theta_v_sixdof[i] if i == 4 else theta_v_sixdof[i]
        # velocity = theta_v_sixdof[i]
        p.setJointMotorControl2(Robot_1, i, controlMode=p.VELOCITY_CONTROL, targetVelocity=velocity, force=500)

def Set_joint_velocity_5_dof(theta_v_sixdof):
    for i in range(5):
        velocity = theta_v_sixdof[i]
        p.setJointMotorControl2(Robot_2, i, controlMode=p.VELOCITY_CONTROL, targetVelocity=velocity, force=500)

def Get_theta_dot_6_dof(x_target, y_target, z_target, vmax, joint_position_local):
    global Flag_take_longest_distance_6_dof, att_position_max_6_dof
    t1, t2, t3, t4, t5, t6 = joint_position_local
    Jtool_sixdof = Jtool_sixdof_function(t1, t2, t3, t4, t5, t6)
    Goal_coordinate_sixdof = np.array([float(x_target), float(y_target), float(z_target)])
    coordinate_sixdof = Coordinate_sixdof(t1, t2, t3, t4, t5)
    # x, y, z = coordinate_sixdof[-2, :3]
    pos = p.getLinkState(bodyUniqueId = Robot_1, linkIndex = 5)[0]
    x, y, z = pos
    att_sixdof = [x, y, z] - Goal_coordinate_sixdof

    if Flag_take_longest_distance_6_dof == 0:
        att_position_max_6_dof = np.linalg.norm(att_sixdof)
        Flag_take_longest_distance_6_dof = 1
    
    Rx = 0
    Rz = 0

    Rx_now = t2 + t3 + t5
    Rz_now = -t1 + t6
    att_Rx = Rx_now - Rx
    att_Rz = Rz_now -Rz

    percent = 1 - np.linalg.norm(att_sixdof) / att_position_max_6_dof
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


def Get_theta_dot_5_dof(x_target, y_target, z_target, vmax, joint_position_local):
    global Flag_take_longest_distance_5_dof, att_position_max_5_dof
    d_5 = 0.08
    t1, t2, t3, t4, t5 = joint_position_local
    Jtool_sixdof = Jtool_fanuc_function(t1, t2, t3, t4, t5, d_5)
    Goal_coordinate_sixdof = np.array([float(x_target), float(y_target), float(z_target)])
    pos = p.getLinkState(bodyUniqueId = Robot_2, linkIndex = 4)[0]
    x, y, z = pos
    att_5dof = [x, y, z] - Goal_coordinate_sixdof

    if Flag_take_longest_distance_5_dof == 0:
        att_position_max_5_dof = np.linalg.norm(att_5dof)
        Flag_take_longest_distance_5_dof = 1
    
    Rx = 0
    Rz = 0

    Rx_now = t2 + t3 + t4
    print(Rx_now)
    Rz_now = -t1 + t5
    att_Rx = Rx_now - Rx
    att_Rz = Rz_now -Rz

    percent = 1 - np.linalg.norm(att_5dof) / att_position_max_5_dof
    if percent <= 0.1:
        v_position_sixdof = max(vmax * percent / 0.1, vmax * 0.5)
    elif percent < 0.9:
        v_position_sixdof = vmax
    else:
        v_position_sixdof = vmax * (1 - percent) / 0.1

    v_att_tool_position_5dof = -v_position_sixdof * (att_5dof / np.linalg.norm(att_5dof))
    v_att_tool_orientation_5dof = -10*np.array([att_Rx, att_Rz])
    c_tool = np.hstack((v_att_tool_position_5dof, v_att_tool_orientation_5dof))
    theta_v_sixdof = np.dot(np.linalg.pinv(Jtool_sixdof), c_tool.T)

    for i in range(5):
        limit = np.pi / 10
        theta_v_sixdof[i] = np.clip(theta_v_sixdof[i], -limit, limit)

    theta_dot = np.array([theta_v_sixdof[0],theta_v_sixdof[1], theta_v_sixdof[2], 
                 theta_v_sixdof[3], theta_v_sixdof[4]])
    return theta_dot, np.linalg.norm(att_5dof), att_Rx, att_Rz

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

def Convert_position_5_dof_base(Px, Py, Pz):
### Hàm convert này mục đích để ta chuyển giao dễ dàng giữa hệ trục toạ độ global và hệ trục toạ độ của robot. 
### Không cần quá lấn cấn về nó, khi tính error về vị trí và hướng, chỉ cần cùng hệ quy chiếu là được, mục tiêu là tìm ra đúng hướng di chuyển đến đích

    Px_convert = Px - 1
    return Px_convert, Py, Pz
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
p.loadURDF("plane.urdf")

Robot_1 = p.loadURDF("./2_robot/URDF_file_2/urdf/6_Dof.urdf", [0, 0, 0], useFixedBase=True)
Robot_2 = p.loadURDF("./2_robot/Robot/urdf/5_Dof.urdf", [0, 0, 0], useFixedBase=True)

p.resetBasePositionAndOrientation(Robot_1, [0, 0, 0], p.getQuaternionFromEuler([-np.pi/2, 0, 0]))
p.resetBasePositionAndOrientation(Robot_2, [0.8, 0, 0], p.getQuaternionFromEuler([-np.pi/2, 0, 0]))

sphereId = p.createMultiBody(baseMass=0, baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[1, 0, 0, 1]), basePosition=[0, 0, 0])
# Flag_mode_simulation = int(input('0 - real time | 1 - non-real time: '))
Flag_mode_simulation = 1
t_sampling = 1.0 / 240.0
if Flag_mode_simulation == 0:
    p.setRealTimeSimulation(1)
elif Flag_mode_simulation == 1:
    p.setRealTimeSimulation(0)
 
###############################################################################################################################################
###############################################################################################################################################

def decode_index(index):
    t1 = index // (3**2)
    remainder = index % (3**2)
    t2 = remainder // 3
    t3 = remainder % 3
    return t1, t2, t3

###############################################################################################################
radius = 1 
dt = 1/240

############################### INIT 2 ROBOT ##################################################################
############################### INIT 2 ROBOT ##################################################################
############################### INIT 2 ROBOT ##################################################################

initial_joint_angles_6_dof = [0, np.pi/2, 0, 0, 0, 0]
for i, angle in enumerate(initial_joint_angles_6_dof):
    p.resetJointState(Robot_1, i, targetValue=angle)

initial_joint_angles_5_dof = [np.pi, np.pi/2, -np.pi/2, np.pi/2, 0]
for i, angle in enumerate(initial_joint_angles_5_dof):
    p.resetJointState(Robot_2, i, targetValue=angle)
###############################################################################################################
###############################################################################################################
###############################################################################################################

Flag_take_longest_distance_5_dof = 0
att_position_max_5_dof = 0
Flag_take_longest_distance_6_dof = 0
att_position_max_6_dof = 0

Px_6_dof = 0.3
Py_6_dof = 0.3
Pz_6_dof = 0.3

Px_5_dof = 1.3
Py_5_dof = 0.3
Pz_5_dof = 0.3

vmax_6_dof = 0.1
vmax_5_dof = 0.1

Px_5_dof_converted, Py_5_dof_converted, Py_5_dof_converted = Convert_position_5_dof_base(Px_5_dof, Py_5_dof, Py_5_dof )

sphereId = p.createMultiBody(baseMass=0, baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[1, 0, 0, 1]), basePosition=[0, 0, 0])

# chuyển mm → m, và tính nửa kích thước
num_red=5
size_mm=50
size_m = size_mm / 1000.0
half = size_m / 2.0
base_height = half

box_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half, half, half])
green_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half, half, half], rgbaColor=[0, 1, 0, 1])
red_vis   = p.createVisualShape(p.GEOM_BOX, halfExtents=[half, half, half], rgbaColor=[1, 0, 0, 1])

X_object_1, Y_object_1 = 0.4, -0.2
X_object_2, Y_object_2 = 0, 0.4
X_object_3, Y_object_3 = 0.4, 0.2
X_object_4, Y_object_4 = 0.8, -0.6
object_1 = p.createMultiBody(
    baseMass=1,
    baseCollisionShapeIndex=box_col,
    baseVisualShapeIndex=green_vis,
    basePosition=[X_object_1, Y_object_1, base_height]
)

object_2 = p.createMultiBody(
    baseMass=1,
    baseCollisionShapeIndex=box_col,
    baseVisualShapeIndex=green_vis,
    basePosition=[X_object_2, Y_object_2 , base_height]
)

object_3 = p.createMultiBody(
    baseMass=1,
    baseCollisionShapeIndex=box_col,
    baseVisualShapeIndex=red_vis,
    basePosition=[X_object_3, Y_object_3 , base_height]
)

object_4 = p.createMultiBody(
    baseMass=1,
    baseCollisionShapeIndex=box_col,
    baseVisualShapeIndex=red_vis,
    basePosition=[X_object_4, Y_object_4 , base_height]
)

Case_6dof = 0
Case_5dof = 0
sphere_id = create_drag_only_sphere()

while True:

    if Case_6dof == 0:
        Px_6_dof = X_object_1
        Py_6_dof = Y_object_1
        Pz_6_dof = 0.4

    elif Case_6dof == 1:
        Px_6_dof = X_object_1
        Py_6_dof = Y_object_1
        Pz_6_dof = 0.2

    elif Case_6dof == 2:
        Px_6_dof = X_object_1
        Py_6_dof = Y_object_1
        Pz_6_dof = 0.4

    elif Case_6dof == 3:
        Px_6_dof = X_object_2
        Py_6_dof = Y_object_2
        Pz_6_dof = 0.4

    elif Case_6dof == 4:
        Px_6_dof = X_object_2
        Py_6_dof = Y_object_2
        Pz_6_dof = 0.2

    elif Case_6dof == 5:
        Px_6_dof = X_object_2
        Py_6_dof = Y_object_2
        Pz_6_dof = 0.4

    ### 5dof
    if Case_5dof == 0:
        Px_5_dof = X_object_3
        Py_5_dof = Y_object_3
        Pz_5_dof = 0.4

    if Case_5dof == 1:
        Px_5_dof = X_object_3
        Py_5_dof = Y_object_3
        Pz_5_dof = 0.2

    if Case_5dof == 2:
        Px_5_dof = X_object_3
        Py_5_dof = Y_object_3
        Pz_5_dof = 0.4

    elif Case_5dof == 3:
        Px_5_dof = X_object_4
        Py_5_dof = Y_object_4
        Pz_5_dof = 0.4

    elif Case_5dof == 4:
        Px_5_dof = X_object_4
        Py_5_dof = Y_object_4
        Pz_5_dof = 0.2

    elif Case_5dof == 5:
        Px_5_dof = X_object_4
        Py_5_dof = Y_object_4
        Pz_5_dof = 0.4

    joint_position_6_dof = np.array(Take_joint_position_6_dof())
    joint_position_6_dof = (joint_position_6_dof + np.pi) % (2 * np.pi) - np.pi

    joint_position_5_dof = np.array(Take_joint_position_5_dof())
    joint_position_5_dof = (joint_position_5_dof + np.pi) % (2 * np.pi) - np.pi
    theta_dot_6_dof, Position_error_6_dof, att_Rx_6_dof, att_Rz_6_dof = Get_theta_dot_6_dof(x_target=Px_6_dof, 
                                                                                            y_target=Py_6_dof, 
                                                                                            z_target=Pz_6_dof, 
                                                                                            vmax= vmax_6_dof, 
                                                                                            joint_position_local= joint_position_6_dof)

    theta_dot_5_dof, Position_error_5_dof, att_Rx_5_dof, att_Rz_5_dof = Get_theta_dot_5_dof(x_target=Px_5_dof, 
                                                                                            y_target=Py_5_dof, 
                                                                                            z_target=Pz_5_dof, 
                                                                                            vmax= vmax_5_dof, 
                                                                                            joint_position_local= joint_position_5_dof)
    # print(att_Rx_5_dof)
    # print(Position_error_5_dof)
    Set_joint_velocity_6_dof(theta_dot_6_dof) 
    Set_joint_velocity_5_dof(theta_dot_5_dof) 

    if Position_error_6_dof < 0.0001 and att_Rx_6_dof < 0.0175:
        Case_6dof = Case_6dof + 1
    
    if Position_error_5_dof < 0.0001 and att_Rx_5_dof < 0.0175:
        Case_5dof = Case_5dof + 1
    
    if Case_6dof > 5: 
        Case_6dof = 0

    if Case_5dof > 5:
        Case_5dof = 0

    pos_collision, orn= p.getBasePositionAndOrientation(sphere_id)
    p.resetBasePositionAndOrientation(sphere_id, [pos_collision[0], pos_collision[1], pos_collision[2]], orn)
    p.resetBasePositionAndOrientation(sphereId, [Px_5_dof, Py_5_dof, Pz_5_dof], [0, 0, 0, 1])
    if Flag_mode_simulation == 1:
        p.stepSimulation()
        time.sleep(1/240)
    