import numpy as np
import socket
from function_robot.Capsuls_sixdof import Capsuls_sixdof
from function_robot.Capsuls_fanuc import Capsuls_fanuc
from function_robot.Coordinate_fanuc import Coordinate_fanuc
from function_robot.Coordinate_sixdof import Coordinate_sixdof
from function_robot.T06_sixdof_function import T06_sixdof_function

from function_robot.J01_fanuc_function import J01_fanuc_function
from function_robot.J02_fanuc_function import J02_fanuc_function
from function_robot.J03_fanuc_function import J03_fanuc_function
from function_robot.Jtool_fanuc_function import Jtool_fanuc_function
from function_robot.Jtool_fanuc_function_1 import Jtool_fanuc_function_1
from function_robot.Jtool_fanuc_function_2 import Jtool_fanuc_function_2
from function_robot.T06_sixdof_function_main_2 import T06_sixdof_function_main_2
from function_robot.J01_sixdof_function import J01_sixdof_function
from function_robot.J02_sixdof_function import J02_sixdof_function
from function_robot.J03_sixdof_function import J03_sixdof_function
from function_robot.J04_sixdof_function import J04_sixdof_function
from function_robot.Jtool_sixdof_function import Jtool_sixdof_function
from function_robot.QR_distance import QR_distance
from function_robot.T05_fanuc_function import T05_fanuc_function
import pybullet as p
import time
import pybullet_data
import numpy as np
import math
from datetime import datetime

def rotx(theta):
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])

def roty(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def rotz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])

def rotation_matrix_to_euler_angles(rot_matrix):
    """
    Chuyển đổi từ ma trận xoay 3x3 thành giá trị Euler (Roll, Pitch, Yaw)
    Ma trận xoay phải thỏa mãn điều kiện SO(3): det(R) = 1 và R.T * R = I
    """
    sy = math.sqrt(rot_matrix[0, 0] * rot_matrix[0, 0] + rot_matrix[1, 0] * rot_matrix[1, 0])

    singular = sy < 1e-6  # Kiểm tra xem ma trận có gần singular không

    if not singular:
        roll = math.atan2(rot_matrix[2, 1], rot_matrix[2, 2])
        pitch = math.atan2(-rot_matrix[2, 0], sy)
        yaw = math.atan2(rot_matrix[1, 0], rot_matrix[0, 0])
    else:
        roll = math.atan2(-rot_matrix[1, 2], rot_matrix[1, 1])
        pitch = math.atan2(-rot_matrix[2, 0], sy)
        yaw = 0

    return np.array([roll, pitch, yaw])  # Trả về Roll, Pitch, Yaw dưới dạng numpy array

def plot_ref_axis(Rx, Ry, Rz, link_pos):
        # Chuyển quaternion thành ma trận xoay
    axis_length = 0.1
    # link_pos = link_state[4]  # Lấy vị trí của link
    rot_matrix = rotz(Rz) @ roty(Ry) @ rotx(Rx)
    rot_matrix = np.array([[rot_matrix[0, 0],rot_matrix[0, 1], rot_matrix[0, 2]], 
                   [rot_matrix[1, 0],rot_matrix[1, 1], rot_matrix[1, 2]], 
                   [rot_matrix[2, 0],rot_matrix[2, 1], rot_matrix[2, 2]]])

    x_axis = rot_matrix[:,0]
    y_axis = rot_matrix[:,1]
    z_axis = rot_matrix[:,2]

    # Xóa các đường debug cũ nếu cần thiết (để không bị lặp lại các đường không mong muốn)
    # p.removeAllUserDebugItems()

    # Vẽ lại trục X
    p.addUserDebugLine(link_pos, 
                        [link_pos[0] + axis_length * x_axis[0],
                        link_pos[1] + axis_length * x_axis[1],
                        link_pos[2] + axis_length * x_axis[2]],
                        [1, 0, 0], lineWidth=2)
    
    # Vẽ lại trục Y
    p.addUserDebugLine(link_pos, 
                        [link_pos[0] + axis_length * y_axis[0],
                        link_pos[1] + axis_length * y_axis[1],
                        link_pos[2] + axis_length * y_axis[2]],
                        [0, 1, 0], lineWidth=2)
    
    # Vẽ lại trục Z
    p.addUserDebugLine(link_pos, 
                        [link_pos[0] + axis_length * z_axis[0],
                        link_pos[1] + axis_length * z_axis[1],
                        link_pos[2] + axis_length * z_axis[2]],
                        [0, 0, 1], lineWidth=2)
    
def plot_axis(robot_id, joint_index):
    axis_length = 0.1
    link_state = p.getLinkState(robot_id, joint_index)
    link_pos = link_state[4]  # Lấy vị trí của link
    link_orientation = link_state[5]  # Lấy orientation của link (quaternion)

    # Chuyển quaternion thành ma trận xoay
    rot_matrix = p.getMatrixFromQuaternion(link_orientation)
    rot_matrix = np.array([[rot_matrix[0],rot_matrix[1], rot_matrix[2]], 
                   [rot_matrix[3],rot_matrix[4], rot_matrix[5]], 
                   [rot_matrix[6],rot_matrix[7], rot_matrix[8]]])
    
    flip_matrix = np.array([[-1, 0, 0],
                                [0, 1, 0],
                                [0, 0, -1]])
    
    rot_matrix = np.dot(rot_matrix, flip_matrix)
    x_axis = rot_matrix[:,0]
    y_axis = rot_matrix[:,1]
    z_axis = rot_matrix[:,2]

    # Xóa các đường debug cũ nếu cần thiết (để không bị lặp lại các đường không mong muốn)
    p.removeAllUserDebugItems()

    # Vẽ lại trục X
    p.addUserDebugLine(link_pos, 
                        [link_pos[0] + axis_length * x_axis[0],
                        link_pos[1] + axis_length * x_axis[1],
                        link_pos[2] + axis_length * x_axis[2]],
                        [1, 0, 0], lineWidth=2)
    
    # Vẽ lại trục Y
    p.addUserDebugLine(link_pos, 
                        [link_pos[0] + axis_length * y_axis[0],
                        link_pos[1] + axis_length * y_axis[1],
                        link_pos[2] + axis_length * y_axis[2]],
                        [0, 1, 0], lineWidth=2)
    
    # Vẽ lại trục Z
    p.addUserDebugLine(link_pos, 
                        [link_pos[0] + axis_length * z_axis[0],
                        link_pos[1] + axis_length * z_axis[1],
                        link_pos[2] + axis_length * z_axis[2]],
                        [0, 0, 1], lineWidth=2)
    return rot_matrix

def Set_joint_position(theta_v_fanuc, theta_v_sixdof):
    p.setJointMotorControl2(bodyUniqueId=Robot_1,
                        jointIndex=0,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition= theta_v_sixdof[0],
                        force = 500)  # Giới hạn lực
    p.setJointMotorControl2(bodyUniqueId=Robot_1,
                        jointIndex=1,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition= theta_v_sixdof[1],
                        force = 500)  # Giới hạn lực
    p.setJointMotorControl2(bodyUniqueId=Robot_1,
                    jointIndex=2,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition= theta_v_sixdof[2],
                    force = 500)  # Giới hạn lực
    p.setJointMotorControl2(bodyUniqueId=Robot_1,
                jointIndex=3,
                controlMode=p.POSITION_CONTROL,
                targetPosition= -theta_v_sixdof[3],
                force = 500)  # Giới hạn lực
    
    p.setJointMotorControl2(bodyUniqueId=Robot_1,
                jointIndex=4,
                controlMode=p.POSITION_CONTROL,
                targetPosition= -theta_v_sixdof[4],
                force = 500)  # Giới hạn lực
    
    p.setJointMotorControl2(bodyUniqueId=Robot_1,
                jointIndex=5,
                controlMode=p.POSITION_CONTROL,
                targetVelocity= theta_v_sixdof[5],
                force = 500)  # Giới hạn lực
    
def Set_joint_velocity(theta_v_fanuc, theta_v_sixdof):
    p.setJointMotorControl2(bodyUniqueId=Robot_1,
                        jointIndex=0,
                        controlMode=p.VELOCITY_CONTROL,
                        targetVelocity= theta_v_sixdof[0],
                        force = 500)  # Giới hạn lực
    p.setJointMotorControl2(bodyUniqueId=Robot_1,
                        jointIndex=1,
                        controlMode=p.VELOCITY_CONTROL,
                        targetVelocity= theta_v_sixdof[1],
                        force = 500)  # Giới hạn lực
    p.setJointMotorControl2(bodyUniqueId=Robot_1,
                    jointIndex=2,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity= theta_v_sixdof[2],
                    force = 500)  # Giới hạn lực
    p.setJointMotorControl2(bodyUniqueId=Robot_1,
                jointIndex=3,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity= -theta_v_sixdof[3],
                force = 500)  # Giới hạn lực
    
    p.setJointMotorControl2(bodyUniqueId=Robot_1,
                jointIndex=4,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity= -theta_v_sixdof[4],
                force = 500)  # Giới hạn lực
    
    p.setJointMotorControl2(bodyUniqueId=Robot_1,
                jointIndex=5,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity= theta_v_sixdof[5],
                force = 500)  # Giới hạn lực
  
    

def Take_joint_position():
    t_1 = 0
    t_2 = 0
    t_3 = 0
    t_4 = 0
    t_5 = 0
    t_6 = p.getJointState(Robot_1, 0)[0]
    t_7 = p.getJointState(Robot_1, 1)[0]
    t_8 = p.getJointState(Robot_1, 2)[0]
    t_9 = p.getJointState(Robot_1, 3)[0]
    t_10 = p.getJointState(Robot_1, 4)[0]
    t_11 = p.getJointState(Robot_1, 5)[0]
    joint_position_recv = np.array([t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9, t_10, t_11])
    # Save joint_position
    return joint_position_recv

#################################################################################################################

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")
startPos_1 = [-0.05,0.05,0]
startOrientation_1 = p.getQuaternionFromEuler([-np.pi/2,0,0])
startPos_2 = [1,0,0]
startOrientation_2 = p.getQuaternionFromEuler([-np.pi/2,0,0])
Robot_1 = p.loadURDF("./2_robot/URDF_file_2/urdf/6_Dof.urdf", [0, 0, 0], useFixedBase=True)
Robot_2 = p.loadURDF("./2_robot/Robot/urdf/5_Dof.urdf", [0, 0, 0], useFixedBase=True)

#set the center of mass frame (loadURDF sets base link frame)
p.resetBasePositionAndOrientation(Robot_1, startPos_1, startOrientation_1)
p.resetBasePositionAndOrientation(Robot_2, startPos_2, startOrientation_2)
# for i in range(p.getNumJoints(Robot_1)):
#     p.resetJointState(bodyUniqueId = Robot_1, jointIndex = i, targetValue = math.pi/2)
i = 0
t_sampling = 0.005

t1 = 0
t2 = np.pi / 2
t3 = -np.pi / 2
t4 = np.pi / 2
t5 = 0

t6 = 0
t7 = np.pi / 2
t8 = 0
t9 = 0
t10 = 0
t11 = 0

p.resetJointState(bodyUniqueId = Robot_1, jointIndex = 0, targetValue = t6)  
p.resetJointState(bodyUniqueId = Robot_1, jointIndex = 1, targetValue = t7)  
p.resetJointState(bodyUniqueId = Robot_1, jointIndex = 2, targetValue = t8)  
p.resetJointState(bodyUniqueId = Robot_1, jointIndex = 3, targetValue = t9)  
p.resetJointState(bodyUniqueId = Robot_1, jointIndex = 4, targetValue = t10)  
p.resetJointState(bodyUniqueId = Robot_1, jointIndex = 5, targetValue = t11)  
# jointPoses = p.calculateInverseKinematics(Robot_1,5, pos)
t = 0
trailDuration = 15
hasPrevPose = 1
pos = p.getLinkState(bodyUniqueId = Robot_1, linkIndex = 5)
prevPose = pos

# Tạo hình cầu trực quan mà không chịu ảnh hưởng vật lý
radius = 0.01  # Bán kính của hình cầu
mass = 0  # Đặt khối lượng là 0 để không chịu ảnh hưởng của vật lý
sphereVisualShape = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[1, 0, 0, 1]) 
# Tạo "vật thể" chỉ để hiển thị (mass=0, không chịu lực tác động)
sphereId = p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphereVisualShape, basePosition=[0, 0, 0])

# ----------------------------------------------- Khai bao gia hinh hoc cua robot --------------------------------------
a_1_fanuc = 0.15
a_2_fanuc = 0.25
a_3_fanuc = 0.22
d_1_fanuc = 0.35
d_5_fanuc = 0.08

a_2_sixdof = 0.26
a_3_sixdof = 0.0305
d_1_sixdof = 0.29
d_4_sixdof = 0.27
d_tool_sixdof = 0.09

# ----------------------------------------------------------------------------------------------------------------------

# ------------------------------------------ Khai bao vi tri home real cua robot ---------------------------------------

# ---------------------------------- Khai bao ma tran khoang cach tuong doi giua 2 robot -------------------------------

Arm_2_matrix = np.zeros((8, 3))
for j in range(8):
    Arm_2_matrix[j, 1] = 1

# ----------------------------------------------------------------------------------------------------------------------

# --------------------------- Khai bao cac gia tri dau tien, tinh toan cac ket qua dau tien ----------------------------

Case_fanuc = 1
Case_sixdof = 1
Flag_fanuc = 0
Flag_sixdof = 0
Flag_network_using = 0
# toa do cac diem cua capsul
Capsul_fanuc = Capsuls_fanuc(t1, t2, t3, t4) + Arm_2_matrix
Capsul_sixdof = Capsuls_sixdof(t6, t7, t8, t9, t10)
coordinate_fanuc = Coordinate_fanuc(t1, t2, t3, t4)
coordinate_sixdof = Coordinate_sixdof(t6, t7, t8, t9, t10)
Rad_1 = np.array([0.1095, 0.1, 0.08, 0.04])
Rad_2 = np.array([0.075, 0.1, 0.04, 0.02])

# tinh toan cac khoang cach nho nhat
so_thu_tu = [0, 2, 4, 6, 8]
q = 0
distances = np.zeros((16, 1))
for n in range(4):
    for m in range(4):
        distances[q], = QR_distance(Capsul_fanuc[so_thu_tu[n], :3],
                                    Capsul_fanuc[so_thu_tu[n] + 1, :3],
                                    Capsul_sixdof[so_thu_tu[m], :3],
                                    Capsul_sixdof[so_thu_tu[m] + 1, :3],
                                    Rad_1[n], Rad_2[m])
        q = q + 1

# ----------------------------------------------------------------------------------------------------------------------
p.setRealTimeSimulation(1)
while True:
    # Tinh Jancobian -------------------------------------------------------------------------------------------------

    joint_position = Take_joint_position()
    t1 = joint_position[0]
    t2 = joint_position[1]
    t3 = joint_position[2]
    t4 = joint_position[3]
    t5 = joint_position[4]

    t6 = joint_position[5]
    t7 = joint_position[6]
    t8 = joint_position[7]
    t9 = joint_position[8]
    t10 = joint_position[9]
    t11 = joint_position[10]

    J01_fanuc = J01_fanuc_function(t1, a_1_fanuc)
    J02_fanuc = J02_fanuc_function(t1, t2, a_1_fanuc, a_2_fanuc)
    J03_fanuc = J03_fanuc_function(t1, t2, t3, a_3_fanuc)
    Jtool_fanuc = Jtool_fanuc_function(t1, t2, t3, t4, t5, d_5_fanuc)
    Jtool_fanuc_1 = Jtool_fanuc_function_1(t1, t2, t3, t4, t5)
    Jtool_fanuc_2 = Jtool_fanuc_function_2(t1, t2, t3, t4, t5)

    J01_sixdof = J01_sixdof_function()
    J02_sixdof = J02_sixdof_function(t6, t7, a_2_sixdof)
    J03_sixdof = J03_sixdof_function(t6, t7, t8, a_3_sixdof)
    J04_sixdof = J04_sixdof_function(t6, t7, t8, d_4_sixdof)
    Jtool_sixdof = Jtool_sixdof_function(t6, t7, t8, t9, t10, t11)

    # Toi diem lay vat
    if Case_sixdof == 1:
        x_target = 0.4
        y_target = 0
        z_target = 0.3
        Goal_sixdof_orientation = np.array([ 0, np.pi/2, 0])
        Goal_sixdof_rotation = rotz(Goal_sixdof_orientation[2]) @ roty(Goal_sixdof_orientation[1]) @ rotx(Goal_sixdof_orientation[0])
        Goal_sixdof_orientation = rotation_matrix_to_euler_angles(Goal_sixdof_rotation)
        v_position_sixdof = 0.1
        v_orientation_sixdof = 0.3
    # ---------------------------------- Step 2: Determine errors and Flag_sixdof --------------------------------------
    # ---------------------------------- Step 2: Determine errors and Flag_sixdof --------------------------------------
    # ---------------------------------- Step 2: Determine errors and Flag_sixdof --------------------------------------
    Capsul_sixdof = Capsuls_sixdof(t6, t7, t8, t9, t10)
    Goal_coordinate_sixdof = np.array([x_target, y_target, z_target])
    coordinate_sixdof = Coordinate_sixdof(t6, t7, t8, t9, t10)
    # x = coordinate_sixdof[-1, 0]
    # y = coordinate_sixdof[-1, 1]
    # z = coordinate_sixdof[-1, 2]
    pos = p.getLinkState(bodyUniqueId = Robot_1, linkIndex = 5)[0]
    x = pos[0]
    y = pos[1]
    z = pos[2]
    att_sixdof = [x, y, z] - Goal_coordinate_sixdof
    # att_sixdof_jacobian = T06_sixdof_function(t6, t7, t8, t9, t10, t11) - Goal_coordinate_sixdof_jacobian
    # Lấy trạng thái của link trong PyBullet
    quaternion = p.getLinkState(bodyUniqueId=Robot_1, linkIndex=5)[1]
    # Chuyển đổi từ quaternion sang Roll, Pitch, Yaw
    # roll, pitch, yaw = p.getEulerFromQuaternion(quaternion)
    rot_matrix = plot_axis(Robot_1, 5)
    plot_ref_axis(Goal_sixdof_orientation[0], Goal_sixdof_orientation[1], Goal_sixdof_orientation[2],[x_target, y_target, z_target] )
    roll, pitch, yaw = rotation_matrix_to_euler_angles(rot_matrix)
    att_sixdof_jacobian = T06_sixdof_function_main_2(roll, pitch, yaw, Goal_sixdof_orientation[0], Goal_sixdof_orientation[1], Goal_sixdof_orientation[2])
    # orientation_sixdof = T06_sixdof_function(t6, t7, t8, t9, t10, t11)
    # --------------------------------------- Flag condition and Case_sixdof -------------------------------------------
    # Toi diem lay vat
    # if np.linalg.norm(att_sixdof) < 0.001 and np.abs(att_sixdof_jacobian[0]) < 0.0175 and np.abs(att_sixdof_jacobian[1]) < 0.0175 and np.abs(att_sixdof_jacobian[2]) < 0.0175:
    #     Case_sixdof = Case_sixdof + 1
    

    if np.linalg.norm(att_sixdof) < 0.005:
        v_position_sixdof = 0.05
    if np.linalg.norm(att_sixdof) < 0.001:
        v_position_sixdof = 0

    #  ----------------------- V_position and orientation of sixdof--------------------------------------------
    if np.linalg.norm(att_sixdof) < v_position_sixdof * t_sampling:
        v_position_sixdof = np.linalg.norm(att_sixdof) / t_sampling


        # ------------------------ V_att_tool_position and orientation of sixdof -----------------------------------
    if np.linalg.norm(att_sixdof) < 0.000001:
        v_att_tool_position_sixdof = np.zeros(3)
    else:
        v_att_tool_position_sixdof = -v_position_sixdof * (att_sixdof / np.linalg.norm(att_sixdof))

    if np.abs(att_sixdof_jacobian[0]) < 0.0175 and np.abs(att_sixdof_jacobian[1]) < 0.0175 and np.abs(att_sixdof_jacobian[2]) < 0.0175:
        v_orientation_sixdof = 0

    v_att_tool_orientation_sixdof = v_orientation_sixdof * (att_sixdof_jacobian / np.linalg.norm(att_sixdof_jacobian))

    # --------------------------Step 4: Determine theta values and avoid singularity -----------------------------------
    # --------------------------Step 4: Determine theta values and avoid singularity -----------------------------------
    # --------------------------Step 4: Determine theta values and avoid singularity -----------------------------------

    det_sixdof_matrix = np.linalg.det(Jtool_sixdof)
    det_sixdof_matrix_3 = np.linalg.det(J04_sixdof)
    # ----------------------------------- Calculate theta_v_tool_sixdof --------------------------------------------

    c_tool = np.hstack((v_att_tool_position_sixdof, v_att_tool_orientation_sixdof))

    # if abs(det_sixdof_matrix) < 0.0001:
    #     theta_v_tool_sixdof = Jtool_sixdof.T.dot(
    #         np.linalg.inv(Jtool_sixdof.dot(Jtool_sixdof.T) + 0.00001 * np.eye(6))).dot(c_tool)
    # else:
    #     theta_v_tool_sixdof = np.dot(np.linalg.inv(Jtool_sixdof), c_tool.T)
    theta_v_tool_sixdof = np.dot(np.linalg.pinv(Jtool_sixdof), c_tool.T)
    
    theta_v_sixdof = theta_v_tool_sixdof
    theta_v_fanuc = [0, 0, 0, 0, 0]
    
    # ----------------------------------------- set joint velocity -----------------------------------------------------
    # t6, t7, t8, t9, t10, t11 = [0, 0, math.pi/2, 0, 0, 0]
    
    # p.resetJointState(bodyUniqueId = Robot_1, jointIndex = 0, targetValue = t6)  
    # p.resetJointState(bodyUniqueId = Robot_1, jointIndex = 1, targetValue = t7)  
    # p.resetJointState(bodyUniqueId = Robot_1, jointIndex = 2, targetValue = t8)  
    # p.resetJointState(bodyUniqueId = Robot_1, jointIndex = 3, targetValue = t9)  
    # p.resetJointState(bodyUniqueId = Robot_1, jointIndex = 4, targetValue = t10)  
    # p.resetJointState(bodyUniqueId = Robot_1, jointIndex = 5, targetValue = t11)  

    Set_joint_velocity(theta_v_fanuc, theta_v_sixdof)
    Capsul_fanuc = Capsuls_fanuc(t1, t2, t3, t4) + Arm_2_matrix
    so_thu_tu = [0, 2, 4, 6, 8]
    q = 0
    distances = np.zeros((16, 1))
    for n in range(4):
        for m in range(4):
            distances[q], = QR_distance(Capsul_fanuc[so_thu_tu[n], :3],
                                        Capsul_fanuc[so_thu_tu[n] + 1, :3],
                                        Capsul_sixdof[so_thu_tu[m], :3],
                                        Capsul_sixdof[so_thu_tu[m] + 1, :3],
                                        Rad_1[n], Rad_2[m])
            q = q + 1
    # print(Case_sixdof)
    # print('Linear_error', np.linalg.norm(att_sixdof))
    # print(att_sixdof_jacobian)
    p.resetBasePositionAndOrientation(sphereId, [x_target, y_target, z_target] , [0, 0, 0, 1])
    print('v_orientation', v_orientation_sixdof, 'v_linear', v_position_sixdof)
