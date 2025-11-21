import pybullet as p
import time
import pybullet_data
import numpy as np
import math
from datetime import datetime

#################################################################################################################

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")
startPos_1 = [0,0,0.5]
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
p.resetJointState(bodyUniqueId = Robot_1, jointIndex = 0, targetValue = 0)  
p.resetJointState(bodyUniqueId = Robot_1, jointIndex = 1, targetValue = 0)  
p.resetJointState(bodyUniqueId = Robot_1, jointIndex = 2, targetValue = math.pi/2)  
p.resetJointState(bodyUniqueId = Robot_1, jointIndex = 3, targetValue = 0)  
p.resetJointState(bodyUniqueId = Robot_1, jointIndex = 4, targetValue = 0)  
p.resetJointState(bodyUniqueId = Robot_1, jointIndex = 5, targetValue = 0)  
# jointPoses = p.calculateInverseKinematics(Robot_1,5, pos)
t = 0
trailDuration = 15
hasPrevPose = 1
pos = p.getLinkState(bodyUniqueId = Robot_1, linkIndex = 5)
prevPose = pos

# Tạo hình cầu trực quan mà không chịu ảnh hưởng vật lý
radius = 0.05  # Bán kính của hình cầu
mass = 0  # Đặt khối lượng là 0 để không chịu ảnh hưởng của vật lý
sphereVisualShape = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[0, 1, 0, 1])  # Màu xanh lá
# Tạo "vật thể" chỉ để hiển thị (mass=0, không chịu lực tác động)
sphereId = p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphereVisualShape, basePosition=[0, 0, 0])

target_velocities = [math.pi/2, 0, 0, 0, 0, 0]
# start_time = time.time()
for i in range (1000000):
    elapsed_time = i*1/240
    # elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        target_velocities = [0, 0, 0, 0, 0, 0]

    # pos_desired = [ 0.5 + 0.2 * math.sin(t),  0.2 * math.cos(t), 0.4 ]
    pos_desired = [ 0.3,  0.3, 0.4 ]
    # print('pos', pos)
    jointPoses = p.calculateInverseKinematics(Robot_1,
                                            endEffectorLinkIndex = 5,
                                            targetPosition = pos_desired)
    for joint_index in range(6):
        p.setJointMotorControl2(bodyUniqueId=Robot_1,
                            jointIndex=joint_index,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=target_velocities[joint_index],
                            force=500)  # Giới hạn lực
        
    pos = p.getLinkState(bodyUniqueId = Robot_1, linkIndex = 5)
    print(pos[1])
    if (hasPrevPose):
        p.addUserDebugLine(prevPose[0], pos[0], [0, 0, 1], 1)

    p.resetBasePositionAndOrientation(sphereId, [pos_desired[0], pos_desired[1], pos_desired[2]], [0, 0, 0, 1])

    prevPose = pos
    p.stepSimulation()
    time.sleep(1./240.)
    # print('6_dof', p.getNumJoints(Robot_1))
    # print('5_dof', p.getNumJoints(Robot_2))
