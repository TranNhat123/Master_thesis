import numpy as np


def Robot5_observer(t1, t2, t3, t4):
    a_1 = 0.15
    a_2 = 0.25
    a_3 = 0.22
    d_1 = 0.35
    d_tool = 0.08
    Base = np.array([0, 0, 0])

    # joint_1
    x_1 = a_1 * np.cos(t1)
    y_1 = a_1 * np.sin(t1)
    z_1 = d_1
    joint_1 = np.array([x_1, y_1, z_1])

    # Joint_2
    x_2 = a_1 * np.cos(t1) + a_2 * np.cos(t1) * np.cos(t2)
    y_2 = a_1 * np.sin(t1) + a_2 * np.sin(t1) * np.cos(t2)
    z_2 = d_1 + a_2 * np.sin(t2)
    joint_2 = np.array([x_2, y_2, z_2])

    # joint_3
    x_3 = a_1 * np.cos(t1) + a_2 * np.cos(t1) * np.cos(t2) + a_3 * np.cos(t1) * np.cos(t2) * np.cos(t3) - a_3 * np.cos(
        t1) * np.sin(t2) * np.sin(t3)
    y_3 = a_1 * np.sin(t1) + a_2 * np.sin(t1) * np.cos(t2) + a_3 * np.sin(t1) * np.cos(t2) * np.cos(t3) - a_3 * np.sin(
        t1) * np.sin(t2) * np.sin(t3)
    z_3 = d_1 + a_3 * np.sin(t2 + t3) + a_2 * np.sin(t2)
    joint_3 = np.array([x_3, y_3, z_3])

    # joint_tool
    x_tool = np.cos(t1) * (a_1 + d_tool * np.sin(t2 + t3 + t4) + a_2 * np.cos(t2) + a_3 * np.cos(t2 + t3))
    y_tool = np.sin(t1) * (a_1 + d_tool * np.sin(t2 + t3 + t4) + a_2 * np.cos(t2) + a_3 * np.cos(t2 + t3))
    z_tool = d_1 + a_2 * np.sin(t2) + a_3 * np.sin(t2 + t3) - d_tool * np.cos(t2 + t3 + t4)
    joint_tool = np.array([x_tool, y_tool, z_tool])
    # Vi khong xoay t5 nen Link 4 chinh la thang T05 luon (minh cho t5=0)
    
    point_1 = np.array([0, 0, d_1])
    point_2 = joint_1
    point_3 = np.mean(np.vstack([joint_1, joint_2]), axis=0)
    point_4 = joint_2 
    point_5 = np.mean(np.vstack([joint_2, joint_3]), axis=0)
    point_6 = joint_3
    point_7 = joint_tool
    Arm_1 = np.array([
        point_1, 
        point_2, 
        point_3, 
        point_4, 
        point_5, 
        point_6, 
        point_7
        ])

    return Arm_1

