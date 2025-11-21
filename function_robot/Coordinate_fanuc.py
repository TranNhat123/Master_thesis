import numpy as np


def Coordinate_fanuc(t1, t2, t3, t4):
    a_1 = 0.15
    a_2 = 0.25
    a_3 = 0.22
    d_1 = 0.35
    d_tool = 0.08
    Base = [0, 0, 0]

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
    Arm_1 = np.array([joint_1, joint_2, joint_3, joint_3, joint_tool])

    return Arm_1

