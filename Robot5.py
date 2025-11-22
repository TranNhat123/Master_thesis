import numpy as np
import pybullet as p
from function_robot.Jtool_fanuc_function import Jtool_fanuc_function
from function_robot.Robot5_observer import Robot5_observer

class Robot_5_Dof:
    def __init__(self, urdf_path, base_position, base_orientation_euler, use_fixed_base=True):
        self.robot_id = p.loadURDF(
            urdf_path,
            base_position,
            p.getQuaternionFromEuler(base_orientation_euler),
            useFixedBase=use_fixed_base
        )
        # Link tool để lấy pose (ở code cũ là linkIndex = 5)
        self.tool_link_index = 4

        self.base_position = base_position
        # Các biến dùng cho luật điều khiển
        self.flag_take_longest_distance = False
        self.att_position_max = 0.0
        self.att_Rx = 0.0
        self.att_Rz = 0.0

        # Góc khởi tạo cho robot 6 bậc
        initial_joint_angles = [0, np.pi / 2, -np.pi/2, np.pi/2, 0]
        for i, angle in enumerate(initial_joint_angles):
            p.resetJointState(self.robot_id, i, targetValue=angle)

        # ====== TẠO 5 MARKER TẠI CÁC LINK ======
        self.link_markers = []
        marker_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.1,                 # chỉnh nhỏ / lớn tùy bạn
            rgbaColor=[0, 0, 1, 0.5]       # màu xanh lá
        )
        for _ in range(7):
            marker_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=marker_visual,
                basePosition=[0, 0, 0]
            )
            self.link_markers.append(marker_id)

        ## Maker_points
        self.marker_points = []

    def update_link_markers(self):
        """
        Cập nhật vị trí 6 marker tại trung tâm 6 link.
        Giả sử linkIndex = 0..5 tương ứng 6 khớp của robot.
        """
        for link_index in range(len(self.marker_points[:, 0])):

            marker_id = self.link_markers[link_index]
            link_pos = self.marker_points[link_index, :]
            p.resetBasePositionAndOrientation(
                marker_id,
                link_pos,
                [0, 0, 0, 1]
            )


    def take_joint_position(self):
        """Lấy vector góc khớp hiện tại của robot 5 bậc."""
        positions = []
        for i in range(5):
            joint_value = p.getJointState(self.robot_id, i)[0]
            positions.append(joint_value)
        return np.array(positions)

    def set_joint_velocity(self, theta_v_sixdof):
        """Set vận tốc cho từng khớp robot 6 bậc."""
        for i in range(5):
            velocity = theta_v_sixdof[i]
            p.setJointMotorControl2(
                self.robot_id,
                i,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=velocity,
                force=500
            )

    def get_theta_dot(self, x_target, y_target, z_target, vmax, joint_position_local):
        """
        Tính theta_dot dựa trên:
        - lỗi vị trí end-effector so với (x_target, y_target, z_target)
        - lỗi orientation Rx, Rz
        """
        t1, t2, t3, t4, t5 = joint_position_local
    
        # Jacobian tại tool
        Jtool_fivedof = Jtool_fanuc_function(t1, t2, t3, t4)
        self.marker_points = Robot5_observer(t1, t2, t3, t4) + self.base_position
        # Tọa độ đích (goal)
        goal = np.array([float(x_target), float(y_target), float(z_target)])

        # Lấy vị trí tool từ PyBullet
        pos = p.getLinkState(bodyUniqueId=self.robot_id, linkIndex=self.tool_link_index)[0]
        x, y, z = pos
        x, y, z = x*1000, y*1000, z*1000 # Convert tu m -> mm
        att = np.array([x, y, z]) - goal

        # Lưu khoảng cách lớn nhất để chuẩn hóa việc tăng/giảm tốc
        if not self.flag_take_longest_distance:
            self.att_position_max = np.linalg.norm(att)
            self.flag_take_longest_distance = True

        # Hướng mong muốn cho Rx, Rz (ở đây đặt = 0)
        Rx_des = 0.0
        Rz_des = 0.0

        Rx_now = t2 + t3 + t4
        Rz_now = -t1 + t5
        self.att_Rx = Rx_now - Rx_des
        self.att_Rz = Rz_now - Rz_des

        # Tính phần trăm quãng đường đã đi để điều chỉnh vận tốc
        percent = 1.0 - np.linalg.norm(att) / max(self.att_position_max, 1e-6)

        if percent <= 0.1:
            v_position_sixdof = max(vmax * percent / 0.1, vmax * 0.5)
        elif percent < 0.9:
            v_position_sixdof = vmax
        else:
            v_position_sixdof = vmax * (1.0 - percent) / 0.1

        # Vận tốc hấp dẫn theo vị trí
        v_att_tool_position = -v_position_sixdof * (att / np.linalg.norm(att))
        # Vận tốc hấp dẫn theo orientation
        v_att_tool_orientation = -50.0 * np.array([self.att_Rx, self.att_Rz])

        # Vector tốc độ mong muốn ở không gian task
        c_tool = np.hstack((v_att_tool_position, v_att_tool_orientation))

        # Tính tốc độ khớp bằng pseudo-inverse Jacobian
        theta_v_sixdof = np.dot(np.linalg.pinv(Jtool_fivedof), c_tool.T)

        # Giới hạn tốc độ cho 5 khớp đầu
        limit = np.pi / 10.0
        for i in range(5):
            theta_v_sixdof[i] = np.clip(theta_v_sixdof[i], -limit, limit)

        # Ở code gốc: khớp 4 bị set 0, khớp 5,6 map lại
        theta_dot = np.array([
            theta_v_sixdof[0],
            theta_v_sixdof[1],
            theta_v_sixdof[2],
            theta_v_sixdof[3],
            theta_v_sixdof[4]
        ])

        return theta_dot, np.linalg.norm(att), self.att_Rx, self.att_Rz