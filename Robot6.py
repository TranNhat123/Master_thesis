import numpy as np
import pybullet as p
from function_robot.Coordinate_sixdof import Coordinate_sixdof
from function_robot.Jtool_sixdof_function import Jtool_sixdof_function
from function_robot.Robot6_observer import Robot6_observer

class Robot_6_Dof:
    def __init__(self, urdf_path, base_position, base_orientation_euler, use_fixed_base=True):
        """Khởi tạo robot 6 bậc trong PyBullet."""
        self.robot_id = p.loadURDF(
            urdf_path,
            base_position,
            p.getQuaternionFromEuler(base_orientation_euler),
            useFixedBase=use_fixed_base
        )

        # Link tool để lấy pose (ở code cũ là linkIndex = 5)
        self.tool_link_index = 5


        # Các biến dùng cho luật điều khiển
        self.flag_take_longest_distance = False
        self.att_position_max = 0.0
        self.att_Rx = 0.0
        self.att_Rz = 0.0
        # Giới hạn tốc độ cho 5 khớp đầu
        self.vel_limit = np.pi / 20.0
        # Góc khởi tạo cho robot 6 bậc
        initial_joint_angles = [0, np.pi / 2, 0, 0, 0, 0]
        for i, angle in enumerate(initial_joint_angles):
            p.resetJointState(self.robot_id, i, targetValue=angle)

            # ====== TẠO 6 MARKER TẠI CÁC LINK ======
        self.link_markers = []
        marker_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.1,                 # chỉnh nhỏ / lớn tùy bạn
            rgbaColor=[1, 0, 0, 0.5]       # màu xanh lá
        )
        for _ in range(5):
            marker_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=marker_visual,
                basePosition=[0, 0, 0]
            )
            self.link_markers.append(marker_id)
        # =======================================

        self.jointpos = []
        self.marker_points = []
    # ------------------ Hàm xử lý robot ------------------ #

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
        """Lấy vector góc khớp hiện tại của robot 6 bậc."""
        positions = []
        for i in range(6):
            joint_value = p.getJointState(self.robot_id, i)[0]
            if i == 4:
                joint_value = -joint_value  # đổi dấu nếu là khớp số 4
            positions.append(joint_value)
        return np.array(positions)

    def take_joint_velocity(self): 
        """Lấy vector vận tốc góc khớp hiện tại của robot 6 bậc."""
        velocity = []
        for i in range(6):
            joint_value = p.getJointState(self.robot_id, i)[1]
            if i == 4:
                joint_value = -joint_value  # đổi dấu nếu là khớp số 4
            velocity.append(joint_value)
        return np.array(velocity)

    def set_joint_velocity(self, theta_v_sixdof):
        """Set vận tốc cho từng khớp robot 6 bậc."""
        for i in range(6):
            velocity = -theta_v_sixdof[i] if i == 4 else theta_v_sixdof[i]
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
        t1, t2, t3, t4, t5, t6 = joint_position_local
        # Jacobian tại tool
        Jtool_sixdof = Jtool_sixdof_function(t1, t2, t3, t4, t5, t6)

        # Tọa độ đích (goal)
        goal = np.array([float(x_target), float(y_target), float(z_target)])

        self.jointpos = Coordinate_sixdof(t1, t2, t3, t4, t5)
        self.marker_points = Robot6_observer(t1, t2, t3, t4, t5)
        # Lấy vị trí tool từ PyBullet
        pos = p.getLinkState(bodyUniqueId=self.robot_id, linkIndex=self.tool_link_index)[0]
        x, y, z = pos
        att_sixdof = np.array([x, y, z]) - goal

        # Lưu khoảng cách lớn nhất để chuẩn hóa việc tăng/giảm tốc
        if not self.flag_take_longest_distance:
            self.att_position_max = np.linalg.norm(att_sixdof)
            self.flag_take_longest_distance = True

        # Hướng mong muốn cho Rx, Rz (ở đây đặt = 0)
        Rx_des = 0.0
        Rz_des = 0.0
        t4_des = 0
        Rx_now = t2 + t3 + t5
        Rz_now = -t1 + t6
        self.att_Rx = Rx_now - Rx_des
        self.att_Rz = Rz_now - Rz_des
        att_t4 = t4 - t4_des
        theta_v_t4 = -np.pi/20*np.sign(att_t4)
        if abs(att_t4) < 1e-3:
            theta_v_t4 = 0
        # Tính phần trăm quãng đường đã đi để điều chỉnh vận tốc
        percent = 1.0 - np.linalg.norm(att_sixdof) / max(self.att_position_max, 1e-6)

        if percent <= 0.1:
            v_position_sixdof = max(vmax * percent / 0.1, vmax * 0.5)
        elif percent < 0.9:
            v_position_sixdof = vmax
        else:
            v_position_sixdof = vmax * (1.0 - percent) / 0.1

        # Vận tốc hấp dẫn theo vị trí
        v_att_tool_position = -v_position_sixdof * (att_sixdof / np.linalg.norm(att_sixdof))
        # Vận tốc hấp dẫn theo orientation
        v_att_tool_orientation = -50.0 * np.array([self.att_Rx, self.att_Rz])

        # Vector tốc độ mong muốn ở không gian task
        c_tool = np.hstack((v_att_tool_position, v_att_tool_orientation))

        # Tính tốc độ khớp bằng pseudo-inverse Jacobian
        theta_v_sixdof = np.dot(np.linalg.pinv(Jtool_sixdof), c_tool.T)

        for i in range(5):
            theta_v_sixdof[i] = np.clip(theta_v_sixdof[i], -self.vel_limit, self.vel_limit)

        # Ở code gốc: khớp 4 bị set 0, khớp 5,6 map lại
        theta_dot = np.array([
            theta_v_sixdof[0],
            theta_v_sixdof[1],
            theta_v_sixdof[2],
            theta_v_t4,
            theta_v_sixdof[3],
            theta_v_sixdof[4]
        ])

        return theta_dot, np.linalg.norm(att_sixdof), self.att_Rx, self.att_Rz