import numpy as np
import pybullet as p
import pybullet_data
from function_robot.Coordinate_sixdof import Coordinate_sixdof
from function_robot.Jtool_sixdof_function import Jtool_sixdof_function
from function_robot.Jtool_fanuc_function import Jtool_fanuc_function
from function_robot.Robot6_observer import Robot6_observer
from function_robot.Robot5_observer import Robot5_observer

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

        # (Có thể dùng nếu bạn cần) – hiện tại không dùng giá trị trả về
        self.jointpos = Coordinate_sixdof(t1, t2, t3, t4, t5)
        self.marker_points = Robot6_observer(t1, t2, t3, t4, t5)
        # Lấy vị trí tool từ PyBullet
        pos = p.getLinkState(bodyUniqueId=self.robot_id, linkIndex=self.tool_link_index)[0]
        x, y, z = pos
        x, y, z = x*1000, y*1000, z*1000 # Convert tu m -> mm
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

        # Giới hạn tốc độ cho 5 khớp đầu
        limit = np.pi / 10.0
        for i in range(5):
            theta_v_sixdof[i] = np.clip(theta_v_sixdof[i], -limit, limit)

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

class Environment:
    def __init__(self, use_gui=True):
        """Khởi tạo PyBullet, mặt phẳng, robot và quả cầu target."""
        self.use_gui = use_gui

        # Kết nối PyBullet
        if use_gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

        # Mặt phẳng
        self.plane_id = p.loadURDF("plane.urdf")

        # Robot 6 bậc
        self.robot6 = Robot_6_Dof(
            urdf_path="./2_robot/URDF_file_2/urdf/6_Dof.urdf",
            base_position=[0, 0, 0],
            base_orientation_euler=[-np.pi / 2, 0, 0],
            use_fixed_base=True
        )

        # Robot 5 bậc
        self.robot5 = Robot_5_Dof(
            urdf_path="./2_robot/Robot/urdf/5_Dof.urdf",
            base_position=[0, -0.75, 0],
            base_orientation_euler=[-np.pi / 2, 0, 0],
            use_fixed_base=True
        )

        # Quả cầu hiển thị mục tiêu
        self.sphere_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.01,
                rgbaColor=[1, 0, 0, 1]
            ),
            basePosition=[0, 0, 0]
        )

        self.workspace_data = np.array([0, 0, 0.2, 0.7]) # x, y, z, radius
        # Quả cầu hiển thị mục tiêu
        self.workspace = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_SPHERE,
                radius= self.workspace_data[3],
                rgbaColor=[0, 1, 0, 0.2]
            ),
            basePosition=[self.workspace_data[0], self.workspace_data[1], self.workspace_data[2]]
        )

        # Tham số mô phỏng và điều khiển
        self.vmax_robot6 = 50
        self.vmax_robot5 = 50
        self.dt = 1.0 / 240.0

        # Target mặc định
        self.Px_robot6 = 0.0
        self.Py_robot6 = 0.0
        self.Pz_robot6 = 0.0

                # Target mặc định
        self.Px_robot5 = 0.0
        self.Py_robot5 = 0.0
        self.Pz_robot5 = 0.0
        # 0 - real time, 1 - non-real time
        self.flag_mode_simulation = 1

    # ------------------ Cài đặt target & mode ------------------ #

    def set_target_robot6(self, Px, Py, Pz):
        self.Px_robot6 = float(Px)
        self.Py_robot6 = float(Py)
        self.Pz_robot6 = float(Pz)

    def set_target_robot5(self, Px, Py, Pz):
        self.Px_robot5 = float(Px)
        self.Py_robot5 = float(Py)
        self.Pz_robot5 = float(Pz)

    def set_mode(self, flag_mode_simulation):
        """
        0 - real time
        1 - non-real time (mình sẽ tự stepSimulation)
        """
        self.flag_mode_simulation = int(flag_mode_simulation)
        if self.flag_mode_simulation == 0:
            # real-time
            p.setRealTimeSimulation(1)
        else:
            # non-real-time
            p.setRealTimeSimulation(0)

    def draw_lines(self):
        obstacles = []
        makers_robot5 = self.robot5.marker_points.copy()
        makers_robot6 = self.robot6.marker_points.copy()
        for i in range(len(makers_robot5[:, 0])):
            if np.linalg.norm(makers_robot5[i, :] - self.workspace_data[0:3]) < self.workspace_data[3]:
                obstacles.append(makers_robot5[i, :])

        
        if len(obstacles) == 0:
            return  # không có obstacle thì thôi

        obstacles = np.array(obstacles)
    
        # Vẽ các đường thẳng nối từ từng obstacle tới tất cả marker của robot6
        for o in obstacles:
            for j in range(makers_robot6.shape[0]):
                pt6 = makers_robot6[j, :]

                p.addUserDebugLine(
                    o.tolist(),          # điểm bắt đầu (obstacle)
                    pt6.tolist(),        # điểm kết thúc (marker robot6)
                    [1, 0, 0],           # màu đỏ
                    lineWidth=1.5
                )
    # ------------------ Một bước mô phỏng ------------------ #
    def step(self):
        p.removeAllUserDebugItems()

        #### Robot6 ##########
        #### Robot6 ##########

        # Lấy joint và wrap về [-pi, pi]
        joint_position_robot6 = self.robot6.take_joint_position()
        joint_position_robot6 = (joint_position_robot6 + np.pi) % (2 * np.pi) - np.pi

        # Controller
        theta_dot_robot6, pos_err_robot6, att_Rx_robot6, att_Rz_robot6 = self.robot6.get_theta_dot(
            x_target=self.Px_robot6,
            y_target=self.Py_robot6,
            z_target=self.Pz_robot6,
            vmax=self.vmax_robot6,
            joint_position_local=joint_position_robot6
        )

        # Điều kiện dừng
        if pos_err_robot6 < 1e-4 and abs(att_Rx_robot6) < 1e-3 and abs(att_Rz_robot6) < 1e-3:
            theta_dot_robot6 = np.zeros(6)

        # Gửi lệnh vận tốc
        self.robot6.set_joint_velocity(theta_dot_robot6)

        # Cập nhật vị trí quả cầu mục tiêu
        p.resetBasePositionAndOrientation(
            self.sphere_id,
            [self.Px_robot6/1000, self.Py_robot6/1000, self.Pz_robot6/1000],
            [0, 0, 0, 1]
        )
        #################################################################################
        #################################################################################
        #### Robot5 ##########
        #### Robot5 ##########

        # Lấy joint và wrap về [-pi, pi]
        joint_position_robot5 = self.robot5.take_joint_position()
        joint_position_robot5 = (joint_position_robot5 + np.pi) % (2 * np.pi) - np.pi

        # Controller
        theta_dot_robot5, pos_err_robot5, att_Rx_robot5, att_Rz_robot5 = self.robot5.get_theta_dot(
            x_target=self.Px_robot5,
            y_target=self.Py_robot5,
            z_target=self.Pz_robot5,
            vmax=self.vmax_robot5,
            joint_position_local=joint_position_robot5
        )

        # Điều kiện dừng
        if pos_err_robot5 < 1e-4 and abs(att_Rx_robot5) < 1e-3 and abs(att_Rz_robot5) < 1e-3:
            theta_dot_robot5 = np.zeros(5)

        # Gửi lệnh vận tốc
        self.robot5.set_joint_velocity(theta_dot_robot5)

        # Cập nhật vị trí quả cầu mục tiêu
        p.resetBasePositionAndOrientation(
            self.sphere_id,
            [self.Px_robot5/1000, self.Py_robot5/1000, self.Pz_robot5/1000],
            [0, 0, 0, 1]
        )
        #################################################################################
        #################################################################################

        # >>> Cập nhật vị trí 6 điểm quan sát trên link
        self.robot6.update_link_markers()
        self.robot5.update_link_markers()
        self.draw_lines()
        # Nếu non-real-time thì phải tự step
        if self.flag_mode_simulation == 1:
            p.stepSimulation()
            # time.sleep(self.dt)  # nếu muốn chậm lại

    def run(self):
        """Vòng lặp vô hạn."""
        while True:
            self.step()


if __name__ == "__main__":
    env = Environment(use_gui=True)

    # Nhập mục tiêu
    Px = float(input("Nhap gia tri cua Px: "))
    Py = float(input("Nhap gia tri cua Py: "))
    Pz = float(input("Nhap gia tri cua Pz: "))
    env.set_target_robot6(Px, Py, Pz)
    env.set_target_robot5(Px, Py, Pz)
    # Chọn mode
    flag_mode = int(input("0 - real time | 1 - non-real time: "))
    env.set_mode(flag_mode)

    # Chạy mô phỏng
    env.run()
