import numpy as np
import pybullet as p
import pybullet_data
from Robot5 import Robot_5_Dof
from Robot6 import Robot_6_Dof

class Environment:
    def __init__(self, use_gui=True):
        """Khởi tạo PyBullet, mặt phẳng, robot và quả cầu target."""
        """Đây là môi trường để sau này deploy chương trình"""
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
        self.sphere_robot6 = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.01,
                rgbaColor=[1, 0, 0, 1]
            ),
            basePosition=[0, 0, 0]
        )

                # Quả cầu hiển thị mục tiêu
        self.sphere_robot5 = p.createMultiBody(
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
        self.vmax_robot6 = 0.05 # (m/s)
        self.vmax_robot5 = 0.05  # (m/s)
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
        self.Px_robot6 = float(Px)/1000
        self.Py_robot6 = float(Py)/1000
        self.Pz_robot6 = float(Pz)/1000

    def set_target_robot5(self, Px, Py, Pz):
        self.Px_robot5 = float(Px)/1000
        self.Py_robot5 = float(Py)/1000
        self.Pz_robot5 = float(Pz)/1000

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
            self.sphere_robot6,
            [self.Px_robot6, self.Py_robot6, self.Pz_robot6],
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
            self.sphere_robot5,
            [self.Px_robot5, self.Py_robot5, self.Pz_robot5],
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
