# robot6_sac_env.py
# Môi trường PyBullet cho SAC:
#   - Robot6 là agent, điều khiển vận tốc 6 khớp.
#   - Robot5 là vật cản động (ở đây để đứng yên, sau này bạn có thể cho nó di chuyển).
# State:
#   q6 (6), dq6 (6), e_pos (3),
#   5 điểm observer, mỗi điểm: [nx, ny, nz, d_norm] -> 20
#   Tổng: 35 chiều.
# Reward:
#   R = k1 * R1 + k2 * R2 + k3 * R3 - k4 * ||a||^2
#   R1: tránh va chạm giữa 2 robot (dựa trên các điểm observer)
#   R2: không vượt giới hạn khớp
#   R3: tiến gần mục tiêu

import time
import numpy as np
import pybullet as p
import pybullet_data
import gym
from gym import spaces
from function_robot.Robot6_observer import Robot6_observer
from function_robot.Robot5_observer import Robot5_observer
from Robot5 import Robot_5_Dof
from Robot6 import Robot_6_Dof

class Robot6SacEnv(gym.Env):
    """
    Môi trường RL cho bài toán:
        Robot6 tránh va chạm với Robot5 và tiến tới mục tiêu.

    Observation (35 chiều):
        - q6_norm: 6  (joint angle, chia cho pi, ∈ [-1, 1])
        - dq6_norm: 6 (joint velocity, chia cho v_max, ∈ [-1, 1])
        - e_pos_norm: 3 (sai số vị trí tool - goal, chuẩn hóa theo 0.5 m)
        - observer_features: 5 * [nx, ny, nz, d_norm] = 20
            nx, ny, nz: vector đơn vị từ điểm trên robot6 tới điểm gần nhất trên robot5
            d_norm = min(d / d_safe, 1)  (0: rất gần, 1: xa hơn d_safe)
    Action (6 chiều):
        - a ∈ [-1, 1]^6
        - joint velocity = a * v_max
    """
    def __init__(self, use_gui: bool = False):
        super().__init__()

        self.use_gui = use_gui

        # Kết nối PyBullet
        if use_gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

        # 1. Tần số robot thật (Bạn chọn 25Hz cho an toàn)
        self.control_freq = 50.0 
        # 2. Tần số vật lý PyBullet (Nên là bội số của 25)
        # 200Hz là con số đẹp (đủ mịn cho va chạm, nhẹ cho CPU)
        self.sim_freq = 200.0   
        self.dt = 1.0 / self.sim_freq
        p.setTimeStep(self.dt)
        # 3. Tính số bước lặp (Substeps)
        # Với sim=200Hz, control=25Hz => n_substeps = 8
        # Nghĩa là: AI ra 1 quyết định -> Robot thực hiện trong 8 bước vật lý (40ms)
        self.n_substeps = int(self.sim_freq / self.control_freq)
        print(f"Setup: Control={self.control_freq}Hz | Sim={self.sim_freq}Hz | Substeps={self.n_substeps}")

        # Mặt phẳng
        self.plane_id = p.loadURDF("plane.urdf")

        # ===== Robot 6 bậc =====
        # Robot 6 bậc
        self.robot6 = Robot_6_Dof(
            urdf_path="./2_robot/URDF_file_2/urdf/6_Dof.urdf",
            base_position=[0, 0, 0],
            base_orientation_euler=[-np.pi / 2, 0, 0],
            use_fixed_base=True
        )

        self.robot6_tool_link = 5  # như trong Robot_6_Dof

        # ===== Robot 5 bậc =====
        self.robot5 = Robot_5_Dof(
            urdf_path="./2_robot/Robot/urdf/5_Dof.urdf",
            base_position=[0, -0.75, 0],
            base_orientation_euler=[-np.pi / 2, 0, 0],
            use_fixed_base=True
        )

        self.robot5_tool_link = 4

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
                rgbaColor=[0, 0, 1, 1]
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


        # Base offset (để cộng với kết quả từ observer)
        self.base6 = np.array([0.0, 0.0, 0.0])
        self.base5 = np.array([0.0, -0.75, 0.0])

        # TODO: # max_steps se tuy thuoc vao tan so cua chuong trinh 
        # Vì tần số lấy mẫu robot là 50hz -> 1 step tương ứng với 1/50 s. 
        # Ta muốn mỗi lần di chuyển cho phép khoảng 20 - 30s -> 30/ (1/50) = 1500 step
        self.max_steps = 1500
        self.step_count = 0

        #### Các đại lượng để chuẩn hóa
        # Vận tốc khớp tối đa (rad/s)
        self.v_max = np.pi / 5.0
        self.q_norm = np.pi
        self.pos_norm = 1 # khoang canh den goal
        self.observe_range = 0.5 # khoang cach den vat can 
        # self.

        # Ngưỡng an toàn / va chạm cho khoảng cách robot6–robot5
        self.d_safe = 0.15   # 15 cm: bắt đầu phạt
        self.d_coll = 0.05   # 5 cm: coi như va chạm
        # Bán kính "nhìn thấy" vật cản cho observer
        # self.observe_range = 0.25  # 25 cm

        # Ngưỡng hoàn thành mục tiêu
        self.goal_tolerance = 0.05  # 5 cm

        # Mục tiêu (mm, sẽ convert sang m khi dùng)
        self.goal_mm = np.array([300.0, 0.0, 300.0])
        # ----- Giới hạn khớp cho R2 (bạn chỉnh theo robot thực nếu cần) -----
        # Giới hạn an toàn (rad)
        self.q_lim = np.array(
            [np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]
        )
        # Giới hạn cơ khí tối đa (rad) để chuẩn hóa phạt
        self.q_max = np.array(
            [np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]
        )

        # ----- Chuẩn hóa khoảng cách tới goal cho R3 -----
        # L_max: khoảng cách lớn nhất trong workspace (m)
        self.L_max = 0.7

        # ----- Trọng số reward -----
        self.k1 = 2.0   # tránh va chạm
        self.k2 = 1.0   # giới hạn khớp
        self.k3 = 2.0   # tới đích
        self.k4 = 0.05  # phạt hành động

        # ===== Gym spaces =====
        # Action: 6 giá trị [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # Observation: 35 chiều
        obs_dim = 6 + 6 + 3 + 5 * 4
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
    # ======================================================================
    #                    CÁC HÀM TIỆN ÍCH CHO ENV
    # ======================================================================

    # ------------------ Cài đặt target & mode ------------------ #

    def set_target_robot6(self, Px, Py, Pz):
        self.Px_robot6 = float(Px)/1000
        self.Py_robot6 = float(Py)/1000
        self.Pz_robot6 = float(Pz)/1000

    def set_target_robot5(self, Px, Py, Pz):
        self.Px_robot5 = float(Px)/1000
        self.Py_robot5 = float(Py)/1000
        self.Pz_robot5 = float(Pz)/1000

    def draw_lines(self):
            """
            Vẽ các đường debug nối từ observer trên Robot 6 tới điểm gần nhất trên Robot 5.
            - Màu ĐỎ: Vật cản nằm trong vùng quan sát (d < observe_range).
            - Màu XANH: Vật cản nằm ngoài vùng quan sát.
            """
            # 1. Lấy trạng thái khớp hiện tại để tính toán vị trí các điểm
            q6, _ = self._get_joint_state_robot6()
            q5_states = p.getJointStates(self.robot5.robot_id, list(range(5)))
            q5 = np.array([s[0] for s in q5_states], dtype=np.float32)

            # 2. Tính toán tọa độ các điểm quan sát (Observer Points)
            # pts6: Các điểm gắn trên thân Robot 6 (Sensor)
            # pts5: Các điểm gắn trên thân Robot 5 (Obstacle)
            # Lưu ý: Phải cộng thêm base offset như trong hàm _get_observer_features
            pts6 = Robot6_observer(q6[0], q6[1], q6[2], q6[3], q6[4]) + self.base6
            pts5 = Robot5_observer(q5[0], q5[1], q5[2], q5[3]) + self.base5

            # 3. Duyệt qua từng điểm trên Robot 6 để vẽ đường
            for p6 in pts6:
                # Tính khoảng cách từ điểm p6 tới TẤT CẢ các điểm trên Robot 5
                diffs = pts5 - p6
                dists = np.linalg.norm(diffs, axis=1)
                
                # Tìm điểm gần nhất trên Robot 5
                min_idx = np.argmin(dists)
                d_min = dists[min_idx]
                closest_pt5 = pts5[min_idx]

                # 4. Xác định màu sắc dựa trên observe_range
                if d_min < self.observe_range:
                    # Nguy hiểm/Đang quan sát thấy -> Màu ĐỎ, nét đậm
                    line_color = [1, 0, 0]
                    line_width = 2.0
                else:
                    # An toàn/Ngoài tầm nhìn -> Màu XANH, nét mảnh
                    line_color = [0, 1, 0]
                    line_width = 1.0

                # 5. Vẽ đường thẳng (UserDebugLine)
                p.addUserDebugLine(
                    lineFromXYZ=p6.tolist(),
                    lineToXYZ=closest_pt5.tolist(),
                    lineColorRGB=line_color,
                    lineWidth=line_width,
                    lifeTime=0  # 0 nghĩa là tồn tại cho đến khi bị xóa (bởi p.removeAllUserDebugItems)
                )

    def _get_observer_features(self):
        """
        Tính 5 điểm observer trên robot6 và mỗi điểm sẽ quan sát điểm gần nhất
        trên robot5. Trả về:
            - features: (20,) = [nx, ny, nz, d_norm] * 5
            - d_min_global: khoảng cách nhỏ nhất giữa 2 robot (m)
            - d_list: list 5 khoảng cách từ mỗi observer của robot6 tới robot5 (m)
        """
        q6, _ = self._get_joint_state_robot6()
        q5_states = p.getJointStates(self.robot5.robot_id, list(range(5)))
        q5 = np.array([s[0] for s in q5_states], dtype=np.float32)

        # 5 điểm trên robot6, 7 điểm trên robot5 (từ 2 hàm observer)
        pts6 = Robot6_observer(q6[0], q6[1], q6[2], q6[3], q6[4]) + self.base6
        pts5 = Robot5_observer(q5[0], q5[1], q5[2], q5[3]) + self.base5

        self.robot6.marker_points = pts6.copy()
        self.robot5.marker_points = pts5.copy()

        features = []
        all_dists = []
        d_list = []

        for p6 in pts6:  # 5 điểm
            diffs = pts5 - p6  # (7, 3)
            dists = np.linalg.norm(diffs, axis=1)
            idx = np.argmin(dists)
            d_min = float(dists[idx])

            all_dists.append(d_min)
            d_list.append(d_min)

            # Nếu vật cản quá xa, coi như không có vật cản
            if d_min > self.observe_range:
                nx, ny, nz = 0.0, 0.0, 0.0
                d_norm = 1.0
            else:
                vec = diffs[idx]
                if d_min > 1e-6:
                    nx, ny, nz = (vec / d_min).tolist()
                else:
                    nx, ny, nz = 0.0, 0.0, 0.0
                # d_norm = min(d / d_safe, 1)
                d_norm = min(d_min / self.d_safe, 1.0)

            features.extend([nx, ny, nz, d_norm])

        if all_dists:
            d_min_global = min(all_dists)
        else:
            d_min_global = self.observe_range

        return (
            np.array(features, dtype=np.float32),
            float(d_min_global),
            [float(d) for d in d_list],
        )

    def _get_joint_state_robot6(self):
        q = self.robot6.take_joint_position()
        dq = self.robot6.take_joint_velocity()
        return q, dq
    
    def _get_obs(self):
        """
        Trả về:
            - obs: vector state 35 chiều
            - d_min: khoảng cách nhỏ nhất giữa 2 robot (m)
            - d_list: list 5 khoảng cách cho R1
            - q6: joint angle hiện tại (6,)
        """
        q6, dq6 = self._get_joint_state_robot6()
        q6_norm = np.clip(q6 / self.q_norm, -1.0, 1.0)
        dq6_norm = np.clip(dq6 / self.v_max, -1.0, 1.0)

        tool_pos = p.getLinkState(bodyUniqueId=self.robot6.robot_id, linkIndex=self.robot6_tool_link)[0]
        goal = self.goal_mm / 1000.0                 # mm -> m
        e_pos = tool_pos - goal                      # m
        e_pos_norm = np.clip(e_pos / self.pos_norm, -1.0, 1.0) # workspace ≈ 0.5 m

        obs_observer, d_min, d_list = self._get_observer_features()

        obs = np.concatenate(
            [q6_norm, dq6_norm, e_pos_norm, obs_observer], axis=0
        )
        return obs.astype(np.float32), float(d_min), d_list, q6

    # ======================================================================
    #                         REWARD COMPONENTS
    # ======================================================================

    def _compute_R1(self, d_list):
        """
        R1: tránh va chạm giữa 2 robot, ∈ [-1, 1]
        d_list: list 5 khoảng cách từ 5 điểm observer của robot6 tới robot5 (m)
        """
        r1_list = []
        for d_cur in d_list:
            if d_cur >= self.d_safe:
                r = 1.0
            elif self.d_coll <= d_cur < self.d_safe:
                # tuyến tính từ 0 -> 1 trong [d_coll, d_safe]
                r = (d_cur - self.d_coll) / (self.d_safe - self.d_coll + 1e-6)
            else:  # d_cur < d_coll
                r = -1.0
            r1_list.append(r)
        return float(np.mean(r1_list))

    def _compute_R2(self, q):
        """
        R2: giới hạn khớp, ∈ [-1, 0]
        q: vector 6 joint angle (rad)
        """
        r2_list = []
        for th, th_lim, th_max in zip(q, self.q_lim, self.q_max):
            if abs(th) <= th_lim:
                r = 0.0
            else:
                r = - (abs(th) - th_lim) / (th_max - th_lim + 1e-6)
                r = max(r, -1.0)
            r2_list.append(r)
        return float(np.mean(r2_list))

    def _compute_R3(self, d_goal):
        """
        R3: tiến gần mục tiêu, ∈ [0, 1]
        d_goal: khoảng cách tool–goal (m)
        """
        R3 = 1.0 - d_goal / (self.L_max + 1e-6)
        return float(np.clip(R3, 0.0, 1.0))

    # ======================================================================
    #                           GYM API
    # ======================================================================

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.plane_id = p.loadURDF("plane.urdf")
        self.step_count = 0

        xbase_robot5 = self.base5[0]
        ybase_robot5 = self.base5[1]
        zbase_robot5 = self.base5[2]

        self.robot6 = Robot_6_Dof(
            urdf_path="./2_robot/URDF_file_2/urdf/6_Dof.urdf",
            base_position=[0, 0, 0],
            base_orientation_euler=[-np.pi / 2, 0, 0],
            use_fixed_base=True
        )
        # ===== Robot 5 bậc =====
        self.robot5 = Robot_5_Dof(
            urdf_path="./2_robot/Robot/urdf/5_Dof.urdf",
            base_position=[0, -0.75, 0],
            base_orientation_euler=[-np.pi / 2, 0, 0],
            use_fixed_base=True
        )

        # 1. Quả cầu mục tiêu Robot 6
        self.sphere_robot6 = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.01,  # Có thể giảm xuống 0.05 cho đỡ vướng mắt
                rgbaColor=[1, 0, 0, 1] # Màu Đỏ
            ),
            basePosition=[0, 0, 0]
        )

        # 2. Quả cầu mục tiêu Robot 5
        self.sphere_robot5 = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.01,
                rgbaColor=[0, 0, 1, 1] # Màu Xanh dương (cho khác robot 6)
            ),
            basePosition=[0, 0, 0]
        )

        # 3. Quả cầu Workspace (nếu cần hiển thị)
        self.workspace = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_SPHERE,
                radius=self.workspace_data[3],
                rgbaColor=[0, 1, 0, 0.2] # Xanh lá mờ
            ),
            basePosition=[self.workspace_data[0], self.workspace_data[1], self.workspace_data[2]]
        )

        # Random góc khớp ban đầu cho robot6
        for i in range(6):
            q = np.random.uniform(-np.pi / 4, np.pi / 4)
            p.resetJointState(self.robot6.robot_id, i, q, 0.0)

        # Robot5: để ở cấu hình 0 cho đơn giản (có thể random nếu muốn)
        for i in range(5):
            p.resetJointState(self.robot5.robot_id, i, 0.0, 0.0)

        # Random mục tiêu (mm) trong vùng làm việc hợp lý
        self.goal_robot6 = np.array([
            np.random.uniform(-300, 300),   # x
            np.random.uniform(-200, -400),  # y
            np.random.uniform(250, 450),   # z
        ])

        self.goal_mm = self.goal_robot6

        self.goal_robot5 = np.array([
            np.random.uniform(-300, 300),   # x
            np.random.uniform(-200, -400),  # y
            np.random.uniform(250, 450),   # z
        ])

        self.set_target_robot6(self.goal_robot6[0], self.goal_robot6[1], self.goal_robot6[2] )
        self.set_target_robot5(self.goal_robot5[0], self.goal_robot5[1], self.goal_robot5[2] )

        obs, _, _, _ = self._get_obs()
        return obs, {}

    def step(self, action):
        p.removeAllUserDebugItems()
        self.step_count += 1
        # Scale action -> joint velocity
        action = np.clip(action, -1.0, 1.0)
        theta_dot_robot6 = action * self.v_max
        self.robot6.set_joint_velocity(theta_dot_robot6)
        
        theta_dot_robot5 = [0, 0, 0, 0, 0]
        self.robot5.set_joint_velocity(theta_dot_robot5)

        # >>> Cập nhật vị trí 6 điểm quan sát trên link
        self.robot6.update_link_markers()
        self.robot5.update_link_markers()
        self.draw_lines()

        # =============================================
        # Điều khiển robot5 
        # =============================================
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
        if pos_err_robot5 < 0.05:
            self.goal_robot5 = np.array([
                np.random.uniform(-300, 300),   # x
                np.random.uniform(-200, -400),  # y
                np.random.uniform(250, 450),   # z
            ])
            self.set_target_robot5(self.goal_robot5[0], self.goal_robot5[1], self.goal_robot5[2] )

        # Gửi lệnh vận tốc
        self.robot5.set_joint_velocity(theta_dot_robot5)


        # =============================================
        # Điều khiển robot5 
        # =============================================

        # Duy trì lệnh đó trong suốt 40ms (8 bước sim)
        for _ in range(self.n_substeps):
            # trong 1 chu ky lenh, thuc hien n_substeps.
            # Đây là môi trường mô phỏng, miễn tần số của chương trình và tần số robot chênh lệch là được. 
            # Trong thực tế thì ta còn phải set đến thời gian nữa. 
            p.stepSimulation()
            if self.use_gui:
                # Sleep để mắt thường nhìn thấy đúng tốc độ thực (chỉ dùng khi debug)
                time.sleep(self.dt)
        # Cập nhật vị trí quả cầu mục tiêu

        p.resetBasePositionAndOrientation(
            self.sphere_robot5,
            [self.Px_robot5, self.Py_robot5, self.Pz_robot5],
            [0, 0, 0, 1]
        )

        # Cập nhật vị trí quả cầu mục tiêu
        p.resetBasePositionAndOrientation(
            self.sphere_robot6,
            [self.Px_robot6, self.Py_robot6, self.Pz_robot6],
            [0, 0, 0, 1]
        )
        
        # Lấy state + thông tin reward
        obs, d_min, d_list, q6 = self._get_obs()
        tool_pos = p.getLinkState(bodyUniqueId=self.robot6.robot_id, linkIndex=self.robot6_tool_link)[0]
        goal = self.goal_mm / 1000.0
        d_goal = float(np.linalg.norm(tool_pos - goal))

        # ----- Tính R1, R2, R3 -----
        R1 = self._compute_R1(d_list)
        R2 = self._compute_R2(q6)
        R3 = self._compute_R3(d_goal)

        # ----- Reward tổng -----
        reward = (
            self.k1 * R1 +
            self.k2 * R2 +
            self.k3 * R3 -
            self.k4 * (float(np.linalg.norm(action) ** 2))
        )

        # ----- Điều kiện kết thúc -----
        terminated = False
        truncated = False

        # Va chạm
        if d_min < self.d_coll:
            terminated = True

        # Tới đích
        if d_goal < self.goal_tolerance:
            terminated = True

        # Hết số bước
        if self.step_count >= self.max_steps:
            truncated = True

        info = {
            "d_goal": d_goal,
            "d_min": d_min,
            "R1": R1,
            "R2": R2,
            "R3": R3,
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        p.disconnect()


# ======================================================================
#                         TEST NHANH ENV
# ======================================================================
if __name__ == "__main__":
    env = Robot6SacEnv(use_gui=True)
    obs, _ = env.reset()
    print("Obs dim:", obs.shape)
    i = 0 
    for epoch in range(300):
        while True: 
            a = env.action_space.sample()
            obs, r, done, trunc, info = env.step(a)
            if done or trunc:
                print("Episode ended:", info)
                obs, _ = env.reset()
                break

    env.close()
