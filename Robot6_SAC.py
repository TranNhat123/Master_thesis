# Robot6_SAC.py
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
    Môi trường RL Hybrid cho Robot 6 bậc tự do:
    - Input (41 chiều): q, dq, error, observer, u_base (tín hiệu phi tuyến).
    - Output (7 chiều): 6 vận tốc RL + 1 hệ số Alpha.
    - Cơ chế: Pha trộn tín hiệu giữa bộ điều khiển nền (Base) và RL.
    """
    def __init__(self, use_gui: bool = False):
        super().__init__()
        self.use_gui = use_gui
        if use_gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

        # Cấu hình tần số
        self.control_freq = 50.0 
        self.sim_freq = 200.0   
        self.dt = 1.0 / self.sim_freq
        p.setTimeStep(self.dt)
        self.n_substeps = int(self.sim_freq / self.control_freq)

        self.plane_id = p.loadURDF("plane.urdf")

        # Khởi tạo Robot
        self.robot6 = Robot_6_Dof(
            urdf_path="./2_robot/URDF_file_2/urdf/6_Dof.urdf",
            base_position=[0, 0, 0],
            base_orientation_euler=[-np.pi / 2, 0, 0],
            use_fixed_base=True
        )
        self.robot6_tool_link = 5 

        self.robot5 = Robot_5_Dof(
            urdf_path="./2_robot/Robot/urdf/5_Dof.urdf",
            base_position=[0, -0.75, 0],
            base_orientation_euler=[-np.pi / 2, 0, 0],
            use_fixed_base=True
        )
        self.robot5_tool_link = 4

        # Các đối tượng hiển thị (Target, Workspace)
        self.sphere_robot6 = p.createMultiBody(baseMass=0, baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0, 0, 0.6]), basePosition=[0,0,0])
        self.sphere_robot5 = p.createMultiBody(baseMass=0, baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[0, 0, 1, 0.6]), basePosition=[0,0,0])
        
        self.workspace_data = np.array([0, 0, 0.2, 0.7])
        self.workspace = p.createMultiBody(baseMass=0, baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=self.workspace_data[3], rgbaColor=[0, 1, 0, 0.1]), basePosition=self.workspace_data[:3])

        # Tham số vận tốc & Mục tiêu
        self.vmax_robot6 = 0.05
        self.vmax_robot5 = 0.05
        self.goal_mm = np.array([300.0, 0.0, 300.0])
        self.Px_robot6, self.Py_robot6, self.Pz_robot6 = 0,0,0
        self.Px_robot5, self.Py_robot5, self.Pz_robot5 = 0,0,0

        self.base6 = np.array([0.0, 0.0, 0.0])
        self.base5 = np.array([0.0, -0.75, 0.0])

        self.max_steps = 1500
        self.step_count = 0

        # Tham số chuẩn hóa
        self.v_max = self.robot6.vel_limit  # Lấy từ Robot6 class
        self.q_norm = np.pi
        self.pos_norm = 1.0
        self.observe_range = 0.5 
        self.d_safe = 0.15
        self.d_coll = 0.05
        self.goal_tolerance = 0.05
        self.L_max = 0.7

        # Reward weights
        self.k1 = 2.0   # Va chạm
        self.k2 = 1.0   # Giới hạn khớp
        self.k3 = 2.0   # Tới đích
        self.k4 = 0.1   # Tiết kiệm năng lượng
        self.k_switch = 1.0 # Phạt khi dùng RL (Alpha lớn)

        # ===== GYM SPACES =====
        # Action: 7 chiều (6 vận tốc + 1 alpha). Tất cả đều [-1, 1] do Tanh.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        # Observation: q6(6) + dq6(6) + e_pos(3) + observer(5*1*4=20) + u_base(6) = 41 chiều
        self.observer_points = 5
        self.observer_k = 1  # CHỈ LẤY 1 ĐIỂM GẦN NHẤT
        
        obs_dim = 6 + 6 + 3 + (self.observer_points * self.observer_k * 4) + 6
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

    def set_target_robot6(self, Px, Py, Pz):
        self.Px_robot6 = float(Px)/1000
        self.Py_robot6 = float(Py)/1000
        self.Pz_robot6 = float(Pz)/1000
    
    def set_target_robot5(self, Px, Py, Pz):
        self.Px_robot5 = float(Px)/1000
        self.Py_robot5 = float(Py)/1000
        self.Pz_robot5 = float(Pz)/1000

    def draw_lines(self):
        q6, _ = self._get_joint_state_robot6()
        q5_states = p.getJointStates(self.robot5.robot_id, list(range(5)))
        q5 = np.array([s[0] for s in q5_states], dtype=np.float32)
        pts6 = Robot6_observer(q6[0], q6[1], q6[2], q6[3], q6[4]) + self.base6
        pts5 = Robot5_observer(q5[0], q5[1], q5[2], q5[3]) + self.base5
        for p6 in pts6:
            diffs = pts5 - p6
            dists = np.linalg.norm(diffs, axis=1)
            min_idx = np.argmin(dists)
            closest_pt5 = pts5[min_idx]
            d_min = dists[min_idx]
            color = [1, 0, 0] if d_min < self.observe_range else [0, 1, 0]
            width = 2.0 if d_min < self.observe_range else 1.0
            p.addUserDebugLine(p6.tolist(), closest_pt5.tolist(), lineColorRGB=color, lineWidth=width, lifeTime=0)

    def _get_joint_state_robot6(self):
        q = self.robot6.take_joint_position()
        dq = self.robot6.take_joint_velocity()
        return q, dq

    def _get_observer_features(self):
        q6, _ = self._get_joint_state_robot6()
        q5_states = p.getJointStates(self.robot5.robot_id, list(range(5)))
        q5 = np.array([s[0] for s in q5_states], dtype=np.float32)
        pts6 = Robot6_observer(q6[0], q6[1], q6[2], q6[3], q6[4]) + self.base6
        pts5 = Robot5_observer(q5[0], q5[1], q5[2], q5[3]) + self.base5
        features = []
        d_list = []
        all_dists = []
        k = getattr(self, "observer_k", 1)
        for p6 in pts6:
            diffs = pts5 - p6
            dists = np.linalg.norm(diffs, axis=1)
            d_min = float(np.min(dists))
            d_list.append(d_min)
            all_dists.append(d_min)

            idx_sorted = np.argsort(dists)[:k]
            for idx in idx_sorted:
                d_cur = float(dists[idx])
                if d_cur > self.observe_range:
                    nx, ny, nz, d_norm = 0.0, 0.0, 0.0, 1.0
                else:
                    vec = diffs[idx]
                    if d_cur > 1e-6:
                        nx, ny, nz = (vec / d_cur).tolist()
                    else:
                        nx, ny, nz = 0.0, 0.0, 0.0
                    d_norm = min(d_cur / self.d_safe, 1.0)
                features.extend([nx, ny, nz, d_norm])

        d_min_global = min(all_dists) if all_dists else self.observe_range
        return np.array(features, dtype=np.float32), float(d_min_global), d_list

    def _get_obs(self):
        # 1. Trạng thái khớp
        q6, dq6 = self._get_joint_state_robot6()
        q6_norm = np.clip(q6 / self.q_norm, -1.0, 1.0)
        dq6_norm = np.clip(dq6 / self.v_max, -1.0, 1.0)

        # 2. Sai số vị trí
        tool_pos = p.getLinkState(self.robot6.robot_id, self.robot6_tool_link)[0]
        goal = self.goal_mm / 1000.0
        e_pos = tool_pos - goal
        e_pos_norm = np.clip(e_pos / self.pos_norm, -1.0, 1.0)

        # 3. Observer
        obs_observer, d_min, d_list = self._get_observer_features()

        # 4. U_Base (Tín hiệu điều khiển phi tuyến dự báo)
        joint_position_robot6 = self.robot6.take_joint_position()
        joint_position_robot6 = (joint_position_robot6 + np.pi) % (2 * np.pi) - np.pi
        
        u_base_vel, _, _, _ = self.robot6.get_theta_dot(
            x_target=self.Px_robot6, y_target=self.Py_robot6, z_target=self.Pz_robot6,
            vmax=self.vmax_robot6, joint_position_local=joint_position_robot6
        )
        
        # Chuẩn hóa u_base về [-1, 1]
        u_base_norm = np.clip(np.array(u_base_vel) / self.v_max, -1.0, 1.0)

        # 5. Ghép lại (Tổng 41 chiều)
        obs = np.concatenate([q6_norm, dq6_norm, e_pos_norm, obs_observer, u_base_norm], axis=0)
        return obs.astype(np.float32), float(d_min), d_list, q6

    def reset(self, *, seed=None, options=None):
        if seed is not None: np.random.seed(seed)
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.plane_id = p.loadURDF("plane.urdf")
        self.step_count = 0

        # Load Robots
        self.robot6 = Robot_6_Dof(urdf_path="./2_robot/URDF_file_2/urdf/6_Dof.urdf", base_position=[0, 0, 0], base_orientation_euler=[-np.pi/2, 0, 0], use_fixed_base=True)
        self.robot5 = Robot_5_Dof(urdf_path="./2_robot/Robot/urdf/5_Dof.urdf", base_position=[0, -0.75, 0], base_orientation_euler=[-np.pi/2, 0, 0], use_fixed_base=True)

        # Reset Objects
        self.sphere_robot6 = p.createMultiBody(baseMass=0, baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0, 0, 0.6]), basePosition=[0,0,0])
        self.sphere_robot5 = p.createMultiBody(baseMass=0, baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[0, 0, 1, 0.6]), basePosition=[0,0,0])
        self.workspace = p.createMultiBody(baseMass=0, baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=self.workspace_data[3], rgbaColor=[0, 1, 0, 0.1]), basePosition=self.workspace_data[:3])

        # Random Pose & Goals
        for i in range(6): p.resetJointState(self.robot6.robot_id, i, np.random.uniform(-np.pi/4, np.pi/4), 0.0)
        for i in range(5): p.resetJointState(self.robot5.robot_id, i, 0.0, 0.0)

        self.goal_robot6 = np.array([np.random.uniform(-300, 300), np.random.uniform(-200, -400), np.random.uniform(250, 450)])
        self.goal_mm = self.goal_robot6
        self.goal_robot5 = np.array([np.random.uniform(-300, 300), np.random.uniform(-200, -400), np.random.uniform(250, 450)])
        
        self.set_target_robot6(*self.goal_robot6)
        self.set_target_robot5(*self.goal_robot5)

        obs, _, _, _ = self._get_obs()
        return obs, {}

    def step(self, action):
        p.removeAllUserDebugItems()
        self.step_count += 1
        
        # 1. Giải mã Action (Tất cả đều từ Tanh [-1, 1])
        u_rl_norm = action[0:6] 
        alpha_raw = action[6]
        
        # Map alpha từ [-1, 1] sang [0, 1] cho việc pha trộn
        alpha = (alpha_raw + 1.0) / 2.0
        alpha = np.clip(alpha, 0.0, 1.0)

        # 2. Lấy tín hiệu Base Controller
        joint_position_robot6 = self.robot6.take_joint_position()
        joint_position_robot6 = (joint_position_robot6 + np.pi) % (2 * np.pi) - np.pi
        
        u_base_vel, _, _, _ = self.robot6.get_theta_dot(
            x_target=self.Px_robot6, y_target=self.Py_robot6, z_target=self.Pz_robot6,
            vmax=self.vmax_robot6, joint_position_local=joint_position_robot6
        )
        
        # 3. Pha trộn tín hiệu (Hybrid Mixing)
        u_rl_vel = u_rl_norm * self.v_max
        theta_dot_robot6 = (1 - alpha) * np.array(u_base_vel) + alpha * u_rl_vel
        theta_dot_robot6 = np.clip(theta_dot_robot6, -self.v_max, self.v_max)

        # 4. Gửi lệnh
        self.robot6.set_joint_velocity(theta_dot_robot6)

        # --- Robot 5 & Simulation ---
        joint_position_robot5 = self.robot5.take_joint_position()
        joint_position_robot5 = (joint_position_robot5 + np.pi) % (2 * np.pi) - np.pi
        theta_dot_robot5, pos_err_robot5, _, _ = self.robot5.get_theta_dot(
            x_target=self.Px_robot5, y_target=self.Py_robot5, z_target=self.Pz_robot5,
            vmax=self.vmax_robot5, joint_position_local=joint_position_robot5
        )
        if pos_err_robot5 < 0.05:
            self.goal_robot5 = np.array([np.random.uniform(-300, 300), np.random.uniform(-200, -400), np.random.uniform(250, 450)])
            self.set_target_robot5(*self.goal_robot5)
        self.robot5.set_joint_velocity(theta_dot_robot5)

        self.robot6.update_link_markers()
        self.robot5.update_link_markers()
        self.draw_lines()

        for _ in range(self.n_substeps):
            p.stepSimulation()
            if self.use_gui: time.sleep(self.dt)

        p.resetBasePositionAndOrientation(self.sphere_robot5, [self.Px_robot5, self.Py_robot5, self.Pz_robot5], [0,0,0,1])
        p.resetBasePositionAndOrientation(self.sphere_robot6, [self.Px_robot6, self.Py_robot6, self.Pz_robot6], [0,0,0,1])

        # 5. Reward
        obs, d_min, d_list, q6 = self._get_obs()
        tool_pos = p.getLinkState(self.robot6.robot_id, self.robot6_tool_link)[0]
        goal = self.goal_mm / 1000.0
        d_goal = float(np.linalg.norm(tool_pos - goal))

        r1_list = []
        for d in d_list:

            if d >= self.d_safe: 
                r = 1.0
            elif self.d_coll <= d < self.d_safe: 
                r = (d - self.d_coll)/(self.d_safe - self.d_coll + 1e-6)
            else: 
                r = -1.0

            r1_list.append(r)

        R1 = float(np.min(r1_list)) # Lay phan thuong theo diem yeu nhat, tránh viec huy sinh 1 khớp để cao điểm khớp khác 
        R2 = 0.0  # Phan thuong theo joint limit, hoi kho để define nên mình chưa define. 
        R3 = float(np.clip(1.0 - d_goal / (self.L_max + 1e-6), 0.0, 1.0))
        R4 = - (alpha ** 2) # Phạt Alpha lớn

        reward = self.k1 * R1 + self.k2 * R2 + self.k3 * R3 + self.k_switch * R4

        terminated = True if (d_min < self.d_coll or d_goal < self.goal_tolerance) else False
        truncated = True if self.step_count >= self.max_steps else False
        
        info = {"d_goal": d_goal, "d_min": d_min, "alpha": alpha, "R1": R1, "R4": R4}
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
