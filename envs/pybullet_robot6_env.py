import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces

from main import Robot_6_Dof, Robot_5_Dof
from function_robot.Robot6_observer import Robot6_observer
from function_robot.Robot5_observer import Robot5_observer


class PyBullet6DOFEnv(gym.Env):
    """Gym environment for the 6-DOF robot.

    Observation (flattened):
      - 6 joint angles (rad)
      - 6 joint velocities (rad/s) (estimated)
      - 3 target position (mm)
      - 5 markers * 5 nearest obstacles * 4 values = 100 dims
        (for each obstacle: nx, ny, nz, d) where nx,ny,nz is direction vector
        from marker to obstacle and d = distance / safety_radius (clipped to [0,1])

    Action:
      - 6 joint velocity commands (rad/s)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, render=False, max_vel=1.0, episode_length=240, n_obstacles=30, safety_radius_mm=150.0, use_robot5_obstacles=False,
                 use_nonlinear_ctrl=True, use_rl_ctrl=True, blend_alpha=0.5, behavior_mode='blend'):
        self.render_mode = render
        self.max_vel = float(max_vel)
        self.episode_length = int(episode_length)
        self.n_obstacles = int(n_obstacles)
        self.safety_radius = float(safety_radius_mm)
        self.use_robot5_obstacles = bool(use_robot5_obstacles)
        # Controller switches
        self.use_nonlinear_ctrl = bool(use_nonlinear_ctrl)
        self.use_rl_ctrl = bool(use_rl_ctrl)
        self.blend_alpha = float(blend_alpha)  # alpha in [0,1], final_action = alpha*rl + (1-alpha)*nonlinear
        # behavior_mode: 'rl' | 'nonlinear' | 'blend' | 'none'
        self.behavior_mode = str(behavior_mode)

        if self.render_mode:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        p.loadURDF("plane.urdf")

        # Create robots
        self.robot6 = Robot_6_Dof(
            urdf_path="./2_robot/URDF_file_2/urdf/6_Dof.urdf",
            base_position=[0, 0, 0],
            base_orientation_euler=[-np.pi / 2, 0, 0],
            use_fixed_base=True,
        )

        if self.use_robot5_obstacles:
            self.robot5 = Robot_5_Dof(
                urdf_path="./2_robot/Robot/urdf/5_Dof.urdf",
                base_position=[0, -0.75, 0],
                base_orientation_euler=[-np.pi / 2, 0, 0],
                use_fixed_base=True,
            )
        else:
            self.robot5 = None

        # If robot5 exists, prepare per-joint phase offsets for dynamic motion
        if self.robot5 is not None:
            # 5 joints for robot5
            self._robot5_phase = np.random.uniform(0, 2 * np.pi, size=5).astype(np.float32)
            self._robot5_amp = 0.5  # amplitude (rad/s) for joint velocity
            self._robot5_freq = 0.05  # frequency multiplier for time step

        # Action / observation spaces
        self.action_space = spaces.Box(low=-self.max_vel, high=self.max_vel, shape=(6,), dtype=np.float32)

        obs_dim = 6 + 6 + 3 + (5 * 5 * 4)  # joints + joint_vels + target + markers*neighbors*4
        high = np.ones(obs_dim, dtype=np.float32) * np.finfo(np.float32).max
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Internal state
        self._prev_joints = np.zeros(6, dtype=np.float32)
        self._step = 0

        # Random static obstacles (in mm) within workspace sphere around robot base
        self._static_obstacles = self._sample_obstacles(self.n_obstacles)

        # Target default (mm)
        self._target = np.array([0.0, 0.0, 200.0], dtype=np.float32)

    def _sample_obstacles(self, n):
        # Sample obstacles uniformly in a cylinder / sphere above the ground (in mm)
        obs = []
        for _ in range(n):
            theta = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(100.0, 700.0)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = np.random.uniform(50.0, 400.0)
            obs.append([x, y, z])
        return np.array(obs, dtype=np.float32)

    def _get_joint_positions(self):
        return self.robot6.take_joint_position().astype(np.float32)

    def _get_joint_velocities(self, joints, dt=1.0 / 240.0):
        # simple finite difference w.r.t previous stored joints
        vel = (joints - self._prev_joints) / max(dt, 1e-6)
        return vel.astype(np.float32)

    def _get_marker_points_mm(self, joints):
        # Robot6_observer expects t1..t5 (meters units and radians) and returns meters positions
        t1, t2, t3, t4, t5, t6 = joints
        pts_m = Robot6_observer(t1, t2, t3, t4, t5)
        pts_mm = pts_m * 1000.0
        return pts_mm  # shape (5,3)

    def _gather_obstacle_points(self):
        # Return obstacles in mm as (N,3)
        if self.use_robot5_obstacles and self.robot5 is not None:
            # take robot5 marker points (Robot5_observer returns meters, main adds base)
            # use current joint states of robot5
            jp = self.robot5.take_joint_position()
            pts = Robot5_observer(jp[0], jp[1], jp[2], jp[3])  # meters
            pts = pts * 1000.0 + np.array(self.robot5.base_position) * 1000.0
            return pts.astype(np.float32)
        else:
            return self._static_obstacles

    def _marker_observations(self, marker_points_mm, obstacles_mm, n_neighbors=5):
        # For each marker, find n_neighbors nearest obstacles and compute [nx,ny,nz,d]
        M = marker_points_mm.shape[0]
        K = n_neighbors
        result = np.zeros((M, K, 4), dtype=np.float32)
        if obstacles_mm is None or len(obstacles_mm) == 0:
            return result.flatten()

        for i in range(M):
            marker = marker_points_mm[i]
            vecs = obstacles_mm - marker  # (N,3)
            dists = np.linalg.norm(vecs, axis=1)
            idx = np.argsort(dists)[:K]
            for j, k in enumerate(idx):
                v = vecs[k]
                dist = dists[k]
                if dist > 1e-6:
                    dir_vec = v / dist
                else:
                    dir_vec = np.zeros(3, dtype=np.float32)
                d_norm = float(min(dist / max(self.safety_radius, 1e-6), 1.0))
                result[i, j, 0:3] = dir_vec
                result[i, j, 3] = d_norm

        return result.flatten()

    def _get_obs(self):
        joints = self._get_joint_positions()
        joint_vels = self._get_joint_velocities(joints)
        target = self._target.astype(np.float32)

        marker_points = self._get_marker_points_mm(joints)
        obstacles = self._gather_obstacle_points()
        marker_obs = self._marker_observations(marker_points, obstacles, n_neighbors=5)

        obs = np.concatenate([joints, joint_vels, target, marker_obs], axis=0).astype(np.float32)
        return obs

    def reset(self, target=None):
        # reset time
        self._step = 0

        # reset robot joints to a nominal pose
        initial_joint_angles = [0, np.pi / 2, 0, 0, 0, 0]
        for i, angle in enumerate(initial_joint_angles):
            p.resetJointState(self.robot6.robot_id, i, targetValue=angle)

        if self.robot5 is not None:
            initial_j5 = [0, np.pi / 2, -np.pi / 2, np.pi / 2, 0]
            for i, angle in enumerate(initial_j5):
                p.resetJointState(self.robot5.robot_id, i, targetValue=angle)

        # reset previous joints
        self._prev_joints = self._get_joint_positions()

        # reset / randomize target
        if target is None:
            tx = float(np.random.uniform(-200.0, 200.0))
            ty = float(np.random.uniform(-200.0, 200.0))
            tz = float(np.random.uniform(100.0, 400.0))
            self._target = np.array([tx, ty, tz], dtype=np.float32)
        else:
            self._target = np.array(target, dtype=np.float32)

        # resample static obstacles
        self._static_obstacles = self._sample_obstacles(self.n_obstacles)

        return self._get_obs()

    def step(self, action):
        self._step += 1
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -self.max_vel, self.max_vel)

        # Decide final commanded joint velocities based on controller switches / behavior_mode
        final_cmd = np.zeros(6, dtype=np.float32)

        # compute nonlinear controller action if enabled
        nonlinear_cmd = None
        if self.use_nonlinear_ctrl:
            try:
                # robot.get_theta_dot expects target in mm and joint positions
                theta_dot, pos_err, att_Rx, att_Rz = self.robot6.get_theta_dot(
                    x_target=self._target[0],
                    y_target=self._target[1],
                    z_target=self._target[2],
                    vmax=50,  # vmax used internally by the original controller; kept default
                    joint_position_local=joints,
                )
                nonlinear_cmd = np.asarray(theta_dot, dtype=np.float32)
            except Exception:
                nonlinear_cmd = None

        rl_cmd = None
        if self.use_rl_ctrl:
            rl_cmd = action.astype(np.float32)

        # Behavior mode selection (hard switch or blend)
        mode = str(self.behavior_mode).lower()
        if mode == 'rl':
            final_cmd = rl_cmd if rl_cmd is not None else np.zeros(6, dtype=np.float32)
        elif mode == 'nonlinear':
            final_cmd = nonlinear_cmd if nonlinear_cmd is not None else np.zeros(6, dtype=np.float32)
        elif mode == 'blend':
            # fallback: if a controller is disabled, use the other
            if rl_cmd is None and nonlinear_cmd is None:
                final_cmd = np.zeros(6, dtype=np.float32)
            elif rl_cmd is None:
                final_cmd = nonlinear_cmd
            elif nonlinear_cmd is None:
                final_cmd = rl_cmd
            else:
                alpha = np.clip(self.blend_alpha, 0.0, 1.0)
                final_cmd = alpha * rl_cmd + (1.0 - alpha) * nonlinear_cmd
        elif mode == 'none':
            final_cmd = np.zeros(6, dtype=np.float32)
        else:
            # unknown mode: default to blend behavior
            alpha = np.clip(self.blend_alpha, 0.0, 1.0)
            if rl_cmd is None:
                final_cmd = nonlinear_cmd if nonlinear_cmd is not None else np.zeros(6, dtype=np.float32)
            elif nonlinear_cmd is None:
                final_cmd = rl_cmd
            else:
                final_cmd = alpha * rl_cmd + (1.0 - alpha) * nonlinear_cmd

        # apply velocities
        self.robot6.set_joint_velocity(final_cmd)
        if self.robot5 is not None:
            # Provide a simple periodic motion for robot5 so its marker points move.
            t = float(self._step)
            vel5 = (self._robot5_amp * np.sin(self._robot5_freq * t + self._robot5_phase)).astype(np.float32)
            # Robot5 expects 5 joint velocities
            try:
                self.robot5.set_joint_velocity(vel5)
            except Exception:
                # fallback: ignore if API differs
                pass

        # advance physics several small steps for stability
        for _ in range(4):
            p.stepSimulation()

        joints = self._get_joint_positions()
        joint_vels = self._get_joint_velocities(joints)
        obs = self._get_obs()

        ee_pos = p.getLinkState(self.robot6.robot_id, self.robot6.tool_link_index)[0]
        ee_mm = np.array(ee_pos) * 1000.0
        dist = np.linalg.norm(ee_mm - self._target)

        # New reward shaping:
        # - distance penalty (smaller magnitude to encourage exploration)
        # - continuous avoidance penalty from marker observations
        # - hard collision penalty if any neighbor too close
        # - success bonus for reaching target
        gamma = 2.0
        distance_penalty = -0.005 * float(dist)

        # extract marker d values from observation
        marker_obs_flat = obs[6 + 6 + 3 :]
        marker_obs = marker_obs_flat.reshape((5, 5, 4))  # (markers, neighbors, [nx,ny,nz,d])
        d_vals = marker_obs[:, :, 3]  # normalized distances in [0,1]

        # avoidance continuous penalty: penalize closeness (1 - d)^2 when d < 1
        closeness = np.clip(1.0 - d_vals, 0.0, 1.0)
        avoidance_continuous = -gamma * float(np.sum(closeness ** 2))

        # hard collision if any neighbor normalized distance below threshold
        d_collision_thresh = 0.2
        collision_hard_penalty = 0.0
        if np.any(d_vals < d_collision_thresh):
            collision_hard_penalty = -200.0

        reward = distance_penalty + avoidance_continuous + collision_hard_penalty

        done = False
        if dist < 5.0:
            reward += 500.0
            done = True

        if self._step >= self.episode_length:
            done = True

        # store previous joints for next velocity estimate
        self._prev_joints = joints.copy()

        info = {"distance_mm": float(dist)}
        return obs, float(reward), bool(done), info

    def render(self, mode="human"):
        pass

    def close(self):
        try:
            p.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    env = PyBullet6DOFEnv(render=False)
    o = env.reset()
    print("obs shape", o.shape)
    a = env.action_space.sample()
    o2, r, d, info = env.step(a)
    print("step ->", o2.shape, r, d, info)
    env.close()
