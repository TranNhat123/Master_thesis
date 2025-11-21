# Hàm MPPI với Gaussian Distribution
import numpy as np 
import torch 

def robot_dynamics(state, velocity, dt):
    next_state = state + velocity * dt
    return next_state

# Hàm tính chi phí
def compute_cost(trajectories_round, goal, distances, device):
    """
    Tính chi phí cho mỗi trajectory dựa trên khoảng cách tới đích và tránh va chạm.

    trajectories_round: Tensor [N, T, 3] - danh sách các quỹ đạo rời rạc (chỉ số nguyên)
    goal: Tensor [3] - vị trí đích đến (x, y, z)
    distances: Tensor [D_x, D_y, D_z] - bản đồ khoảng cách (distance field)
    device: thiết bị tính toán (CPU/GPU)

    Trả về: cost Tensor [N]
    """
    # ------------------------------
    # 1. Chi phí goal: khoảng cách tại điểm cuối tới đích
    # ------------------------------
    final_pos = trajectories_round[:, -1, :]  # [N, 3]
    goal_cost = torch.norm(final_pos - goal[None, :], dim=-1)  # [N]

    # ------------------------------
    # 2. Chi phí tránh va chạm (obstacle cost)
    # ------------------------------
    indices = trajectories_round.cpu().long()  # [N, T, 3]

    # Lấy giá trị khoảng cách từ bản đồ distance field
    obs_cost = distances[indices[..., 0], indices[..., 1], indices[..., 2]]  # [N, T]
    obs_cost = torch.tensor(obs_cost).to(device)

    # Tính soft reward (khuyến khích đi xa vật cản)
    obs_reward = -obs_cost  # đi càng xa vật cản càng tốt

    # Áp dụng hình phạt mạnh nếu tới gần vật cản (dưới ngưỡng)
    safe_distance = 1.0
    hard_penalty = 10.0
    penalty = torch.where(obs_cost <= safe_distance, hard_penalty, 0.0)  # [N, T]

    # Tổng chi phí tránh va chạm
    obs_total_cost = torch.sum(obs_reward + penalty, dim=1)  # [N]

    # ------------------------------
    # 3. Tổng hợp chi phí
    # ------------------------------
    alpha = 5.0  # trọng số cho goal (càng gần đích càng tốt)
    beta = 10   # trọng số cho tránh va chạm

    total_cost = alpha * goal_cost + beta * obs_total_cost  # [N]
    # cost o day la chi phi, cang xa cang cao, minh chon cai nho nhat
    return total_cost


def Take_cur_position_index(joint_position):
    theta = joint_position
    theta_degree = theta*180/np.pi
    theta_index_round = np.round((theta_degree + 180)/5)
    theta_index_not_round = np.round((theta_degree + 180)/5, 3)
    return theta_index_round.astype(int), theta_index_not_round

# Hàm MPPI
def mppi(state, dt, T, N, goal, lambda_mppi, v_max, std_v_max, device, distances, vector_avoid):    # Vector gốc và chuẩn hóa
    v = vector_avoid
    v = v / np.linalg.norm(v)
    epsilon = 1  # Độ nhiễu
    # Tạo nhiễu Gaussian
    noise = np.random.randn(N, T, 3)
    # Thêm nhiễu vào vector gốc
    v_new = v + epsilon * noise  
    # Chuẩn hóa từng vector riêng lẻ
    norms = np.linalg.norm(v_new, axis=-1, keepdims=True)  # Tính norm cho từng vector (N, T, 1)
    v_new = v_new / norms  # Chuẩn hóa thành vector đơn vị
    # Chuyển sang tensor của PyTorch
    v_new = torch.tensor(v_new, dtype=torch.float32)
    ##  Tạo giá trị ngẫu nhiên cho vận tốc
    # mean = v_max/2  # Giá trị trung bình mong muốn
    # value = np.random.normal(loc=mean, scale=std_v_max, size=(N, T, 1))  # Tạo số ngẫu nhiên từ phân phối Gaussian

    value = np.full((N, T, 1), fill_value=v_max, dtype=np.float32)
    value_velocity = torch.tensor(value, dtype=torch.float32)
    # value_velocity = torch.tensor(value, dtype=torch.float32)
    u_samples = v_new * value_velocity
    u_samples = u_samples.to(device)
    trajectories = torch.zeros((N, T, 3), device=device)
    # Biến tạm để lưu trạng thái hiện tại
    temp_state = state.expand(N, -1).clone()
    # Tính toán quỹ đạo qua từng bước thời gian
    for t in range(T):
        temp_state = robot_dynamics(temp_state, u_samples[:, t, :], dt)
        trajectories[:, t, :] = temp_state

    trajectories = trajectories.cpu().numpy()
    trajectories_round, trajectories_not_round = Take_cur_position_index(trajectories)
    trajectories_round_tensor = torch.tensor(trajectories_round).to(device)
    costs = compute_cost(trajectories_round_tensor, goal, distances, device)
    beta = torch.min(costs)
    weights = torch.exp(-(costs - beta) / lambda_mppi)
    weights /= torch.sum(weights)

    u_optimal = torch.sum(u_samples * weights[:, None, None], dim=0)
    best_trajectory = trajectories[torch.argmin(costs)]
    # Do luc minh ve thi minh ve tren mien tu 0 -> 71 nen phai dung trajectories_round 
    # hoac trajectories_not round de visualize chu trajectories la no tu 0->pi thoi
    return u_optimal, trajectories_not_round