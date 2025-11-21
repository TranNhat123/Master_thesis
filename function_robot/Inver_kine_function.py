import numpy as np 

def calculate_position(t1, t2, t3):
    d1, d4 = 0.29 * 1000, 0.27 * 1000
    l2, l3 = 0.26 * 1000, 0.0305 * 1000
    l1 = 0

    dx = (d4 * (np.cos(t1) * np.cos(t2) * np.sin(t3) + np.cos(t1) * np.cos(t3) * np.sin(t2)) +
          l2 * np.cos(t1) * np.cos(t2) + l3 * np.cos(t1) * np.cos(t2) * np.cos(t3) -
          l3 * np.cos(t1) * np.sin(t2) * np.sin(t3))

    dy = (d4 * (np.cos(t2) * np.sin(t1) * np.sin(t3) + np.cos(t3) * np.sin(t1) * np.sin(t2)) +
          l2 * np.cos(t2) * np.sin(t1) + l3 * np.cos(t2) * np.cos(t3) * np.sin(t1) -
          l3 * np.sin(t1) * np.sin(t2) * np.sin(t3))

    dz = (d1 - d4 * (np.cos(t2) * np.cos(t3) - np.sin(t2) * np.sin(t3)) +
          l2 * np.sin(t2) + l3 * np.cos(t2) * np.sin(t3) + l3 * np.cos(t3) * np.sin(t2))
    return dx, dy, dz

def Inverse_kinematic(dx, dy, dz):
    dx = float(dx)
    dy = float(dy)
    dz = float(dz)
    print('print dx, dy, dz', [ dx, dy, dz])
    d1, d4 = 0.29 * 1000, 0.27 * 1000
    l2, l3 = 0.26 * 1000, 0.0305 * 1000
    l1 = 0
    theta_1 = np.zeros(8)
    theta_3 = np.zeros(8)
    theta_2 = np.zeros(8)
    sign_value = np.zeros(8)

    # Theta_1
    if abs(dy - 0) < 0.000001:
        if dx >= 0:
            theta_1[0:8] = 0
        else: 
            theta_1[1:8] = np.pi

    elif abs(dx - 0) < 0.000001: 
        if dy > 0:
            theta_1[0:8] = np.pi/2
        else: 
            theta_1[0:8] = -np.pi/2
        
    else:
        for i in range(4):
            theta_1[i] = np.arctan2(dy, dx)
            theta_1[i + 4] = -np.arctan2(dy, dx)

    # Theta_3
    for i in range(8):
        if np.abs(dy - 0) < 0.001:
            A = dx / np.cos(theta_1[i]) - l1
        elif np.abs(dx - 0) < 0.001:
            A = dy / np.sin(theta_1[i]) - l1
        else:
            A = dx / np.cos(theta_1[i]) - l1
            
        B = dz - d1
        a = 2 * l2 * l3
        b = 2 * l2 * d4
        c = A ** 2 + B ** 2 - l3 ** 2 - l2 ** 2 - d4 ** 2
        r = np.sqrt(a ** 2 + b ** 2)
        
        sign_value[i] = -1 if i % 2 == 1 else 1
        
        if i in [0, 1, 4, 5]:
            theta_3[i] = np.arctan2(c / r, np.sqrt(abs(1 - c ** 2 / r ** 2))) - np.arctan2(a, b)
        else:
            theta_3[i] = np.arctan2(c / r, -np.sqrt(abs(1 - c ** 2 / r ** 2))) - np.arctan2(a, b)
        
        u = dz - d1
        k = -d4 * np.cos(theta_3[i]) + l3 * np.sin(theta_3[i])
        l = d4 * np.sin(theta_3[i]) + l3 * np.cos(theta_3[i]) + l2
        p = np.sqrt(k ** 2 + l ** 2)
        
        theta_2[i] = np.arctan2(u / p, sign_value[i] * np.sqrt(1 - u ** 2 / p ** 2)) - np.arctan2(k, l)

        # Tạo mảng kết quả
    theta_123 = np.column_stack((theta_1, theta_2, theta_3))

    Results = []
    for theta in theta_123: 
        theta1 = theta[0]
        theta2 = theta[1]
        theta3 = theta[2]
        x_new, y_new, z_new =  calculate_position(theta1, theta2, theta3)
        if abs(x_new - dx) < 0.01 and abs(y_new - dy) < 0.01 and abs(z_new - dz) < 0.01:
            Results.append([theta1, theta2, theta3])

    # Ưu tiên chọn kết quả trong phạm vi [-pi/2, pi/2]
    Filtered_Results = [res for res in Results if all(-np.pi/2 <= t <= np.pi/2 for t in res)]

    return_value = Filtered_Results if Filtered_Results else Results
    return return_value
