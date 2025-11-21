import numpy as np

def Jtool_fanuc_function(t1, t2, t3, t4, t5, d_5):
    # Link parameters
    a_1 = 0.15
    a_2 = 0.25
    a_3 = 0.22
    d_2 = 0
    k_1 = 0

    # Calculate the elements of the Jacobian matrix
    k11 = (-(a_1 + a_2 * np.cos(t2) + a_3 * np.cos(t2 + t3) + d_5 * np.sin(t2 + t3 + t4))) * np.sin(t1) + (d_2 + k_1) * np.cos(t1)
    k12 = (-a_2 * np.sin(t2) - a_3 * np.sin(t2 + t3) + d_5 * np.cos(t2 + t3 + t4)) * np.cos(t1)
    k13 = -a_3 * np.cos(t1) * np.sin(t2 + t3) + d_5 * np.cos(t1) * np.cos(t2 + t3 + t4)
    k21 = (a_1 + a_2 * np.cos(t2) + a_3 * np.cos(t2 + t3) + d_5 * np.sin(t2 + t3 + t4)) * np.cos(t1) + (d_2 + k_1) * np.sin(t1)
    k22 = -a_2 * np.sin(t1) * np.sin(t2) - a_3 * np.sin(t1) * np.sin(t2 + t3) + d_5 * np.sin(t1) * np.cos(t2 + t3 + t4)
    k23 = -a_3 * np.sin(t1) * np.sin(t2 + t3) + d_5 * np.sin(t1) * np.cos(t2 + t3 + t4)
    k31 = 0
    k32 = a_2 * np.cos(t2) + a_3 * np.cos(t2 + t3) + d_5 * np.sin(t2 + t3 + t4)
    k33 = a_3 * np.cos(t2 + t3) + d_5 * np.sin(t2 + t3 + t4)

    k41 = 0
    k42 = 1
    k43 = 1
    k44 = 1
    k45 = 0

    k51 = -1
    k52 = 0
    k53 = 0
    k54 = 0
    k55 = 1

    Jtool = np.array([[k11, k12, k13, 0, 0],
                      [k21, k22, k23, 0, 0],
                      [k31, k32, k33, 0, 0],
                      [k41, k42, k43, k44, k45],
                      [k51, k52, k53, k54, k55],])

    return Jtool

