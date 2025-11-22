import numpy as np

def Jtool_sixdof_function(t1, t2, t3, t4, t5, t6):
    a_2 = 0.26*1000
    a_3 = 0.0305*1000
    d_4 = 0.27*1000
    
    k11 = -d_4 * np.sin(t1) * np.cos(t2) * np.sin(t3) - d_4 * np.sin(t1) * np.cos(t3) * np.sin(t2) - a_2 * np.sin(
        t1) * np.cos(t2) - a_3 * np.sin(t1) * np.cos(t2) * np.cos(t3) + a_3 * np.sin(t1) * np.sin(t2) * np.sin(t3)

    k12 = -d_4 * np.cos(t1) * np.sin(t2) * np.sin(t3) + d_4 * np.cos(t1) * np.cos(t2) * np.cos(t3) - a_2 * np.cos(
        t1) * np.sin(t2) - a_3 * np.cos(t1) * np.sin(t2) * np.cos(t3) - a_3 * np.cos(t1) * np.cos(t2) * np.sin(t3)

    k13 = d_4 * np.cos(t1) * np.cos(t2) * np.cos(t3) - d_4 * np.cos(t1) * np.sin(t2) * np.sin(t3) - a_3 * np.cos(
        t1) * np.cos(t2) * np.sin(t3) - a_3 * np.cos(t1) * np.sin(t2) * np.cos(t3)

    k21 = d_4 * np.cos(t1) * np.cos(t2) * np.sin(t3) + d_4 * np.cos(t1) * np.cos(t3) * np.sin(t2) + a_2 * np.cos(
        t1) * np.cos(t2) + a_3 * np.cos(t1) * np.cos(t2) * np.cos(t3) - a_3 * np.cos(t1) * np.sin(t2) * np.sin(t3)

    k22 = -d_4 * np.sin(t1) * np.sin(t2) * np.sin(t3) + d_4 * np.sin(t1) * np.cos(t2) * np.cos(t3) - a_2 * np.sin(
        t1) * np.sin(t2) - a_3 * np.sin(t1) * np.sin(t2) * np.cos(t3) - a_3 * np.sin(t1) * np.cos(t2) * np.sin(t3)

    k23 = d_4 * np.sin(t1) * np.cos(t2) * np.cos(t3) - d_4 * np.sin(t1) * np.sin(t2) * np.sin(t3) - a_3 * np.sin(
        t1) * np.cos(t2) * np.sin(t3) - a_3 * np.sin(t1) * np.sin(t2) * np.cos(t3)

    k31 = 0

    k32 = d_4 * np.sin(t2) * np.cos(t3) + d_4 * np.cos(t2) * np.sin(t3) + a_2 * np.cos(t2) - a_3 * np.sin(t2) * np.sin(
        t3) + a_3 * np.cos(t2) * np.cos(t3)

    k33 = d_4 * np.cos(t2) * np.sin(t3) + d_4 * np.sin(t2) * np.cos(t3) + a_3 * np.cos(t2) * np.cos(t3) - a_3 * np.sin(
        t2) * np.sin(t3)

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
