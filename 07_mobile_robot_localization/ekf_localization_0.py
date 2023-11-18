"""Algorithm EKF_localization_known_correspondences
   Table 7.2, p.169
"""

import numpy as np
import matplotlib.pyplot as plt

landmarks = [
    [0, 10, 0],
    [10, 10, 1],
    [10, 0, 2]
]

x_pose = np.array([[1, 1, 0]], dtype=np.float32).T
P_pose = np.array([[5**2, 0, 0], 
                   [0, 5**2, 0],
                   [0, 0, 1**2]], dtype=np.float32)

res_map = 0.1   # m/pixel
w_world = 10    # m
h_world = 10    # m
w_map = int(w_world / res_map)  # pixels
h_map = int(h_world / res_map)  # pixels

def draw_pdf(pose, var_pose):
    xx = np.linspace(0, w_map, w_map+1) * res_map
    yy = np.linspace(0, h_map, h_map+1) * res_map

    x, y = np.meshgrid(xx, yy)

    pdf = np.zeros([h_map, w_map], dtype=np.float32)

    mx = pose[0,0]
    my = pose[1,0]
    ma = pose[2,0]

    sx = (var_pose[0,0])
    sy = (var_pose[1,1])
    sa = (var_pose[2,2])

    S = np.array([[sx, 0], [0, sy]], dtype=np.float32)
    print('S')
    print(S)
    detS = np.linalg.det(S)
  
    t = (x-mx)**2/sx + (y-my)**2/sy
    pdf = 1/np.sqrt(((2*np.pi)**2) * detS) * np.exp(-1/2*t)

    return pdf

def draw_map():
    pdf = draw_pdf(x_pose, P_pose)
    
    plt.figure('map')
    plt.imshow(pdf)

    # draw robot
    x_robot = x_pose[0,0]/res_map
    y_robot = x_pose[1,0]/res_map
    ha_robot = x_pose[2,0]

    plt.scatter(x_robot, y_robot, c='b')
    # draw the robot's heading angle
    dx = 10*np.cos(ha_robot)
    dy = 10*np.sin(ha_robot)
    plt.plot([x_robot, x_robot+dx], [y_robot, y_robot+dy], c='b')
    for idx_lm in range(len(landmarks)):
        lm = landmarks[idx_lm]
        x_lm = int(lm[0]/res_map)
        y_lm = int(lm[1]/res_map)
        s_lm = lm[2]    # signature

        plt.scatter(x_lm, y_lm, c='r')
        plt.text(x_lm, y_lm, 'LM{}'.format(idx_lm))
    plt.gca().invert_yaxis()
    plt.xlabel('x {}m'.format(res_map))
    plt.ylabel('y {}m'.format(res_map))
    plt.show()


draw_map()

dt = 0.1
for t in range(10):
    x  = x_pose[0,0]
    y  = x_pose[1,0]
    ha = x_pose[2,0]

    vt = 5
    wt = np.pi/6

    # motion update
    xp = x - vt/wt*np.sin(ha) + vt/wt*np.sin(ha+wt*dt)
    yp = y + vt/wt*np.cos(ha) - vt/wt*np.cos(ha+wt*dt)
    hap = ha + wt*dt
    x_pred = np.array([[xp],
                       [yp],
                       [hap]])
    
    Gt = np.array([[1, 0, vt/wt*np.cos(ha) - vt/wt*np.cos(ha+wt*dt)],
                   [0, 1, vt/wt*np.sin(ha) - vt/wt*np.sin(ha+wt*dt)],
                   [0, 0, 1]])
    Rt = np.array([[0.1**2, 0, 0],
                   [0, 0.1**2, 0],
                   [0, 0, 0.1**2]])
    
    P_pred = np.matmul(np.matmul(Gt, P_pose), Gt.T) + Rt

    print(P_pred)

    # measurement update
    sr = 1.0**2
    sf = 1.0**2
    ss = 1.0**2

    Qt = np.array([[sr, 0, 0],
                   [0, sf, 0],
                   [0, 0, ss]])
    
    sum_dx = np.zeros([3,1], dtype=np.float32)
    sum_dp = np.zeros([3,3], dtype=np.float32)
    for i in range(len(landmarks)):
        j = i
        mx, my, ms = landmarks[j]
        dx = mx - xp
        dy = my - yp
        q = dx**2 + dy**2

        # observe feature i
        z = np.array([
            [np.sqrt(q)],
            [np.arctan2(dy, dx) - hap],
            [ms],
        ])

        # predict feature i
        z_pred = np.array([
            [np.sqrt(q)],
            [np.arctan2(dy, dx) - hap],
            [ms],
        ])

        Ht = 1/q*np.array([
            [np.sqrt(q)*dx, -np.sqrt(q)*dy,  0],
            [           dy,             dx, -1],
            [            0,              0,  0],
        ])

        print(Ht)

        # HPH^T + Q
        St = np.matmul(np.matmul(Ht, P_pred), Ht.T) + Qt
        invSt = np.linalg.inv(St)
        # PH^T*S^-1
        Kt = np.matmul(np.matmul(P_pred, Ht.T), invSt)
        
        print('St')
        print(St)
        print('inv St')
        print(invSt)
        print('Kt')
        print(Kt)

        sum_dx += np.matmul(Kt, (z-z_pred))
        sum_dp += np.matmul(Kt, Ht)

    print(P_pred)
    print(sum_dx)
    print(sum_dp)

    x_pose = x_pred + sum_dx
    P_pose = np.matmul(np.eye(3) - sum_dp, P_pred)

    print(x_pose)
    print(P_pose)
    
    draw_map()


    

