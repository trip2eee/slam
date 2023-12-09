"""Algorithm EKF_localization
   Table 7.3, p.217
"""

import numpy as np
import matplotlib.pyplot as plt

landmarks = [
    [0, 10, 0],
    [10, 10, 1],
    [10, 0, 2]
]

dt = 0.1

class Robot:
    def __init__(self):

        x_pose = np.array([[1, 1, 0]], dtype=np.float32).T
        P_pose = np.array([[3**2, 0, 0], 
                           [0, 3**2, 0],
                           [0, 0, 1**2]], dtype=np.float32)

        self.x = x_pose
        self.P = P_pose

        self.x_pred = x_pose.copy()
        self.P_pred = P_pose.copy()

        self.x_gt = x_pose.copy()
    
    def predict(self, ut):
        # control ut
        vt, wt = ut

        # compute ground truth
        x_gt = self.x_gt[0,0]
        y_gt = self.x_gt[1,0]
        ha_gt = self.x_gt[2,0]
        self.x_gt[0,0] = x_gt - vt/wt*np.sin(ha_gt) + vt/wt*np.sin(ha_gt+wt*dt)
        self.x_gt[1,0] = y_gt + vt/wt*np.cos(ha_gt) - vt/wt*np.cos(ha_gt+wt*dt)
        self.x_gt[2,0] = ha_gt + wt*dt

        # motion update (prediction)
        
        a1 = 0.001
        a2 = 0.0
        a3 = 0.0
        a4 = 0.0001
        vt += np.random.randn()*(np.sqrt(a1)*vt)
        wt += np.random.randn()*(np.sqrt(a4)*wt)

        x  = self.x[0,0]
        y  = self.x[1,0]
        ha = self.x[2,0]   # heading angle        
        
        xp = x - vt/wt*np.sin(ha) + vt/wt*np.sin(ha+wt*dt)
        yp = y + vt/wt*np.cos(ha) - vt/wt*np.cos(ha+wt*dt)
        hap = ha + wt*dt
        x_pred = np.array([[xp], [yp], [hap]])

        # dx/dx, dx/dy, dx/da
        # dy/dx, dy/dy, dy/da
        #     0,     0,      1
        Gt = np.array([[1, 0, -vt/wt*np.cos(ha) + vt/wt*np.cos(ha+wt*dt)],
                       [0, 1, -vt/wt*np.sin(ha) + vt/wt*np.sin(ha+wt*dt)],
                       [0, 0, 1]])

        # dx/dv, dx/dw
        # dy/dv, dy/dw
        # da/dv, da/dw
        Vt = np.array([
            [(-np.sin(ha)+np.sin(ha+wt*dt))/wt, vt*(np.sin(ha)-np.sin(ha+wt*dt))/wt**2 + vt*np.cos(ha+wt*dt)*dt/wt],
            [(np.cos(ha)-np.cos(ha+wt*dt))/wt, -vt*(np.cos(ha)-np.cos(ha+wt*dt))/wt**2 + vt*np.sin(ha+wt*dt)*dt/wt],
            [0.0, dt]
        ])

        Mt = np.array([
            [a1*vt**2+a2*wt**2, 0],
            [0, a2*vt**2+a3*wt**2]
        ])
        
        P_pose = self.P
        P_pred = np.matmul(np.matmul(Gt, P_pose), Gt.T) + np.matmul(np.matmul(Vt,Mt),Vt.T)

        self.x_pred = x_pred
        self.P_pred = P_pred

    def measure(self, landmark):
        x_pred = self.x_pred
        mx, my, ms = landmark
        dx = mx - x_pred[0,0]
        dy = my - x_pred[1,0]
        q = dx**2 + dy**2

        z = np.array([
            [np.sqrt(q) + np.random.randn()*np.sqrt(q)*0.01],
            [np.arctan2(dy, dx) - x_pred[2,0] + np.random.randn()*0.01],
            [ms],
        ])

        return z

    def update(self):
        # measurement update
        x_pred = self.x_pred
        P_pred = self.P_pred

        # for all landmarks k in the map m
        N = len(landmarks)
        for k in range(N):
            mx, my, ms = landmarks[k]
            dx = mx - x_pred[0,0]
            dy = my - x_pred[1,0]
            q = dx**2 + dy**2

            sr = np.sqrt(q)*0.1
            sf = 0.1
            ss = 1.0

            Qt = np.array([[sr**2, 0,     0],
                           [0,     sf**2, 0],
                           [0,     0,     ss**2]])

            # predict feature i
            z_pred = np.array([
                [np.sqrt(q)],
                [np.arctan2(dy, dx) - x_pred[2,0]],
                [ms],
            ])

            Ht = np.array([
                [dx/np.sqrt(q), -dy/np.sqrt(q),  0],
                [         dy/q,           dx/q, -1],
                [            0,              0,  0],
            ])

            # HPH^T + Q
            St = np.matmul(np.matmul(Ht, P_pred), Ht.T) + Qt
            invSt = np.linalg.inv(St)

            # for all observed features i
            ck = 0
            zk = None
            min_d2 = 1e10
            for i in range(N):
                zi = self.measure(landmarks[i])
                r = zi - z_pred

                d2 = np.matmul(np.matmul(r.T, invSt), r)
                if d2 < min_d2:
                    min_d2 = d2
                    ck = i
                    zk = zi

            assert(k == ck)

            # PH^T*S^-1
            Kt = np.matmul(np.matmul(P_pred, Ht.T), invSt)
            KHt = np.matmul(Kt, Ht)

            x_pred = x_pred + np.matmul(Kt, (zk-z_pred))
            I = np.eye(3)
            P_pred = np.matmul(I - KHt, P_pred)

        self.x = x_pred
        self.P = P_pred

res_map = 0.1   # m/pixel
w_world = 10    # m
h_world = 10    # m
w_map = int(w_world / res_map)  # pixels
h_map = int(h_world / res_map)  # pixels

def draw_pdf(robot):
    xx = np.linspace(0, w_map, w_map+1) * res_map
    yy = np.linspace(0, h_map, h_map+1) * res_map

    x, y = np.meshgrid(xx, yy)

    pdf = np.zeros([h_map, w_map], dtype=np.float32)

    mx = robot.x[0,0]
    my = robot.x[1,0]
    ma = robot.x[2,0]

    S = robot.P[:2,:2]
    invS = np.linalg.inv(S)
    isx = invS[0,0]
    isy = invS[1,1]
    isxy = (invS[0,1] + invS[1,0])*0.5

    detS = np.linalg.det(S)

    t = (x-mx)**2*isx + (y-my)**2*isy + 2*(x-mx)*(y-my)*isxy
    pdf = 1/np.sqrt(((2*np.pi)**2) * detS) * np.exp(-1/2*t)

    return pdf

def draw_robot(x_pose, color):
    x_robot = x_pose[0,0]/res_map
    y_robot = x_pose[1,0]/res_map
    ha_robot = x_pose[2,0]

    plt.scatter(x_robot, y_robot, c=color)
    # draw the robot's heading angle
    dx = 10*np.cos(ha_robot)
    dy = 10*np.sin(ha_robot)
    plt.plot([x_robot, x_robot+dx], [y_robot, y_robot+dy], c=color)

def draw_map(robot):
    
    pdf = draw_pdf(robot)

    fig = plt.figure('map')
    plt.imshow(pdf)

    # draw robot
    draw_robot(robot.x_gt,   'g')
    draw_robot(robot.x_pred, 'c')
    draw_robot(robot.x, 'b')

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

    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)

robot = Robot()
draw_map(robot)

for t in range(30):

    # control ut
    vt = 3
    wt = np.pi/6
    ut = [vt, wt]

    # motion update
    robot.predict(ut)
    
    # measurement update
    robot.update()
    
    draw_map(robot)


    

