""" EKF SLAM with unknown correspondences
    Table 10.2 on page 321.
"""
import matplotlib.pyplot as plt
import numpy as np

# x, y, signature
landmarks = [
    [ 5, 10, 0],
    [10, 10, 1],
    [15, 10, 2],
    [20, 10, 3],
    [20,  0, 4],
    [15,  0, 5],
    [10,  0, 6],
    [ 5,  0, 7],
]
landmarks = np.array(landmarks, dtype=np.float32)

INFINITE = 5**2

# The number of landmarks (measurements)
M = landmarks.shape[0]

class EKF_SLAM:
    def __init__(self):
        # M+1 space for new measurement (line 9)
        self.x = np.zeros([3 + 3*(M+1), 1], dtype=np.float32)
        self.x[0,0] = 0     # x
        self.x[1,0] = 11    # y
        self.x[2,0] = 0     # theta
        
        self.x_gt = self.x.copy()
        
        self.P = np.eye(3 + 3*(M+1), 3 + 3*(M+1), dtype=np.float32)*INFINITE
        self.P[0,0] = 1.0
        self.P[1,1] = 1.0
        self.P[2,2] = 1.0
        

        self.x_pred = self.x.copy()
        self.P_pred = self.P.copy()
        self.N = 0

    def predict(self, ut, dt=0.1):
        vt, wt = ut

        theta = self.x[2,0]

        dx = np.array([
            [-vt/wt*np.sin(theta) + vt/wt*np.sin(theta + wt*dt)],
            [ vt/wt*np.cos(theta) - vt/wt*np.cos(theta + wt*dt)],
            [wt*dt],
        ])
        self.x_gt[:3] += dx # update ground truth


        vt += np.random.randn()*0.3
        wt += np.random.randn()*0.3
        dx = np.array([
            [-vt/wt*np.sin(theta) + vt/wt*np.sin(theta + wt*dt)],
            [ vt/wt*np.cos(theta) - vt/wt*np.cos(theta + wt*dt)],
            [wt*dt],
        ])

        Fx = np.eye(3, 3+3*(M+1), dtype=np.float32)
        self.x_pred = self.x + np.matmul(Fx.T, dx)

        #     dx/dx,     dx/dy,     dx/dtheta
        #     dy/dx,     dy/dy,     dy/dtheta
        # dtheta/dx, dtheta/dy, dtheta/dtheta
        g = np.array([
            [0, 0, -vt/wt*np.cos(theta) + vt/wt*np.cos(theta + wt*dt)],
            [0, 0, -vt/wt*np.sin(theta) + vt/wt*np.sin(theta + wt*dt)],
            [0, 0, 0],
        ])
        Gt = np.eye(3+3*(M+1), 3+3*(M+1)) + np.matmul(np.matmul(Fx.T, g), Fx)

        Rt = np.array([
            [0.1**2, 0, 0],
            [0, 0.1**2, 0],
            [0, 0, 0.01**2]
        ], dtype=np.float32)

        self.P_pred = np.matmul(np.matmul(Gt, self.P), Gt.T) + np.matmul(np.matmul(Fx.T, Rt), Fx)

    def measure(self, mi):
        mx, my, ms = mi
        x = self.x_pred[0,0]
        y = self.x_pred[1,0]
        theta = self.x_pred[2,0]

        dx = mx - x
        dy = my - y
        r = np.sqrt(dx**2 + dy**2)
        phi = np.arctan2(dy, dx) - theta
        s = ms

        r += np.random.randn()*0.1
        phi += np.random.randn()*0.001

        if r <= 6:
            return np.array([[r, phi, s]], dtype=np.float32).T
        else:
            return None

    def update(self):

        x_pred = self.x_pred
        P_pred = self.P_pred

        s_r = 1.0
        s_phi = 0.1
        s_s = 0.1
        Q = np.array([
                [s_r**2, 0, 0],
                [0, s_phi**2, 0],
                [0, 0, s_s**2],
            ], dtype=np.float32)
            
        # for all observed feature zi_meas
        for i in range(M):
            zi_meas = self.measure(landmarks[i])
            # if not observed, skip
            if zi_meas is None:
                continue
            
            r, phi, s = zi_meas
            N = self.N
            x_pred[3+N*3+0,0] = x_pred[0,0] + r*np.cos(phi + x_pred[2,0])
            x_pred[3+N*3+1,0] = x_pred[1,0] + r*np.sin(phi + x_pred[2,0])
            x_pred[3+N*3+2,0] = s

            j = -1          # the index of track with the minimum distance
            min_d = 1e10    # minimum distance
            min_invS = None
            min_H = None    # Measurement matrix with the minimum distance
            min_y = None    # innovation vector with the minimum distance
            alpha = 2.0**2  # association threshold

            for k in range(N+1):
            
                dx = x_pred[3+k*3+0,0] - x_pred[0,0]
                dy = x_pred[3+k*3+1,0] - x_pred[1,0]
                q = dx**2 + dy**2
                # predicted measurement
                # r_i, phi_i, s_i
                zi_pred = np.array([
                    [np.sqrt(q)],
                    [np.arctan2(dy,dx) - x_pred[2,0]],
                    [x_pred[3+k*3+2,0]]
                ])

                #
                Fxk = np.zeros([6, 3+3*(M+1)], dtype=np.float32)
                Fxk[0,0] = 1
                Fxk[1,1] = 1
                Fxk[2,2] = 1
                Fxk[3,3+k*3+0] = 1
                Fxk[4,3+k*3+1] = 1
                Fxk[5,3+k*3+2] = 1

                # dr/dx,   dr/dy,   dr/dtheta,   dr/dmx_j,   dr/dmy_j,   dr/dms_j
                # dphi/dx, dphi/dy, dphi/dtheta, dphi/dmx_j, dphi/dmy_j, dphi/dms_j
                # ds/dx,   ds/dy,   ds/dtheta,   ds/dmx_j,   ds/dmy_j,   ds/dms_j
                hk = 1/q*np.array([
                    [-np.sqrt(q)*dx, -np.sqrt(q)*dy,  0, np.sqrt(q)*dx, np.sqrt(q)*dy, 0],
                    [            dy,            -dx, -q,           -dy,            dx, 0],
                    [             0,              0,  0,             0,             0, q],  # 1/q*q = 1
                ])
                Hk = np.matmul(hk, Fxk)

                Sk = np.matmul(np.matmul(Hk, P_pred), Hk.T) + Q
                invS = np.linalg.inv(Sk)

                r = zi_meas-zi_pred                                

                if k == N:
                    d = alpha   # line 19 gives the maximum distance to N+1
                else:
                    d = np.matmul(np.matmul(r.T, invS), r)  # line 17 computes Mahalanobis distance

                if d <= alpha and d < min_d:
                    min_d = d
                    j = k
                    min_invS = invS
                    min_H = Hk
                    min_y = r

            print('  assoc', i, j)
            self.N = max(j+1, self.N)
            
            Kt = np.matmul(np.matmul(P_pred, min_H.T), min_invS)
            x_pred = x_pred + np.matmul(Kt, min_y)
            P_pred = np.matmul((np.eye(3+3*(M+1), 3+3*(M+1)) - np.matmul(Kt, min_H)), P_pred)

        self.x = x_pred
        self.P = P_pred

    def plot(self):
        
        # draw landmarks
        plt.scatter(landmarks[:,0], landmarks[:,1], c='k')
        for i in range(M):
            j = 3+i*3

            if self.P[j,j] < INFINITE:
                self.draw_cov_ellipse(self.x[j:j+3], self.P[j:j+3,j:j+3], color='r')

        # draw robot
        self.draw_robot(self.x_gt, color='g')
        self.draw_robot(self.x_pred, color='c')
        self.draw_robot(self.x, color='b')        
        
        self.draw_cov_ellipse(self.x, self.P[:2,:2], color='b')

    def draw_robot(self, x, color):
        x_robot = x[0,0]
        y_robot = x[1,0]
        theta_robot = x[2,0]

        d=1.0
        dx = d*np.cos(theta_robot)
        dy = d*np.sin(theta_robot)
        plt.scatter([x_robot], [y_robot], c=color)
        plt.plot([x_robot, x_robot+dx], [y_robot, y_robot+dy], c=color)


    def draw_cov_ellipse(self, x, cov, color):
        xc = x[0,0]
        yc = x[1,0]

        W, V = np.linalg.eig(cov)
        a = np.sqrt(W[0])
        b = np.sqrt(W[1])

        v0 = V[0,:]
        theta = np.arctan2(v0[1], v0[0])

        xs = []
        ys = []
        for d in range(0, 360):
            t = d * np.pi / 180
            x0 = a*np.cos(t)
            y0 = b*np.sin(t)

            x1 = xc + x0*np.cos(theta) - y0*np.sin(theta)
            y1 = yc + x0*np.sin(theta) + y0*np.cos(theta)

            xs.append(x1)
            ys.append(y1)
        
        plt.plot(xs, ys, c=color)

ekf_slam = EKF_SLAM()

tm = 3  # time multiplier

for t in range(25*tm):
    fig = plt.figure('map')

    print('time:',t)

    # Robot maneuver
    if t <= 7*tm:
        ut = [25, 0.001]
    elif t <= 9*tm:
        ut = [25, -np.pi/2*5]
    elif t <= 10*tm:
        ut = [25, 0.001]
    elif t <= 12*tm:
        ut = [25, -np.pi/2*5]
    elif t <= 16*tm:
        ut = [25, 0.001]
    elif t <= 17*tm:
        ut = [25, -np.pi/2*5]
    else:
        ut = [25, 0.001]

    ekf_slam.predict(ut, dt=0.1/tm)
    ekf_slam.update()

    ekf_slam.plot()

    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)



