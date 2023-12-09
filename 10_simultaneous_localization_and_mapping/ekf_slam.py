""" EKF SLAM with known correspondences
    Table 10.1 on page 314.
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

# The number of landmarks
N = landmarks.shape[0]
c = [-1] * N    # correspondence matrix

class EKF_SLAM:
    def __init__(self):
        self.x = np.zeros([3 + 3*N, 1], dtype=np.float32)
        self.x[0,0] = 0     # x
        self.x[1,0] = 11    # y
        self.x[2,0] = 0     # theta
        
        self.x_gt = self.x.copy()
        
        self.P = np.eye(3 + 3*N, 3 + 3*N, dtype=np.float32)*INFINITE
        self.P[0,0] = 1.0
        self.P[1,1] = 1.0
        self.P[2,2] = 1.0
        

        self.x_pred = self.x.copy()
        self.P_pred = self.P.copy()

    def predict(self, ut, dt=0.1):
        vt, wt = ut

        theta = self.x[2,0]

        dx = np.array([
            [-vt/wt*np.sin(theta) + vt/wt*np.sin(theta + wt*dt)],
            [ vt/wt*np.cos(theta) - vt/wt*np.cos(theta + wt*dt)],
            [wt*dt],
        ])
        self.x_gt[:3] += dx # update ground truth


        # vt += np.random.randn()*0.001
        # wt += np.random.randn()*0.0001
        dx = np.array([
            [-vt/wt*np.sin(theta) + vt/wt*np.sin(theta + wt*dt)],
            [ vt/wt*np.cos(theta) - vt/wt*np.cos(theta + wt*dt)],
            [wt*dt],
        ])

        Fx = np.eye(3, 3+3*N, dtype=np.float32)
        self.x_pred = self.x + np.matmul(Fx.T, dx)

        #     dx/dx,     dx/dy,     dx/dtheta
        #     dy/dx,     dy/dy,     dy/dtheta
        # dtheta/dx, dtheta/dy, dtheta/dtheta
        g = np.array([
            [0, 0, -vt/wt*np.cos(theta) + vt/wt*np.cos(theta + wt*dt)],
            [0, 0, -vt/wt*np.sin(theta) + vt/wt*np.sin(theta + wt*dt)],
            [0, 0, 0],
        ])
        Gt = np.eye(3+3*N, 3+3*N) + np.matmul(np.matmul(Fx.T, g), Fx)

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

        # r += np.random.randn()*0.1
        # phi += np.random.randn()*phi*0.01

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

        # for all observed feature zi_meas
        for i in range(N):
            zi_meas = self.measure(landmarks[i])
            if zi_meas is None:
                continue
            
            r, phi, s = zi_meas
            Q = np.array([
                [s_r**2, 0, 0],
                [0, s_phi**2, 0],
                [0, 0, s_s**2],
            ], dtype=np.float32)
            
            j = c[i]
            # if landmark j is never seen before
            if -1 == j:
                j = i
                c[i] = j
                
                x_pred[3+j*3+0,0] = x_pred[0,0] + r*np.cos(phi + x_pred[2,0])
                x_pred[3+j*3+1,0] = x_pred[1,0] + r*np.sin(phi + x_pred[2,0])
                x_pred[3+j*3+2,0] = s
            
            print('  update', j)

            dx = x_pred[3+j*3+0,0] - x_pred[0,0]
            dy = x_pred[3+j*3+1,0] - x_pred[1,0]
            q = dx**2 + dy**2
            # predicted measurement
            # r_i, phi_i, s_i
            zi_pred = np.array([
                [np.sqrt(q)],
                [np.arctan2(dy,dx) - x_pred[2,0]],
                [x_pred[3+j*3+2,0]]
            ])

            #
            Fxj = np.zeros([6, 3+3*N], dtype=np.float32)
            Fxj[0,0] = 1
            Fxj[1,1] = 1
            Fxj[2,2] = 1
            Fxj[3,3+j*3+0] = 1
            Fxj[4,3+j*3+1] = 1
            Fxj[5,3+j*3+2] = 1

            # dr/dx,   dr/dy,   dr/dtheta,   dr/dmx_j,   dr/dmy_j,   dr/dms_j
            # dphi/dx, dphi/dy, dphi/dtheta, dphi/dmx_j, dphi/dmy_j, dphi/dms_j
            # ds/dx,   ds/dy,   ds/dtheta,   ds/dmx_j,   ds/dmy_j,   ds/dms_j
            ht = 1/q*np.array([
                [-np.sqrt(q)*dx, -np.sqrt(q)*dy,  0, np.sqrt(q)*dx, np.sqrt(q)*dy, 0],
                [            dy,            -dx, -q,           -dy,            dx, 0],
                [             0,              0,  0,             0,             0, q],  # 1/q*q = 1
            ])
            Ht = np.matmul(ht, Fxj)

            St = np.matmul(np.matmul(Ht, P_pred), Ht.T) + Q
            Kt = np.matmul(np.matmul(P_pred, Ht.T), np.linalg.inv(St))
            x_pred = x_pred + np.matmul(Kt, zi_meas-zi_pred)
            P_pred = np.matmul((np.eye(3+3*N, 3+3*N) - np.matmul(Kt, Ht)), P_pred)

        self.x = x_pred
        self.P = P_pred

    def plot(self):
        
        # draw landmarks
        plt.scatter(landmarks[:,0], landmarks[:,1], c='k')
        for i in range(N):
            j = 3+i*3

            if self.P[j,j] < INFINITE:
                self.draw_cov_ellipse(self.x[j:j+3], self.P[j:j+3,j:j+3], color='r')

        # draw robot
        self.draw_robot(self.x_gt, color='g')
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


for t in range(25):
    fig = plt.figure('map')

    print('time:',t)

    if t <= 7:
        ut = [25, 0.001]
    elif t <= 9:
        ut = [25, -np.pi/2*5]
    elif t <= 10:
        ut = [25, 0.001]
    elif t <= 12:
        ut = [25, -np.pi/2*5]
    elif t <= 16:
        ut = [25, 0.001]
    elif t <= 17:
        ut = [25, -np.pi/2*5]
    else:
        ut = [25, 0.001]

    ekf_slam.predict(ut)
    ekf_slam.update()

    ekf_slam.plot()

    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)

print(c)




