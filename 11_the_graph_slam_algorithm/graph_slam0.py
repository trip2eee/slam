""" GraphSLAM with known correspondence
    Table 11.1, 11.2, 11.3, 11.4, and 11.5
    on page 347~350
"""

import numpy as np
import matplotlib.pyplot as plt

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
landmarks[:,1] -= 11.0

M = len(landmarks)          # the number of measurements
c = [ci for ci in range(M)] # correspondence matrix

INFINITE = 100**2
    
tm = 3  # time multiplier
dt = 0.1/tm

class GraphSLAM:
    def __init__(self):
        self.x = []
        self.x_gt = []
        self.u = []

    def initialize(self, u):
        """ This method initializes mean pose vectors (Table 11.1)
            u: u_1:t
        """
        T = len(u)

        self.x = np.zeros([T+1, 3], dtype=np.float32)
        self.x_gt = np.zeros([T+1, 3], dtype=np.float32)

        self.x[0,:] = [0, 0, 0]
        self.x_gt[0,:] = [0, 0, 0]

        for t in range(T):
            ut = u[t]
            vt, wt = ut

            # Generate ground truth.
            theta0 = self.x_gt[t,2]
            self.x_gt[t+1,0] = self.x_gt[t,0] + -vt/wt*np.sin(theta0) + vt/wt*np.sin(theta0 + wt*dt)
            self.x_gt[t+1,1] = self.x_gt[t,1] + vt/wt*np.cos(theta0) - vt/wt*np.cos(theta0 + wt*dt)
            self.x_gt[t+1,2] = self.x_gt[t,2] + wt*dt
            

            # initial poses with noise in control
            vt += np.random.randn()*0.3
            wt += np.random.randn()*0.3
            self.u.append([vt, wt])     # add noisy control

            theta0 = self.x[t,2]
            self.x[t+1,0] = self.x[t,0] + -vt/wt*np.sin(theta0) + vt/wt*np.sin(theta0 + wt*dt)
            self.x[t+1,1] = self.x[t,1] + vt/wt*np.cos(theta0) - vt/wt*np.cos(theta0 + wt*dt)
            self.x[t+1,2] = self.x[t,2] + wt*dt

    # def measure(self, mi, t):
    #     mx, my, ms = mi
    #     x = self.x_pred[0,0]
    #     y = self.x_pred[1,0]
    #     theta = self.x_pred[2,0]

    #     dx = mx - x
    #     dy = my - y
    #     r = np.sqrt(dx**2 + dy**2)
    #     phi = np.arctan2(dy, dx) - theta
    #     s = ms

    #     r += np.random.randn()*0.1
    #     phi += np.random.randn()*0.001

    #     if r <= 6:
    #         return np.array([[r, phi, s]], dtype=np.float32).T
    #     else:
    #         return None
         
    def linearize(self):
        """ This method calculates O and xi (Table 11.2).
        """
        n = len(self.x)+1 # the number of poses (number of controls + initial pose).
        
        # initialize information matrix and information vector to 0 (line 2)
        self.O = np.zeros([n*3+M*3, n*3+M*3], dtype=np.float32)     # information matrix (Omega)
        self.xi = np.zeros([n*3+M*3, 1], dtype=np.float32)      # information vector (xi)

        self.O[0:3, 0:3] += INFINITE    # line 3

        T = len(u)  # the number of controls
        for t in range(1, T+1):
            x0 = self.x[t-1].reshape(3,1)

            theta0 = x0[2,0]

            ut = self.u[t-1]
            vt, wt = ut

            dx = np.array([
                [-vt/wt*np.sin(theta0) + vt/wt*np.sin(theta0 + wt*dt)],
                [ vt/wt*np.cos(theta0) - vt/wt*np.cos(theta0 + wt*dt)],
                [wt*dt],
            ])

            xt = x0 + dx

            #     dx/dx,     dx/dy,     dx/dtheta
            #     dy/dx,     dy/dy,     dy/dtheta
            # dtheta/dx, dtheta/dy, dtheta/dtheta
            Gt = np.array([
                [1, 0, -vt/wt*np.cos(theta0) + vt/wt*np.cos(theta0 + wt*dt)],
                [0, 1, -vt/wt*np.sin(theta0) + vt/wt*np.sin(theta0 + wt*dt)],
                [0, 0, 1],
            ])
            
            Rt = np.array([
                [0.3**2, 0, 0],
                [0, 0.3**2, 0],
                [0, 0, 0.3**2]
            ], dtype=np.float32)
            invRt = np.linalg.inv(Rt)
            
            # Line 7~8
            # on page 356
            # (xt - Gt x_{t-1})^T = x^T_{t-1:t}(-Gt 1)^T
            # (-Gt x_{t-1} + xt) = (-Gt 1)x_{t-1:t}
            # x_{t-1:t} = [x_{t-1}, x_t]^T
            
            # Augmented matrix A = (-Gt I)
            # line 7
            I3 = np.eye(3, 3)            
            A = np.hstack([-Gt, I3])
            O_t = np.matmul(np.matmul(A.T, invRt), A)
            self.O[(t-1)*3:t*3+3, (t-1)*3:t*3+3] += O_t

            # line 8
            xi_t = np.matmul(np.matmul(A.T, invRt), xt - np.matmul(Gt, x0))
            self.xi[(t-1)*3:t*3+3, :] += xi_t

        # for all measurements zt (line 10)
        for t in range(0, T+1):
            s_r = 1.0
            s_phi = 0.1
            s_s = 0.1
            Q = np.array([
                [s_r**2, 0, 0],
                [0, s_phi**2, 0],
                [0, 0, s_s**2],
            ], dtype=np.float32)
            invQ = np.linalg.inv(Q)

            x = self.x[t,0]
            y = self.x[t,1]
            theta = self.x[t,2]

            # for all observed features z^i_t = (r^i_t, phi^i_t, s^i_t)
            for i in range(M):
                j = c[i]
                mx, my, ms = landmarks[j]

                dx = mx - x
                dy = my - y
                q = dx**2 + dy**2

                # if the feature is out of sensor range, skip
                if q > 6.0:
                    continue
                
                zi = np.array([
                    [np.sqrt(q)],
                    [np.arctan2(dy,dx) - theta],
                    [ms]
                ])
                
                zi_pred = np.array([
                    [np.sqrt(q)],
                    [np.arctan2(dy,dx) - theta],
                    [ms]
                ])

                # dr/dx,   dr/dy,   dr/dtheta,   dr/dmx_j,   dr/dmy_j,   dr/dms_j
                # dphi/dx, dphi/dy, dphi/dtheta, dphi/dmx_j, dphi/dmy_j, dphi/dms_j
                # ds/dx,   ds/dy,   ds/dtheta,   ds/dmx_j,   ds/dmy_j,   ds/dms_j
                Hi = 1/q*np.array([
                    [-np.sqrt(q)*dx, -np.sqrt(q)*dy,  0, np.sqrt(q)*dx, np.sqrt(q)*dy, 0],
                    [            dy,            -dx, -q,           -dy,            dx, 0],
                    [             0,              0,  0,             0,             0, q],  # 1/q*q = 1
                ])

                # Information at xt and mj
                O_tj = np.matmul(np.matmul(Hi.T, invQ), Hi)

                a = np.array([[x, y, theta, mx, my, ms]]).T
                xi_tj = np.matmul(Hi.T, invQ)







            


    
    def reduce(self, O, xi):
        """ This method reduces the size of the information.
        """
        pass

    def solve(self, O_pose, xi_pose, O, xi):
        """ This method updates the posterior x(mu) (Table 11.4).
        """
        pass

    def plot(self):
        plt.scatter(landmarks[:,0], landmarks[:,1], c='k')

        plt.plot(self.x_gt[:,0], self.x_gt[:,1], c='g')
        plt.plot(self.x[:,0], self.x[:,1], c='r')

if __name__ == '__main__':

    u = []
    for t in range(21*tm):

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
        
        u.append(ut)
    
    slam = GraphSLAM()
    slam.initialize(u)

    # repeat    
    slam.linearize()

    fig = plt.figure('map')
    
    slam.plot()

    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)


