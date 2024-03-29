""" GraphSLAM with known correspondence
    Table 11.1, 11.2, 11.3, 11.4, and 11.5
    on page 347~350
"""

import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(123)
landmarks = [
    [ 5, 10, 0],
    [10, 10, 0],
    [15, 10, 0],
    [20, 10, 0],
    [20,  0, 0],
    [15,  0, 0],
    [10,  0, 0],
    [ 5,  0, 0],
]
landmarks = np.array(landmarks, dtype=np.float32)
landmarks[:,1] -= 11.0

M = len(landmarks)          # the number of measurements
c = [ci for ci in range(M)] # correspondence matrix

INFINITE = 100**2
    
tm = 2  # time multiplier
dt = 0.1/tm

r_max = 10.0 # maximum detection range

DIM_POSE = 3
DIM_MEAS = 2

STD_V = 0.3
STD_W = 0.1

STD_R = 0.02
STD_PHI = 0.02

class GraphSLAM:
    def __init__(self):
        # ground truth pose
        self.x_gt = [
            [0, 0, 0]
        ]
        self.x = [
            [0, 0, 0]
        ]

        self.S = None   # Sigma 0:t

        self.u_gt = []  # controls (ground truth)
        self.u = [
            [0.0, 0.0]
        ]     # controls (with noise)
        
        # tau(j) : set of poses xt at which j was observed.
        self.z = [      # measurements
            []
        ]

        self.tau = []
        for i in range(M):
            self.tau.append([])
        
        self.m = None   # landmarks

    def control(self, ut):
        vt, wt = ut
        self.u_gt.append([vt, wt])
        
        # compute ground truth pose
        x0 = self.x_gt[-1][0]
        y0 = self.x_gt[-1][1]
        theta0 = self.x_gt[-1][2]

        x_gt = x0 + -vt/wt*np.sin(theta0) + vt/wt*np.sin(theta0 + wt*dt)
        y_gt = y0 + vt/wt*np.cos(theta0) - vt/wt*np.cos(theta0 + wt*dt)
        theta_gt = theta0 + wt*dt

        self.x_gt.append([x_gt, y_gt, theta_gt])
        
        # compute accumulated pose with noisy control
        vt += np.random.randn()*STD_V
        wt += np.random.randn()*STD_W

        x0 = self.x[-1][0]
        y0 = self.x[-1][1]
        theta0 = self.x[-1][2]

        x = x0 + -vt/wt*np.sin(theta0) + vt/wt*np.sin(theta0 + wt*dt)
        y = y0 +  vt/wt*np.cos(theta0) - vt/wt*np.cos(theta0 + wt*dt)
        theta = theta0 + wt*dt

        self.x.append([x, y, theta])
        
        zt = []
        cur_t = len(self.x) - 1
        for i in range(M):
            zi = self.measure(landmarks[i], x_gt, y_gt, theta_gt)
            zt.append(zi)

            # if the measurement is valid
            if zi[0,0] <= r_max:
                j = i
                self.tau[j].append(cur_t)

        self.z.append(zt)

        # noisy control signal
        self.u.append([vt, wt])
    
    def measure(self, mi, x, y, theta):
        """ This method measures landmark at pose (x, y, theta)
            mi: landmark
        """
        mx, my, ms = mi
        
        dx = mx - x
        dy = my - y
        r = np.sqrt(dx**2 + dy**2)
        phi = np.arctan2(dy, dx) - theta

        r += np.random.randn()*STD_R
        phi += np.random.randn()*STD_PHI
        
        # phi shall be [-pi, pi]
        if phi > np.pi:
            phi -= 2*np.pi
        elif phi < -np.pi:
            phi += 2*np.pi

        s = ms

        # maximum detection range: 6m
        if 0 < r <= r_max:
            # valid sensing
            return np.array([[r, phi]], dtype=np.float32).T
        else:
            # invalid sensing
            return np.array([[r_max + 1.0, phi]], dtype=np.float32).T

    def initialize(self):
        """ This method initializes mean pose vectors (Table 11.1)
            u: u_1:t
        """
        
        self.x_gt = np.array(self.x_gt)
        self.x = np.array(self.x)
        self.u = np.array(self.u)

        # map
        self.m = np.zeros([M, DIM_MEAS], dtype=np.float32)
        m_cnt = np.zeros([M])

        for t in range(len(self.z)):

            zt = self.z[t]
            x, y, theta = self.x[t]
            for i in range(len(zt)):
                zi = zt[i]

                r = zi[0,0]
                phi = zi[1,0]

                if r <= r_max:

                    mx = r*np.cos(phi + theta) + x
                    my = r*np.sin(phi + theta) + y

                    self.m[i,0] += mx
                    self.m[i,1] += my
                    # self.m[i,2] += s
                    m_cnt[i] += 1

        for i in range(M):
            self.m[i] /= m_cnt[i]

    def linearize(self):
        """ This method calculates O and xi (Table 11.2).
        """
        n = len(self.x) # the number of poses
        
        # initialize information matrix and information vector to 0 (line 2)
        self.O = np.zeros([n*DIM_POSE + M*DIM_MEAS, n*DIM_POSE + M*DIM_MEAS], dtype=np.float32)     # information matrix (Omega)
        self.xi = np.zeros([n*DIM_POSE + M*DIM_MEAS, 1], dtype=np.float32)      # information vector (xi)

        self.O[0:DIM_POSE, 0:DIM_POSE] += np.eye(DIM_POSE, DIM_POSE) * INFINITE    # line 3

        T = len(self.u)  # the number of controls
        for t in range(1, T):
            x0 = self.x[t-1].reshape(DIM_POSE,1)

            theta0 = x0[2,0]

            ut = self.u[t]
            vt, wt = ut

            dx = np.array([
                [-vt/wt*np.sin(theta0) + vt/wt*np.sin(theta0 + wt*dt)],
                [ vt/wt*np.cos(theta0) - vt/wt*np.cos(theta0 + wt*dt)],
                [ wt*dt],
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
            
            s_v = STD_V*dt
            s_w = STD_W*dt
            Rt = np.array([
                [s_v**2, 0,      0],
                [0,      s_v**2, 0],
                [0,      0,      s_w**2]
            ], dtype=np.float32)
            invRt = np.linalg.inv(Rt)
            
            # Line 7~8
            # on page 356
            # (xt - Gt x_{t-1})^T = x^T_{t-1:t}(-Gt 1)^T
            # (-Gt x_{t-1} + xt) = (-Gt 1)x_{t-1:t}
            # where x_{t-1:t} = [x_{t-1}, x_t]^T
            
            # Augmented matrix A = (-Gt I)
            # line 7
            I3 = np.eye(DIM_POSE, DIM_POSE)
            A = np.hstack([-Gt, I3])
            O_t = np.matmul(np.matmul(A.T, invRt), A)

            idx_t = (t-1)*DIM_POSE

            self.O[idx_t:idx_t+2*DIM_POSE, idx_t:idx_t+2*DIM_POSE] += O_t

            # line 8
            xi_t = np.matmul(np.matmul(A.T, invRt), xt - np.matmul(Gt, x0))
            self.xi[idx_t:idx_t+2*DIM_POSE, :] += xi_t

        # for all measurements zt (line 10)
        for t in range(0, T):
            zt = self.z[t]

            s_r = STD_R
            s_phi = STD_PHI
            # Q = np.array([
            #     [s_r**2, 0, 0],
            #     [0, s_phi**2, 0],
            #     [0, 0, s_s**2],
            # ], dtype=np.float32)

            Q = np.array([
                [s_r**2, 0],
                [0,      s_phi**2],
            ], dtype=np.float32)

            invQ = np.linalg.inv(Q)

            x = self.x[t,0]
            y = self.x[t,1]
            theta = self.x[t,2]

            # for all observed features z^i_t = (r^i_t, phi^i_t, s^i_t)
            for i in range(len(zt)):
                # measurement i corresponds to landmark j.
                j = i
                zi = zt[j]
                if zi[0,0] > r_max:
                    continue

                # m_jx, m_jy, m_js = self.m[j]
                m_jx, m_jy = self.m[j]
                dx = m_jx - x
                dy = m_jy - y
                q = dx**2 + dy**2
                r = np.sqrt(q)

                phi_pred = np.arctan2(dy,dx) - theta

                # phi_pred shall be [-pi, pi]
                if phi_pred > np.pi:
                    phi_pred -= 2*np.pi
                elif phi_pred < -np.pi:
                    phi_pred += 2*np.pi

                zi_pred = np.array([
                    [np.sqrt(q)],
                    [phi_pred],
                    # [m_js]
                ])

                # dr/dx,   dr/dy,   dr/dtheta,   dr/dmx_j,   dr/dmy_j
                # dphi/dx, dphi/dy, dphi/dtheta, dphi/dmx_j, dphi/dmy_j
                Hi = np.array([
                    [-dx/r, -dy/r,  0,  dx/r, dy/r],
                    [ dy/q, -dx/q, -1, -dy/q, dx/q],
                ])

                # line 18
                # Information at xt and mj
                O_tj = np.matmul(np.matmul(Hi.T, invQ), Hi)
                # Omega at xt and mj
                #     x0 x1 x2 m0 m1
                #  x0  x
                #  x1     x
                #  x2        x    xx
                #  m0
                #  m1
                
                # 6x6 matrix
                # O_xt O_xm
                # O_mx O_mj
                # As described on page 351, chop matrix and vector; add to O and xi.

                # O_tj = [A   B]
                #        [B^T C]                
                idx_t = t*DIM_POSE
                idx_j = n*DIM_POSE + j*DIM_MEAS
                
                # A
                self.O[idx_t:idx_t+DIM_POSE, idx_t:idx_t+DIM_POSE] += O_tj[:DIM_POSE,:DIM_POSE]

                # B
                self.O[idx_t:idx_t+DIM_POSE, idx_j:idx_j+DIM_MEAS] += O_tj[:DIM_POSE,DIM_POSE:]

                # B^T
                self.O[idx_j:idx_j+DIM_MEAS, idx_t:idx_t+DIM_POSE] += O_tj[DIM_POSE:,:DIM_POSE]

                # C
                self.O[idx_j:idx_j+DIM_MEAS, idx_j:idx_j+DIM_MEAS] += O_tj[DIM_POSE:,DIM_POSE:]
                                

                # line 19
                mu = np.array([[x, y, theta, m_jx, m_jy]]).T
                delta = zi - zi_pred + np.matmul(Hi, mu)
                xi_tj = np.matmul(np.matmul(Hi.T, invQ), delta)

                self.xi[idx_t:idx_t+DIM_POSE] += xi_tj[:DIM_POSE]
                self.xi[idx_j:idx_j+DIM_MEAS] += xi_tj[DIM_POSE:]

    def reduce(self):
        """ This method reduces the size of the information (Table 11.3).
        """
        n = len(self.x) # the number of poses
        self.O_red = self.O.copy()     # line 2, reduced information matrix
        self.xi_red = self.xi.copy()   # line 3, reduced information vector

        # for each feature j
        for j in range(M):
           
            idx_j = n*DIM_POSE + j*DIM_MEAS
            # 2x2
            O_jj = self.O_red[idx_j:idx_j+DIM_MEAS, idx_j:idx_j+DIM_MEAS]
            # 2x2
            invO_jj = np.linalg.inv(O_jj)
            # 2x1
            xi_j = self.xi_red[idx_j:idx_j+DIM_MEAS]

            n_tau = len(self.tau[j])
            # 3k x 2
            O_kj = np.zeros([DIM_POSE*n_tau, DIM_MEAS], dtype=np.float32)
            # 2 x 3k
            O_jk = np.zeros([DIM_MEAS, DIM_POSE*n_tau], dtype=np.float32)
            
            # for each pose tau(j)
            idx_row = 0
            for k in self.tau[j]:

                # tau(j) = {1, 2}
                # xi: [xi_1
                #      xi_2
                #      xi_j]
                
                # O: [O_11,     , O_1j, 
                #         , O_22, O_2j]
                #     O_j1, O_j2, O_jj

                # O_kk = O_kk - O_kj * (O_jj)^-1 * O_jk
                # xi_k = xi_k - O_kj * (O_jj)^-1 * xi_j

                idx_k = k*DIM_POSE

                # 3k x 2
                O_kj[idx_row:idx_row+DIM_POSE, :] += self.O_red[idx_k:idx_k+DIM_POSE, idx_j:idx_j+DIM_MEAS]
                # 2 x 3k
                O_jk[:, idx_row:idx_row+DIM_POSE] += self.O_red[idx_j:idx_j+DIM_MEAS, idx_k:idx_k+DIM_POSE]

                idx_row += DIM_POSE
                
            
            # Marginal of pose k: 3x2 * 2x2* 2x1 = 3x1
            xi_kj = np.matmul(np.matmul(O_kj, invO_jj), xi_j)

            # Marginal of pose k: 3x2 * 2x2 * 2x3 = 3x3
            O_kk = np.matmul(np.matmul(O_kj, invO_jj), O_jk)
            

            idx_row = 0
            for k in self.tau[j]:

                idx_k = k*DIM_POSE
                self.xi_red[idx_k:idx_k+DIM_POSE] -= xi_kj[idx_row:idx_row+DIM_POSE]

                idx_col = 0
                for k2 in self.tau[j]:
                    idx_k2 = k2*DIM_POSE

                    self.O_red[idx_k:idx_k+DIM_POSE, idx_k2:idx_k2+DIM_POSE] -= O_kk[idx_row:idx_row+DIM_POSE, idx_col:idx_col+DIM_POSE]

                    idx_col += DIM_POSE

                idx_row += DIM_POSE

        self.O_red = self.O_red[:n*DIM_POSE, :n*DIM_POSE]
        self.xi_red = self.xi_red[:n*DIM_POSE,:]


    def solve(self):
        """ This method updates the posterior x(mu) (Table 11.4).
        """
        n = len(self.x) # the number of poses
    
        self.S = np.linalg.inv(self.O_red)
        self.x = np.matmul(self.S, self.xi_red)

        # self.x = self.x.reshape([-1, DIM_POSE])
        self.x = self.x[:n*DIM_POSE].reshape([-1, DIM_POSE])

        # print('solved pose')
        # print(self.x)

        # for each feature j
        for j in range(M):
            idx_j = n*DIM_POSE + j*DIM_MEAS

            O_jj = self.O[idx_j:idx_j+DIM_MEAS, idx_j:idx_j+DIM_MEAS]
            xi_j = self.xi[idx_j:idx_j+DIM_MEAS]

            invO_jj = np.linalg.inv(O_jj)

            n_tau = len(self.tau[j])
            O_jk = np.zeros([DIM_MEAS, n_tau*DIM_POSE])
            x_k  = np.zeros([n_tau*DIM_POSE, 1])

            idx_stack = 0
            for k in self.tau[j]:
                
                idx_k = k*DIM_POSE
                O_jk0 = self.O[idx_j:idx_j+DIM_MEAS, idx_k:idx_k+DIM_POSE]
                x_k0  = self.x[k].reshape([DIM_POSE,1])

                O_jk[:, idx_stack:idx_stack+DIM_POSE] += O_jk0
                x_k[idx_stack:idx_stack+DIM_POSE,:] += x_k0

                idx_stack += DIM_POSE

            # Please refer to errata: http://probabilistic-robotics.informatik.uni-freiburg.de/corrections/pg361.pdf
            # '+' -> '-'
            mj = np.matmul(invO_jj, xi_j - np.matmul(O_jk, x_k))

            self.m[j,0] = mj[0,0]
            self.m[j,1] = mj[1,0]


    def plot(self):
        
        fig = plt.figure('map')

        plt.scatter(landmarks[:,0], landmarks[:,1], c='k')

        plt.plot(self.x_gt[:,0], self.x_gt[:,1], c='g')
        plt.plot(self.x[:,0], self.x[:,1], c='r')

        plt.scatter(self.m[:,0], self.m[:,1], c='r')

        plt.axis('equal')

        plt.draw()
        plt.waitforbuttonpress(0)
        plt.close(fig)

if __name__ == '__main__':

    slam = GraphSLAM()

    # Robot maneuver
    for t in range(21*tm):
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
        
        slam.control(ut)

    slam.initialize()
    slam.plot()

    # repeat
    for i in range(3):
        slam.linearize()
        slam.reduce()
        slam.solve()
        
        slam.plot()

