""" Fast SLAM 1.0 with known correspondence
    Table 13.1 on page 450.
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

L = len(landmarks)      # the number of landmarks

tm = 2  # time multiplier
dt = 0.1/tm

r_max = 10.0        # maximum detection range

DIM_POSE = 3
DIM_MEAS = 2

def deg2rad(x):
    return x * np.pi / 180.0

STD_R = 0.5
STD_PHI = deg2rad(5)

P0 = 0.1    # default importance weight

alpha = [0.02, 0.01,  # v_t
         0.01, 0.02,  # w_t
         0.01, 0.01]  # hat

class Particle:
    def __init__(self):
        self.x = []               # robot path list
        self.feature = [None]*L   # feature list
        self.w = 0                # importance weight

    def copy(self):
        y = Particle()
        y.x = self.x.copy()
        y.feature = self.feature.copy()
        y.w = self.w

        return y

class FastSLAM:
    def __init__(self):
        self.x_gt = [[0, 0, 0]] # ground truth pose
        self.x = [[0, 0, 0]]    # estimated pose

        self.z = [[]]   # measurements
        
        self.m = None   # map features
        
        self.cor = []     # correspondences (c)
        self.M = 100      # the number of particles
        
        self.Y = []     # list of particles
        for k in range(self.M):
            y = Particle()
            x0 = np.array([[0, 0, 0]]).T
            y.x.append(x0)    # x, y, theta = 0
            self.Y.append(y)

    def control(self, ut):
        vt, wt = ut
       
        # compute ground truth pose
        x0 = self.x_gt[-1][0]
        y0 = self.x_gt[-1][1]
        theta0 = self.x_gt[-1][2]

        x_gt = x0 + -vt/wt*np.sin(theta0) + vt/wt*np.sin(theta0 + wt*dt)
        y_gt = y0 + vt/wt*np.cos(theta0) - vt/wt*np.cos(theta0 + wt*dt)
        theta_gt = theta0 + wt*dt

        self.x_gt.append([x_gt, y_gt, theta_gt])

        self.fast_slam(ut)

    
    def measure(self, mi):
        """ This method measures landmark at pose (x, y, theta)
            mi: landmark
        """
        mx, my, ms = mi
        
        x     = self.x_gt[-1][0]
        y     = self.x_gt[-1][1]
        theta = self.x_gt[-1][2]

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
            return np.array([[r, phi]]).T
        else:
            # invalid sensing
            return np.array([[r_max + 1.0, phi]]).T

    
    def sample_normal_distribution(self, b):
        """ Table 5.4 Algorithm for sampling from normal distribution with zero mean and variance b.
        """
        r = np.random.randint(-1000, 1000, size=12) * 0.001
        return b/6.0*r.sum()

    
    def sample_motion_model_velocity(self, xt_1, ut, dt=0.1):
        """ This method computes motion model in Table 8.1 on p.188.
        """
        # control ut
        vt, wt = ut

        # compute ground truth
        x = xt_1[0]
        y = xt_1[1]
        theta = xt_1[2]
        

        sample = self.sample_normal_distribution

        v_hat = vt + sample(alpha[0]*abs(vt) + alpha[1]*abs(wt))   # line 1
        w_hat = wt + sample(alpha[2]*abs(vt) + alpha[3]*abs(wt))   # line 2
        g_hat =      sample(alpha[4]*abs(vt) + alpha[5]*abs(wt))   # line 3
        
        x_t = x - v_hat/w_hat*np.sin(theta) + v_hat/w_hat*np.sin(theta + w_hat*dt)
        y_t = y + v_hat/w_hat*np.cos(theta) - v_hat/w_hat*np.cos(theta + w_hat*dt)
        theta_t = theta + w_hat*dt + g_hat*dt
        
        return [x_t, y_t, theta_t]

    def inv_h(self, z, xt):
        """ Inverse measurement function
        """
        r   = z[0,0]
        phi = z[1,0]

        x_t = xt[0,0]
        y_t = xt[1,0]
        theta_t = xt[2,0]

        m_x = r*np.cos(phi + theta_t) + x_t
        m_y = r*np.sin(phi + theta_t) + y_t

        return [m_x, m_y]

    def h(self, m_t, xt):
        """ Measurement function h
        """
        dx = m_t[0,0] - xt[0,0]
        dy = m_t[1,0] - xt[1,0]

        r = np.sqrt(dx**2 + dy**2)
        phi = np.arctan2(dy,dx) - xt[2,0]

        z_pred = np.array([[r, phi]]).T

        return z_pred
    
    def jacobian_H(self, xt, m_t):

        dx = m_t[0,0] - xt[0,0]
        dy = m_t[1,0] - xt[1,0]
        q = dx**2 + dy**2
        r = np.sqrt(q)

        # dr/dmx_j,   dr/dmy_j
        # dphi/dmx_j, dphi/dmy_j
        H = np.array([
            [ dx/r, dy/r],
            [-dy/q, dx/q],
        ])

        return H
    
    def fast_slam(self, ut):
        """ Algorithm FastSLAM (Table 13.1 on p.450)
        """
        
        M = self.M
        # loop over all particles
        for k in range(M):
            # retrieve particle from Y_t-1
            y_k: Particle
            y_k = self.Y[k]

            # sample pose (line 4)
            x_k0 = y_k.x[-1]
            x_k1 = self.sample_motion_model_velocity(x_k0, ut, dt=dt)
            x_k1 = np.array(x_k1).reshape([3,1])
            y_k.x.append(x_k1)

            x_t = x_k1[0,0]
            y_t = x_k1[1,0]

            for j in range(L):
                
                zj = self.measure(landmarks[j])
                # if the measurement is valid
                if zj[0,0] <= r_max:

                    # if feature j never seen before
                    if y_k.feature[j] is None:

                        # initialize mean   (line 7)
                        # mu^k_j,t = h^-1(zt, x^k_t)
                        m_jx, m_jy = self.inv_h(zj, x_k1)
                        m_j = np.array([[m_jx, m_jy]]).T

                        # Calculate Jacobian (line 8)
                        H = self.jacobian_H(x_k1, m_j)

                        # Initialize Covariance
                        std_r0 = STD_R*2
                        std_phi0 = STD_PHI*2
                        Qt = np.array([
                            [std_r0**2, 0.0],
                            [0.0, std_phi0**2]
                        ])
                        invH = np.linalg.inv(H)
                        Cov_j = np.matmul(np.matmul(invH, Qt), invH.T)

                        y_k.feature[j] = [m_j, Cov_j]
                        y_k.w = P0    # default importance weight
                    else:
                        m_j, Cov_j = y_k.feature[j]
                        z_pred = self.h(m_j, x_k1)  # line 12

                        # Calculate Jacobian (line 13)
                        H = self.jacobian_H(x_k1, m_j)
                        
                        std_r = 1.0 + STD_R
                        std_phi = 1.0 + STD_PHI
                        Qt = np.array([
                            [std_r**2, 0.0],
                            [0.0, std_phi**2]
                        ])
                        S = np.matmul(np.matmul(H, Cov_j), H.T) + Qt
                        invS = np.linalg.inv(S)
                        K = np.matmul(np.matmul(Cov_j, H.T), invS)
                        I = np.eye(2)
                        m_j = m_j + np.matmul(K, zj - z_pred)
                        Cov_j = np.matmul((I - np.matmul(K, H)), Cov_j)

                        y_k.feature[j] = [m_j, Cov_j]

                        r = zj - z_pred
                        detS = np.linalg.det(S)
                        y_k.w = 1/(2*np.pi*np.sqrt(detS)) * np.exp(-0.5*np.matmul(np.matmul(r.T, invS), r))
                        y_k.w = y_k.w[0,0]                        
                else:
                    # leave unobserved features unchanged
                    pass
            
        # Resample
        Y1 = [] # initialize new particle set (line 25)
        
        print('resampling')
        # normalize importance
        sum_w = 0
        for m in range(M):
            sum_w += self.Y[m].w
        print('sum_w: {}'.format(sum_w))
        sum_w = max(1e-8, sum_w)

        for m in range(M):
            self.Y[m].w /= sum_w
            # print(self.Y[m].w)

        c = self.Y[0].w
        i = 0
        r = np.random.rand() * 1/self.M
        for m in range(self.M):
            U = r + (m / M)   # m: 0 ~ M-1
            while U > c and i < (M-1):
                i = i + 1
                c = c + self.Y[i].w

            y_k = self.Y[i].copy()
            Y1.append(y_k)
        
        self.Y = Y1
                    
        self.plot()

    def draw_covariance(self, m_j, Cov_j, n_sig=2):
        # Draw Covariance
        mx = m_j[0,0]
        my = m_j[1,0]
        
        # eigenvalue, eigenvector
        w, v = np.linalg.eig(Cov_j)
        theta = np.arctan2(v[1,0], v[0,0])

        x = [0] * 361
        y = [0] * 361

        cos = np.cos(theta)
        sin = np.sin(theta)

        for a in range(361):
            x0 = w[0] * np.cos(a*np.pi/180) * n_sig
            y0 = w[1] * np.sin(a*np.pi/180) * n_sig

            x[a] = x0*cos - y0*sin + mx
            y[a] = x0*sin + y0*cos + my

        plt.plot(x, y, c='c')

    def draw_robot(self, x, color):
        x_robot = x[0,0]
        y_robot = x[1,0]
        theta_robot = x[2,0]

        d=1.0
        dx = d*np.cos(theta_robot)
        dy = d*np.sin(theta_robot)
        plt.scatter([x_robot], [y_robot], c=color)
        plt.plot([x_robot, x_robot+dx], [y_robot, y_robot+dy], c=color)

    def plot(self):
        fig = plt.figure('map')

        plt.scatter(landmarks[:,0], landmarks[:,1], c='k')
        
        x_gt = np.array(self.x_gt)
        plt.plot(x_gt[:,0], x_gt[:,1], c='g')
        self.draw_robot(x_gt[-1,:].reshape([3,1]), color='g')

        k_best = 0

        for k in range(self.M):
            y_k:Particle
            y_k = self.Y[k]            

            x_k = np.array(y_k.x)
            plt.plot(x_k[:,0,0], x_k[:,1,0], c='r')

            if y_k.w > self.Y[k_best].w:
                k_best = k
        
        # draw path of the best particle
        y_k:Particle
        y_k = self.Y[k_best]
        x_k = np.array(y_k.x)
        plt.plot(x_k[:,0,0], x_k[:,1,0], c='b')
        self.draw_robot(y_k.x[-1], color='b')

        # draw features of the best particle
        for j in range(L):
            if y_k.feature[j] is not None:
                m_j, Cov_j = y_k.feature[j]
                plt.scatter(m_j[0,0], m_j[1,0], c='r')
                self.draw_covariance(m_j, Cov_j, n_sig=2)


        plt.axis('equal')

        plt.draw()
        plt.waitforbuttonpress(0)
        plt.close(fig)

if __name__ == '__main__':

    slam = FastSLAM()

    # Robot maneuver (with loop closing)
    for t in range(30*tm):
        if t <= 7*tm:
            ut = [25, 0.001]
        elif t <= 9*tm:
            ut = [25, -np.pi/2*5]
        elif t <= 10*tm:
            ut = [25, 0.001]
        elif t <= 12*tm:
            ut = [25, -np.pi/2*5]
        elif t <= 17*tm:
            ut = [25, 0.001]
        elif t <= 19*tm:
            ut = [25, -np.pi/2*5]
        elif t <= 20*tm:
            ut = [25, 0.001]
        elif t <= 22*tm:
            ut = [25, -np.pi/2*5]
        else:
            ut = [25, 0.001]
        
        slam.control(ut)



