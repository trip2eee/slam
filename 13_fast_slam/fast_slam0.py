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

INFINITE = 100**2
    
tm = 2  # time multiplier
dt = 0.1/tm

r_max = 10.0        # maximum detection range
p_min_assoc = 0.3   # minimum association probability
r_assoc = 3.0

DIM_POSE = 3
DIM_MEAS = 2

def deg2rad(x):
    return x * np.pi / 180.0

STD_V = 0.3
STD_W = deg2rad(5)

STD_R = 0.3
STD_PHI = deg2rad(2)


alpha = [0.1, 0.1,  # v_t
         0.1, 0.1,  # w_t
         0.1, 0.1]  # hat


class Particle:
    def __init__(self):
        self.x = []         # robot path list
        self.feature = []   # feature list

class FastSLAM:
    def __init__(self):
        self.x_gt = [[0, 0, 0]] # ground truth pose
        self.x = [[0, 0, 0]]    # estimated pose

        self.z = [[]]   # measurements
        
        self.m = None   # map features

        self.seen = [0]*L
        self.cor = []     # correspondences (c)
        self.M = 100      # the number of particles
        
        self.Y = []     # list of particles
        for k in range(self.M):
            y = Particle()
            y.x.append([0, 0, 0])    # x, y, theta = 0
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


    def inv_h(self, z, xt):
        r   = z[0,0]
        phi = z[1,0]

        x_t = xt[0]
        y_t = xt[1]
        theta_t = xt[2]

        m_x = r*np.cos(phi + theta_t) + x_t
        m_y = r*np.sin(phi + theta_t) + y_t

        return [m_x, m_y]

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

    def fast_slam(self, ut):
        """ Algorithm FastSLAM (Table 13.1 on p.450)
        """
        
        M = self.M
        # loop over all particles
        for k in range(M):
            # retrieve particle from Y_t-1
            y: Particle
            y = self.Y[k]

            # sample pose (line 4)
            x0 = y.x[-1]
            x1 = self.sample_motion_model_velocity(x0, ut, dt=dt)
            y.x.append(x1)

            x_t = x1[0]
            y_t = x1[1]
            theta_t = x1[2]

            for j in range(L):
                
                zj = self.measure(landmarks[j])
                # if the measurement is valid
                if zj[0,0] <= r_max:

                    # if feature j never seen before
                    if 0 == self.seen[j]:
                        # initialize mean   (line 7)
                        # mu^k_j,t = h^-1(zt, x^k_t)
                        m_jx, m_jy = self.inv_h(zj, x1)
                        
                        # Calculate Jacobian (line 8)
                        dx = m_jx - x_t
                        dy = m_jy - y_t
                        q = dx**2 + dy**2
                        r = np.sqrt(q)

                        # dr/dx,   dr/dy,   dr/dtheta,   dr/dmx_j,   dr/dmy_j,   dr/dms_j
                        # dphi/dx, dphi/dy, dphi/dtheta, dphi/dmx_j, dphi/dmy_j, dphi/dms_j
                        H = np.array([
                            [-dx/r, -dy/r,  0,  dx/r, dy/r],
                            [ dy/q, -dx/q, -1, -dy/q, dx/q],
                        ])



        self.plot()

    def draw_covariance(self, j, n_sig=2):
        # Draw Covariance
        if self.O is not None:
            n = len(self.x)
            
            idx_j = n*DIM_POSE + j*DIM_MEAS

            O_jj = self.O[idx_j:idx_j+DIM_MEAS, idx_j:idx_j+DIM_MEAS]
            mx = self.m[j,0]
            my = self.m[j,1]
            
            if np.linalg.det(O_jj) > 0:
                Cov_jj = np.linalg.inv(O_jj)
                # eigenvalue, eigenvector
                w, v = np.linalg.eig(Cov_jj)
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

        for k in range(self.M):
            y = self.Y[k]
            y:Particle

            xk = np.array(y.x)

            plt.plot(xk[:,0], xk[:,1], c='r')
            

        # m_landmarks = []
        # for idx_m in range(len(self.m)):
        #     if self.cor[idx_m] == idx_m:
        #         m = self.m[idx_m]
        #         # plt.text(m[0], m[1], '{:d}'.format(idx_m))
        #         m_landmarks.append(m)
        # m_landmarks = np.array(m_landmarks)

        # plt.scatter(m_landmarks[:,0], m_landmarks[:,1], c='r')

        # print('# measurements:', len(self.m))
        # print('# landmarks:', len(m_landmarks))

        # self.draw_covariance(j=0)
        # self.draw_covariance(j=1)
        # self.draw_covariance(j=2)


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



