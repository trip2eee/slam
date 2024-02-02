""" Fast SLAM 1.0 with unknown correspondence
    Table 13.2 on page 461.
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
thres_sig = 2

alpha = [0.02, 0.01,  # v_t
         0.01, 0.02,  # w_t
         0.01, 0.01]  # hat


class Feature:
    def __init__(self, m=None, Cov=None):
        self.m = m         # mean value
        self.Cov = Cov     # covariance
        self.i = 1         # observed count
        self.z_pred = None # predicted measurement
        self.H = None      # Jacobian of h()        
        self.S = None
        self.invS = None
        self.assoc = True
        self.z = None
        self.w = 0
        self.id = 0

    def copy(self):
        f = Feature()
        f.m = self.m.copy()
        f.Cov = self.Cov.copy()
        f.i = self.i
        f.w = self.w
        f.id = self.id

        if self.z_pred is not None:
            f.z_pred = self.z_pred.copy()
            f.H = self.H.copy()
            f.S = self.S.copy()
            f.invS = self.invS.copy()
            f.assoc = self.assoc
        
        if self.z is not None:
            f.z = self.z.copy()
        return f

class Particle:
    def __init__(self):
        self.x = []         # robot path list
        self.feature = []   # feature list
        self.w = 0          # importance weight
        self.new_feature_id = 0

    def copy(self):
        y = Particle()
        y.x = self.x.copy()
        y.feature = []
        for f in self.feature:
            y.feature.append(f.copy())
        y.w = self.w
        y.new_feature_id = self.new_feature_id

        return y

class FastSLAM:
    def __init__(self):
        self.x_gt = [[0, 0, 0]] # ground truth pose
        self.x = [[0, 0, 0]]    # estimated pose
        self.M = 100            # the number of particles        

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

        if 0 < r <= r_max:
            # valid sensing

            r += np.random.randn()*STD_R
            phi += np.random.randn()*STD_PHI
            
            # phi shall be [-pi, pi]
            if phi > np.pi:
                phi -= 2*np.pi
            elif phi < -np.pi:
                phi += 2*np.pi

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

        if phi > np.pi:
            phi -= np.pi*2
        elif phi < -np.pi:
            phi += np.pi*2

        z_pred = np.array([[r, phi]]).T

        return z_pred
    
    def jacobian_H(self, m_t, xt):

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

            # for each feature fj
            f_j:Feature
            for f_j in y_k.feature:
                f_j.z_pred = self.h(f_j.m, x_k1)        # measurement prediction (line 6)
                f_j.H = self.jacobian_H(f_j.m, x_k1)    # calculate Jacobian (line 7)

                # measurement covariance
                std_r = STD_R*1
                std_phi = STD_PHI*1
                Qt = np.array([
                    [std_r**2, 0.0],
                    [0.0, std_phi**2]
                ])
                H = f_j.H
                Cov_j = f_j.Cov

                # if f_j.id == 0:
                #     print(Cov_j)
                    
                S = np.matmul(np.matmul(H, Cov_j), H.T) + Qt
                invS = np.linalg.inv(S)

                f_j.S = S
                f_j.invS = invS
                f_j.assoc = False
                f_j.z = None

            # for each measurement
            num_features = len(y_k.feature) # the number of valid features N^k_{t-1}
            for lm_id, lm in enumerate(landmarks):

                z = self.measure(lm)

                # if the measurement is valid
                if z[0,0] <= r_max:
                    
                    c = -1        # correspondence
                    w_max = 0    # maximum likelihood of correspondence
                    for j in range(num_features):
                        f_j = y_k.feature[j]
                        z_pred = f_j.z_pred
                        invS = f_j.invS
                        r = z - z_pred

                        if r[1,0] > np.pi:
                            r[1,0] -= np.pi*2
                        elif r[1,0] < -np.pi:
                            r[1,0] += np.pi*2

                        d2_Mahalanobis = np.matmul(np.matmul(r.T, invS), r)

                        detS = np.linalg.det(f_j.S)
                        eta = 1/(2*np.pi*np.sqrt(detS))
                        w = eta*np.exp(-0.5*d2_Mahalanobis)
                        w = w[0,0]

                        # print('d(L{}, F{}): {}, {}'.format(lm_id, j, d2_Mahalanobis, w))
                        if d2_Mahalanobis <= thres_sig**2:
                            if w_max < w:
                                w_max = w
                                c = j

                    # if feature j is a new feature (line 16)
                    if c == -1:
                        # print('L{} not associated, {}, {}'.format(lm_id, min_d, w_max))

                        # initialize mean mu^k_j,t = h^-1(zt, x^k_t)  (line 17)
                        m_jx, m_jy = self.inv_h(z, x_k1)
                        m_j = np.array([[m_jx, m_jy]]).T

                        # Calculate Jacobian (line 8)
                        H = self.jacobian_H(m_j, x_k1)

                        # Initialize Covariance
                        std_r0 = STD_R*1
                        std_phi0 = STD_PHI*1
                        Qt = np.array([
                            [std_r0**2, 0.0],
                            [0.0, std_phi0**2]
                        ])
                        invH = np.linalg.inv(H)
                        Cov_j = np.matmul(np.matmul(invH, Qt), invH.T)

                        f_j = Feature(m_j, Cov_j)
                        f_j.H = H
                        f_j.assoc = True
                        f_j.z = z
                        f_j.w = P0
                        f_j.id = y_k.new_feature_id
                        # print('create new feature {}'.format(f_j.id))
                        y_k.new_feature_id += 1

                        y_k.feature.append(f_j)
                    else:
                        f_j:Feature
                        f_j = y_k.feature[c]

                        # print('L{} associated with F{}'.format(lm_id, f_j.id))
                        
                        z_pred = f_j.z_pred
                        m_j = f_j.m
                        H = f_j.H
                        invS = f_j.invS
                        Cov_j = f_j.Cov

                        K = np.matmul(np.matmul(Cov_j, H.T), invS)
                        I = np.eye(2)

                        r = z - z_pred
                        if r[1,0] > np.pi:
                            r[1,0] -= np.pi*2
                        elif r[1,0] < -np.pi:
                            r[1,0] += np.pi*2

                        f_j.m = m_j + np.matmul(K, r)
                        f_j.Cov = np.matmul((I - np.matmul(K, H)), Cov_j)

                        f_j.i += 1
                        f_j.assoc = True
                        f_j.z = z
                        f_j.w = w_max

            y_k.w = 1.0
            num_assoc = 0
            f_j:Feature
            for f_j in y_k.feature:
                if True == f_j.assoc:
                    y_k.w *= f_j.w
                    num_assoc += 1
                else:
                    m_x = f_j.m[0,0]
                    m_y = f_j.m[1,0]

                    x = x_k1[0,0]
                    y = x_k1[1,0]

                    dx = m_x - x
                    dy = m_y - y
                    r2 = dx**2 + dy**2
                    f_j.w = P0

                    # if the feature j is inside perceptual range of the robot
                    if r2 <= r_max**2:
                        f_j.i -= 1                  # decrement counter (line 31)
                        if f_j.i < 0:
                            # print('F{} discarded'.format(f_j.id))
                            y_k.feature.remove(f_j) # discard feature j (line 33)
            
            if num_assoc == 0:
                y_k.w = P0

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
        plt.clf()

        plt.scatter(landmarks[:,0], landmarks[:,1], c='k')
        
        k_best = 0
        for k in range(self.M):
            y_k:Particle
            y_k = self.Y[k]            

            x_k = np.array(y_k.x)
            plt.plot(x_k[:,0,0], x_k[:,1,0], c='r')
            if y_k.w > self.Y[k_best].w:
                k_best = k

        # draw the ground truth path        
        x_gt = np.array(self.x_gt)
        plt.plot(x_gt[:,0], x_gt[:,1], c='g')
        self.draw_robot(x_gt[-1,:].reshape([3,1]), color='g')

        # draw path of the best particle
        y_k:Particle
        y_k = self.Y[k_best]
        x_k = np.array(y_k.x)
        plt.plot(x_k[:,0,0], x_k[:,1,0], c='b')
        self.draw_robot(y_k.x[-1], color='b')

        # draw features of the best particle
        f_j:Feature
        for f_j in y_k.feature:
            m_j = f_j.m
            z_pred = f_j.z_pred
            z = f_j.z
            S = f_j.S
            H = f_j.H
            x_k1 = y_k.x[-1]

            if z_pred is not None:
                m_jx, m_jy = self.inv_h(z_pred, x_k1)
                m_pred = np.array([[m_jx, m_jy]]).T
                plt.scatter(m_pred[0,0], m_pred[1,0], c='c')

                invH = np.linalg.inv(H)
                Cov_j = np.matmul(np.matmul(invH, S), invH.T)
                self.draw_covariance(m_pred, Cov_j, n_sig=thres_sig)

            if z is not None:                
                m_jx, m_jy = self.inv_h(z, x_k1)
                plt.scatter(m_jx, m_jy, c='r')

            plt.scatter(m_j[0,0], m_j[1,0], c='b')
            plt.text(m_j[0,0], m_j[1,0], 'F{}'.format(f_j.id))

        plt.axis('equal')
        plt.draw()
        plt.waitforbuttonpress(0.1)

if __name__ == '__main__':

    slam = FastSLAM()

    # Robot maneuver (with loop closing)
    for t in range(30*tm):
        print('t:', t)

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
    
    plt.show()



