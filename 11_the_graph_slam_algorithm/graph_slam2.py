""" GraphSLAM with unknown correspondence
    Table 11.1, 11.2, 11.3, 11.4, 11.8, and 11.9
    on page 347~365

    Closed Loop Test
    TODO:
        1. Further improvement using Conjugate Gradient
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
p_min_assoc = 0.5   # minimum association probability
r_assoc = 3.0

DIM_POSE = 3
DIM_MEAS = 2

def deg2rad(x):
    return x * np.pi / 180.0

STD_V = 0.5
STD_W = deg2rad(5)

STD_R = 0.3
STD_PHI = deg2rad(2)

class GraphSLAM:
    def __init__(self):
        self.x_gt = [[0, 0, 0]] # ground truth pose
        self.x = [[0, 0, 0]]    # estimated pose

        self.S = None   # Sigma 0:t

        self.u_gt = []  # controls (ground truth)
        self.u = [[0.0, 0.0]]   # controls (with noise)
                
        self.z = [[]]   # measurements
        
        self.tau = []   # tau(j) : set of poses xt at which j was observed.

        self.m = None   # map features
        self.num_meas = 0
        self.cor = []     # correspondences (c)
        self.M = 0      # the number of features (unknown in the initial state)

        self.O = None   # information matrix (Omega)

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
        # for each landmark
        for i in range(L):
            zi = self.measure(landmarks[i], x_gt, y_gt, theta_gt)

            # if the measurement is valid
            if zi[0,0] <= r_max:
                zt.append(zi)
                j = self.num_meas
                self.cor.append(j)   # ci = j

                tau_t = [cur_t]
                self.tau.append(tau_t)

                self.num_meas += 1        
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
            return np.array([[r, phi]]).T
        else:
            # invalid sensing
            return np.array([[r_max + 1.0, phi]]).T

    def initialize(self):
        """ This method initializes mean pose vectors (Table 11.1)
            u: u_1:t
        """
        
        self.x_gt = np.array(self.x_gt)
        self.x = np.array(self.x)
        self.u = np.array(self.u)

        # map
        self.M = self.num_meas
        M = self.M
        self.m = np.zeros([M, DIM_MEAS], dtype=np.float32)

        idx_m = 0
        for t in range(len(self.z)):

            zt = self.z[t]
            x, y, theta = self.x[t]
            for i in range(len(zt)):
                zi = zt[i]

                r = zi[0,0]
                phi = zi[1,0]

                mx = r*np.cos(phi + theta) + x
                my = r*np.sin(phi + theta) + y

                self.m[idx_m,0] = mx
                self.m[idx_m,1] = my                
                # self.m[idx_m,2] += s
                idx_m += 1

    def linearize(self):
        """ This method calculates O and xi (Table 11.2).
        """
        n = len(self.x) # the number of poses
        M = self.M      # the number of features
        
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

        idx_m = 0
        # for all measurements zt (line 10)
        for t in range(0, T):
            zt = self.z[t]

            s_r = STD_R
            s_phi = STD_PHI

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
                j = self.cor[idx_m]   # correspondence
                idx_m += 1

                zi = zt[i]

                m_jx, m_jy = self.m[j]
                dx = m_jx - x
                dy = m_jy - y
                q = dx**2 + dy**2
                r = np.sqrt(q)

                phi_pred = np.arctan2(dy,dx) - theta                

                # phi_pred shall be [-pi, pi]
                # if phi_pred > np.pi:
                #     phi_pred -= 2*np.pi
                # elif phi_pred < -np.pi:
                #     phi_pred += 2*np.pi

                # Error Case
                # phi = 3.14 - 0.1 = 3.04
                # phi_pred = 3.14 - 0.1 + 0.2(noise) = 3.24 
                # Since 3.24 > np.pi, 3.24 - 2*pi = -3.04
                phi_error = zi[1,0] - phi_pred
                if phi_error < -np.pi:
                    phi_pred -= 2*np.pi
                elif phi_error > np.pi:
                    phi_pred += 2*np.pi

                zi_pred = np.array([
                    [r],
                    [phi_pred],
                ])

                # dr/dx,   dr/dy,   dr/dtheta,   dr/dmx_j,   dr/dmy_j,   dr/dms_j
                # dphi/dx, dphi/dy, dphi/dtheta, dphi/dmx_j, dphi/dmy_j, dphi/dms_j
                # ds/dx,   ds/dy,   ds/dtheta,   ds/dmx_j,   ds/dmy_j,   ds/dms_j
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
        M = self.M
        self.O_red = self.O.copy()     # line 2, reduced information matrix
        self.xi_red = self.xi.copy()   # line 3, reduced information vector

        # for each feature j
        for j in range(M):
            # if the feature is merged with other feature
            if self.cor[j] != j:
                continue
            
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

            # for each pose k = tau(j)
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
        M = self.M

        self.S = np.linalg.inv(self.O_red)  # Sigma_0:t
        self.x = np.matmul(self.S, self.xi_red) # mu_0:t

        # self.x = self.x.reshape([-1, DIM_POSE])
        self.x = self.x[:n*DIM_POSE].reshape([-1, DIM_POSE])
        
        # for each feature j
        for j in range(M):
            # if the feature is merged with other feature
            if self.cor[j] != j:
                continue

            idx_j = n*DIM_POSE + j*DIM_MEAS

            O_jj = self.O[idx_j:idx_j+DIM_MEAS, idx_j:idx_j+DIM_MEAS]
            xi_j = self.xi[idx_j:idx_j+DIM_MEAS]

            invO_jj = np.linalg.inv(O_jj)

            n_tau = len(self.tau[j])
            O_jk = np.zeros([DIM_MEAS, n_tau*DIM_POSE], dtype=np.float32)
            x_k  = np.zeros([n_tau*DIM_POSE, 1], dtype=np.float32)

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
            # mu_j
            self.m[j,0] = mj[0,0]
            self.m[j,1] = mj[1,0]

    def correspondence_test(self, j, k):
        """ This method test for correspondences (Table 11.8 on page 364).
            This method returns True of correspondence is changed. Otherwise False is returned.
        """

        # Applying local submap concept.
        xj = self.m[j,0]
        yj = self.m[j,1]
        
        xk = self.m[k,0]
        yk = self.m[k,1]

        dx = xj - xk
        dy = yj - yk
        r2 = dx**2 + dy**2

        if r2 > r_assoc**2:
            return 0.0
        
        # mu_j = O^-1_j,j * (xi_j - O_j,tau(j)*~mu_tau(j))
        # mu_[j,k] = O^-1_jk,jk * (xi_jk - O_jk,tau(j,k)*mu_tau(j,k))

        n = len(self.x) # the number of poses
        tau_j = self.tau[j]                

        # implementation of Table 11.8 on page 364.
        tau_k = self.tau[k]
        # tau_jk = tau_j + tau_k
        tau_jk = tau_j.copy()
        for tk in tau_k:
            if tk not in tau_jk:
                tau_jk.append(tk)
            else:
                print('{} duplicated'.format(tk))

        # line 2
        idx_j = n*DIM_POSE + j*DIM_MEAS
        idx_k = n*DIM_POSE + k*DIM_MEAS
        O_jj = self.O[idx_j:idx_j+DIM_MEAS, idx_j:idx_j+DIM_MEAS]
        O_kk = self.O[idx_k:idx_k+DIM_MEAS, idx_k:idx_k+DIM_MEAS]
        O_jk = self.O[idx_j:idx_j+DIM_MEAS, idx_k:idx_k+DIM_MEAS]
        O_kj = self.O[idx_k:idx_k+DIM_MEAS, idx_j:idx_j+DIM_MEAS]

        O_jk_jk = np.zeros([DIM_MEAS*2, DIM_MEAS*2], dtype=np.float32)
        
        O_jk_jk[0:DIM_MEAS, 0:DIM_MEAS] = O_jj
        O_jk_jk[0:DIM_MEAS, DIM_MEAS:] = O_jk
        O_jk_jk[DIM_MEAS:, 0:DIM_MEAS] = O_kj   # O_jk = O_kj.T
        O_jk_jk[DIM_MEAS:, DIM_MEAS:] = O_kk

        # S_tjk_tjk: Sigma_tau(j,k),tau(j,k)
        dim_tau = len(tau_jk)
        S_tjk_tjk = np.zeros([dim_tau*DIM_POSE, dim_tau*DIM_POSE], dtype=np.float32)

        for idx_row, t1 in enumerate(tau_jk):
            for idx_col, t2 in enumerate(tau_jk):
                idx_t1 = t1*DIM_POSE
                idx_t2 = t2*DIM_POSE
                
                # S: Sigma_0:t
                S_12 = self.S[idx_t1:idx_t1+DIM_POSE, idx_t2:idx_t2+DIM_POSE]
                S_tjk_tjk[idx_row*DIM_POSE:(idx_row+1)*DIM_POSE, idx_col*DIM_POSE:(idx_col+1)*DIM_POSE] = S_12

        # O_tjk_jk: Omega_tau(j,k),jk
        O_tjk_jk = np.zeros([dim_tau*DIM_POSE, DIM_MEAS*2], dtype=np.float32)
        O_jk_tjk = np.zeros([DIM_MEAS*2, dim_tau*DIM_POSE], dtype=np.float32)
        
        for idx_row, t1 in enumerate(tau_jk):
            idx_t1 = t1*DIM_POSE
            O_tjk_jk[idx_row*DIM_POSE:(idx_row+1)*DIM_POSE, 0:DIM_MEAS] = self.O[idx_t1:idx_t1+DIM_POSE, idx_j:idx_j+DIM_MEAS]
            O_tjk_jk[idx_row*DIM_POSE:(idx_row+1)*DIM_POSE,  DIM_MEAS:] = self.O[idx_t1:idx_t1+DIM_POSE, idx_k:idx_k+DIM_MEAS]

            O_jk_tjk[0:DIM_MEAS,idx_row*DIM_POSE:(idx_row+1)*DIM_POSE] = self.O[idx_j:idx_j+DIM_MEAS, idx_t1:idx_t1+DIM_POSE]
            O_jk_tjk[DIM_MEAS: ,idx_row*DIM_POSE:(idx_row+1)*DIM_POSE] = self.O[idx_k:idx_k+DIM_MEAS, idx_t1:idx_t1+DIM_POSE]

        # O_jk: O_[j,k]
        O_jk = O_jk_jk - np.matmul(np.matmul(O_jk_tjk, S_tjk_tjk), O_tjk_jk)

        # print(O_jk)

        # line 3
        # mu_jk: mu_j,k
        mu_jk = np.zeros([2*DIM_MEAS, 1], dtype=np.float32)
        mu_jk[:DIM_MEAS] = self.m[j].reshape([-1,1])
        mu_jk[DIM_MEAS:] = self.m[k].reshape([-1,1])
        
        # xi_[j,k] = O_[j,k] mu_[j,k]
        # mu_[j,k] = O^-1_jk,jk * (xi_jk - O_jk,tau(j,k)mu_tau(j,k))
        # xi_jk: 4x1 -> mu_[j,k]: 4x1
        # O^-1_jk,jk: 4x4
        # xi_[j,k]: 4x1
        # xi_jk: xi_[j,k], 4x1
        xi_jk = np.matmul(O_jk, mu_jk)

        # print(mu_jk)

        # line 4
        # O_djk: Omega_deltaj,k
        I = np.array([[ 1,  0],
                      [ 0,  1],
                      [-1,  0],
                      [ 0, -1]], dtype=np.float32)
        O_djk = np.matmul(np.matmul(I.T, O_jk), I)

        # line 5
        # xi_djk: xi_deltaj,k, 2x1
        xi_djk = np.matmul(I.T, xi_jk)

        # line 6
        invO_djk = np.linalg.inv(O_djk) # = Sigma_djk
        mu_djk = np.matmul(invO_djk, xi_djk)
        
        # line 7
        det = np.linalg.det(invO_djk)   # = det(Sigma_jk)
        det = max(det, 1e-6)
        eta = 1 / (2*np.pi*np.sqrt(det))

        x = np.matmul(np.matmul(mu_djk.T, O_djk), mu_djk)
        if x >= 0.0:
            p = eta * np.exp(-0.5 * x)
            p = p[0,0]
        else:
            p = 1.0

        return p

    def graph_slam(self):
        """ Algorithm GraphSLAM (Table 11.9 on p.365)
        """
        self.initialize()

        self.plot()

        self.linearize()
        self.reduce()
        self.solve()
        
        self.plot()

        # repeat (line 7)
        pair_found = True
        while pair_found:
            pair_found = False

            # for each pair of non-corresponding features mj, mk (line 8)
            M = self.M

            for j in range(M):
                if self.cor[j] == j:

                    for k in range(M):
                        if j != k and self.cor[k] == k:

                            p = self.correspondence_test(j, k)  # (line 9)
                            if p_min_assoc < p:
                                pair_found = True

                                print('pair {}, {}, p:{}'.format(j, k, p))
                                print('  ', self.m[j].T)
                                print('  ', self.m[k].T)

                                # for all ci=k, set ci=j (line 11)
                                for idx_c in range(len(self.cor)):
                                    if self.cor[idx_c] == k:
                                        self.cor[idx_c] = j

                                self.tau[j] += self.tau[k]
                                self.tau[k] = []

            self.linearize()
            self.reduce()
            self.solve()

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

    def plot(self):
        fig = plt.figure('map')

        plt.scatter(landmarks[:,0], landmarks[:,1], c='k')
        
        plt.plot(self.x_gt[:,0], self.x_gt[:,1], c='g')
        plt.plot(self.x[:,0], self.x[:,1], c='r')

        m_landmarks = []
        for idx_m in range(len(self.m)):
            if self.cor[idx_m] == idx_m:
                m = self.m[idx_m]
                # plt.text(m[0], m[1], '{:d}'.format(idx_m))
                m_landmarks.append(m)
        m_landmarks = np.array(m_landmarks)

        plt.scatter(m_landmarks[:,0], m_landmarks[:,1], c='r')

        print('# measurements:', len(self.m))
        print('# landmarks:', len(m_landmarks))

        self.draw_covariance(j=0)
        self.draw_covariance(j=1)
        self.draw_covariance(j=2)


        plt.axis('equal')

        plt.draw()
        plt.waitforbuttonpress(0)
        plt.close(fig)

if __name__ == '__main__':

    slam = GraphSLAM()

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

    slam.graph_slam()


