"""Robot for Monte Carlo Localization
"""
from range_sensor import RangeSensor
import numpy as np
import matplotlib.pyplot as plt
import time

res_x = 0.15    # 0.15m
res_y = 0.15    # 0.15m
res_theta = 90   # degree

max_x = 15
max_y = 15

w_map = int(max_x / res_x)
h_map = int(max_y / res_y)

max_sensor_range = 5
dt = 0.1
alpha = [0.2, 0.1, 0.2, 0.2, 0.1, 0.1]

pixel_obstacle = 200

class Robot:
    def __init__(self, map, x_pose):

        self.x = x_pose
        self.x_gt = x_pose.copy()

        self.range_sensor = RangeSensor()
        self.range_sensor.z_max = max_sensor_range
    
        self.sensor_angles = [
            0, 15, 30, 45, 60, 75, 90, 360-15, 360-30, 360-45, 360-60, 360-75, 360-90
        ]
        num_angles = len(self.sensor_angles)
        self.sensor_ranges = [0] * num_angles
        self.true_ranges = [0] * num_angles

        self.obstacles = []

        self.map = map

        h_map, w_map = self.map.shape
        dir_map = 360//res_theta

        self.dir_map = dir_map
        self.h_map = h_map
        self.w_map = w_map
    
    def prob_normal_distribution(self, a, b):
        """ Table 5.2 zero-centered normal distribution with variance b
        """
        return 1/np.sqrt(2*np.pi*b) * np.exp(-1/2*(a**2)/b)

    def prob_triangular_distribution(self, a, b):
        """ Table 5.2 triangular distribution with variance b
        """
        
        prob = (np.sqrt(6*b) - abs(a)) / (6*b)
        prob *= abs(a) <= np.sqrt(6*b)
        return prob


    def motion_model_velocity(self, xt, ut, xt_1, dt=1/10):
        """ Table 5.1 Algorith motion_model_velocity
            Algorithm for computing p(xt | ut, xt-1) based on velocity information.
            xt   : pose at t [x1, y1, ha1]
            ut   : [v, w]^T
            xt_1 : pose at t-1 [x0, y0, ha0]
        """
        x1, y1, ha1 = xt        # x', y', theta'
        # x0, y0, ha0 = xt_1      

        # x, y, theta
        x0 = xt_1[:,0]
        y0 = xt_1[:,1]
        ha0 = xt_1[:,2]

        cos_ha0 = np.cos(ha0)
        sin_ha0 = np.sin(ha0)

        t1 = (x0-x1)*cos_ha0 + (y0-y1)*sin_ha0
        t2 = (y0-y1)*cos_ha0 - (x0-x1)*sin_ha0
        # prevent division by zero
        idx_zero = np.where(abs(t2) < 1e-6)
        t2[idx_zero] = 1e-6
        
        mu = 1.0/2.0*t1/t2                  # line 2

        mx = (x0 + x1) / 2
        my = (y0 + y1) / 2

        # center of circle (cx, cy)
        cx = mx + mu*(y0-y1)            # line 3 x*
        cy = my + mu*(x1-x0)            # line 4 y*
        cr = np.sqrt((x0 - cx)**2 + (y0 - cy)**2)   # line 5 r*
        delta_ha = np.arctan2(y1-cy, x1-cx) - np.arctan2(y0-cy, x0-cx)  # line 6
        v_hat = delta_ha / dt * cr                  # line 7
        w_hat = delta_ha / dt                       # line 8
        gamma_hat = (ha1 - ha0) / dt - w_hat        # line 9. final rotation

        v, w = ut
        
        prob = self.prob_normal_distribution

        pv = prob(v-v_hat, alpha[0]*abs(v) + alpha[1]*abs(w))
        pw = prob(w-w_hat, alpha[2]*abs(v) + alpha[3]*abs(w))
        pg = prob(gamma_hat, alpha[4]*abs(v) + alpha[5]*abs(w))

        return (pv * pw * pg)


    def predict(self, ut):
        """ This method computes motion model in Table 8.1 on p.188.
        """

        # control ut
        vt, wt = ut

        # compute ground truth
        x = self.x[0,0]
        y = self.x[1,0]
        ha = self.x[2,0]

        if abs(wt) > 0:
            self.x[0,0] = x - vt/wt*np.sin(ha) + vt/wt*np.sin(ha+wt*dt)
            self.x[1,0] = y + vt/wt*np.cos(ha) - vt/wt*np.cos(ha+wt*dt)
            self.x[2,0] = ha + wt*dt
        else:
            self.x[0,0] = x + vt*np.cos(ha)*dt
            self.x[1,0] = y + vt*np.sin(ha)*dt
            self.x[2,0] = ha

    def reset_obstacle_lut(self):
        self.obstacles = []

    def make_obstacle_lut(self, map):
        if len(self.obstacles) == 0:
            for i in range(map.shape[0]):
                for j in range(map.shape[1]):
                    if map[i,j] < pixel_obstacle:
                        # add (x, y) of obstacles
                        self.obstacles.append([j, i])
            self.obstacles = np.array(self.obstacles, dtype=np.float32)
            if len(self.obstacles) > 0:
                self.obstacles[:,0] *= res_x
                self.obstacles[:,1] *= res_y

    
    def measure(self, xt, map, noise=True):
        x     = xt[0,0]
        y     = xt[1,0]
        theta = xt[2,0]
        self.make_obstacle_lut(map)

        z_true = self.predict_measure(xt, map)
        for idx_angle in range(len(self.sensor_angles)):

            if noise:
                self.range_sensor.compute_pdf(z_true[idx_angle])
                # draw a noisy measure
                self.sensor_ranges[idx_angle] = self.range_sensor.measure()
            else:
                self.sensor_ranges[idx_angle] = z_true[idx_angle]

        return self.sensor_ranges

    def predict_measure(self, xt, map):
        x     = xt[0,0]
        y     = xt[1,0]
        theta = xt[2,0]
        self.make_obstacle_lut(map)

        if len(self.obstacles) > 0:
            dx = self.obstacles[:,0] - x
            dy = self.obstacles[:,1] - y
            r2 = dx**2 + dy**2
            idx_valid = np.where(r2 <= max_sensor_range**2)

            dx = dx[idx_valid]
            dy = dy[idx_valid]

            # range of arctan2: [-pi, pi]
            angle_valid = np.arctan2(dy, dx)
            idx_neg = np.where(angle_valid < 0.0)
            angle_valid[idx_neg] += np.pi*2.0

            # angle_valid = np.arctan2(dy, dx) - theta

            angle_thres = 5*np.pi/180.0

            for idx_angle in range(len(self.sensor_angles)):
                angle = self.sensor_angles[idx_angle]
                beam_angle = angle*np.pi/180.0 + theta
                # beam_angle = angle*np.pi/180.0
                                
                min_dist2 = max_sensor_range**2

                diff_angles = abs(beam_angle - angle_valid)
                idx_valid = np.where((diff_angles <= angle_thres) | ((diff_angles >= np.pi*2-angle_thres) & (diff_angles <= np.pi*2+angle_thres)))

                dx_beam = dx[idx_valid]
                dy_beam = dy[idx_valid]

                r2 = dx_beam**2 + dy_beam**2
                if len(r2) > 0:
                    min_dist2 = r2.min()
                else:
                    min_dist2 = max_sensor_range**2
                
                r = np.sqrt(min_dist2)
                self.true_ranges[idx_angle] = r
        else:
            for idx_angle in range(len(self.sensor_angles)):
                self.true_ranges[idx_angle] = max_sensor_range
        
        return self.true_ranges

    def update(self):
        # measurement update
        z_meas = self.measure(self.x, self.map)

        xt = np.array([[j*res_x, i*res_y, d*res_theta*np.pi/180.0]]).T
        # xt = np.array([[j*res_x, i*res_y, 2*res_theta*np.pi/180.0]]).T

        z_pred = self.predict_measure(xt, self.map)

        N = len(self.sensor_angles)
        q = 1.0
        for k in range(N):
            p = self.range_sensor.compute_px(z_meas[k], z_pred[k])
            q *= p

        return q
    
    def plot(self, color='g', sensor_readings=True):
        x     = self.x[0,0]
        y     = self.x[1,0]
        theta = self.x[2,0]

        plt.scatter(x/res_x, y/res_y, c=color)
        dx = 1*np.cos(theta)
        dy = 1*np.sin(theta)
        plt.plot([x/res_x, (x+dx)/res_x], [y/res_y, (y+dy)/res_y], c=color)
        
        # sensor readings
        if sensor_readings:
            for idx_angle in range(len(self.sensor_angles)):
                angle = self.sensor_angles[idx_angle]
                beam_angle = angle*np.pi/180.0 + theta

                sensor_range = self.sensor_ranges[idx_angle]

                rx = sensor_range*np.cos(beam_angle)
                ry = sensor_range*np.sin(beam_angle)

                plt.plot([x/res_x, (x+rx)/res_x], [y/res_y, (y+ry)/res_y], c='r')




            





