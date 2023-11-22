from range_sensor import RangeSensor
import numpy as np
import matplotlib.pyplot as plt

res_x = 0.15    # 0.15m
res_y = 0.15    # 0.15m
res_theta = 5   # degree

max_sensor_range = 5
dt = 0.1

class Robot:
    def __init__(self):

        x_pose = np.array([[25*res_x, 33*res_y, 0]], dtype=np.float32).T
        P_pose = np.array([[3**2, 0, 0], 
                           [0, 3**2, 0],
                           [0, 0, 1**2]], dtype=np.float32)

        self.x = x_pose
        self.P = P_pose

        self.x_pred = x_pose.copy()
        self.P_pred = P_pose.copy()

        self.x_gt = x_pose.copy()

        self.range_sensor = RangeSensor()
        self.range_sensor.z_max = max_sensor_range
    
        self.sensor_angles = [
            0, 30, 60, 90, -30, -60, -90
            # 0, 30, 60, 90, 
        ]

        self.sensor_ranges = [
            0, 0, 0, 0, 0, 0, 0,
        ]
        self.true_ranges = [
            0, 0, 0, 0, 0, 0, 0,
        ]

        self.obstacles = []

    def predict(self, ut):
        # control ut
        vt, wt = ut

        # compute ground truth
        x_gt = self.x_gt[0,0]
        y_gt = self.x_gt[1,0]
        ha_gt = self.x_gt[2,0]

        if abs(wt) > 0:
            self.x_gt[0,0] = x_gt - vt/wt*np.sin(ha_gt) + vt/wt*np.sin(ha_gt+wt*dt)
            self.x_gt[1,0] = y_gt + vt/wt*np.cos(ha_gt) - vt/wt*np.cos(ha_gt+wt*dt)
            self.x_gt[2,0] = ha_gt + wt*dt
        else:
            self.x_gt[0,0] = x_gt + vt*np.cos(ha_gt)*dt
            self.x_gt[1,0] = y_gt + vt*np.sin(ha_gt)*dt
            self.x_gt[2,0] = ha_gt


        # motion update (prediction)
        vt += np.random.randn()*0.1
        wt += np.random.randn()*0.1

        x  = self.x[0,0]
        y  = self.x[1,0]
        ha = self.x[2,0]   # heading angle        
        
        if abs(wt) > 0:
            xp = x - vt/wt*np.sin(ha) + vt/wt*np.sin(ha+wt*dt)
            yp = y + vt/wt*np.cos(ha) - vt/wt*np.cos(ha+wt*dt)
            hap = ha + wt*dt
        else:
            xp = x + vt*np.cos(ha)*dt
            yp = y + vt*np.sin(ha)*dt
            hap = ha + wt*dt

        x_pred = np.array([[xp], [yp], [hap]])

        if abs(wt) > 0:
            Gt = np.array([[1, 0, vt/wt*np.cos(ha) - vt/wt*np.cos(ha+wt*dt)],
                           [0, 1, vt/wt*np.sin(ha) - vt/wt*np.sin(ha+wt*dt)],
                           [0, 0, 1]])
        else:
            Gt = np.array([[1, 0, -vt*np.sin(ha)],
                           [0, 1, vt*np.cos(ha)],
                           [0, 0, 1]])


        # process noise covariance
        sx  = 0.1
        sy  = 0.1
        sha = 0.02
        Rt = np.array([[sx**2, 0,     0],
                       [0,     sy**2, 0],
                       [0,     0,     sha**2]])
        
        P_pose = self.P
        P_pred = np.matmul(np.matmul(Gt, P_pose), Gt.T) + Rt

        self.x_pred = x_pred
        self.P_pred = P_pred
    
    def measure(self, map):
        x     = self.x[0,0]
        y     = self.x[1,0]
        theta = self.x[2,0]

        if len(self.obstacles) == 0:
            for i in range(map.shape[0]):
                for j in range(map.shape[1]):
                    if map[i,j] < 1:
                        # add (x, y) of obstacles
                        self.obstacles.append([j, i])

        angle_thres = 5*np.pi/180.0
        for idx_angle in range(len(self.sensor_angles)):
            angle = self.sensor_angles[idx_angle]
            beam_angle = angle*np.pi/180.0 + theta

            min_dist2 = max_sensor_range**2
            for pos in self.obstacles:
                mx, my = pos
                mx *= res_x
                my *= res_y

                dx = mx - x
                dy = my - y                
                meas_angle = np.arctan2(dy, dx)

                if abs(beam_angle - meas_angle) <= angle_thres:
                    r2 = dx**2 + dy**2
                    if min_dist2 > r2:
                        min_dist2 = r2
            
            r = np.sqrt(min_dist2)
            self.true_ranges[idx_angle] = r
            
            self.range_sensor.compute_pdf(r)
            self.sensor_ranges[idx_angle] = self.range_sensor.measure()

    def update(self):
        # measurement update
        x_pred = self.x_pred
        P_pred = self.P_pred        

        sum_dx = np.zeros([3,1], dtype=np.float32)
        sum_dp = np.zeros([3,3], dtype=np.float32)

        # for all landmarks k in the map m
        # N = len(landmarks)
        # for k in range(N):
        #     mx, my, ms = landmarks[k]
        #     dx = mx - x_pred[0,0]
        #     dy = my - x_pred[1,0]
        #     q = dx**2 + dy**2

        #     sr = np.sqrt(q)*0.1
        #     sf = 0.1
        #     ss = 1.0

        #     Qt = np.array([[sr**2, 0,     0],
        #                    [0,     sf**2, 0],
        #                    [0,     0,     ss**2]])

        #     # predict feature i
        #     z_pred = np.array([
        #         [np.sqrt(q)],
        #         [np.arctan2(dy, dx) - x_pred[2,0]],
        #         [ms],
        #     ])

        #     Ht = 1/q*np.array([
        #         [np.sqrt(q)*dx, -np.sqrt(q)*dy,  0],
        #         [           dy,             dx, -1],
        #         [            0,              0,  0],
        #     ])

        #     # HPH^T + Q
        #     St = np.matmul(np.matmul(Ht, P_pred), Ht.T) + Qt
        #     invSt = np.linalg.inv(St)

        #     # for all observed features i
        #     ck = 0
        #     zk = None
        #     min_d2 = 1e10
        #     for i in range(N):
        #         zi = self.measure(landmarks[i])
        #         r = zi - z_pred

        #         d2 = np.matmul(np.matmul(r.T, invSt), r)
        #         if d2 < min_d2:
        #             min_d2 = d2
        #             ck = i
        #             zk = zi

        #     assert(k == ck)

        #     # PH^T*S^-1
        #     Kt = np.matmul(np.matmul(P_pred, Ht.T), invSt)
        #     KHt = np.matmul(Kt, Ht)

        #     # weight            
        #     w = 1.0/N
        #     sum_dx += w*np.matmul(Kt, (zk-z_pred))
        #     # the diagonal term of KHt shall be less than 1 to make covariances positive.
        #     sum_dp += w*KHt

        # self.x = x_pred + sum_dx
        # I = np.eye(3)
        # self.P = np.matmul(I - sum_dp, P_pred)
        
        self.x = self.x_pred
        self.P = self.P_pred

    def plot(self):
        x     = self.x_gt[0,0]
        y     = self.x_gt[1,0]
        theta = self.x_gt[2,0]

        # current ground truth
        plt.scatter(x/res_x, y/res_y, c='g')
        dx = 1*np.cos(theta)
        dy = 1*np.sin(theta)
        plt.plot([x/res_x, (x+dx)/res_x], [y/res_y, (y+dy)/res_y], c='g')

        
        x     = self.x[0,0]
        y     = self.x[1,0]
        theta = self.x[2,0]

        # current position
        plt.scatter(x/res_x, y/res_y, c='b')

        # sensor readings
        for idx_angle in range(len(self.sensor_angles)):
            angle = self.sensor_angles[idx_angle]
            beam_angle = angle*np.pi/180.0 + theta

            sensor_range = self.sensor_ranges[idx_angle]

            rx = sensor_range*np.cos(beam_angle)
            ry = sensor_range*np.sin(beam_angle)

            plt.plot([x/res_x, (x+rx)/res_x], [y/res_y, (y+ry)/res_y], c='r')




            





