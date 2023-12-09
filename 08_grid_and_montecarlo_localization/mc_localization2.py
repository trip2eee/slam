"""Augmented Monte Carlo Localization
   Table 8.3 on page 258.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from range_sensor import RangeSensor

res_x = 0.15    # 0.15m
res_y = 0.15    # 0.15m

max_x = 15
max_y = 15
max_sensor_range = 5


# alpha 1 ~ alpha 6
alpha = [4, 2, 4, 4, 4, 2]

def sample_normal_distribution(b):
   """ Table 5.4 Algorithm for sampling from normal distribution with zero mean and variance b.
   """
   r = np.random.randint(-1000, 1000, size=12) * 0.001
   return b/6.0*r.sum()

def sample_triangular_distribution(b):
   """ Table 5.4 Algorithm for sampling from triangular distribution with zero mean and variance b.
   """
   r1 = np.random.randint(-1000, 1000) * 0.001
   r2 = np.random.randint(-1000, 1000) * 0.001
   return b * r1 * r2

sample = sample_normal_distribution

class MCLocalization:
   def __init__(self, x_gt, map):
      self.x_gt = x_gt

      self.M = 500 # The number of particles
      self.X = np.zeros([self.M, 3])  # particles
      self.w = np.ones(self.M) / self.M # Weight (importance)
      self.w_avg = 0
      self.w_slow = 0
      self.w_fast = 0
      self.a_slow = 0.001
      self.a_fast = 0.9

      self.map = map
      self.obstacles = []
   
      self.sensor_angles = [
         0, 15, 30, 45, 60, 75, 90, #105, 120, 135, 150, 
         360-15, 360-30, 360-45, 360-60, 360-75, 360-90, #360-105, 360-120, 360-135, 360-150,
      ]
      num_angles = len(self.sensor_angles)
      self.sensor_ranges = [0] * num_angles
      self.true_ranges = [0] * num_angles

      self.make_obstacle_lut(map)

      self.range_sensor = RangeSensor()
      self.range_sensor.z_max = max_sensor_range

   def reset_obstacle_lut(self):
        self.obstacles = []

   def make_obstacle_lut(self, map):
      if len(self.obstacles) == 0:
         for i in range(map.shape[0]):
            for j in range(map.shape[1]):
               if map[i,j] == 0:
                  # add (x, y) of obstacles
                  self.obstacles.append([j, i])
         self.obstacles = np.array(self.obstacles, dtype=np.float32)
         if len(self.obstacles) > 0:
            self.obstacles[:,0] *= res_x
            self.obstacles[:,1] *= res_y


   def sample_motion_model_velocity(self, xt_1, ut, noise=False, dt=0.1):
      """ This method computes motion model in Table 8.1 on p.188.
      """

      # control ut
      vt, wt = ut

      # compute ground truth
      x = xt_1[0]
      y = xt_1[1]
      theta = xt_1[2]

      xt = np.zeros([3])

      if noise:
         v_hat = vt + sample(alpha[0]*abs(vt) + alpha[1]*abs(wt))   # line 1
         w_hat = wt + sample(alpha[2]*abs(vt) + alpha[3]*abs(wt))   # line 2
         g_hat =      sample(alpha[4]*abs(vt) + alpha[5]*abs(wt))   # line 3
      else:
         v_hat = vt
         w_hat = wt
         g_hat = 0
          
      if abs(w_hat) > 0:
         xt[0] = x - v_hat/w_hat*np.sin(theta) + v_hat/w_hat*np.sin(theta + w_hat*dt)
         xt[1] = y + v_hat/w_hat*np.cos(theta) - v_hat/w_hat*np.cos(theta + w_hat*dt)
         xt[2] = theta + w_hat*dt + g_hat*dt
      else:
         xt[0] = x + v_hat*np.cos(theta)*dt
         xt[1] = y + v_hat*np.sin(theta)*dt
         xt[2] = theta + g_hat*dt

      u_map = int(xt[0]/res_x + 0.5)
      v_map = int(xt[1]/res_y + 0.5)

      if 0 <= u_map < self.map.shape[1] and 0 <= v_map < self.map.shape[0] and self.map[v_map, u_map] == 255:
          p = 1.0
      else:
         p = 0.0
 
      return xt, p

   def measure(self, xt, noise=False):        
      x     = xt[0]
      y     = xt[1]
      theta = xt[2]
      N = len(self.sensor_angles)
      sensor_ranges = [max_sensor_range] * N

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

         angle_thres = 5*np.pi/180.0

         for idx_angle in range(N):
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
            if noise:
               r += np.random.randn()*0.15
            
            sensor_ranges[idx_angle] = r
      else:
        for idx_angle in range(N):
            if noise:
              r = min(max_sensor_range, max_sensor_range+np.random.randn()*0.15)
            sensor_ranges[idx_angle] = r
      
      return sensor_ranges
   
   def initialize(self, x0=None):

      if x0 is None:
         for m in range(self.M):
            # generate random samples
            x = np.random.rand() * max_x
            y = np.random.rand() * max_y

            u_map = int(x/res_x + 0.5)
            v_map = int(y/res_y + 0.5)

            while u_map >= self.map.shape[1] or v_map >= self.map.shape[0] or self.map[v_map, u_map] < 255:
               x = np.random.rand() * max_x
               y = np.random.rand() * max_y

               u_map = int(x/res_x + 0.5)
               v_map = int(y/res_y + 0.5)

            # a = np.random.randint(0, 2) * 180 * np.pi / 180.0
            a = np.random.rand()*np.pi*2.0
            self.X[m,0] = x
            self.X[m,1] = y
            self.X[m,2] = a
      else:
         for m in range(self.M):
            # generate random samples
            x = np.random.randn() * max_x * 0.1 + x0[0]
            y = np.random.randn() * max_y * 0.1 + x0[1]

            u_map = int(x/res_x + 0.5)
            v_map = int(y/res_y + 0.5)

            while True:
               if 0 <= u_map < self.map.shape[1] and 0 <= v_map < self.map.shape[0] and self.map[v_map, u_map] == 255:
                  break
               else:
                  x = np.random.randn() * max_x * 0.1 + x0[0]
                  y = np.random.randn() * max_y * 0.1 + x0[1]

                  u_map = int(x/res_x + 0.5)
                  v_map = int(y/res_y + 0.5)

            # a = np.random.randint(0, 2) * 180 * np.pi / 180.0
            a = np.random.randn()*np.pi + x0[2]
            self.X[m,0] = x
            self.X[m,1] = y
            self.X[m,2] = a

   def sample(self, ut, dt=0.1):
      self.x_gt, p_gt = self.sample_motion_model_velocity(mcl.x_gt, ut, noise=False, dt=dt)
      self.sensor_ranges = self.measure(self.x_gt)

      self.w_avg = 0

      # for each particle
      for m in range(self.M):
         # motion model
         self.X[m], pm = self.sample_motion_model_velocity(self.X[m], ut, noise=True, dt=dt)

         # measurement model
         ranges = self.measure(self.X[m], noise=True)

         N = len(self.sensor_angles)
         q = 1.0
         for k in range(N):
            p = self.range_sensor.compute_px(ranges[k], self.sensor_ranges[k])
            q *= p

         self.w[m] = q*pm
         self.w_avg += self.w[m] / self.M

      self.w_slow = self.w_slow + self.a_slow*(self.w_avg - self.w_slow)
      self.w_fast = self.w_fast + self.a_fast*(self.w_avg - self.w_fast)

   def resample(self):
      Xt = []
      Wt = []
      M = self.M

      r = np.random.rand() * 1/M
      self.w = self.w / np.sum(self.w)

      c = self.w[0]
      i = 0

      p_random = max(0.0, 1 - self.w_fast / self.w_slow)
      print('p_random: {:.2f} {}, {}'.format(p_random, self.w_fast, self.w_slow))

      for m in range(self.M):
         p = np.random.rand()
         if p <= p_random:
            # generate random samples
            x = np.random.rand() * max_x
            y = np.random.rand() * max_y

            u_map = int(x/res_x + 0.5)
            v_map = int(y/res_y + 0.5)

            while u_map >= self.map.shape[1] or v_map >= self.map.shape[0] or self.map[v_map, u_map] < 255:
               x = np.random.rand() * max_x
               y = np.random.rand() * max_y

               u_map = int(x/res_x + 0.5)
               v_map = int(y/res_y + 0.5)

            # a = np.random.randint(0, 2) * 180 * np.pi / 180.0
            a = np.random.rand()*np.pi*2.0
            self.X[m,0] = x
            self.X[m,1] = y
            self.X[m,2] = a
            Xt.append([x,y,a])
            Wt.append(1.0/self.M)
         else:

            U = r + (m / M)   # m: 0 ~ M-1
            while U > c:
               i = i + 1
               c = c + self.w[i]
            Xt.append(self.X[i])
            Wt.append(self.w[i])

      self.X = np.array(Xt)
      self.w = np.array(Wt)

   def plot(self):
      # plot ground truth
      self.draw_robot(self.x_gt, color='g', sensor_readings=True)

      # draw particles      
      x     = self.X[:,0]
      y     = self.X[:,1]
      theta = self.X[:,2]

      plt.scatter(x/res_x, y/res_y, c='b')
      
      dx = 0.5*np.cos(theta)
      dy = 0.5*np.sin(theta)
      plt.plot([x/res_x, (x+dx)/res_x], [y/res_y, (y+dy)/res_y], c='b')
            

   def draw_robot(self, xt, color='g', sensor_readings=False):
      x     = xt[0]
      y     = xt[1]
      theta = xt[2]

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

if __name__ == '__main__':
   map = cv2.imread('grid_map.png')
   
   x0 = np.array([20*res_x, 30*res_y, 0], dtype=np.float32)
   mcl = MCLocalization(x0, map[:,:,1])

   fig = plt.figure()
   plt.imshow(map)

   mcl.initialize(x0)
   # mcl.initialize()

   mcl.plot()
   plt.gca().invert_yaxis()
   plt.draw()
   plt.waitforbuttonpress(0)
   plt.close(fig)

   for t in range(100):
      fig = plt.figure()
      plt.imshow(map)
      vt = 1.5
      wt = 0
      ut = [vt, wt]
      dt = 0.1

      mcl.sample(ut, dt)

      mcl.resample()
      
      mcl.plot()
      
      plt.gca().invert_yaxis()
      plt.draw()
      plt.waitforbuttonpress(0)
      plt.close(fig)
