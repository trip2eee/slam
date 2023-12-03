"""Monte Carlo Localization
   Table 8.2 on page 200.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from robot_mc import Robot
from range_sensor import RangeSensor

res_x = 0.15    # 0.15m
res_y = 0.15    # 0.15m

max_x = 15
max_y = 15

class MCLocalization:
   def __init__(self):
      self.M = 50 # The number of particles
      self.X = []  # particles
      self.w = np.ones(self.M) / self.M # Weight (importance)

   def initialize(self):
      self.X = []
      for m in range(self.M):
         # generate random samples
         x = np.random.rand() * max_x
         y = np.random.rand() * max_y
         a = np.random.rand() * 2*np.pi
         xm = np.array([[x, y, a]]).T
         self.X.append(xm)

   def sample_motion_model(self):
      Xt = []

if __name__ == '__main__':
   map = cv2.imread('grid_map.png')
   
   mcl = MCLocalization()

   fig = plt.figure()
   plt.imshow(map)

   x0 = np.array([[20*res_x, 30*res_y, 0]], dtype=np.float32).T
   robot = Robot(map[:,:,1], x0)
   mcl.initialize()

   vt = 1.5
   wt = 0
   ut = [vt, wt]

   robot.predict(ut)
   robot.plot()
   for m in range(mcl.M):
      particle = Robot(map[:,:,1], mcl.X[m])
      particle.predict(ut)
      particle.plot(color='b')

      particle.update()
      
   # mcl.sample_motion_model()
   

   plt.gca().invert_yaxis()
   plt.draw()
   plt.waitforbuttonpress(0)

