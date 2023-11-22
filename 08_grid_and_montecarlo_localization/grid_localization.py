""" Grid Localization
    Table 8.1, p 188.
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from robot import Robot

res_x = 0.15    # 0.15m
res_y = 0.15    # 0.15m
res_theta = 5   # degree

max_x = 15
max_y = 15

w_map = int(max_x / res_x)
h_map = int(max_y / res_y)

print("map size: {}, {}".format(w_map, h_map))

grid_map = cv2.imread('./grid_map0.png')

map = grid_map[:,:,0]
pdf = np.zeros([h_map, w_map])

robot = Robot()

for t in range(30):

    vt = 1
    wt = 0
    ut = [vt, wt]
    robot.predict(ut)
    robot.update()
    
    robot.measure(map)

    fig = plt.figure()
    plt.imshow(grid_map)
    robot.plot()

    plt.gca().invert_yaxis()

    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)
