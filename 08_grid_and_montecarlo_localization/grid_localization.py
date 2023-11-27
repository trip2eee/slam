""" Grid Localization
    Table 8.1, p 188.
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from robot import Robot

res_x = 0.15    # 0.15m
res_y = 0.15    # 0.15m

max_x = 15
max_y = 15

w_map = int(max_x / res_x)
h_map = int(max_y / res_y)

print("map size: {}, {}".format(w_map, h_map))

grid_map = cv2.imread('./grid_map0.png')

map = grid_map[:,:,0]
pdf = np.zeros([h_map, w_map])

robot = Robot(map)

for t in range(30):

    vt = 1.5
    wt = 0
    ut = [vt, wt]
    robot.predict(ut)
    robot.update()
    
    # robot.measure(robot.x_gt, map)

    p_xy = robot.pk.sum(axis=2)
    # p_xy = robot.pk_pred.sum(axis=2)
    

    grid_map_disp = grid_map.copy().astype(np.int32)

    
    grid_map_disp[:,:,0] = grid_map[:,:,0] * 0.2 + p_xy*(255*0.8) #(p_xy * 255).astype(np.uint8)
    grid_map_disp[:,:,1] = grid_map[:,:,1] * 0.2
    grid_map_disp[:,:,2] = grid_map[:,:,2] * 0.2

    fig = plt.figure()
    plt.imshow(grid_map_disp)
    # robot.plot(sensor_readings=False)

    plt.gca().invert_yaxis()

    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)
