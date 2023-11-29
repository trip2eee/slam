""" Range sensing test code
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from robot import Robot

map = np.ones([100, 100], dtype=np.uint8) * 255

res_x = 0.15    # 0.15m
res_y = 0.15    # 0.15m

# sensor beam angles
# index: 0  1   2   3   4   5   6    7    8    9    10   11   12
# angle: 0, 15, 30, 45, 60, 75, 90, -15, -30, -45, -60, -75, -90

x_pose = np.array([[50*res_x, 50*res_y, 0]], dtype=np.float32).T
robot = Robot(map, x_pose)


def check_beam(id):
    if robot.sensor_ranges[id] >= 4:
        print('Beam {}: error'.format(id))
    else:
        print('Beam {}: OK'.format(id))

print('angle:', robot.x[2,0]*180/np.pi)

map[50, 60] = 0
robot.measure(x_pose, map, noise=False)
check_beam(0)

map[40, 60] = 0
robot.reset_obstacle_lut()
robot.measure(x_pose, map, noise=False)
check_beam(9)


map[60, 60] = 0
robot.reset_obstacle_lut()
robot.measure(x_pose, map, noise=False)
check_beam(3)


map[60, 50] = 0
robot.reset_obstacle_lut()
robot.measure(x_pose, map, noise=False)
check_beam(6)

map[40, 50] = 0
robot.reset_obstacle_lut()
robot.measure(x_pose, map, noise=False)
check_beam(12)

plt.figure('map')
plt.imshow(map)
robot.plot(sensor_readings=True)
plt.gca().invert_yaxis()
plt.show()

# rotate robot 90 deg
x_pose = np.array([[50*res_x, 50*res_y, np.pi/2]], dtype=np.float32).T
robot = Robot(map, x_pose)
print('angle:', robot.x[2,0]*180/np.pi)

robot.reset_obstacle_lut()
robot.measure(x_pose, map, noise=False)

# map = np.ones([100, 100], dtype=np.uint8) * 255

map[50, 60] = 0
robot.reset_obstacle_lut()
robot.measure(x_pose, map, noise=False)
check_beam(12)

check_beam(0)
check_beam(9)
check_beam(12)

map[49, 30] = 0
robot.reset_obstacle_lut()
robot.measure(x_pose, map, noise=False)
check_beam(6)

plt.figure('map')
plt.imshow(map)
robot.plot(sensor_readings=True)
plt.gca().invert_yaxis()
plt.show()


# rotate robot 180 deg
map = np.ones([100, 100], dtype=np.uint8) * 255
x_pose = np.array([[50*res_x, 50*res_y, np.pi]], dtype=np.float32).T
robot = Robot(map, x_pose)
print('angle:', robot.x[2,0]*180/np.pi)

map[40, 40] = 0
robot.reset_obstacle_lut()
robot.measure(x_pose, map, noise=False)
check_beam(3)

map[50, 40] = 0
robot.reset_obstacle_lut()
robot.measure(x_pose, map, noise=False)
check_beam(0)

map[60, 40] = 0
robot.reset_obstacle_lut()
robot.measure(x_pose, map, noise=False)
check_beam(9)

map[40, 50] = 0
robot.reset_obstacle_lut()
robot.measure(x_pose, map, noise=False)
check_beam(0)

map[60, 50] = 0
robot.reset_obstacle_lut()
robot.measure(x_pose, map, noise=False)
check_beam(12)

plt.figure('map')
plt.imshow(map)
robot.plot(sensor_readings=True)
plt.gca().invert_yaxis()
plt.show()

# rotate robot 270 deg
map = np.ones([100, 100], dtype=np.uint8) * 255
x_pose = np.array([[50*res_x, 50*res_y, np.pi*3/2]], dtype=np.float32).T
robot = Robot(map, x_pose)
print('angle:', robot.x[2,0]*180/np.pi)

robot.reset_obstacle_lut()
robot.measure(x_pose, map, noise=False)

map[40, 50] = 0
robot.reset_obstacle_lut()
robot.measure(x_pose, map, noise=False)
check_beam(0)

map[50, 60] = 0
robot.reset_obstacle_lut()
robot.measure(x_pose, map, noise=False)
check_beam(0)

map[50, 40] = 0
robot.reset_obstacle_lut()
robot.measure(x_pose, map, noise=False)
check_beam(12)

plt.figure('map')
plt.imshow(map)
robot.plot(sensor_readings=True)
plt.gca().invert_yaxis()
plt.show()
