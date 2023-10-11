import numpy as np
import matplotlib.pyplot as plt
from robot import Robot

robot = Robot()
robot.set_pos(1, 0, np.pi/2)

x0 = robot.x
y0 = robot.y
ha = robot.ha

plt.figure('world')
robot.plot()
robot.step_vel(10, np.pi/4*10)
x1 = robot.x
y1 = robot.y

robot.plot()
mx = (x0 + x1) / 2
my = (y0 + y1) / 2

t1 = (x0-x1)*np.cos(ha) + (y0-y1)*np.sin(ha)
t2 = (y0-y1)*np.cos(ha) - (x0-x1)*np.sin(ha)
mu = 1/2*t1/t2
a1 = -(y0 - y1) / (x0 - x1)

print('mu:', mu)
print('a1/2:', a1/2)

cx = mx + mu*(y0-y1)
cy = my + mu*(x1-x0)

plt.plot([x0, x1], [y0, y1], c='g')
plt.scatter(mx, my, c='g')
plt.scatter(cx, cy, c='r')

print('trajectory R:', robot.trajectory_r)

plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
