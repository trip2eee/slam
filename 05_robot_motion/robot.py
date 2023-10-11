import numpy as np
import matplotlib.pyplot as plt

class Robot:
    def __init__(self):
        self.x = 0    # the x-coordinate of the robot (m).
        self.y = 0    # the y-coordinate of the robot (m).
        self.ha = 0   # the heading angle of the robot (rad).
        self.v = 0    # velocity (m/s)
        self.w = 0    # angular velocity (rad/s).
        self.r = 0.3  # radius of the robot (m)
        self.xc = None  # The x-coordinate of the center of trajectory (m)
        self.yc = None  # The y-coordinate of the center of trajectory (m)
        self.trajectory_r = None
        
    def set_pos(self, x, y, ha):
        self.x = x
        self.y = y
        self.ha = ha
    
    def step_vel(self, v, w, dt=0.1):
        """ set command for velocity motion model
            v: velocity (m/s)
            w: angular velocity(rad/s)
            dt: delta t (s)
        """
        self.v = v
        self.w = w

        if abs(w) > 0.0:
            r = v / w
            ha = self.ha
            xp  = self.x - r*np.sin(ha) + r*np.sin(ha + w*dt)
            yp  = self.y + r*np.cos(ha) - r*np.cos(ha + w*dt)
            hap = ha + w*dt

            self.xc = self.x - r*np.sin(ha)
            self.yc = self.y + r*np.cos(ha)
            self.trajectory_r = r
                        
            self.x = xp
            self.y = yp
            self.ha = hap

        else:
            ha = self.ha
            vx = v*np.cos(ha)
            vy = v*np.sin(ha)

            self.x += vx
            self.y += vy

            self.xc = None
            self.yc = None

    def plot(self):
        plt.scatter(self.x, self.y, c='r')

        if self.xc is not None and self.yc is not None:
            plt.scatter(self.xc, self.yc, c='b')

        # draw robot circle
        list_x = []
        list_y = []
        angles = [t * np.pi/180.0 for t in range(0, 360)]        
        for i in range(len(angles)-1):
            a = angles[i]
            x = self.x + self.r*np.cos(a)
            y = self.y + self.r*np.sin(a)
            list_x.append(x)
            list_y.append(y)
        
        plt.plot(list_x, list_y, c='b')

        # draw robot heading angle
        x1 = self.x + self.r*np.cos(self.ha)
        y1 = self.y + self.r*np.sin(self.ha)
        ha_x = [self.x, x1]
        ha_y = [self.y, y1]
        plt.plot(ha_x, ha_y, c='r')


