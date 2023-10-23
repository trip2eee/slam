""" 5.3.2 Sampling Algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
from robot import Robot

delta_t = 1/10

# alpha 1 ~ alpha 6
alpha = [0.2, 0.1, 0.2, 0.2, 0.1, 0.1]

def prob_normal_distribution(a, b):
    """ Table 5.2 zero-centered normal distribution with variance b
    """
    return 1/np.sqrt(2*np.pi*b) * np.exp(-1/2*(a**2)/b)

def prob_triangular_distribution(a, b):
    """ Table 5.2 triangular distribution with variance b
    """
    
    prob = (np.sqrt(6*b) - abs(a)) / (6*b)
    prob *= abs(a) <= np.sqrt(6*b)
    return prob


a = np.array([xi*0.1 - 5.0 for xi in range(100)])
b = 1

prob_normal = prob_normal_distribution(a, b)
plt.figure('distribution')
plt.plot(a, prob_normal, label='normal')

prob_tri = prob_triangular_distribution(a, b)
plt.plot(a, prob_tri, label='triangular')
plt.legend()

prob = prob_normal_distribution

def motion_model_velocity(xt, ut, xt_1, dt=1/10):
    """ Table 5.1 Algorith motion_model_velocity
        Algorithm for computing p(xt | ut, xt-1) based on velocity information.
        xt   : pose at t [x1, y1, ha1]
        ut   : [v, w]^T
        xt_1 : pose at t-1 [x0, y0, ha0]
    """
    x1, y1, ha1 = xt        # x', y', theta'
    x0, y0, ha0 = xt_1      # x, y, theta

    t1 = (x0-x1)*np.cos(ha0) + (y0-y1)*np.sin(ha0)
    t2 = (y0-y1)*np.cos(ha0) - (x0-x1)*np.sin(ha0)
    mu = 1/2*t1/t2                  # line 2

    mx = (x0 + x1) / 2
    my = (y0 + y1) / 2

    cx = mx + mu*(y0-y1)            # line 3 x*
    cy = my + mu*(x1-x0)            # line 4 y*
    cr = np.sqrt((x0 - cx)**2 + (y0 - cy)**2)   # line 5 r*
    delta_ha = np.arctan2(y1-cy, x1-cx) - np.arctan2(y0-cy, x0-cx)  # line 6
    v_hat = delta_ha / dt * cr                  # line 7
    w_hat = delta_ha / dt                       # line 8
    gamma_hat = (ha1 - ha0) / dt - w_hat        # line 9. final rotation

    v, w = ut
    
    pv = prob(v-v_hat, alpha[0]*abs(v) + alpha[1]*abs(w))
    pw = prob(w-w_hat, alpha[2]*abs(v) + alpha[3]*abs(w))
    pg = prob(gamma_hat, alpha[4]*abs(v) + alpha[5]*abs(w))

    return pv, pw, pg

def sample_normal_distribution(b):
    """ Table 5.4 Algorithm for sampling from normal distribution with zero mean and variance b.
    """
    r = np.random.random_integers(-1000, 1000, size=12) * 0.001
    return b/6.0*r.sum()


def sample_triangular_distribution(b):
    """ Table 5.4 Algorithm for sampling from triangular distribution with zero mean and variance b.
    """
    r1 = np.random.random_integers(-1000, 1000) * 0.001
    r2 = np.random.random_integers(-1000, 1000) * 0.001

    return b * r1 * r2

sample = sample_normal_distribution

def sample_motion_model_velocity(ut, xt_1, dt=1/10):
    """ Table 5.3 Algorithm for sampling pose xt = (x', y', theta')^T from a pose xt-1 = (x, y, theta)^T
        and a control ut = (v, w)^T.
    """
    v, w = ut

    v_hat = v + sample(alpha[0]*abs(v) + alpha[1]*abs(w))   # line 1
    w_hat = w + sample(alpha[2]*abs(v) + alpha[3]*abs(w))   # line 2
    g_hat =     sample(alpha[4]*abs(v) + alpha[5]*abs(w))   # line 3

    x, y, theta = xt_1 # x_{t-1}

    xp = x - v_hat/w_hat * np.sin(theta) + v_hat/w_hat*np.sin(theta + w_hat * dt)   # line 4
    yp = y + v_hat/w_hat * np.cos(theta) - v_hat/w_hat*np.cos(theta + w_hat * dt)   # line 5
    thetap = theta + w_hat*dt + g_hat*dt

    xt = np.array([xp, yp, thetap])

    return xt


tri = []
norm = []
for i in range(10000):
    s = sample_triangular_distribution(1.0)
    tri.append(s)

    s = sample_normal_distribution(1.0)
    norm.append(s)

plt.figure('triangular')
plt.hist(tri, bins=20)
plt.figure('normal')
plt.hist(norm, bins=20)

robot = Robot()
robot.set_pos(1, 0, np.pi/2)

x0 = robot.x
y0 = robot.y
ha0 = robot.ha

xt_1 = np.array([x0, y0, ha0])
ut = np.array([10, np.pi/4*10])

plt.figure('world')
robot.plot()
robot.step_vel(ut[0], ut[1], dt=delta_t)
x1 = robot.x
y1 = robot.y
ha1 = robot.ha


xt = np.array([x1, y1, ha1])

pv, pw, pgamma = motion_model_velocity(xt, ut, xt_1, dt=delta_t)

samples = []
for s in range(500):
    xt = sample_motion_model_velocity(ut, xt_1, dt=delta_t)
    
    samples.append([xt[0], xt[1]])

samples = np.array(samples)
plt.scatter(samples[:,0], samples[:,1], c='b', marker='.')

robot.plot()

plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
