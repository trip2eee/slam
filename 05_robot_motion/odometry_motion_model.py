""" 5.4 Odometry Motion Model
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

prob = prob_normal_distribution
sample = sample_normal_distribution

def motion_model_odometry(xt, ut, xt_1):
    """ Table 5.5 Algorithm for computing p(xt | ut, xt-1) based on odometry information.
        ut : control (bar xt_1, bar xt)
    """
    x0, y0, theta0 = xt_1
    x1, y1, theta1 = xt

    bx0, by0, btheta0, bx1, by1, btheta1 = ut
    d_rot1 = np.arctan2(by1 - by0, bx1 - bx0) - btheta0     # line 2
    



