import numpy as np
import matplotlib.pyplot as plt

class RangeSensor:
    def __init__(self):
        self.sigma_hit = 0.1
        self.lambda_short = 1
        self.z_max = 5  # 5m

        self.w_hit = 0.8
        self.w_short = 0.05
        self.w_max = 0.1
        self.w_rand = 0.05

        self.resolution = 0.01  # 0.01m
        self.pdf = np.zeros(self.z_max)

    def compute_p_hit(self, x, z):
        s_hit = self.sigma_hit
        p = 1/np.sqrt(2*np.pi*s_hit**2)*np.exp(-1/2*(x-z)**2/s_hit**2)
        return p

    def compute_p_short(self, z):
        l_short = self.lambda_short
        eta = 1/(1 - np.exp(-l_short*z))
        p = eta*l_short*np.exp(-l_short*z)
        return p

    def compute_p_max(self, z):
        if abs(z-self.z_max) < 1e-6:
            p = 1
        else:
            p = 0
        return p

    def compute_p_rand(self, z):
        if 0 <= z <= self.z_max:
            p = 1/self.z_max
        else:
            p = 0
        return p
    
    def compute_pdf(self):
        self.pdf = np.zeros(int(self.z_max / self.resolution))

        
        return self.pdf
    
sensor = RangeSensor()

pdf = sensor.compute_pdf()





        
