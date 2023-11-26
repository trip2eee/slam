import numpy as np
import matplotlib.pyplot as plt

class RangeSensor:
    def __init__(self):
        self.sigma_hit = 0.15
        self.lambda_short = 1.0
        self.z_max = 5  # 5m

        # z_hit, z_short, z_max, and z_rand in p.128, equation 6.13        
        w_hit   = 0.97
        w_short = 0.005
        w_max   = 0.020
        w_rand  = 0.005

        # normalize
        w_sum = w_hit + w_short + w_max + w_rand
        self.w_hit   = w_hit / w_sum
        self.w_short = w_short / w_sum
        self.w_max   = w_max / w_sum
        self.w_rand  = w_rand / w_sum

        self.resolution = 0.01  # 0.01m
        self.pdf = np.zeros(self.z_max)

    def compute_p_hit(self, x, z):
        """x: the range at which p_hit is computed.
           z: true value           
        """
        if 0 <= x <= self.z_max:
            s_hit = self.sigma_hit
            p = 1/np.sqrt(2*np.pi*s_hit**2)*np.exp(-1/2*(x-z)**2/s_hit**2)

        else:
            p = 0

        return p

    def compute_p_short(self, x, z):
        """z: true value
        """
        if 0 <= x <= z:
            l_short = self.lambda_short
            eta = 1/(1 - np.exp(-l_short*z))
            p = eta*l_short*np.exp(-l_short*x)
        else:
            p = 0
        return p

    def compute_p_max(self, x):
        if abs(x-self.z_max) < 1e-6:
            p = 1
        else:
            p = 0
        return p

    def compute_p_rand(self, x):
        if 0 <= x <= self.z_max:
            p = 1/self.z_max
        else:
            p = 0
        return p

    def compute_pdf(self, z):
        self.pdf = np.zeros(int(self.z_max / self.resolution) + 1)
        res = self.resolution

        w_hit = self.w_hit
        w_short = self.w_short
        w_max = self.w_max
        w_rand = self.w_rand

        eta_p_hit = 0
        for xi in range(len(self.pdf)):
            x = xi * res
            p_hit = self.compute_p_hit(x, z)
            eta_p_hit += p_hit
        
        for xi in range(len(self.pdf)):
            x = xi * res
            p_hit = self.compute_p_hit(x, z) / eta_p_hit
            p_short = self.compute_p_short(x, z)
            p_max = self.compute_p_max(x)
            p_rand = self.compute_p_rand(x)

            self.pdf[xi] = p_hit*w_hit + p_short*w_short + p_max*w_max + p_rand*w_rand

        return self.pdf

    def measure(self):
        sum_pdf = self.pdf.sum()
        r = np.random.random()*sum_pdf

        sum_p = 0
        z = 0
        for idx_p in range(len(self.pdf)):
            if sum_p >= r:
                z = idx_p*self.resolution
                break
            p = self.pdf[idx_p]
            sum_p += p
        return z



if __name__ == '__main__':

    sensor = RangeSensor()

    sensor_pdf = sensor.compute_pdf(z=3)
    list_z = []
    list_x = []
    for i in range(1000):
        z = sensor.measure()
        list_x.append(i)
        list_z.append(z)


    plt.figure('pdf')
    plt.title('beam model pdf')
    plt.plot(sensor_pdf)
    plt.xlabel('z')
    plt.ylabel('p(z|z*)')

    plt.figure('measurements')
    plt.scatter(list_x, list_z)
    plt.ylabel('z')
    plt.xlabel('t')

    plt.figure('hist')
    plt.hist(list_z, bins=50)

    plt.show()



        

