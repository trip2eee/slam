import numpy as np
import matplotlib.pyplot as plt


def normal_dist(mean_x, std_x, x):
    fx = 1/(std_x*np.sqrt(2*np.pi)) * np.exp(-1/2*((x-mean_x) / std_x)**2)    
    return fx

class ParticleFilter:
    def __init__(self, M, px):
        self.M = M              # the number of particles
        self.px = px            # probability
        self.w = np.zeros(M)    # weight
        self.X = []

    def sample(self):
        self.X = []

        for m in range(self.M):
            p = np.random.rand()

            # draw sample
            cdf = 0
            for i in range(len(self.px)):
                if i == 0:
                    cdf = self.px[i]
                else:
                    cdf += self.px[i]

                if cdf >= p:
                    x = i
                    break

            self.X.append(x)

    def resample(self):
        """ Low variance sampler
        """
        Xt = []
        r = np.random.rand() * 1/self.M
        self.w = self.w / np.sum(self.w)

        c = self.w[0]
        i = 0

        for m in range(self.M):
            U = r + (m / M)   # m: 0 ~ M-1
            while U > c:
                i = i + 1
                c = c + self.w[i]
            Xt.append(self.X[i])
        
        self.X = Xt

        self.update_pdf()

    def update_pdf(self):
        for i in range(len(self.px)):
            self.px[i] = 0

        for m in range(M):
            xm = self.X[m]
            self.px[xm] += 1
        self.px = self.px / np.sum(self.px)

x_max = 50
x_obj = 30
x_robot = 0

x = np.array([x for x in range(x_max)])
fx_obj = normal_dist(x_obj, 2.5, x)

fx_obj = np.ones([x_max])
fx_obj = fx_obj / np.sum(fx_obj)

M = 15
filter = ParticleFilter(M, fx_obj)

for i in range(20):
    print('iteration: ', i)

    plt.figure()
    filter.sample()
    print('  ', filter.X)
    # draw samples
    plt.subplot(211)
    zeros = [0]*filter.M
    plt.scatter(x_obj, 0, c='r', label='object')
    plt.scatter(x_robot, 0, c='b', label='robot')    
    plt.scatter(filter.X, zeros, c='m', marker='x', label='samples')
    
    for m in range(M):
        xm = filter.X[m]
        dist_error = (x_obj - xm)
        wm = normal_dist(0, 10, dist_error)
        filter.w[m] = wm

    filter.resample()
    
    # draw resampled results
    plt.plot([0, x_max], [0, 0], c='k')
    plt.legend()

    plt.subplot(212)
    plt.plot(filter.px)

    plt.show()

