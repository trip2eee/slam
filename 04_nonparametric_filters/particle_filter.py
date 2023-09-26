import numpy as np
import matplotlib.pyplot as plt


def normal_dist(mean_x, std_x, x):
    fx = 1/(std_x*np.sqrt(2*np.pi)) * np.exp(-1/2*((x-mean_x) / std_x)**2)    
    return fx

class ParticleFilter:
    def __init__(self, M, px):
        self.M = M              # the number of particles
        self.px = px            # probability
        self.w = np.ones(M)     # weight (importance)
        self.X = []

    def sample(self):
        self.X = []

        for m in range(self.M):
            p = np.random.rand()
            print('draw p: ', p)

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

    def sample_uniform(self):
        Xt = []
        for m in range(self.M):
            x = np.random.randint(low=0, high=x_max)
            Xt.append(x)
        self.X = Xt

    def sample_gaussian(self):
        Xt = []
        
        for m in range(self.M):
            std_x = 0.5
            mean_x = self.X[m]

            x = np.random.randn()*std_x + mean_x
            # x = np.clip(x, a_min=0, a_max=x_max-1)
            # x = int(x + 0.5)

            Xt.append(x)
        self.X = Xt

    def resample(self):
        """ Low variance sampler (Table 4.4, p86)
        """
        Xt = []
        Wt = []

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
            Wt.append(self.w[i])
        
        self.X = Xt
        self.w = Wt

        self.update_pdf()

    def update_pdf(self):
        for i in range(len(self.px)):
            self.px[i] = 0.0

        for m in range(M):
            # xm, wm : position and importance of the m-th sample
            xm = self.X[m]
            wm  = self.w[m]
            xm = np.clip(xm, a_min=0, a_max=x_max-1)
            xm = int(xm + 0.5)            
            self.px[xm] += wm

        self.px = self.px / np.sum(self.px)

x_max = 50
x_obj = 30
x_robot = 0

x = np.array([x for x in range(x_max)])
fx_obj = normal_dist(x_obj, 2.5, x)

fx_obj = np.ones([x_max])
fx_obj = fx_obj / np.sum(fx_obj)

M = 50
filter = ParticleFilter(M, fx_obj)

for i in range(20):
    print('iteration: ', i)

    plt.figure()
    # filter.sample()
    if i == 0:
        filter.sample_uniform()
    else:
        filter.sample_gaussian()

    for x in filter.X:
        print('{:.2f}, '.format(x), end='')
    print('')
    
    # draw samples
    plt.subplot(311)
    zeros = [0]*filter.M
    
    plt.plot([0, x_max], [0, 0], c='k')
    plt.scatter(x_obj, 0, c='r', label='object')
    plt.scatter(x_robot, 0, c='b', label='robot')    
    plt.scatter(filter.X, zeros, c='m', marker='x', label='samples')
    plt.legend()

    plt.title('samples')
    
    # compute pdf f(z)
    pz = np.zeros([x_max])
    for m in range(M):
        z = filter.X[m]
        z = np.clip(z, a_min=0, a_max=x_max-1)
        z = int(z+0.5)
        pz[z] += 1
    pz /= M

    for m in range(M):
        z = filter.X[m]
        z = np.clip(z, a_min=0, a_max=x_max-1)
        z = int(z+0.5)
        
        # compute importance factor w_m = p(z | x_m)
        # p(z | x_m) = p(x_m | z) * p(z) / p(x_m)        
        pxz = normal_dist(x_obj, 1, z)
        pzm = pxz * pz[z] / (filter.px[z] + 1e-6)

        filter.w[m] = pzm

    plt.subplot(312)
    plt.plot([0, x_max], [0, 0], c='k')
    plt.stem(filter.X, filter.w)
    plt.title('w(x)')

    filter.resample()
    # draw resampled results
    plt.subplot(313)
    plt.plot([0, x_max], [0, 0], c='k')
    plt.plot(filter.px)
    plt.title('p(x)')

    plt.show()



