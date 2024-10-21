import math
import xxhash
import numpy as np
from math import sqrt, log


class SVME:

    def __init__(self, args):
        self.n = args['client']
        self.d = args['dim']
        self.k = args['sparsity']
        self.eps = args['eps']
        self.delta = args['delta']
        if args['neighbor_dist'] == -1:
            self.L = 2.0 * self.k
        else:
            self.L = args['neighbor_dist']
            self.L = min(self.L, 2.0 * self.k)

        self.mean_vec = []

    def get_name(self):
        return 'SVME'

    def get_mean(self):
        return self.mean_vec

    def predictedd_error(self):
        return max(sqrt(self.k / self.bucket), self.noise_param)

    def sampling(self, x, seed, p):
        t = (xxhash.xxh32(str(x), seed=seed * 0x1998).intdigest() % 10000)
        if t < 10000 * p:
            return 1
        else:
            return 0

    def hash_bin(self, x, seed, bucket):
        t = (xxhash.xxh32(str(x), seed=seed * 0xf022).intdigest() % bucket)
        return t

    def random_sign(self, x, seed):
        t = (xxhash.xxh32(str(x), seed=seed * 0x1352).intdigest() % 2)
        if t == 1:
            return 1
        else:
            return -1

    def rr(self, v, eps, range):
        v = v / range
        e_eps = math.exp(eps)
        c_eps = (e_eps + 1) / (e_eps - 1)
        p = (v * (e_eps - 1) + e_eps + 1) / (2 * (1 + e_eps))
        noisy_v = 0.0
        if np.random.binomial(1, p):
            noisy_v = c_eps
        else:
            noisy_v = -c_eps
        noisy_v *= range
        return noisy_v

    def estimate_range(self, data, top):
        seed = []
        T = []
        bucket = 0
        clip_r = 0.0
        noise_param = 0.0

        L = self.L
        k = self.k
        n = self.n

        if L < sqrt(k):
            bucket = min(int(k / L / L), k)
            noise_param = L
            clip_r = k
        else:
            bucket = 1
            clip_r = sqrt(k * log(n)) / 2.5
            noise_param = min(2 * clip_r, L)

        if (L / bucket) > log(2 * bucket / self.delta):
            noise_param = min(
                noise_param, 3 * sqrt(bucket * L * log(2 * bucket / self.delta)))


        noise_param /= self.eps

        for i in range(n):

            # local randomization process
            seed.append(np.random.randint(100, 10000, dtype=int))
            Ti = np.zeros(bucket)
            kv = data[i]

            for val in kv:
                j = val[0]
                v = val[1]
                sign_j = self.random_sign(j, seed[i])
                hash_j = self.hash_bin(j, seed[i], bucket)
                Ti[hash_j] += sign_j * v

            for j in range(bucket):
                Ti[j] = np.clip(Ti[j], -clip_r, clip_r)
                if noise_param >= clip_r:
                    Ti[j] = self.rr(Ti[j], self.eps, clip_r)
                else:
                    Ti[j] += self.laplacian_noise(noise_param)

            T.append(Ti)

        # aggregation
        cont_vec = np.zeros(top)
        mean_vec = np.zeros(top)
        for i in range(n):
            for j in range(top):
                hash_j = self.hash_bin(j, seed[i], bucket)
                sign_j = self.random_sign(j, seed[i])
                cont_vec[j] += 1
                mean_vec[j] += T[i][hash_j] * sign_j

        for j in range(top):
            mean_vec[j] /= cont_vec[j]
            mean_vec[j] = np.clip(mean_vec[j], -1, 1)

        self.mean_vec = mean_vec

    def laplacian_noise(self, noise_param):
        if noise_param == 0:
            return 0
        else:
            return np.random.laplace(loc=0, scale=noise_param)

    def esitimate(self, data):
        self.estimate_range(data, self.d)



#synthetic data
def get_data(n, k, d):
    data = []
    for i in range(n):
        data_i = []
        indexs = np.random.randint(0, d, size=k)
        for index in indexs:
            val = []
            val.append(index)
            low = -1 + 2 * np.random.random()
            rang = 1-low
            val.append(low+rang*np.random.random())
            data_i.append(val)
        data.append(data_i)
    return data


def get_real_mean(data, n, d):
    real_value = np.zeros(d)
    for i in range(n):
        data_i = data[i]
        for val in data_i:
            real_value[val[0]] += val[1]
    real_value /= n
    return real_value

def mean_square_error(real_mean,estimated_mean):
    if len(real_mean)!=len(estimated_mean):
        return 'Vector lengths dont match'
    L=len(real_mean)
    Mse=0.0
    for i in range(L):
        Mse+=((real_mean[i]-estimated_mean[i])*(real_mean[i]-estimated_mean[i]))
    Mse/=L
    return Mse



