import numpy as np
from sklearn.datasets import make_spd_matrix
import time
from General_LP_OT import *
import random
from General_RKHS_OT import ot_cont_rkhs

K = 1
WASSER_Q = 2  # which L_Q norm to use on R^d
WASSER_P = 2  # which Wasserstein distance to use
MARGS = 2
DIM = 1
GAMMA = 500
MINMAX = -1  # 1 is minimization in the dual, -1 is maximization in the dual
print('DIM = ' + str(DIM))
print('MARGS = ' + str(MARGS))
print('GAMMA = ' + str(GAMMA))
print('__________')
print('Number mixtures each marginal: ' + str(K))
print('Wasser_Q = ' + str(WASSER_Q))
print('Wasser_P = ' + str(WASSER_P))
np.random.seed(0)
p_list = []
for i in range(MARGS):
    p_here = np.random.random_sample(K)
    p_here /= np.sum(p_here)
    p_list.append(p_here)

mus = []
sigs = []
for j in range(MARGS):
    means = []
    sig_list = []
    for i in range(K):
        means.append(10 * (np.random.random_sample(DIM) - 0.5))
        sig_list.append(make_spd_matrix(DIM))
    mus.append(means)
    sigs.append(sig_list)

np.random.seed(round(1000000*random.random()))


def gen_points(ps=p_list, mus=mus, sigs=sigs):
    n_margs = len(mus)
    if hasattr(mus[0][0], "__len__"):
        d = len(mus[0][0])
    else:
        d = 1
    while 1:
        dataset = np.zeros([n_margs, d])
        for i in range(n_margs):
            dataset[i, :] = np.reshape(sample_mixture_gaussian(1, p_array=ps[i], mu_list=mus[i], sig_list=sigs[i]), [-1])
        yield dataset


def sample_mixture_gaussian(batch_size, p_array, mu_list, sig_list):
    """
    samples from a mixture of normals
    :param batch_size: sample size
    :param p_array: np array which includes probability for each component of mix
    :param mu_list: list of means of each component
    :param sig_list: list of covariance matrices of each component
    :return: samples from mixture
    """
    if hasattr(mu_list[0], "__len__"):
        d = len(mu_list[0])  # dimension of distribution
    else:
        d = 1
    k = len(mu_list)  # number of mixtures
    dataset = np.zeros([batch_size, d])
    rh = np.random.choice(range(k), p=p_array, size=batch_size)
    for i in range(batch_size):
        if d > 1:
            dataset[i, :] = np.random.multivariate_normal(mean=mu_list[rh[i]], cov=sig_list[rh[i]])
        else:
            dataset[i, :] = np.random.randn() * sig_list[rh[i]] + mu_list[rh[i]]
    return dataset


def f(x):
    out = 0
    for i in range(MARGS):
        out += x[i, :] * (-1) ** i
    out = np.abs(out ** WASSER_Q)
    out = np.sum(out, axis=0) ** (WASSER_P/WASSER_Q)
    return out


def gauss_kern(x, y, sig=2):
    out = (x-y) ** 2
    out = np.sum(out, 2)
    return np.exp(-out/(2 * sig ** 2))


def lap_kern(x, y, sig=2):
    out = (x-y) ** 2
    out = np.sum(out, 2)
    out = np.sqrt(out)
    return np.exp(-out / sig)


n_runs = 1
save_list = [MARGS, DIM, K, WASSER_P, WASSER_Q, MINMAX, n_runs]
save_list_string = [str(x) for x in save_list]
name_obj = 'Saved/' + '_'.join(save_list_string) + '_objRKHS.txt'
name_times = 'Saved/' + '_'.join(save_list_string) + '_timesRKHS.txt'
name_optiX = 'Saved/' + '_'.join(save_list_string) + '_optiX.txt'
name_optiA = 'Saved/' + '_'.join(save_list_string) + '_optiA.txt'

objs = []
times = []
optisX = []
optisA = []

for i in range(n_runs):
    t0 = time.time()
    value, optimizer = ot_cont_rkhs(gen_points, f, MARGS, DIM, gauss_kern, minmax=-1, gamma=GAMMA, n=600000, start=0.0005,
                                    decay='poly', pen_fun=lambda x: (np.maximum(x, 0)) ** 2, pen_der=lambda x: 2*np.maximum(x, 0))
    times.append(time.time()-t0)
    objs.append(value)


print(objs)
print(times)
np.savetxt(name_obj, objs)
np.savetxt(name_times, times)
