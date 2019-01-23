import numpy as np
from scipy.linalg import sqrtm
from sklearn.datasets import make_spd_matrix
import time
from General_LP_OT import *
import random


K = 6
WASSER_Q = 2  # which L_Q norm to use on R^d
WASSER_P = 1  # which Wasserstein distance to use
MARGS = 10
WHICH_FUN = 'Nichtnormal'
DIM = 3
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


def gen_margs(batch_size, ps, mus, sigs):
    n_margs = len(mus)
    if hasattr(mus[0][0], "__len__"):
        d = len(mus[0][0])
    else:
        d = 1
    while 1:
        dataset = np.zeros([batch_size, d, n_margs])
        for i in range(n_margs):
            dataset[:, :, i] = sample_mixture_gaussian(batch_size, p_array=ps[i], mu_list=mus[i], sig_list=sigs[i])
        yield dataset


def gen_lp_quant(batch_size, i):
    while 1:
        yield sample_mixture_gaussian(batch_size, p_array=p_list[i], mu_list=mus[i], sig_list=sigs[i])


def samp_lp(batch_size):
    l = []
    for i in range(MARGS):
        l.append(sample_mixture_gaussian(batch_size, p_array=p_list[i], mu_list=mus[i], sig_list=sigs[i]))
    return l


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


# # Minimal Example
# import matplotlib.pyplot as plt
# cov1 = np.array([[1, 0], [0, 1]])
# cov2 = np.array([[2, 1.5], [1.5, 2]])
# cov3 = np.array([[1, -0.6], [-0.6, 1]])
# samp = sample_mixture_gaussian(50000, [0.4, 0.2, 0.4], [[-5, -5], [0, 0], [5, 5]], [cov1, cov2, cov3])
# plt.hist2d(samp[:, 0], samp[:, 1], bins=200)
# plt.show()


if WHICH_FUN == 'normal':
    def f(*argv):
        # Here as well, all marginals need to have the same dimension
        out = 0
        ind = 0
        for arg in argv:
            out += arg * (-1) ** ind
            ind += 1
        out = np.abs(out)
        out = out ** WASSER_Q
        out = np.sum(out, 0) ** (WASSER_P/WASSER_Q)
        return out
else:
    def f(*argv):
        # Here as well, all marginals need to have the same dimension
        out = 0
        ind = 0
        sin_term = 0
        for arg in argv:
            out += arg * (-1) ** ind
            ind += 1
            sin_term += arg[0]
        sin_term = np.sin(sin_term)
        out = np.abs(out)
        out = out ** WASSER_Q
        out = np.sum(out, 0) ** (WASSER_P/WASSER_Q)
        return out * sin_term


# if MARGS == 2 and K == 1 and WASSER_P == 2 and WASSER_Q == 2:
#     m1 = mus[0][0]
#     m2 = mus[1][0]
#     C1 = sigs[0][0] ** 2
#     C2 = sigs[1][0] ** 2
#     md = np.sum((m1-m2) ** 2)
#     trt = np.trace(C1 + C2 - 2 * np.sqrt(C1 * C2))
#     print(md + trt)
#     exit()

# Comonotone reference value. Only for two marginals (and dimension 1?)
if MARGS == 2 and DIM == 1:
    sample_com = 1000000
    l = samp_lp(sample_com)
    l0 = l[0]
    l1 = l[1]
    m0 = np.sort(l0, axis=0)
    m1 = np.sort(l1, axis=0)
    ref_val = 0
    for i in range(sample_com):
        ref_val += f(m0[i], m1[i])
    ref_val /= sample_com
    print(ref_val)
    exit()

# Below one also gets a reference vlaue for DIM > 1, but it's a terrible reference value!
# if MARGS == 2 and DIM > 1:
#
#     sample_com = 100000
#     l = samp_lp(sample_com)
#     l0 = l[0]
#     l1 = l[1]
#     s0 = np.sum(l0, axis=1)
#     s1 = np.sum(l1, axis=1)
#     ind0 = np.argsort(s0, axis=0)
#     ind1 = np.argsort(s1, axis=0)
#     m0 = l0[ind0, :]
#     m1 = l1[ind1, :]
#     ref_val = 0
#     for i in range(sample_com):
#         ref_val += f(m0[i, :], m1[i, :])
#     ref_val /= sample_com
#     print(ref_val)
# exit()

# LP IMPLEMENTATION:
n_runs = 1
objs = []
times = []
tsize = 10 ** 6.1  # maximal total number of variables in LP.
MINMAX = 1  # 1 to minimize, -1 to maximize (for the primal: over measures)
DISC_METHOD = 'MC'  # Either 'MC' or 'quant'

for run in range(n_runs):
    print('RUN NUMBER: ' + str(run))
    t0 = time.time()
    if DISC_METHOD == 'MC':
        obj, opti = ot_continuous(samp_lp, f, MARGS, tsize=tsize, minmax=MINMAX, disc='MC')
    elif DISC_METHOD == 'quant':
        obj, opti = ot_continuous(gen_lp_quant, f, MARGS, tsize=tsize, minmax=MINMAX, disc='quant')
    else:
        print('Discretization method fail')
        obj = 0
        exit()

    time_taken = time.time()-t0
    objs.append(obj)
    times.append(time_taken)

print('Number marginals: ' + str(MARGS))
print('Dimension: ' + str(DIM))
print('Number mixtures: ' + str(K))
print('Wasser_P: ' + str(WASSER_P))
print('Wasser_Q: ' + str(WASSER_Q))
print('Minmax = ' + str(MINMAX))
print('Discretization method: ' + DISC_METHOD)
print('Objective value: ' + str(objs))
print('Time taken: ' + str(times))

save_list = [MARGS, DIM, K, WASSER_P, WASSER_Q, MINMAX, DISC_METHOD, tsize, n_runs, WHICH_FUN]
save_list_string = [str(x) for x in save_list]
name_obj = 'Saved/' + '_'.join(save_list_string) + '_obj.txt'
name_times = 'Saved/' + '_'.join(save_list_string) + '_times.txt'
np.savetxt(name_obj, objs)
np.savetxt(name_times, times)


# grid = np.zeros([len(opti[0][0]), len(opti[0][1]), 2])
# for i in range(len(opti[0][0])):
#     for j in range(len(opti[0][1])):
#         grid[i, j, 0] = opti[0][0][i]
#         grid[i, j, 1] = opti[0][1][j]
#
# import matplotlib.pyplot as plt
# plt.scatter(grid[:, :, 0], grid[:, :, 1], opti[1] * 500)
# plt.show()
# print(opti[0])
# print(opti[1])
