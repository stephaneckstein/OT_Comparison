import numpy as np


def ot_cont_rkhs(margs_fun, f, k, d, kernel, gamma=100, n=250000, decay='poly', p=2, start=0.0005, final=10**(-7),
                 minmax=1, pen_fun=lambda x: np.exp(x-1), pen_der=lambda x: np.exp(x-1)):
    """
    calculates a multi-marginal OT problem by a reproducing kernel hilbert space approach
    We follow algorithm 3 in Genevay, Cuturi, Peyré, Bach "Stochastic Optimization for Large-scale OT"
    :param margs_fun: marginals generator functions.
    Should take no inputs and generate [k, d] marginals
    :param f: function in OT objective. Evaluated at [k, d] shape entries
    :param k: number of marginals
    :param d: dimension of marginals. Should be the same for each marginal
    :param kernel: kernel basis, should take two np array inputs of shape [n, k, d], [n, k, d]
    and return [n, k] shape
    :param gamma: factor for penalization
    :param n: number of iterations of training
    :param decay: how step size of SGD declines
    :param p: if decay='poly', this is the exponent of the polynomial decay
    :param start: initial step size
    :param final: final step size (if polynomial decay)
    :param minmax: if 1, solves (sup measures, inf dual) problem. if -1, solves (inf measures, sup dual) problem
    :param pen_fun: penalty function
    :param pen_der: derivative of penalty function
    :return: objective, optimizer
    """
    alpha_list = np.zeros([n, 1])
    xyzlist = np.zeros([n, k, d])
    ulist = np.zeros([n, k])
    pen_list = np.zeros(n)
    gen = margs_fun()
    sample = next(gen)
    xyzlist[0, :, :] = sample

    rate = start
    alpha_list[0] = rate * (pen_der(gamma * minmax * f(sample)) - 1)
    eps = 1/gamma
    pen_list[0] = eps * pen_fun(gamma * minmax * f(sample))

    decay_update = 10000
    if decay == 'exp':
        decay_rate = (final/start) ** (decay_update/n)

    for i in range(1, n):
        sample = next(gen)
        xyzlist[i, :, :] = sample
        ulist[i, :] = np.sum(np.tile(alpha_list[:i], [1, k]) * kernel(np.tile(sample, [i, 1, 1]), xyzlist[:i, :, :]), 0)
        alpha_list[i] = rate * (pen_der(gamma * (minmax * f(sample)-np.sum(ulist[i, :]))) - 1)
        pen_list[i] = eps * pen_fun(gamma * (minmax * f(sample) - np.sum(ulist[i, :])))
        # alpha_list[i] = rate * (np.exp((minmax * f(sample) - np.sum(ulist[i, :])) * gamma) - 1)
        # pen_list[i] = eps * np.exp((minmax * f(sample) - np.sum(ulist[i, :])) * gamma)
        if i % decay_update == 0:
            if decay == 'poly':
                rate = (start - final) * (1 - i / n) ** p + final
            elif decay == 'exp':
                rate *= decay_rate
        if i % 5000 == 0:
            print(i)
            print('Current rate: ' + str(rate))
            print(minmax * np.mean(np.sum(ulist[i-5000:i, :], 1)))
            print(minmax * np.mean(pen_list[i-5000:i]))
            print(minmax * np.mean(np.sum(ulist[i-5000:i, :], 1)) + minmax * np.mean(pen_list[i-5000:i]))


    return minmax * np.mean(np.sum(ulist[-50000:, :], 1)) + np.mean(pen_list[-50000:]), [xyzlist, alpha_list]



# Minimal example:
# MARGS = 2
# DIM = 1
# # The marginals
# seed = np.random.get_state()
# np.random.seed(0)
# # means = np.random.random_sample([MARGS, DIM]) * 2 - 1
# means = np.zeros([MARGS, DIM])
# # sigmas = np.random.random_sample([MARGS, DIM]) + 0.5
# sigmas = np.ones([MARGS, DIM]) * 2 * np.pi
# np.random.set_state(seed)
#
#
# def marg_gen(type='uniform'):
#     if type == 'uniform':
#         while 1:
#             yield np.random.random_sample([MARGS, DIM]) * sigmas + means
#     if type == 'normal':
#         while 1:
#             yield np.random.randn(MARGS, DIM) * sigmas + means
#
#
# def f(p):
#     out = np.sin(np.sum(p[0, :]))
#     s = np.sum(p[1:, :])
#     return out * np.cos(s)
#
#
# def gauss_kern(x, y, sig=2):
#     out = (x-y) ** 2
#     out = np.sum(out, 2)
#     return np.exp(-out/(2 * sig ** 2))
#
#
# def lap_kern(x, y, sig=2):
#     out = (x-y) ** 2
#     out = np.sum(out, 2)
#     out = np.sqrt(out)
#     return np.exp(-out / sig)
#
#
# ot_cont_rkhs(marg_gen, f, MARGS, DIM, gauss_kern, minmax=1)