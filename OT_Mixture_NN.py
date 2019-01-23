import tensorflow as tf
import numpy as np
import random
from sklearn.datasets import make_spd_matrix
from sklearn.mixture import GaussianMixture
from scipy.linalg import inv
import time

MARGS = 5
DIM = 2
K = 6  # number of mixtures in each marginal
GAMMA = 500 * MARGS * DIM  # + 1  # +1 at the end if one wants to avoid duplicates with saving
SC = 0  # special case for final case
if SC == 1:
    BATCH_SIZE = min(2 ** 8 * (2 ** int(round(np.log2(DIM*MARGS + K)))), 2 ** 12) * 4
    BATCH_MARG = min(2 ** 8 * (2 ** int(round(np.log2(DIM*MARGS/2 + K)))), 2 ** 11) * 2
    N = 100000
else:
    BATCH_SIZE = min(2 ** 8 * (2 ** int(round(np.log2(DIM * MARGS + K)))), 2 ** 12)
    BATCH_MARG = min(2 ** 8 * (2 ** int(round(np.log2(DIM * MARGS / 2 + K)))), 2 ** 11)
    N = min(50000 + 10000 * DIM * MARGS, 95000)

FEATURES_1 = ['identity']  # ['identity', 'fourier']  # ['identity', 'relu', 'square', 'sign']  # , 'square', 'relu', 'sign', 'pow4'
FEATURES_N = []  # ['sum', 'sumfourier']  # ['signdiff']  # 'prod'
ACTIVATION = 'ReLu'  # Adjust learning rate etc. when changing activation function
NEURONS = 32
N_LAYERS = 5
N_FOURIER = 0
MINMAX = -1  # 1 for minimization (in the dual) and -1 for maximization (in the dual)
PEN = 'L2'  # Either 'L2' or 'exp'
WHICH_FUN = 'normal'  # either 'normal' (for function f in paper) or something else (for function \tilde{f} in paper)


WASSER_Q = 2  # which L_Q norm to use on R^d
WASSER_P = 1  # which Wasserstein distance to use


print('DIM = ' + str(DIM))
print('MARGS = ' + str(MARGS))
print('BATCH_SIZE = ' + str(BATCH_SIZE))
print('BATCH_MARG = ' + str(BATCH_MARG))
print('GAMMA = ' + str(GAMMA))
print('HIDDEN = ' + str(NEURONS))
print('LAYERS: ' + str(N_LAYERS))
print('ACTIVATION: ' + ACTIVATION)
print('__________')
print('Number mixtures each marginal: ' + str(K))
print('Wasser_Q = ' + str(WASSER_Q))
print('Wasser_P = ' + str(WASSER_P))


# The marginals
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


# very old function
def gen_points_alt(batch_size, ps=p_list, mus=mus, ss=sigs):
    n_margs = len(mus)
    if hasattr(mus[0][0], "__len__"):
        d = len(mus[0][0])
    else:
        d = 1
    while 1:
        dataset = np.zeros([batch_size, n_margs, d])
        for i in range(n_margs):
            dataset[:, i, :] = sample_mixture_gaussian(batch_size, p_array=ps[i], mu_list=mus[i], sig_list=ss[i])
        yield dataset


# old function
def gen_points(batch_size, ps=p_list, mus=mus, sigs=sigs, k=K):
    n_margs = len(mus)
    if hasattr(mus[0][0], "__len__"):
        d = len(mus[0][0])
    else:
        d = 1

    model_list = []
    for i in range(n_margs):
        gmm = GaussianMixture(k, covariance_type='full')
        gmm.fit(np.zeros([k, d]))
        means_sk = np.zeros([k, d])
        for j in range(k):
            means_sk[j, :] = mus[i][j]
        gmm.means_ = means_sk
        gmm.weights_ = ps[i]
        covs = np.zeros([k, d, d])
        for j in range(k):
            covs[j, :, :] = sigs[i][j]
            if d == 1:
                covs[j, :, :] = sigs[i][j] ** 2  # To be consistent with an earlier messed-up implementation...
        gmm.covariances_ = covs
        # precisions = np.array([inv(x) for x in sigs[i]])
        # gmm.precisions_cholesky_ = precisions
        model_list.append(gmm)
    while 1:
        dataset = np.zeros([batch_size, n_margs, d])
        for i in range(n_margs):
            dataset[:, i, :] = model_list[i].sample(batch_size)[0]
        yield dataset

# currently used
def gen_points_best(batch_size, ps=p_list, mus=mus, ss=sigs, k=K):
    n_margs = len(mus)
    if hasattr(mus[0][0], "__len__"):
        d = len(mus[0][0])
    else:
        d = 1

    psc = []
    for i in range(n_margs):
        csp = np.cumsum(ps[i])
        csp = np.concatenate(([0], csp))
        psc.append(csp)

    sh = []
    if d == 1:
        for i in range(n_margs):
            sh.append(np.array(ss[i]) ** 2)
    else:
        sh = sigs.copy()

    while 1:
        dataset = np.zeros([batch_size, n_margs, d])
        for i in range(n_margs):
            sel_idx = np.random.random_sample(batch_size)
            for kind in range(k):
                idx = (sel_idx > psc[i][kind]) * (sel_idx < psc[i][kind+1])
                ksamples = np.sum(idx)
                dataset[idx, i, :] = np.random.multivariate_normal(mus[i][kind], sh[i][kind], ksamples)
        yield dataset


def sample_mixture_gaussian(batch_size, p_array, mu_list, sig_list, k=K, d=DIM):
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


# Build tf graph
if WHICH_FUN == 'normal':
    def f_1(x):
        out = 0
        for i in range(MARGS):
            out += x[:, i, :] * (-1) ** i
        out = tf.nn.relu(tf.pow(out, WASSER_Q))  # For training |x^p| is apparently better than |x|^p...
        # out = out ** WASSER_Q
        out = tf.pow(tf.reduce_sum(out, axis=1), WASSER_P/WASSER_Q)
        # out = tf.reduce_sum(out, axis=1) ** (WASSER_P/WASSER_Q)
        return MINMAX * out


    def f_1_np(x):
        out = 0
        for i in range(MARGS):
            out += x[:, i, :] * (-1) ** i
        out = np.abs(out) ** WASSER_Q
        out = np.sum(out, axis=1) ** (WASSER_P / WASSER_Q)
        return MINMAX * out
else:
    def f_1(x):
        out = 0
        for i in range(MARGS):
            out += x[:, i, :] * (-1) ** i
        out = tf.nn.relu(tf.pow(out, WASSER_Q))  # For training |x^p| is apparently better than |x|^p...
        # out = out ** WASSER_Q
        out = tf.pow(tf.reduce_sum(out, axis=1), WASSER_P/WASSER_Q) * tf.sin(tf.reduce_sum(x[:, :, 0], 1))
        # out = tf.reduce_sum(out, axis=1) ** (WASSER_P/WASSER_Q)
        return MINMAX * out

    def f_1_np(x):
        out = 0
        for i in range(MARGS):
            out += x[:, i, :] * (-1) ** i
        out = np.abs(out) ** WASSER_Q
        out = np.sum(out, axis=1) ** (WASSER_P/WASSER_Q) * np.sin(np.sum(x[:, :, 0], 1))
        return MINMAX * out


def layer(x, layernum, input_dim, output_dim, activation=ACTIVATION):
    ua_w = tf.get_variable('ua_w'+str(layernum), shape=[input_dim, output_dim],
                           initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
    ua_b = tf.get_variable('ua_b'+str(layernum), shape=[output_dim], initializer=tf.contrib.layers.xavier_initializer(),
                               dtype=tf.float64)
    z = tf.matmul(x, ua_w) + ua_b
    if activation == 'ReLu':
        return tf.nn.relu(z)
    if activation == 'tanh':
        return tf.nn.tanh(z)
    if activation == 'leakyReLu':
        return tf.nn.leaky_relu(z)
    if activation == 'softplus':
        return tf.nn.softplus(z)
    if activation == 'sin':
        return tf.sin(z)
    else:
        return z


def map_feat(x, feat, mean=0, scale=1):
    # input is shape [batch_size, d]
    # output is shape [batch_size, d]
    if feat == 'identity':
        return (x-mean)/scale
    if feat == 'square':
        return (x-mean)/scale ** 2
    if feat == 'sign':
        return tf.sign((x-mean)/scale)
    if feat == 'relu':
        return tf.nn.relu((x-mean)/scale)
    if feat == 'pow4':
        return x ** 4
    if feat == 'sin':
        return tf.sin((x-mean)/scale)
    if feat == 'fourier':
        fourier_freq = tf.get_variable(name='fourier_freq', shape=[N_FOURIER * 2], dtype=tf.float64)
        out = tf.sin(fourier_freq[0] * 2 * np.pi * (x-mean)/scale)
        out = tf.concat([out, tf.cos(fourier_freq[N_FOURIER] * 2 * np.pi * (x-mean)/scale)], axis=1)
        for i in range(1, N_FOURIER):
            out = tf.concat([out, tf.sin(fourier_freq[i] * 2 * np.pi * (x-mean)/scale)], axis=1)
            out = tf.concat([out, tf.cos(fourier_freq[N_FOURIER + i] * 2 * np.pi * (x-mean)/scale)], axis=1)
        return out
    else:
        print('potential ERROR: feature not supplied')
        return 0


def map_feat_n(x, feat, mean=0, scale=1):
    # input is shape [batch_size, d]
    # output is shape [batch_size, 1]
    if feat == 'max':
        return tf.reduce_max((x-mean)/scale, 1, keepdims=True)
    if feat == 'prod':
        return tf.reduce_prod((x-mean)/scale, 1, keepdims=True)
    if feat == 'signdiff':
        return tf.sign(tf.reduce_sum((x-mean)/scale, 1, keepdims=True))
    if feat == 'sum':
        return tf.reduce_sum((x-mean)/scale, 1, keepdims=True)
    if feat == 'sumfourier':
        fourier_freq_s = tf.get_variable(name='fourier_freq_s', shape=[N_FOURIER * 2], dtype=tf.float64)
        out = tf.sin(fourier_freq_s[0] * 2 * np.pi * tf.reduce_sum((x-mean)/scale, 1, keepdims=True))
        out = tf.concat([out, tf.cos(fourier_freq_s[N_FOURIER] * 2 * np.pi * tf.reduce_sum((x-mean)/scale, 1, keepdims=True))], axis=1)
        for i in range(1, N_FOURIER):
            out = tf.concat([out, tf.sin(fourier_freq_s[i] * 2 * np.pi * tf.reduce_sum((x-mean)/scale, 1, keepdims=True))], axis=1)
            out = tf.concat([out, tf.cos(fourier_freq_s[N_FOURIER + i] * 2 * np.pi * tf.reduce_sum((x-mean)/scale, 1, keepdims=True))], axis=1)
        return out


    else:
        print('potential ERROR: feature not supplied')
        return 0


def univ_approx(x, name, n_layers=N_LAYERS, hidden_dim=NEURONS*DIM, input_dim=DIM, output_dim=1, mean=0, scale=1):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        inp_dim = 0
        if len(FEATURES_1) > 0:
            inp_x = map_feat(x, FEATURES_1[0])
            snbo = 0
            inp_dim += input_dim
            if FEATURES_1[0] == 'fourier':
                inp_dim += (2 * N_FOURIER - 1)*input_dim
        else:
            inp_x = map_feat_n(x, FEATURES_N[0])
            snbo = 1
            inp_dim += 1
            if FEATURES_N[0] == 'sumfourier':
                inp_dim += (2 * N_FOURIER - 1)
        for i in range(1, len(FEATURES_1)):
            inp_dim += input_dim
            inp_x = tf.concat([inp_x, map_feat(x, FEATURES_1[i], mean=mean, scale=scale)], axis=1)
            if FEATURES_1[i] == 'fourier':
                inp_dim += (2 * N_FOURIER - 1)*input_dim
        for i in range(snbo, len(FEATURES_N)):
            inp_dim += 1
            inp_x = tf.concat([inp_x, map_feat_n(x, FEATURES_N[i], mean=mean, scale=scale)], axis=1)
            if FEATURES_N[i] == 'sumfourier':
                inp_dim += (2 * N_FOURIER - 1)

        if n_layers == 1:
            return layer(inp_x, 0, inp_dim, output_dim, activation='')
        else:
            a = layer(inp_x, 0, inp_dim, hidden_dim)
            for i in range(1, n_layers-1):
                a = layer(a, i, hidden_dim, hidden_dim)
            a = layer(a, n_layers-1, hidden_dim, output_dim, activation='')
            return a


S_marg = tf.placeholder(dtype=tf.float64, shape=[None, MARGS, DIM])
S_theta = tf.placeholder(dtype=tf.float64, shape=[None, MARGS, DIM])

sum_ints = 0
for i in range(MARGS):
    sum_ints += tf.reduce_sum(univ_approx(S_marg[:, i, :], str(i), mean=0, scale=1), axis=1)
ints = tf.reduce_mean(sum_ints)

sum_pen = 0
for i in range(MARGS):
    sum_pen += tf.reduce_sum(univ_approx(S_theta[:, i, :], str(i), mean=0, scale=1), axis=1)
    if i == 0:
        h0 = tf.reduce_sum(univ_approx(S_theta[:, i, :], str(i), mean=0, scale=1), axis=1)
    if i == 1:
        h1 = tf.reduce_sum(univ_approx(S_theta[:, i, :], str(i), mean=0, scale=1), axis=1)


fvar = f_1(S_theta)
diff = fvar - sum_pen
if PEN == 'L2':
    obj_fun = ints + GAMMA * tf.reduce_mean(tf.square(tf.nn.relu(diff)))
elif PEN == 'exp':
    obj_fun = ints + 1./GAMMA * tf.reduce_mean(tf.exp(GAMMA * diff - 1))
else:
    print('Reassigned PEN to L2 as current version is not implemented')
    PEN = 'L2'
    obj_fun = ints + GAMMA * tf.reduce_mean(tf.square(tf.nn.relu(diff)))

global_step = tf.Variable(0, trainable=False)
decay_steps = 50000 + 10000*DIM
decay_start = 5000
start_rate = 0.0001
final_rate = 0.0000005
exp = 2
rate = tf.train.polynomial_decay(start_rate, global_step, decay_steps, final_rate, power=exp)
train_op = tf.train.AdamOptimizer(learning_rate=rate, beta1=0.99, beta2=0.995).minimize(obj_fun)

t0 = time.time()
value_list = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    gen_marg = gen_points_best(BATCH_MARG)
    gen_theta = gen_points_best(BATCH_SIZE)

    maxv = []
    good_samples = np.zeros([0, MARGS, DIM])
    for t in range(1, N+1):
        gm = next(gen_marg)
        gt = next(gen_theta)

        (_, c, dv) = sess.run([train_op, obj_fun, diff], feed_dict={S_marg: gm, S_theta: gt, global_step: max(min(decay_steps, t-decay_start), 1)})

        value_list.append(c)
        maxv.append(np.max(dv))
        if t > 10000:
            if PEN == 'L2':
                den_max = max(maxv[-1000:])
                u = np.random.random_sample([BATCH_SIZE])
                good_samples = np.append(good_samples, gt[u * den_max <= dv, :, :], axis=0)
            elif PEN == 'exp':
                den_max = np.exp(GAMMA * max(maxv[-500:]))
                u = np.random.random_sample([BATCH_SIZE])
                good_samples = np.append(good_samples, gt[u * den_max <= np.exp(GAMMA * dv), :, :], axis=0)

        if t % 500 == 0:
            print(t)
            print('Current dual value: ' + str(np.mean(value_list[max(0, t-2000):])))
            if t > 15000:
                print('Current primal value: ' + str(np.mean(f_1_np(good_samples[max(0, t-50000):]))))
                print('Number of good samples: ' + str(len(good_samples)))
            crate = sess.run(rate, feed_dict={global_step: max(min(decay_steps, t-decay_start), 1)})
            print('Current learning rate: ' + str(crate))

save_list = [MARGS, DIM, K, WASSER_P, WASSER_Q, MINMAX, N, GAMMA, NEURONS, N_LAYERS, BATCH_SIZE, BATCH_MARG, PEN, WHICH_FUN]
save_list_string = [str(x) for x in save_list]
name_vals = 'Saved/' + '_'.join(save_list_string) + '_vals.txt'
name_samples = 'Saved/' + '_'.join(save_list_string) + '_samples.txt'
name_times = 'Saved/' + '_'.join(save_list_string) + '_time.txt'
np.savetxt(name_vals, value_list)
np.savetxt(name_samples, np.reshape(good_samples[max(0, t-100000):], [-1]))
np.savetxt(name_times, [time.time()-t0])
