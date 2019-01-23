from gurobipy import *
import numpy as np
import sklearn.cluster
import time


def optimal_sample(p_gen, n_out, metric='Lp', p=2, tol=0.5 * 10**-2, max_iter=500000, r=2, n_sample_step=10**4):
    """
    An approximative version of Algorithm 4.2 in Pflug & Pichler "Multistate stochastic optimization"
    :param p_gen:
    generator that takes a single input (batch size).
    Produces points of a distribution on R^d. Outputs [n, d] np-array
    :param n_out: number of support points of output measure
    :param metric: according to which Wasserstein distance the optimality is defined
    :param p: parameter of L_p metric
    :param tol: stopping criterion for algorithm.
    :param max_iter: maximal number of iterations of algorithm
    :param r: order of Wasserstein distance. Notably doesn't have to be the same as for metric. > 1 required
    :return: [n_out, d] numpy array and [n_out] array of weights representing a measure supported on n_out points
    """
    if metric == 'Lp':
        # inputs are shaped either
        # 1) [d], [d] or
        # 2) [n, d], [n, d]
        # returns either [1] or [n] shape
        def d_metric(x, y):
            out = abs(x-y)
            out = out ** p
            if len(out.shape) > 1:
                out = np.sum(out, 1)
            else:
                out = np.sum(out)
            out = out ** (1/p)
            return out
        def d_grad_2(x, y):
            # returns gradient of L_p(x-y) w.r.t second component evaluated at x, y
            return abs(x-y) ** (p-1) * np.sign(x-y) * (1/d_metric(x, y))

    else:
        print('Error: Selected metric not implemented')
        return 0

    gen_large = p_gen(n_sample_step)
    p_init = next(gen_large)
    print('Calculating initial K-Means cluster ... ')
    cluster_init = sklearn.cluster.KMeans(n_clusters=n_out, n_init=10, max_iter=300).fit(p_init)
    print('Done!')
    centers = cluster_init.cluster_centers_
    diff = 100
    old_centers = centers.copy()
    ind = 0
    gen_small = p_gen(1)
    start_rate = 0.001
    final_rate = 0.000005
    rate_update = 5000
    decay_steps = max_iter
    power = 3
    cur_rate = (start_rate - final_rate) * (1 - ind/decay_steps) ** power + final_rate
    print('Starting main algorithm')
    while diff > tol and ind < max_iter:
        ind += 1
        s = next(gen_small)
        i = np.argmin(d_metric(centers, s))
        centers[i, :] = centers[i, :] - cur_rate * r * d_metric(centers[i, :], s) ** (r-1) * d_grad_2(centers[i, :], s)

        if ind % rate_update == 0:
            cur_rate = (start_rate - final_rate) * (1 - ind / decay_steps) ** power + final_rate
            diff = np.sum(np.abs(centers - old_centers))
            old_centers = centers.copy()
            print('Current iteration: ' + str(ind))
            print('Current rate: ' + str(cur_rate))
            print('Current difference stopping criteria: ' + str(diff))
    weights = np.zeros(n_out)
    final_sample = next(gen_large)
    dmat = np.zeros([n_sample_step, n_out])
    for i in range(n_out):
        dmat[:, i] = d_metric(final_sample, centers[i, :])
    ind = np.argmin(dmat, 1)
    for i in range(n_out):
        weights[i] = sum(ind == i)
    weights /= n_sample_step
    print('Done!')
    return centers, weights


# import matplotlib.pyplot as plt
# D = 2
# def gen_unif(batch_size):
#     while 1:
#         yield np.random.random_sample([batch_size, D]) * 2 - 1
#
#
# def gen_normal(batch_size):
#     while 1:
#         yield np.random.randn(batch_size, D)
# c, w = optimal_sample(gen_normal, 12)
# print(c)
# print(w)
# plt.scatter(c[:, 0], c[:, 1], s=w * (1 / np.max(w)) * 100)
# plt.show()


def ot_continuous(margs_fun, f, k, n=0, tsize=10**5, disc='MC', minmax=1):
    """
    Computes a multi-marginal optimal transport problem using gurobipy and a discretization method as specified
    :param margs_fun:
    A function which takes a single argument (batchsize). Returns a list of samples,
    where each entry is a np-array (sample of one marginal)
    :param f:
    The function in the integral of the OT problem
    Has to be able to evaluate k np-array inputs of size d_1, ..., d_k
    :param k: number of marginals
    :param d: dimension of marginals. A list of dimensions if different for each marginal
    :param tsize: Total size of the resulting linear program (if discretized)
    :param disc: discretization scheme. Currently only 'MC' (monte carlo)
    :return: objective_value, optimizer
    """
    if n < 1:
        n = int(np.floor(tsize ** (1/k)))

    if disc == 'MC':
        l_sample = margs_fun(n)
        return ot_lp(l_sample, f, minmax=minmax)
    if disc == 'quant':
        # here, margs_fun has to be a list of generator functions.
        l_sample = []
        w_list = []
        for i in range(k):
            c, w = optimal_sample(lambda x: margs_fun(x, i), n)
            l_sample.append(c)
            w_list.append(w)
        return ot_lp(l_sample, f, minmax=minmax, weights=w_list)
    return 0


def ot_lp(margs, f, weights=(), minmax=1):
    """
    Computes a discrete multi-marginal optimal transport problem using gurobipy
    :param margs:
    list of k numpy arrays. Number of entries are the number of marginals. Each numpy array is of the form
    [n_i, d_i] where n_i is the number of support points for the i-th marginal and d_i is its dimension.
    :param f:
    The function in the integral of the OT problem
    Has to be able to evaluate k np-array inputs of size d_1, ..., d_k
    :param weights:
    list of k numpy arrays. Each array is of size n_i and has nonnegative entries that sum to one.
    There are the weights of the points of each marginal
    If not specified, i.e. if (not weights) == True, then all points will be treated as equally weighted.
    :return:
    objective_value, optimizer
    """

    # Get dimensions of the problem:
    k = len(margs)
    n_sample_list = []
    d_list = []
    for i in range(k):
        n_sample_list.append(margs[i].shape[0])
        if len(margs[i].shape) > 1:
            d_list.append(margs[i].shape[1])
        else:
            d_list.append(1)

    # Build cost matrix
    code_start = 'cost_mat = np.zeros(['
    for i in range(k):
        code_start += str(n_sample_list[i])
        if i < k-1:
            code_start += ', '
    code_start += '])\n'

    eval_line_1 = 'cost_mat['
    eval_line_2 = 'f('
    for i in range(k):
        eval_line_1 += 'i'+str(i)
        eval_line_2 += 'margs[' + str(i) + '][i' + str(i) + ', :]'
        if i < k-1:
            eval_line_1 += ', '
            eval_line_2 += ', '
        else:
            eval_line_1 += ']'
            eval_line_2 += ')'
        ts = ''
        for j in range(i):
            ts += '\t'
        code_start += ts + 'for i' + str(i) + ' in range(' + str(n_sample_list[i]) + '):\n'
    ts = ''
    for j in range(k):
        ts += '\t'
    eval_line = ts + eval_line_1 + ' = ' + eval_line_2
    code_mat = code_start + eval_line
    exec(code_mat)
    cm = locals()['cost_mat']

    # Build Gurobi model
    m = Model('MultiOT')

    pi_code = 'pi_var = m.addVars('
    for i in range(k):
        pi_code += str(n_sample_list[i]) + ', '
    pi_code += 'lb = 0, name="pi_var")'
    exec(pi_code)
    m.update()

    if not weights:
        weights = []
        for i in range(k):
            weights.append(np.ones(n_sample_list[i])/n_sample_list[i])

    # Add constraints:
    index_string = '(' + '"*", ' * (k-1) + '"*")'
    for i in range(k):
        for j in range(n_sample_list[i]):
            if i < k - 1:
                ihere = index_string[:1 + 5 * i] + str(j) + ', ' + index_string[1 + 5 * (i + 1):]
            else:
                ihere = index_string[:1 + 5 * i] + str(j) + ')'

            ci_line = 'm.addConstr(locals()["pi_var"].sum' + ihere + ' == ' + 'weights['+str(i)+'][' + str(j) + '], name="W' + str(i) + '_' + str(j) + '")'
            exec(ci_line)

    # Specify objective of gurobi
    obj = LinExpr()
    code_obj = ''
    index = '['
    for i in range(k):
        index += 'i'+str(i)
        if i < k-1:
            index += ', '
        else:
            index += ']'
        ts = ''
        for j in range(i):
            ts += '\t'
        code_obj += ts + 'for i' + str(i) + ' in range(' + str(n_sample_list[i]) + '):\n'
    ts = ''
    for j in range(k):
        ts += '\t'
    code_obj += ts + 'obj += locals()["pi_var"]' + index + ' * cm' + index
    exec(code_obj)
    if minmax == 1:
        m.setObjective(obj, GRB.MINIMIZE)
    else:
        m.setObjective(obj, GRB.MAXIMIZE)
    m.optimize()

    objective_val = m.ObjVal
    opti = np.zeros(n_sample_list)
    for index, x in np.ndenumerate(opti):
        opti[index] = locals()["pi_var"][index].x

    return m.ObjVal, [margs, opti]



# # Minimal Example:
# def my_f(*argv):
#     out = 0
#     for arg in argv:
#         out += np.sum(arg)
#     return out
#
#
# obj, opti = ot_lp([np.ones([10, 5]) for i in range(6)], my_f)
