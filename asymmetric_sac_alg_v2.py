import numpy as np
from asymmetric_sac_alg_v1 import check_delta
from options import Options

from nuclear_function import get_nuclear_func_val
from test_functions import TestFunc


class Point:
    def __init__(self, coord, func_val, u, idx):
        self.idx = idx
        self.coord = coord
        self.func_val = func_val
        self.u = u


def split_points(test_points, op_point):
    for i in range(len(test_points)):
        idx = 0
        for d in range(len(op_point)):
            if test_points[i].coord[d] > op_point[d]:
                idx += np.power(2, len(op_point) - d - 1)
        test_points[i].idx = idx
    return test_points


def initialization_op_point_and_delta(test_func: TestFunc, options: Options):
    dim = test_func.dim
    op_point = np.empty((dim,))
    low = test_func.down
    high = test_func.up
    if options.initialization_mode == 'center':
        op_point = np.array([(high[i] - low[i]) / 2 + low[i] for i in range(dim)])
    elif options.initialization_mode == 'random':
        op_point = np.array([np.random.uniform(0, 1) * (high[i] - low[i]) + low[i] for i in range(dim)])
    elif options.initialization_mode == 'manual':
        op_point = np.array(options.get_x)

    delta = np.array([[(op_point[i] - low[i]), (high[i] - op_point[i])] for i in range(dim)])

    return op_point, delta


def get_quad_idx(center, point):
    idx = 0
    for i in range(len(center)):
        if point[i] > center[i]:
            idx += np.power(2, len(center) - i - 1)
    return idx


def initialization_test_point(test_func: TestFunc, number_point: int, delta, op_point):
    dim = test_func.dim
    quad_tree = []

    for i in range(number_point):
        test_point = np.zeros((dim, ))
        idx = 0
        for j in range(dim):
            low = op_point[j] - delta[j][0]
            high = op_point[j] + delta[j][1]
            test_point[j] = np.random.uniform(0, 1) * (high - low) + low
            if test_point[j] > op_point[j]:
                idx += np.power(2, dim - j - 1)
        quad_tree.append(Point(test_point, 0, 0, idx))

    return quad_tree


def find_norm_nuclear_func(test_points, options: Options):
    sum_val = 0
    for point in test_points:
        val = get_nuclear_func_val(options.index_nf, point.func_val, options.s_factor)
        point.func_val = val
        sum_val += val

    for point in test_points:
        point.func_val = point.func_val / sum_val

    return test_points


def find_g(test_points, min_flag: int):
    max_fit_tp = np.max([point.func_val for point in test_points])
    min_fit_tp = np.min([point.func_val for point in test_points])

    if min_flag == 1:
        best = min_fit_tp
        worst = max_fit_tp
        for point in test_points:
            val = (point.func_val - best) / (worst - best)
            point.func_val = val
    else:
        best = max_fit_tp
        worst = min_fit_tp
        for point in test_points:
            val = (best - point.func_val) / (best - worst)
            point.func_val = val

    return test_points


def find_fitness_func_value(test_points, op_point, test_func: TestFunc, options: Options):
    amp_noise = np.random.uniform(-1, 1) * test_func.amp * options.k_noise
    fitness_op_value = test_func.get_value(op_point)
    fitness_op_value += amp_noise

    for i in range(len(test_points)):
        amp_noise = np.random.uniform(-1, 1) * test_func.amp * options.k_noise
        val = test_func.get_value(test_points[i].coord) + amp_noise
        test_points[i].func_val = val

    return fitness_op_value


def move(op_point, test_points, delta, options: Options):
    dim = len(op_point)

    new_delta = np.zeros(delta.shape)
    new_op_point = np.zeros(op_point.shape)

    sum_u_p = np.zeros((np.power(2, dim), dim))
    u_norm = np.zeros((dim, ))

    for i in range(len(test_points)):
        u = np.zeros((dim,))
        for d in range(dim):
            temp = test_points[i].coord[d]
            u[d] = (temp - op_point[d]) / delta[d][int(temp >= op_point[d])]
            u_norm[d] += test_points[i].func_val * u[d]
        test_points[i].u = u

    for d in range(dim):
        new_op_point[d] = op_point[d] + delta[d][int(u_norm[d] >= 0)] * u_norm[d]

    split_points(test_points, new_op_point)
    for j in range(len(sum_u_p)):
        cluster = [point for point in test_points if point.idx == j]
        for i in range(len(cluster)):
            for d in range(dim):
                sum_u_p[j][d] += cluster[i].func_val * np.power(np.abs(cluster[i].u[d]), options.q)

    for d in range(dim):
        for i in range(len(new_delta[d])):
            sum_idx = [k for k in range(np.power(2, dim)) if bin(k)[2:].zfill(dim)[d] == str(i)]
            u_p = sum([sum_u_p[j][d] for j in sum_idx])
            new_delta[d][i] = options.g * delta[d][i] * np.power(u_p, 1 / options.q)

    return new_op_point, new_delta


def sac(test_func: TestFunc, options: Options, epsilon=pow(10, -5), max_compression=0.03):
    op_point, delta = initialization_op_point_and_delta(test_func, options)
    number_measurements = 0

    best_chart = np.zeros((options.max_iter, ))
    x_bests = np.zeros((options.max_iter, len(op_point)))
    stop_iter = options.max_iter

    for i in range(options.max_iter):
        iteration = i + 1

        test_points = initialization_test_point(test_func, options.number_points, delta, op_point)
        fit_op_val = find_fitness_func_value(test_points, op_point, test_func, options)
        number_measurements += len(test_points) + 1

        x_bests = np.copy(op_point)
        best_chart[i] = fit_op_val

        if iteration > 2:
            if (np.sum(delta[0]) < epsilon) or (np.sum(delta[1]) < epsilon):
                stop_iter = iteration
                break

        find_g(test_points, options.min_flag)

        find_norm_nuclear_func(test_points, options)

        # line1 = np.array([[(op_point[0] - delta[0][0]), i - 5] for i in range(11)])
        # line2 = np.array([[(op_point[0] + delta[0][1]), i - 5] for i in range(11)])
        # line3 = np.array([[i - 5, (op_point[1] - delta[1][0])] for i in range(11)])
        # line4 = np.array([[i - 5, (op_point[1] + delta[1][1])] for i in range(11)])
        # a1 = np.array([point.coord for point in test_points if point.idx == 0])
        # a2 = np.array([point.coord for point in test_points if point.idx == 1])
        # a3 = np.array([point.coord for point in test_points if point.idx == 2])
        # a4 = np.array([point.coord for point in test_points if point.idx == 3])
        # graph.graph12(5, op_point, a1, a2, a3, a4, line1, line2, line3, line4, test_points)

        op_point, delta = move(op_point, test_points, delta, options)
        delta = check_delta(delta, op_point, test_func)
        for d in range(len(delta)):
            if sum(delta[d]) / (test_func.up[d] - test_func.down[d]) < max_compression:
                if delta[d][0] > delta[d][1]:
                    delta[d][1] = delta[d][0]
                else:
                    delta[d][0] = delta[d][1]

    return x_bests, best_chart, number_measurements, stop_iter


def in_rectangle(point, borders):
    return all(list(map(lambda x, b: True if b[0] <= x <= b[1] else False, point, borders)))


def test():
    from inform_tf import TEST_FUNC_1
    op = Options(20, 300, 1, 1, 20, 1, initialization_mode='center', k_noise=0, min_flag=1)
    tf = TestFunc(TEST_FUNC_1)

    p = 0
    nm = 0
    average_numb_iter = 0
    ep = 0.2
    g_min = tf.global_min
    for i in range(100):
        x_bests, best_chart, number_measurements, stop_iter = sac(tf, op, epsilon=pow(10, -5), max_compression=0.03)
        print('Решение', x_bests)
        print('Количество итераций', stop_iter)
        nm += number_measurements
        average_numb_iter += stop_iter
        if (g_min[0] - ep <= x_bests[0] <= g_min[0] + ep) and (g_min[1] - ep <= x_bests[1] <= g_min[1] + ep):
            p += 1
    print('Оценка вероятности', p / 100.0)
    print('Среднее количество измерений', nm / 100.0)
    print('Среднее количество итераций:', average_numb_iter / 100.0)


def main():
    test()


if __name__ == '__main__':
    main()
