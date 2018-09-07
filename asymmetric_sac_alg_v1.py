import numpy as np

from graph import visualization
from options import Options

from nuclear_function import get_nuclear_func_val
from test_functions import TestFunc


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
    number_quadrants = np.power(2, test_func.dim)
    quad_tree = [[] for _ in range(number_quadrants)]

    for i in range(number_point):
        test_point = np.zeros((dim, ))
        idx = 0
        for j in range(dim):
            low = op_point[j] - delta[j][0]
            high = op_point[j] + delta[j][1]
            test_point[j] = np.random.uniform(0, 1) * (high - low) + low
            if test_point[j] > op_point[j]:
                idx += np.power(2, dim - j - 1)
        quad_tree[idx].append(test_point)

    return quad_tree


def find_norm_nuclear_func(g, options: Options):
    nuclear_func_norm_val = np.empty((len(g),))
    sum_val = 0
    for i in range(len(g)):
        nuclear_func_norm_val[i] = get_nuclear_func_val(options.index_nf, g[i], options.s_factor)
        sum_val += nuclear_func_norm_val[i]

    for i in range(len(nuclear_func_norm_val)):
        nuclear_func_norm_val[i] = nuclear_func_norm_val[i] / sum_val

    return nuclear_func_norm_val


def find_g(fit_tp_value, min_flag: int):
    max_fit_tp = np.max(fit_tp_value)
    min_fit_tp = np.min(fit_tp_value)

    if min_flag == 1:
        best = min_fit_tp
        worst = max_fit_tp
        g = np.array([(fit_tp_value[i] - best) / (worst - best) for i in range(len(fit_tp_value))])
    else:
        best = max_fit_tp
        worst = min_fit_tp
        g = np.array([(best - fit_tp_value[i]) / (best - worst) for i in range(len(fit_tp_value))])

    return g


def find_fitness_func_value(test_points, op_point, test_func: TestFunc, options: Options):
    fitness_tp_value = np.empty((options.number_points,))
    idx = 0

    amp_noise = np.random.uniform(-1, 1) * test_func.amp * options.k_noise
    fitness_op_value = test_func.get_value(op_point)
    fitness_op_value += amp_noise

    for i in range(len(test_points)):
        for j in range(len(test_points[i])):
            amp_noise = np.random.uniform(-1, 1) * test_func.amp * options.k_noise
            fitness_tp_value[idx] = test_func.get_value(test_points[i][j])
            fitness_tp_value[idx] += amp_noise
            idx += 1

    return fitness_op_value, fitness_tp_value


def check_delta(delta, op_point, test_func: TestFunc):
    low = test_func.down
    high = test_func.up
    for i in range(len(op_point)):
        real_low = op_point[i] - delta[i][0]
        real_high = op_point[i] + delta[i][1]

        if real_low < low[i]:
            delta[i][0] = op_point[i] - low[i]
        if real_high > high[i]:
            delta[i][1] = high[i] - op_point[i]

    return delta


def move(op_point, nf_norm_val, test_points, delta, options: Options):
    dim = len(op_point)

    new_delta = np.zeros(delta.shape)
    new_op_point = np.zeros(op_point.shape)

    sum_u_p = np.zeros((np.power(2, dim), dim))
    u_norm = np.zeros((dim,))

    u = np.zeros((options.number_points, dim))
    idx = 0
    for i in range(len(test_points)):
        for j in range(len(test_points[i])):
            for d in range(dim):
                temp = test_points[i][j][d]
                u[idx][d] = (temp - op_point[d]) / delta[d][int(temp >= op_point[d])]
                sum_u_p[i][d] += nf_norm_val[idx] * np.power(np.abs(u[idx][d]), options.q)
                u_norm[d] += nf_norm_val[idx] * u[idx][d]
            idx += 1

    for d in range(dim):
        new_op_point[d] = op_point[d] + delta[d][int(u_norm[d] >= 0)] * u_norm[d]

        for i in range(len(new_delta[d])):
            sum_idx = [k for k in range(len(test_points)) if bin(k)[2:].zfill(dim)[d] == str(i)]
            u_p = sum([sum_u_p[j][d] for j in sum_idx])
            new_delta[d][i] = options.g * delta[d][i] * np.power(u_p, 1 / options.q)

    return new_op_point, new_delta


def sac(test_func: TestFunc, options: Options, epsilon=pow(10, -5)):
    op_point, delta = initialization_op_point_and_delta(test_func, options)
    number_measurements = 0

    best_chart = np.zeros((options.max_iter, ))
    x_bests = np.zeros((options.max_iter, len(op_point)))
    stop_iter = options.max_iter

    for i in range(options.max_iter):
        iteration = i + 1

        test_points = initialization_test_point(test_func, options.number_points, delta, op_point)
        fit_op_val, fit_tp_val = find_fitness_func_value(test_points, op_point, test_func, options)
        number_measurements += len(fit_tp_val) + 1

        x_bests = np.copy(op_point)
        best_chart[i] = fit_op_val

        if iteration > 2:
            if (np.sum(delta[0]) < epsilon) or (np.sum(delta[1]) < epsilon):
                stop_iter = iteration
                break

        g = find_g(fit_tp_val, options.min_flag)

        nf_val = find_norm_nuclear_func(g, options)

        # Раскомментировать следующие строки для пошаговой визуализации
        # points = [np.array(p) for p in test_points]
        # visualization(op_point, delta, test_func._function, points, test_func.down, test_func.up, d=0.1)

        op_point, delta = move(op_point, nf_val, test_points, delta, options)
        delta = check_delta(delta, op_point, test_func)

    return x_bests, best_chart, number_measurements, stop_iter


def in_rectangle(point, borders):
    return all(list(map(lambda x, b: True if b[0] <= x <= b[1] else False, point, borders)))


def test():
    from inform_tf import TEST_FUNC_1
    op = Options(20, 400, 1, 1, 50, 4, initialization_mode='center', k_noise=0, min_flag=1)
    tf = TestFunc(TEST_FUNC_1)

    p = 0
    nm = 0
    average_numb_iter = 0
    ep = 0.2
    g_min = tf.global_min
    for i in range(100):
        x_bests, best_chart, number_measurements, stop_iter = sac(tf, op, epsilon=pow(10, -5))
        print('Решение:', x_bests)
        print('Количество итераций:', stop_iter)
        nm += number_measurements
        average_numb_iter += stop_iter
        if (g_min[0] - ep <= x_bests[0] <= g_min[0] + ep) and (g_min[1] - ep <= x_bests[1] <= g_min[1] + ep):
            p += 1
    print('Оценка вероятности:', p / 100.0)
    print('Среднее количество измерений:', nm / 100.0)
    print('Среднее количество итераций:', average_numb_iter / 100.0)


def main():
    test()


if __name__ == '__main__':
    main()
