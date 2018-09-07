import numpy as np
from asymmetric_sac_alg_v1 import find_norm_nuclear_func, find_g
from graph import visualization
from options import Options

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

    delta = np.array([np.maximum(np.abs(op_point[i] - low[i]), np.abs(high[i] - op_point[i])) for i in range(dim)])

    return op_point, delta


def initialization_test_point(test_func: TestFunc, number_point: int, delta, op_point):
    dim = test_func.dim
    test_points = np.zeros((number_point, dim))

    for i in range(number_point):
        for j in range(dim):
            low = op_point[j] - delta[j]
            high = op_point[j] + delta[j]
            test_points[i][j] = np.random.uniform(0, 1) * (high - low) + low

    return test_points


def find_fitness_func_value(test_points, op_point, test_func: TestFunc, options: Options):
    fitness_tp_value = np.empty((options.number_points,))

    amp_noise = np.random.uniform(-1, 1) * test_func.amp * options.k_noise
    fitness_op_value = test_func.get_value(op_point)
    fitness_op_value += amp_noise

    for i in range(len(test_points)):
        amp_noise = np.random.uniform(-1, 1) * test_func.amp * options.k_noise
        fitness_tp_value[i] = test_func.get_value(test_points[i])
        fitness_tp_value[i] += amp_noise

    return fitness_op_value, fitness_tp_value


def move(op_point, nf_norm_val, test_points, delta, options: Options):
    dim = len(op_point)

    new_delta = np.zeros(delta.shape)
    new_op_point = np.zeros(op_point.shape)

    sum_u_p = np.zeros((dim,))
    u_norm = np.zeros((dim,))

    u = np.zeros((options.number_points, dim))
    for i in range(len(test_points)):
        for d in range(dim):
            temp = test_points[i][d]
            u[i][d] = (temp - op_point[d]) / delta[d]
            sum_u_p[d] += nf_norm_val[i] * np.power(np.abs(u[i][d]), options.q)
            u_norm[d] += nf_norm_val[i] * u[i][d]

    for d in range(dim):
        new_op_point[d] = op_point[d] + delta[d] * u_norm[d]
        u_p = np.sum(sum_u_p[d])
        new_delta[d] = options.g * delta[d] * np.power(u_p, 1 / options.q)

    return new_op_point, new_delta


def standard_sac(test_func: TestFunc, options: Options, epsilon=pow(10, -5)):
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
        visualization(op_point, delta, test_func._function, test_points, test_func.down, test_func.up, d=0.1)

        op_point, delta = move(op_point, nf_val, test_points, delta, options)

    return x_bests, best_chart, number_measurements, stop_iter


def test():
    from inform_tf import TEST_FUNC_1
    op = Options(20, 300, 1, 1.2, 10, 4, initialization_mode='center', k_noise=0, min_flag=1)
    tf = TestFunc(TEST_FUNC_1)

    p = 0
    nm = 0
    average_numb_iter = 0
    ep = 0.2
    g_min = tf.global_min
    for i in range(100):
        x_bests, best_chart, number_measurements, stop_iter = standard_sac(tf, op, epsilon=pow(10, -5))
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

