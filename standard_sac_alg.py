import numpy as np
from asymmetric_sac_alg_v1 import find_norm_nuclear_func, find_g
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


def standard_sac(test_func: TestFunc, options: Options):
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
            if (np.sum(delta[0]) < pow(10, -5)) or (np.sum(delta[1]) < pow(10, -5)):
                stop_iter = iteration
                break

        g = find_g(fit_tp_val, options.min_flag)

        nf_val = find_norm_nuclear_func(g, options)

        # line1 = np.array([[(op_point[0] - delta[0]), i - 5] for i in range(11)])
        # line2 = np.array([[(op_point[0] + delta[0]), i - 5] for i in range(11)])
        # line3 = np.array([[i - 5, (op_point[1] - delta[1])] for i in range(11)])
        # line4 = np.array([[i - 5, (op_point[1] + delta[1])] for i in range(11)])
        # graph.graph13(5, op_point, test_points, line1, line2, line3, line4, g)

        op_point, delta = move(op_point, nf_val, test_points, delta, options)
        # delta = check_delta(delta, op_point, test_func)

    return x_bests, best_chart, number_measurements, stop_iter


def test():
    from inform_tf import TEST_FUNC_1
    op = Options(20, 300, 1, 1.2, 10, 4, initialization_mode='center', k_noise=0, min_flag=1)
    tf = TestFunc(TEST_FUNC_1)

    p = 0
    # p1 = 0
    nm = 0
    # nm1 = 0
    for i in range(100):
        x_bests, best_chart, number_measurements, stop_iter = standard_sac(tf, op)
        # x_bests1, best_chart1, number_measurements1, stop_iter1 = sac(tf, op)
        # print('Новый', x_bests1)
        # print('Новый', stop_iter1)
        print('Старый', x_bests)
        print('Старый', stop_iter)
        nm += number_measurements
        # nm1 += number_measurements1
        if (-2.2 <= x_bests[0] <= -1.8) and (3.8 <= x_bests[1] <= 4.2):
            p += 1
        # if (-2.2 <= x_bests1[0] <= -1.8) and (3.8 <= x_bests1[1] <= 4.2):
        #     p1 += 1
    print('Старый Вероятность', p)
    print('Старый Среднее количество измерений', nm / 100.0)

    # print('Новый Вероятность', p1)
    # print('Новый Среднее количество измерений', nm1 / 100.0)

    # print(x_bests)
    # print(best_chart[:stop_iter])
    # print(number_measurements)
    # print(stop_iter)


def main():
    test()


if __name__ == '__main__':
    main()

