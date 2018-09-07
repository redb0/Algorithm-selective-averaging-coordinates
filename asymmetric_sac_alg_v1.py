import numpy as np
from options import Options

# from norm import graph
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


def sac(test_func: TestFunc, options: Options):
    # np.random по умолчанию засеевается временем
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

        # line1 = np.array([[(op_point[0] - delta[0][0]), i - 5] for i in range(11)])
        # line2 = np.array([[(op_point[0] + delta[0][1]), i - 5] for i in range(11)])
        # line3 = np.array([[i - 5, (op_point[1] - delta[1][0])] for i in range(11)])
        # line4 = np.array([[i - 5, (op_point[1] + delta[1][1])] for i in range(11)])
        # graph.graph12(5, op_point, np.array(test_points[0]), np.array(test_points[1]), np.array(test_points[2]), np.array(test_points[3]), line1, line2, line3, line4, g)

        op_point, delta = move(op_point, nf_val, test_points, delta, options)
        delta = check_delta(delta, op_point, test_func)

    return x_bests, best_chart, number_measurements, stop_iter


# def magic(delta, test_points, op_point, new_op_point, nf_val):
#     # angle
#     a1 = np.array([op_point[0] - delta[0][0], op_point[1] - delta[1][0]])
#     a2 = np.array([op_point[0] - delta[0][0], op_point[1] + delta[1][1]])
#     a3 = np.array([op_point[0] + delta[0][1], op_point[1] + delta[1][1]])
#     a4 = np.array([op_point[0] + delta[0][1], op_point[1] - delta[1][0]])
#     a1_idx = get_quad_idx(op_point, a1)
#     a2_idx = get_quad_idx(op_point, a2)
#     a3_idx = get_quad_idx(op_point, a3)
#     a4_idx = get_quad_idx(op_point, a4)
#     quad_indexes = {a1_idx, a2_idx, a3_idx, a4_idx}
#
#     constraints = np.zeros(delta.shape)
#
#     for i in range(len(delta)):
#         constraints[i] = np.array([new_op_point[i] - delta[i][0], new_op_point[i] - delta[i][1]])
#
#     all_points = []
#     func_val = []
#
#     for idx in quad_indexes:
#         points = test_points[idx]
#         for j in range(len(points)):
#             if in_rectangle(points[j], constraints):
#                 all_points.append(points[j])
#                 k = sum([len(nf_val[i]) for i in range(idx)])
#                 k += j
#                 func_val.append(nf_val[k])


def in_rectangle(point, borders):
    return all(list(map(lambda x, b: True if b[0] <= x <= b[1] else False, point, borders)))


def test():
    from inform_tf import TEST_FUNC_1
    op = Options(20, 400, 1, 1, 50, 4, initialization_mode='center', k_noise=0, min_flag=1)
    tf = TestFunc(TEST_FUNC_1)

    p = 0
    # p1 = 0
    nm = 0
    # nm1 = 0
    for i in range(100):
        x_bests, best_chart, number_measurements, stop_iter = sac(tf, op)
        # x_bests1, best_chart1, number_measurements1, stop_iter1 = standard_sac(tf, op)
        print('Новый', x_bests)
        print('Новый', stop_iter)
        # print('Старый', x_bests1)
        # print('Старый', stop_iter1)
        nm += number_measurements
        # nm1 += number_measurements1
        if (-2.2 <= x_bests[0] <= -1.8) and (3.8 <= x_bests[1] <= 4.2):
            p += 1
        # if (-2.2 <= x_bests1[0] <= -1.8) and (3.8 <= x_bests1[1] <= 4.2):
        #     p1 += 1
    print('Вероятность', p)
    print('Среднее количество измерений', nm / 100.0)

    # print('Вероятность', p1)
    # print('Среднее количество измерений', nm1 / 100.0)

    # print(x_bests)
    # print(best_chart[:stop_iter])
    # print(number_measurements)
    # print(stop_iter)


def main():
    test()


if __name__ == '__main__':
    main()
