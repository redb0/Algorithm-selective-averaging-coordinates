from time import time
import numpy as np

import graph
from sac_alg import SAC
from test_function_range import get_range


def get_true_min(f_index, dim):
    if f_index == 4:
        true_min = np.array([4, 4])
    if f_index == 5:
        true_min = np.array([-2, 4])
    if f_index == 6:
        true_min = np.array([1 for i in range(dim)])
    if f_index == 7:
        true_min = np.array([0 for i in range(dim)])
    if f_index == 8:
        true_min = np.array([0 for i in range(dim)])
    if f_index == 9:
        true_min = np.array([0 for i in range(dim)])
    if f_index == 10:
        true_min = np.array([-32, -32])
    if f_index == 11:
        true_min = np.array([0, 0])
    if f_index == 12:
        true_min = np.array([9, 8.6])
    if f_index == 13:
        true_min = np.array([0, 0])
    if f_index == 14:
        true_min = np.array([-5, -5])
    if f_index == 15:
        true_min = np.array([3, -5])
    if f_index == 16:
        true_min = np.array([2, -3])
    if f_index == 17:
        true_min = np.array([-5, -5])
    if f_index == 18:
        true_min = np.array([-2, -5])
    if f_index == 19:
        true_min = np.array([-1, 3])
    return true_min


def main():
    min_flag = 1
    max_iter = 10

    # N = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    N = [50]
    gamma = 1.2
    selectivity_factor = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
    nuclear_func_index = [1, 2, 3, 4]
    q = 2
    epsilon = 5 * pow(10, -8)
    eps = [0.25]  # 0.05, 0.1, 0.25
    k_noise = 0

    number_of_runs = 100
    # functions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]
    functions = [5, 11, 15]
    probability = np.zeros((len(selectivity_factor), len(nuclear_func_index)))  # N eps
    mean_time = []

    for j in range(len(functions)):
        f_index = functions[j]
        low, up, dim = get_range(f_index)

        for n in range(len(N)):

            for sf in range(len(selectivity_factor)):
                for nf in range(len(nuclear_func_index)):
                    file_name = "test_data_F" + str(f_index) + str(N[n]) + str(selectivity_factor[sf]) + str(nuclear_func_index[nf]) + ".txt"
                    f_name = 'F' + str(f_index)
                    file = open(file_name, 'w')

                    time_runs = np.zeros((number_of_runs,))
                    agents = np.zeros((number_of_runs, dim))

                    file.write("N = " + str(N[n]) + "; max_iter = " + str(max_iter) + "; selectivity_factor = " + str(selectivity_factor[sf]) + "; q = " + str(q) + "; ядро = " + str(nuclear_func_index[nf]) + "\n")

                    f = []
                    stop_it_mean = []

                    for i in range(number_of_runs):
                        print("Прогон: " + str(i + 1) + "...")

                        start = time()
                        res = SAC(f_index, max_iter, N[n], min_flag, nuclear_func_index[nf], selectivity_factor[sf], gamma, q, epsilon, k_noise)
                        end = time()

                        func_best, agent_best, best_chart, mean_chart, coord_x, stop_iteration, last_value_func, coord_x_test = res

                        time_runs[i] = end - start
                        agents[i] = agent_best

                        f.append(func_best)
                        stop_it_mean.append(stop_iteration)

                        if i == 0:
                            f_min = func_best
                        else:
                            if f_min > func_best:
                                f_min = func_best

                        # print(timeit.timeit(stmt=GSA(f_index, N, max_iter, elitist_check, min_flag, r_power), number=1))
                        print(str(time_runs[i]))

                        s = f_name + ": Fopt = " + str(func_best) + "; Xopt = " + str(agent_best) + "; time = " + str(time_runs[i]) + "; stop_iteration = " + str(stop_iteration) + "\n"
                        file.write(s)

                    true_min = get_true_min(f_index, dim)
                    mean_time.append(np.mean(time_runs))

                    s = "Лучший результат прогонов: " + str(min(f)) + "\n"
                    file.write(s)

                    s = "Среднее время поиска: " + str(np.mean(time_runs)) + "\n"
                    file.write(s)

                    s = "Среднее количество итераций: " + str(np.mean(stop_it_mean)) + "\n"
                    file.write(s)

                    for c in range(len(eps)):
                    # for c in range(len(nuclear_func_index)):
                        # t = True
                        p = 0
                        for d in range(len(agents)):
                            t = True
                            for k in range(len(true_min)):
                                a = true_min[k] + eps[c] >= agents[d][k]
                                b = true_min[k] - eps[c] <= agents[d][k]
                                # a = true_min[k] + eps[0] >= agents[d][k]
                                # b = true_min[k] - eps[0] <= agents[d][k]
                                if a and b:
                                    t = t * True
                                else:
                                    t = t * False

                            if t == 1:
                                p = p + 1
                        probability[sf][nf] = (p / number_of_runs)
                        s = "Вероятность попадания в epsilon(" + str(eps[c]) + ") окрестность экстремума: " + str(
                            probability[sf][nf]) + "\n"
                        # probability[n][c] = (p / number_of_runs)
                        # s = "Вероятность попадания в epsilon(" + str(eps[c]) + ") окрестность экстремума: " + str(
                        #     probability[n][c]) + "\n"
                        file.write(s)

                    file.close()

        print("Конец прогонов для функции: F" + str(f_index) + ".")

        graph.graph_probability(probability, nuclear_func_index, selectivity_factor, f_index, max_iter, number_of_runs, "Ядро", "Коэффициент селективности")
    # graph.graph_probability(probability, eps, N, f_index, max_iter, number_of_runs, "epsilon")
    # graph.graph_time(mean_time, N, f_index, max_iter, number_of_runs)

    print("Конец всех прогонов.")


if __name__ == "__main__":
    main()

