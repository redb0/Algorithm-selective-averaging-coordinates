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
    # N = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
    # N = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    N = [500]
    max_iter = 200
    min_flag = 1  # 1 - минимизация, 0 - максимизация
    gamma = 1.2
    selectivity_factor = 300
    nuclear_func_index = 1
    q = 2
    epsilon = 5 * pow(10, -8)

    # k_noise = 1
    k_noise = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
    # k_noise = [0, 0.2, 0.4]
    # k_noise = [1, 2, 3, 4, 5]

    number_of_runs = 100
    # epsilon = [0.05, 0.1, 0.25]
    # epsilon = [0.25, 0.5]
    epsilon = [0.5]

    # probability = np.zeros((len(N), len(r_power)))
    # probability = np.zeros((len(N), len(epsilon)))
    # probability = np.zeros((len(N), len(k_noise)))
    # probability = np.zeros((len(epsilon), len(k_noise)))
    time_runs = np.zeros((number_of_runs,))
    mean_time = []

    functions = [5] # 5 16    , 15, 16
    probability = np.zeros((len(functions), len(k_noise)))

    for j in range(len(functions)):  # перебор функций
        f_index = functions[j]
        low, up, dim = get_range(f_index)

        for r in range(len(k_noise)):  # r_power
            for n in range(len(N)):  # перебор коичества точек
                # file_name = "test_data_F" + str(f_index) + "_N" + str(N[n]) + "_r_power" + str(r_power[r]) + ".txt" # r_power[r]
                file_name = "test_data_F" + str(f_index) + "_N" + str(N[n]) + "_error" + str(k_noise[r]) + ".txt"
                f_name = 'F' + str(f_index)
                file = open(file_name, 'w')

                agents = np.zeros((number_of_runs, dim))

                # file.write(
                #     "N = " + str(N[n]) + "; max_iter = " + str(max_iter) + "; r_power = " + str(r_power[r]) + "\n")

                file.write(
                    "N = " + str(N[n]) + "; max_iter = " + str(max_iter) + "; error = " + str(k_noise[r]) + "\n")

                for i in range(number_of_runs):
                    print("Прогон: " + str(i + 1) + "...")

                    start = time()
                    res = SAC(f_index, max_iter, N[n], min_flag, nuclear_func_index, selectivity_factor, gamma,
                              q, epsilon, k_noise[r])
                    end = time()

                    func_best, agent_best, best_chart, _, coord_x, stop_iteration, last_value_func, _ = res

                    time_runs[i] = end - start
                    print(str(time_runs[i]))
                    print(str(stop_iteration))
                    print("func_best=" + str(func_best) + " - " + "last_value_func=" + str(last_value_func))
                    print(coord_x[-1])

                    agents[i] = agent_best

                    if i == 0:
                        f_min = func_best
                    else:
                        if f_min > func_best:
                            f_min = func_best

                    s = f_name + ": Fopt = " + str(func_best) + "; Xopt = " + str(agent_best) + "\n"
                    file.write(s)

                true_min = get_true_min(f_index, dim)
                mean_time.append(np.mean(time_runs))

                s = "Лучший результат прогонов: " + str(f_min) + "\n"
                file.write(s)
                s = "Среднее время поиска: " + str(np.mean(time_runs)) + "\n"
                file.write(s)

                for c in range(len(epsilon)):
                    # t = True
                    p = 0
                    for d in range(len(agents)):
                        t = True
                        for k in range(len(true_min)):
                            a = true_min[k] + epsilon[c] >= agents[d][k]
                            b = true_min[k] - epsilon[c] <= agents[d][k]
                            if a and b:
                                t = t * True
                            else:
                                t = t * False

                        if t == 1:
                            p = p + 1

                    # probability[n][c] = (p / number_of_runs)
                    # print("Вероятность попадания в экстремум epsilon(" + str(epsilon[c]) + ") при N = " + str(
                    #     N[n]) + ": " + str(probability[n][c]) + ".")

                    # probability[n][r] = (p / number_of_runs)
                    # print("Вероятность попадания в экстремум epsilon(" + str(epsilon[c]) + ") при N = " + str(N[n]) + ": " + str(probability[n][r]) + ".")

                    probability[c][r] = (p / number_of_runs)
                    print("Вероятность попадания в экстремум epsilon(" + str(epsilon[c]) + ") при N = " + str(
                        N[n]) + " и ошибке = " + str(k_noise[r]) + ": " + str(probability[c][r]) + ".")

                    s = "Вероятность попадания в epsilon(" + str(epsilon[c]) + ") окрестность экстремума: " + str(
                        p / number_of_runs) + "\n"
                    file.write(s)

                file.close()
                print("Конец прогонов для функции: F" + str(f_index) + " при N = " + str(N[n]) + ".")

        # graph.graph_probability(probability, r_power, N, f_index, max_iter, number_of_runs, "r_power")

        # graph.graph_probability_with_error(probability, epsilon, N, f_index, max_iter, number_of_runs, k_noise,
        #                                    "epsilon")

        # graph.graph_probability(probability, epsilon, N, f_index, max_iter, number_of_runs, "epsilon")
        # graph.graph_time(mean_time, N, f_index, max_iter, number_of_runs)

    graph.graph_probability_with_error(probability, functions, N, f_index, max_iter, number_of_runs, k_noise,
                                       "Func")

    print("Конец всех прогонов.")


if __name__ == "__main__":
    main()
