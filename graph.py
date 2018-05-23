import numpy as np

from matplotlib import animation, cm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import ticker

from mpl_toolkits.mplot3d import Axes3D

import test_function
import test_function_range


def get_data(delta, low, up, f_index, dimension):
    if (type(up) == int) or (type(up) == float):
        x = np.arange(low, up, delta)
        y = np.arange(low, up, delta)
    else:
        x = np.arange(low[0], up[0], delta)
        y = np.arange(low[1], up[1], delta)

    X, Y = np.meshgrid(x, y)

    Z = test_function.test_function(np.array([X, Y]), f_index, dimension)

    return X, Y, Z


def draw_isolines(low, up, dimension, f_index):

    delta = 0.05
    # для 4 функции 0.05

    if (type(up) == int) or (type(up) == float):
        x = np.arange(low, up, delta)
        y = np.arange(low, up, delta)
    else:
        x = np.arange(low[0], up[0], delta)
        y = np.arange(low[1], up[1], delta)

    X, Y = np.meshgrid(x, y)

    Z = test_function.test_function(np.array([X, Y]), f_index, dimension)

    levels = np.arange(np.min(Z), np.max(Z), delta)  # * 65

    return X, Y, Z, levels


def data_gen(max_iter, coord, num=0):
    # i = num
    while num < max_iter:
        xlist = np.zeros((len(coord[num]), 1))
        ylist = np.zeros((len(coord[num]), 1))
        for j in range(len(coord[num])):
            xlist[j] = coord[num][j][0]
            ylist[j] = coord[num][j][1]
        num = num + 1
        yield xlist, ylist


def make_init(low, up, xdata, ydata, line, ax):

    def init():
        del xdata[:]
        del ydata[:]
        line.set_data(xdata, ydata)
        if (type(up) == int) or (type(up) == float):
            ax.set_ylim(low, up)
            ax.set_xlim(low, up)
        else:
            ax.set_ylim(low[0], up[0])
            ax.set_xlim(low[0], up[0])
        return line,

    return init


def run(data, line):
    # обновление данных
    xlist, ylist = data
    # для автоматического масштабирования точечного графика раскоментировать следующие строки
    # xmin = np.min(xlist)
    # xmax = np.max(xlist)
    # ymin = np.min(ylist)
    # ymax = np.max(ylist)
    # xmin, xmax = ax.get_xlim()
    # ymin, ymax = ax.get_ylim()
    # ax.set_xlim(xmin, xmax)
    # ax.set_ylim(ymin, ymax)
    # ax.figure.canvas.draw()

    line.set_data(xlist, ylist)

    return line,


# Функция построения анимированного графика поиска агентами оптимума.
# В виде фона используется график изолиний, значения для которого получаеются через функцию draw_isolines.
# Для анимирования используются вспомогательные функции: data_gen, make_init, run.
# data_gen  - генерирует данные для одного кадра анимации (координаты N агентов за 1 итерацию).
# make_init - функция выполняется один раз перед первым кадром, для инициализации начальных значений.
# run       - функция задает значения каждого кадра.
# @param f_index           - индекс функции в файле test_function
# @param rate_change_graph - скорость изменения графика в милисекундах
# @param coord             - трехмерный массив координат агентов по итерациям.
# Вид: coord[индекс итерации 0 - max_iter-1][индекс агента 0 - N-1][индекс измерения 0 - dim-1]
# @param max_iter          - общее количество итераций
def print_graph(f_index, rate_change_graph, coord, max_iter):
    low, up, dim = test_function_range.get_range(f_index)

    # make sure the full paths for ImageMagick and ffmpeg are configured
    rcParams['animation.convert_path'] = r'C:\Program Files\ImageMagick-7.0.7-Q16\magick.exe'
    rcParams['animation.ffmpeg_path'] = r'C:\Program Files\ImageMagick-7.0.7-Q16\ffmpeg.exe'

    # точечный анимированный график для трехмерных функций
    if dim == 2:
        fig, ax = plt.subplots()
        xdata, ydata = [], []
        line, = ax.plot([], [], lw=2, color='r', linestyle=' ', marker='o', label='Агенты')
        plt.legend(loc='upper left')
        X, Y, Z, levels = draw_isolines(low, up, dim, f_index)
        # рисование графика изолиний исходной функции
        CS = plt.contour(X, Y, Z, levels=levels)
        ax.grid()
        # создание анимированного точечного графика
        # blit контролирует используется ли blitting. Если True не будет работать масштабирование и перемещение графика
        ani = animation.FuncAnimation(fig, run, frames=data_gen(max_iter, coord), blit=False,
                                      interval=rate_change_graph, repeat=False,
                                      init_func=make_init(low, up, xdata, ydata, line, ax), fargs=(line,), save_count=max_iter)

        metadata = dict(title='Movie Test', artist='Matplotlib',
                        comment='Movie support!')
        file_name = 'animation_F_' + str(f_index) + '.mpeg'
        rcParams['animation.bitrate'] = 2000
        ani.save(file_name, writer=animation.ImageMagickFileWriter(metadata=metadata, fps=5))  # ImageMagickFileWriter, extra_args=['-vcodec', 'libx264']
        plt.show()


# Функция рисования графика динамики лучших решений по итерациям.
# Если в решениях примутствуют отрицательные значения используется линейная шкала, иначе логарифмическая.
# @param best_chart - массив значений лучших решений (значений оптимизируемой функции) по итерациям
def graph_best_chart(best_chart):
    fig, ax = plt.subplots()
    ax.plot(best_chart, "g", label='Лучшие решения')
    plt.grid(True, color="k")
    if np.min(best_chart) >= 0:
        ax.set_yscale('log', basey=10)
    else:
        ax.set_yscale('linear')
    plt.legend(loc='upper right')
    plt.xlabel("Итерации")
    plt.ylabel("Лучшие значения")
    plt.title("Изменение лучших значений минимизируемой функции", loc='center')

    file_name = 'convergence_F_' + str(13) + '.png'
    plt.savefig(file_name)

    plt.show()


def get_data_func_3d(f_index, delta=0.2):
    low, up, dim = test_function_range.get_range(f_index)

    X, Y, Z = get_data(delta, low, up, f_index, dim)

    return X, Y, Z


def graph_motion_points_3d(f_index, rate_change_graph, coord, max_iter):
    low, up, dim = test_function_range.get_range(f_index)

    # make sure the full paths for ImageMagick and ffmpeg are configured
    rcParams['animation.convert_path'] = r'C:\Program Files\ImageMagick-7.0.7-Q16\magick.exe'
    rcParams['animation.ffmpeg_path'] = r'C:\Program Files\ImageMagick-7.0.7-Q16\ffmpeg.exe'

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xdata, ydata, zdata = [], [], []
    line = ax.scatter([], [], [], marker='o', color='r', label='Агенты')

    plt.legend(loc='upper left')

    # получение данных для рисования фона
    X, Y, Z = get_data_func_3d(f_index, delta=0.1)

    # рисование 3D графика исходной функции как фон
    # plot_surface - сплошная поверхность с подсветкой высот, contour3D - сетка с подсветкой высот (изолинии)
    # plot_wireframe - сетка одного цвета
    ax.plot_surface(X, Y, Z, rstride=4, cstride=4, cmap=cm.jet, alpha=0.5)
    ax.grid()

    # создание анимированного точечного графика
    # blit контролирует используется ли blitting. Если True не будет работать масштабирование и перемещение графика
    ani = animation.FuncAnimation(fig, run_3d, frames=data_gen_for_3d_func(max_iter, coord, f_index, dim), blit=False,
                                  interval=rate_change_graph, repeat=False,
                                  init_func=make_init_3d(low, up, xdata, ydata, zdata, ax, line), fargs=(line, ax))

    plt.show()


def data_gen_for_3d_func(max_iter, coord, f_index, dimension, num=0):
    while num < max_iter:
        # не работает если написать так np.zeros((len(coord[num]), 1))
        xlist = np.zeros((len(coord[num]),))
        ylist = np.zeros((len(coord[num]),))
        zlist = np.zeros((len(coord[num]),))
        for j in range(len(coord[num])):
            xlist[j] = coord[num][j][0]
            ylist[j] = coord[num][j][1]
            zlist[j] = test_function.test_function(coord[num][j], f_index, dimension)
        num = num + 1
        yield xlist, ylist, zlist


def make_init_3d(low, up, xdata, ydata, zdata, ax,  line):

    def init():
        del xdata[:]
        del ydata[:]
        del zdata[:]
        line._offsets3d = (xdata, ydata, zdata)
        if (type(up) == int) or (type(up) == float):
            ax.set_ylim(low, up)
            ax.set_xlim(low, up)
        else:
            ax.set_ylim(low[1], up[1])
            ax.set_xlim(low[0], up[0])
        return line

    return init


def run_3d(data, line, ax):
    # обновление данных
    xlist, ylist, zlist = data
    # для автоматического масштабирования точечного графика раскоментировать следующие строки
    # xmin = np.min(xlist)
    # xmax = np.max(xlist)
    # ymin = np.min(ylist)
    # ymax = np.max(ylist)
    # xmin, xmax = ax.get_xlim()
    # ymin, ymax = ax.get_ylim()
    # ax.set_xlim(xmin, xmax)
    # ax.set_ylim(ymin, ymax)

    # ax.set_zlim(np.min(zlist), np.max(zlist))
    # ax.figure.canvas.draw()

    # line.set_data(xlist, ylist)
    # line.set_3d_properties(zlist)

    line._offsets3d = (xlist, ylist, zlist)

    return line


def comparative_graph_convergence(f_index, **best_chart_alg):
    """Функция построения сравнительного графика сходимости алгоритмов."""
    colors = ['b', 'g', 'r', 'm', 'k', 'y', 'c']
    lisestyles = ['-', '--', '-.']
    markers = ['o', 's', '^', 'x', 'p', 'v']
    fig, ax = plt.subplots()
    plt.grid(True, color="k")
    min_char = []
    for b in best_chart_alg:
        min_char.append(np.min(best_chart_alg.get(b)))

    if np.min(min_char) >= 0:
        ax.set_yscale('log', basey=10)
    else:
        ax.set_yscale('linear')
    plt.xlabel("Итерации")
    plt.ylabel("Лучшие значения")
    plt.title("Изменение лучших значений минимизируемой функции", loc='center')

    # ax.set_yscale('log', basey=10)
    ind = 0
    for alg in best_chart_alg:
        plt.plot(best_chart_alg.get(alg), color=colors[ind], lw=1, linestyle=lisestyles[ind])  #  marker=markers[ind],

        if ind <= len(lisestyles):
            ind = ind + 1
        else:
            ind = 0

    plt.legend(best_chart_alg.keys(), loc='upper right')
    file_name = 'convergence_F_' + str(f_index) + '.png'
    plt.savefig(file_name)

    plt.show()


def graph_time(mean_time, N, f_index, max_iter, number_of_runs):
    fig, ax = plt.subplots()
    ax.plot(mean_time, "g", label='Время', marker='o', lw=2)
    plt.grid(True, color="k")
    plt.legend(loc='upper right')
    plt.xlabel("Количество точек")
    plt.ylabel("Время, сек")
    plt.title("Зависимость сремени выполнения от количества точек", loc='center')
    # Создаем экземпляр класса, который будет отвечать за расположение меток
    locator = ticker.MultipleLocator(base=1)
    # Установим локатор для главных меток
    ax.xaxis.set_major_locator(locator)

    # Создаем форматер
    formatter = ticker.FixedFormatter([0] + N)
    # Установка форматера для оси X
    ax.xaxis.set_major_formatter(formatter)
    file_name = 'time_F_' + str(f_index) + '_Iter_' + str(max_iter) + '_Runs_' + str(number_of_runs) + '.png'
    plt.savefig(file_name)

    plt.show()


def graph_probability(probability, epsilon, N, f_index, max_iter, number_of_runs, title, xlabel):
    colors = ['b', 'g', 'r', 'm', 'k', 'y', 'c']
    lisestyles = ['-', '--', '-.', ':']
    markers = ['o', 'x', 'D', '^', 'd']
    file_name = title + '_F_' + str(f_index) + '_Iter_' + str(max_iter) + '_Runs_' + str(number_of_runs) + '.png'
    fig, ax = plt.subplots()
    ax.grid(True, color="k")
    params = {'legend.fontsize': 'medium',
              'figure.figsize': (15, 5),
              'axes.labelsize': 'medium',
              'axes.titlesize': 'medium',
              'xtick.labelsize': 'large',
              'ytick.labelsize': 'large'}
    plt.rcParams.update(params)

    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel("Оценка вероятности", fontsize=13)
    plt.title("Зависимость оценки вероятности попадания в экстремум от количества точек", loc='center')
    ind = 0
    for k in range(len(epsilon)):
        ax.plot(probability[:, k], color=colors[ind], lw=1.5, marker=markers[k],
                linestyle=lisestyles[ind], label=str(epsilon[k]))  # marker=markers[ind],
        if ind <= len(lisestyles):
            ind = ind + 1
        else:
            ind = 0
    plt.legend(title=title, loc=(1, 0.4))  # loc='upper right' 'center right'

    # Создаем экземпляр класса, который будет отвечать за расположение меток
    locator = ticker.MultipleLocator(base=1)
    # Установим локатор для главных меток
    ax.xaxis.set_major_locator(locator)

    # Создаем форматер
    formatter = ticker.FixedFormatter([0] + N)
    # Установка форматера для оси X
    ax.xaxis.set_major_formatter(formatter)
    plt.savefig(file_name, bbox_inches='tight')
    plt.show()


def graph_probability_with_error(probability, epsilon, N, f_index, max_iter, number_of_runs, errors, title):
    colors = ['b', 'g', 'r', 'm', 'k', 'y', 'c']
    lisestyles = ['-', '--', '-.', ':']
    markers = ['o', 'x', 'D', '^', 'd']
    file_name = title + '_F_' + str(f_index) + '_Iter_' + str(max_iter) + '_Runs_' + str(number_of_runs) + '.png'
    fig, ax = plt.subplots()
    ax.grid(True, color="k")
    params = {'legend.fontsize': 'medium',
              'figure.figsize': (15, 5),
              'axes.labelsize': 'medium',
              'axes.titlesize': 'medium',
              'xtick.labelsize': 'large',
              'ytick.labelsize': 'large'}
    plt.rcParams.update(params)

    plt.xlabel("Отношение шум/сигнал", fontsize=13)
    plt.ylabel("Оценка вероятности", fontsize=13)
    plt.title("Зависимость оценки вероятности попадания в экстремум от влияния шума", loc='center')
    ind = 0
    for k in range(len(epsilon)):
        ax.plot(probability[k], color=colors[ind], lw=1.5, marker=markers[k],
                linestyle=lisestyles[ind], label=str(epsilon[k]))  # marker=markers[ind],
        if ind <= len(lisestyles):
            ind = ind + 1
        else:
            ind = 0
    plt.legend(title=title, loc=(1, 0.4))  # loc='upper right' 'center right'

    # Создаем экземпляр класса, который будет отвечать за расположение меток
    locator = ticker.MultipleLocator(base=1)
    # Установим локатор для главных меток
    ax.xaxis.set_major_locator(locator)

    # Создаем форматер
    formatter = ticker.FixedFormatter([0] + errors)
    # Установка форматера для оси X
    ax.xaxis.set_major_formatter(formatter)
    plt.savefig(file_name, bbox_inches='tight')
    plt.show()
