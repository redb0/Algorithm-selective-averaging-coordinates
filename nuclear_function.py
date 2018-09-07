import numpy as np


def get_nuclear_func_val(nuclear_func_index: int, g, selectivity_factor):
    """
    Функция вычисления знаячения ядра.
    Тип ядра, а также коэффициет селективности (selectivity_factor) определяют,
    как быстро будет убывать функция, 
    соответственно как сильно будет выделяться точка в сторону которой необходимо двигаться.
    Чем ближе g к 1, тем результат ближе к 0.
    Ядра:
        1) Линейное
        2) Параболическое
        3) Кубическое
        4) Экспоненциальное
    
    Пример вызова:
        >>> get_nuclear_func_val(1, 0.2, 10)
        >>> 0.10737418240000006
        >>> get_nuclear_func_val(1, 0.2, 1)
        >>> 0.8
        >>> get_nuclear_func_val(1, 1, 10)
        >>> 0
    
    :param nuclear_func_index: индекс ядерной фукции
    :param g: усредненное значение (аргумент ядра), должен нвходиться в отрезке [0, 1]
    :param selectivity_factor: коэффициент селективности
    :return: значение ядра
    """

    # линейное
    if nuclear_func_index == 1:
        return (1 - g)**selectivity_factor

    # параболическое
    if nuclear_func_index == 2:
        return (1 - g**2)**selectivity_factor

    # кубическо
    if nuclear_func_index == 3:
        return (1 - g**3)**selectivity_factor

    # экспоненциальное
    if nuclear_func_index == 4:
        return np.exp(-selectivity_factor * g)

    # гипербалическое
    # if nuclear_func_index == 5:
    #     return g ** (-selectivity_factor)
