from typing import List

import numpy as np


def get_bocharov_feldbaum_func(n: int, a: List[List[float]], c: List[List[float]],
                               p: List[List[float]], b: List[float]):
    def func(x):
        l = []
        for i in range(n):
            res = 0
            for j in range(len(x)):
                res = res + a[i][j] * np.abs(x[j] - c[i][j]) ** p[i][j]
            res = res + b[i]
            l.append(res)
        res = np.array(l)
        return np.min(res)
    return func


def get_hyperbolic_potential_abs(n: int, a: List[float], c: List[List[float]],
                                 p: List[List[float]], b: List[float]):
    def func(x):
        value = 0
        for i in range(n):
            res = 0
            for j in range(len(x)):
                res = res + np.abs(x[j] - c[i][j]) ** p[i][j]
            res = a[i] * res + b[i]
            res = -(1 / res)
            value = value + res
        return value
    return func


def get_hyperbolic_potential_sqr(n: int, a: List[List[float]], c: List[List[float]], p, b):
    def func(x):
        value = 0
        for i in range(n):
            res = 0
            for j in range(len(x)):
                res = res + a[i][j] * (x[j] - c[i][j]) ** 2  # правильно ли стоит a???????
            res = res + b[i]
            res = -(1 / res)
            value = value + res
        return value
    return func


def get_exponential_potential(n: int, a: List[float], c: List[List[float]],
                              p: List[List[float]], b: List[float]):
    def func(x):
        value = 0
        for i in range(n):
            res = 0
            for j in range(len(x)):
                res = res + np.abs(x[j] - c[i][j]) ** p[i][j]
            res = (-b[i]) * np.exp((-a[i]) * res)
            value = value + res
        return value
    return func


default_type_function = {
    'bocharov_feldbaum': get_bocharov_feldbaum_func,
    'hyperbolic_potential_abs': get_hyperbolic_potential_abs,
    'hyperbolic_potential_sqr': get_hyperbolic_potential_sqr,
    'exponential_potential': get_exponential_potential
}


def get_func(inform: dict):
    for key, func in default_type_function.items():
        if inform['type'] == key:
            return func(inform['number_extrema'], inform['coefficients_abruptness'],
                        inform['coordinates'], inform['degree_smoothness'], inform['func_values'])
