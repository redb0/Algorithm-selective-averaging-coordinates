from typing import Union

from functions import get_func
from inform_tf import get_inform, is_valid_inform

number = Union[int, float]


class TestFunc:
    def __init__(self, inform):
        self._index = inform['index']
        self._type = inform['type']
        self._number_extrema = inform['number_extrema']
        self._coordinates = inform['coordinates']
        self._func_values = inform['func_values']
        self._degree_smoothness = inform['degree_smoothness']
        self._steepness_coefficients = inform['coefficients_abruptness']  # coefficients_abruptness
        self._constraints_high = inform['constraints_high']
        self._constraints_down = inform['constraints_down']
        self._global_min = inform['global_min']
        self._global_max = inform['global_max']
        self._min_value = inform['min_value']
        self._max_value = inform['max_value']
        self._amp = inform['amp_noise']
        self._dimension = inform['dimension']

        self._function = get_func(inform)

    def __str__(self):
        return f"--------------------\n" \
               f"TestFunc - тестовая функция\n" \
               f"индекс: {self._index}\n" \
               f"тип: {self._type}\n" \
               f"количество экстремумов: {self._number_extrema}\n" \
               f"координаты экстремумов: {self._coordinates}\n" \
               f"значения функции в точках экстремумов: {self._func_values}\n" \
               f"степени гладкости: {self._degree_smoothness}\n" \
               f"коэффициенты крутизны: {self._steepness_coefficients}\n" \
               f"ограничения сверху для каждой координаты: {self._constraints_high}\n"\
               f"ограничения снизу для каждой координаты: {self._constraints_down}\n" \
               f"координаты глобального минимума: {self._global_min}\n" \
               f"координаты глобального максимума: {self._global_max}\n" \
               f"значение в точке глобального минимума: {self._min_value}\n" \
               f"значение в точке глобального максимума: {self._max_value}\n" \
               f"амплитуда сигнала: {self._amp}\n" \
               f"размерность: {self._dimension}\n" \
               f"--------------------"

    def __repr__(self):
        return f'TestFunc(dict(index={self._index}, type={self._type}, ' \
               f'number_extrema={self._number_extrema}, coordinates={self._coordinates}, ' \
               f'func_values={self._func_values}, degree_smoothness={self._degree_smoothness}, ' \
               f'coefficients_abruptness={self._steepness_coefficients}, constraints_high={self._constraints_high}, ' \
               f'constraints_down={self._constraints_down}, global_min={self._global_min}, ' \
               f'global_max={self._global_max}, min_value={self._min_value}, max_value={self._max_value}, ' \
               f'amp_noise={self._amp}, dimension={self._dimension}))'

    @classmethod
    def get_function(cls, idx: int):
        inform = get_inform(idx)
        if inform:
            func = cls(inform)
            return func
        else:
            raise IndexError('Некорректный индекс')

    @classmethod
    def create_function(cls, information: dict):
        if not is_valid_inform(information):
            raise ValueError('Передано неверное описание функции')
        else:
            return cls(information)

    def get_value(self, x):
        if self._function:
            return self._function(x)

    @property
    def index(self) -> int:
        return self._index

    @index.setter
    def index(self, value: int) -> None:
        if value >= 0:
            self._index = value
        else:
            self._index = 1

    @property
    def dim(self) -> int:
        return self._dimension

    @dim.setter
    def dim(self, value: int) -> None:
        if value > 0:
            self._dimension = value
        else:
            raise ValueError('Размерность не может быть <= 0')

    @property
    def func_type(self) -> str:
        return self._type

    @property
    def numb_extrema(self) -> int:
        return self._number_extrema

    @property
    def coordinates(self):
        return self._coordinates

    @property
    def func_values(self):
        return self._func_values

    @property
    def degree_smoothness(self):
        return self._degree_smoothness

    @property
    def steepness_coefficients(self):
        return self._steepness_coefficients

    @property
    def up(self):
        return self._constraints_high

    @property
    def down(self):
        return self._constraints_down

    @property
    def global_min(self):
        return self._global_min

    @property
    def global_max(self):
        return self._global_max

    @property
    def min_value(self):
        return self._min_value

    @property
    def max_value(self):
        return self._max_value

    @property
    def amp(self):
        return self._amp
