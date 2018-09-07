from typing import Union, List


number = Union[int, float]


class Options:
    def __init__(self, max_iter, number_point, q, gamma, s_factor, nf_index,
                 initialization_mode='center', k_noise=0, min_flag=1):
        self._min_flag = min_flag
        self._max_iter = max_iter
        self._np = number_point
        self._q = q
        self._g = gamma
        self._s_factor = s_factor
        self._idx_nf = nf_index
        self._k_noise = k_noise
        self._initialization_mode = initialization_mode
        self._x = None

    def __repr__(self):
        return f"Options({self._max_iter}, {self._np}, {self._q}, {self._g}, {self._s_factor}, {self._idx_nf}, " \
               f"initialization_mode={self._initialization_mode}, k_noise={self._k_noise}, min_flag={self._min_flag})"

    def __str__(self):
        return f"-------------------------------\n" \
               f"Options - Параметры алгоритма\n" \
               f"Количество точек: {self._np}\n" \
               f"Максимальное количество итераций: {self._max_iter}\n" \
               f"Параметр q: {self._q}\n" \
               f"Параметр gamma: {self._g}\n" \
               f"Коэффициент селективного усреднения: {self._s_factor}\n" \
               f"Индекс тестовой функции: {self._idx_nf}\n" \
               f"Уровень зашумления основного сигнала: {self._k_noise}\n" \
               f"Режим расположения рабочей точки: {self._initialization_mode}\n" \
               f"Флаг минимизации: {self._min_flag}\n" \
               f"-------------------------------"

    @property
    def min_flag(self) -> int:
        return self._min_flag

    @min_flag.setter
    def min_flag(self, value: int) -> None:
        if value <= 0:
            self._min_flag = 0
        else:
            self._min_flag = 1

    @property
    def max_iter(self) -> int:
        return self._max_iter

    @max_iter.setter
    def max_iter(self, value: int) -> None:
        if value >= 1:
            self._max_iter = value
        else:
            self._max_iter = 20

    @property
    def number_points(self) -> int:
        return self._np

    @number_points.setter
    def number_points(self, value: int) -> None:
        if value >= 1:
            self._np = value
        else:
            self._np = 100

    @property
    def q(self) -> number:
        return self._q

    @q.setter
    def q(self, value: number) -> None:
        if value > 0:
            self._q = value
        else:
            self._q = 1

    @property
    def g(self) -> number:
        return self._g

    @g.setter
    def g(self, value: number) -> None:
        if value > 0:
            self._g = value
        else:
            self._g = 1

    @property
    def s_factor(self) -> int:
        return self._s_factor

    @s_factor.setter
    def s_factor(self, value: int) -> None:
        if value > 0:
            self._s_factor = value
        else:
            self._s_factor = 50

    @property
    def index_nf(self) -> int:
        return self._idx_nf

    @index_nf.setter
    def index_nf(self, value: int) -> None:
        if value >= 0:
            self._idx_nf = value
        else:
            self._idx_nf = 1

    @property
    def k_noise(self) -> number:
        return self._k_noise

    @k_noise.setter
    def k_noise(self, value: number) -> None:
        if value >= 0:
            self._k_noise = value
        else:
            self._k_noise = 0

    @property
    def initialization_mode(self) -> str:
        return self._initialization_mode

    @initialization_mode.setter
    def initialization_mode(self, value: str, x: Union[None, List[number]]=None) -> None:
        if value in ['center', 'random']:
            self._initialization_mode = value
        elif value == 'manual':
            if x is not None:
                self._x = x
            else:
                print('Не установлена начальная точка, режим изменен на "center"!')
                self._initialization_mode = 'center'
        else:
            self._initialization_mode = 'center'

    @property
    def get_x(self):
        return self._x


# def main():
#     obj = Options(20, 100, 1, 1, 50, 1, initialization_mode='center', k_noise=0, min_flag=1)
#     print(obj)
#     print(obj.__repr__())
#
# if __name__ == "__main__":
#     main()
