# Algorithm selective averaging coordinates (SAC)

Реализация алгоритма селективного усреднения координат (SAC). 


More information [here](https://cyberleninka.ru/article/v/metod-globalnoy-optimizatsii-osnovannyy-na-selektivnom-usrednenii-koordinat-pri-nalichii-ogranicheniy).

## Последовательность

1) Инициализация рабочей точки, размеров delta-области;
2) Основной цикл:
    1) Инициализация тестовых точек;
    2) Измерение значений целевой функции
    3) Проверка условия останова (размер дельта области, изменение значения целевой функции, 
    количество итераций, любой другой разумный критерий);
    4) Вычисление нормированных на [0, 1] значений целевой функции;
    5) Вычисление нормированных на [0, 1] весов тестовых точек (используя ядерную функцию);
    6) Вычисление новых координат рабочей точки;
    7) Вычисление новых размеров delta-области;
    8) Повторение шагов i - viii до выполнения критерия останова.

## Тестовые функции

Для конструирования своих тестовых функций (Бочарова-Фельдбаума, гиперболических, 
експоненциальных потенциалов) можно использовать [этот проект](https://github.com/redb0/tf-generator). 

Предлагаемые методы позволяют очень гибко конструировать (настраивать количество эктстремумов, 
их расположение, величину, и характер функции в их окрестностях) тестовые примеры.

## Пример

Для работы вам может понадобиться библиотеки: `numpy`, `matplotlib`.

По умолчанию будет запущено 100 прогонов для определения оценки вероятности попадания 
в окрестность глобального минимума, определения среднего количества измерений 
и среднего количества итераций.

```commandline
git clone https://github.com/redb0/Algorithm-selective-averaging-coordinates
cd ./SAC
python standard_sac_alg.py
```

Для пошаговой визуализации раскомментируйте указанные строки в файле с реализацией интересующего алгоритма.
Не рекомендуется запускать тесты алгоритма с активной визуализацией.

Ниже представлена визуализация первых 4 итерация алгоритма:

![Первая итерация алгоритма](https://github.com/redb0/Algorithm-selective-averaging-coordinates/blob/master/png/standard_sac_step_1.png)

![Вторая  итерация алгоритма](https://github.com/redb0/Algorithm-selective-averaging-coordinates/blob/master/png/standard_sac_step_2.png)

![Третья  итерация алгоритма](https://github.com/redb0/Algorithm-selective-averaging-coordinates/blob/master/png/standard_sac_step_3.png)

![Четвертая итерация алгоритма](https://github.com/redb0/Algorithm-selective-averaging-coordinates/blob/master/png/standard_sac_step_4.png)

Полная работа алгоритма:

![Анимация работы алгоритма](https://github.com/redb0/Algorithm-selective-averaging-coordinates/blob/master/png/Animation.gif)


## Структура файлов

`functions.py` - реализация конструкторов тестовых функций.

`inform_tf.py` - файл с тестовыми функциями.

`standard_sac_alg.py` - реализация стандартного алгоритма.

`nuclear_function.py` - реализация ядерных функций.

`asymmetric_sac_alg_v1.py` - реализация алгоритма с модификацией № 1.

`asymmetric_sac_alg_v2.py` - реализация алгоритма с модификацией № 2.

Вспомогательные:

`options.py` - класс для описания настроек алгоритма.

`test_functions.py` - класс для описания тестовой функции.

`graph.py` - реализация визуализации.

