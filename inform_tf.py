from functions import default_type_function


required_fields = ['type', 'index', 'number_extrema', 'coordinates', 'func_values',
                   'degree_smoothness', 'coefficients_abruptness', 'constraints_high',
                   'constraints_down', 'global_min', 'global_max', 'min_value',
                   'max_value', 'amp_noise', 'dimension']


TEST_FUNC_1 = {  # number 5
    "type": "bocharov_feldbaum",
    "index": 1,
    "number_extrema": 10,  # n
    "coordinates": [  # c
        [-2, 4], [0, 0], [4, 4], [4, 0], [-2, 0],
        [0, -2], [-4, 2], [2, -4], [2, 2], [-4, -2]
    ],
    "func_values": [0, 3, 5, 6, 7, 8, 9, 10, 11, 12],  # b
    "degree_smoothness": [  # p
        [0.6, 1.6], [1.6, 2], [0.6, 0.6], [1.1, 1.8], [0.5, 0.5],
        [1.3, 1.3], [0.8, 1.2], [0.9, 0.3], [1.1, 1.7], [1.2, 0.5]
    ],
    "coefficients_abruptness": [  # a
        [6, 6], [6, 7], [6, 7], [5, 5], [5, 5],
        [5, 5], [4, 3], [2, 4], [6, 4], [3, 3]
    ],
    "constraints_high": [6, 6],
    "constraints_down": [-6, -6],
    "global_min": [-2, 4],
    "global_max": [-6, -6],
    "min_value": 0.0,
    "max_value": 24.89,
    "amp_noise": 12.445,
    "dimension": 2
}

default_test_func = [TEST_FUNC_1]


def get_inform(idx: int):
    for tf in default_test_func:
        if idx == tf['index']:
            return tf
    return None


def validation_type(t: str) -> bool:
    if t in default_type_function.keys():
        return True
    return False


def presence_fields(inform: dict) -> bool:
    if set(inform.keys()) == set(required_fields):
        return True
    return False


def validation_coordinate(coordinate, dim: int) -> bool:
    if len(coordinate) == dim:
        for c in coordinate:
            if not (isinstance(c, int) or isinstance(c, float)):
                return False
        return True
    return False


def validation_sublist_len(l, length: int) -> bool:
    for item in l:
        if len(item) != length:
            return False
    return True


def is_valid_inform(inform: dict) -> bool:
    if not presence_fields(inform):
        return False

    t = inform['type']
    dim = inform['dimension']
    numb_extrema = inform['number_extrema']
    coordinates = inform['coordinates']
    ds = inform['degree_smoothness']
    steepness_coefficients = inform['coefficients_abruptness']

    if not validation_type(t):
        return False
    elif not isinstance(inform['index'], int) and (inform['index'] < 0):
        return False
    elif not isinstance(dim, int) and (dim <= 0):
        return False
    elif (inform['amp_noise'] < 0) or (numb_extrema <= 0):
        return False
    elif not validation_coordinate(inform['global_min'], dim):
        return False
    elif not validation_coordinate(inform['global_max'], dim):
        return False
    elif len(inform['constraints_high']) != len(inform['constraints_down']) != 2:
        return False
    elif len(coordinates) != len(inform['func_values']) != len(steepness_coefficients) != numb_extrema:
        return False
    elif not validation_sublist_len(coordinates, dim):
        return False
    elif (t != 'hyperbolic_potential_sqr') and (not validation_sublist_len(ds, dim) or len(ds) != numb_extrema):
        return False
    elif (t != 'hyperbolic_potential_abs') and not validation_sublist_len(steepness_coefficients, dim):
        return False
    else:
        return True
