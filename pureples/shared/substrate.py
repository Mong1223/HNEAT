"""
Класс для субстрата, то есть сети являющейся фенотипом.
"""


class Substrate(object):
    """
    Класс который представляет субстрат: На вход координаты нейронов слоёв и значение разрешения сети.
    """

    def __init__(self, input_coordinates, output_coordinates, hidden_coordinates=(), res=10.0):
        self.input_coordinates = input_coordinates
        self.hidden_coordinates = hidden_coordinates
        self.output_coordinates = output_coordinates
        self.res = res
