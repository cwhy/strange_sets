# coding: utf-8
from typing import Tuple, Optional

import numpy as np
from MLkit.dataset import CategoricalDataSet, BaseDataSet, DataSetType, Encoding


def draw_square(xc, yc, w):
    _dot_area = ([], [])
    r = int(w // 2)
    xc = int(xc)
    yc = int(yc)
    for x in range(xc - r, xc - r + w):
        for y in range(yc - r, yc - r + w):
            _dot_area[0].append(x)
            _dot_area[1].append(y)
    return _dot_area


class CountSquares(BaseDataSet):
    def __init__(self,
                 _dim_X: Tuple[int, int] = (64, 64),
                 max_y: int = 20,
                 max_size: Optional[int] = 5,
                 encode_y:Encoding= Encoding.T_1HOT,
                 flatten_x:bool=False,
                 name="count_squares"):

        self.flatten_x = flatten_x
        self._name = name
        self._dim_X = list(_dim_X)
        self._dim_x = np.prod(self.dim_X)
        
        self.encode_y = encode_y
        self._dim_y = max_y
        self._dim_Y = [max_y]
        self.max_size = max_size
        if self.max_size is None:
            self.max_size = int(self.dim_x / max_y / max_y + 3)
        else:
            self.max_size = max_size

    @property
    def dim_X(self):
        return self._dim_X

    @property
    def dim_Y(self):
        return self._dim_Y

    @property
    def dim_x(self):
        return self._dim_x

    @property
    def dim_y(self):
        return self._dim_y

    @property
    def name(self):
        return self._name

    def sample(self, mb_size, name_mod: Optional[str]=None) -> DataSetType:
        x_array = np.zeros([mb_size] + self.dim_X)
        y_array = np.zeros((mb_size, 1))
        for i in range(mb_size):
            x_array[i, :, :], y_array[i, :] = self._get_one()
        if name_mod is None:
            _name = self.name
        else:
            _name = self.name + '_' + name_mod
        base_set = CategoricalDataSet(x_array, y_array,
                                      n_classes=self.dim_y,
                                      name=_name,
                                      y_encoding=Encoding.T_DENSE)
        if self.flatten_x:
            base_set = base_set.flatten_x()
        if self.encode_y == Encoding.T_1HOT:
            return base_set.update_y_1hot()
        elif self.encode_y == Encoding.T_DENSE:
            return base_set
        else:
            raise Exception("Wrong encoding option for y")

    def _get_one(self):
        base = np.zeros(self.dim_X)
        dots_ctr_0 = np.random.randint(self.dim_X[0], size=self.dim_y)
        dots_ctr_1 = np.random.randint(self.dim_X[1], size=self.dim_y)
        dot_size = np.random.randint(1, self.max_size, size=self.dim_y)
        dots_area_lists = ([], [])
        n_dots = 0
        for i in range(self.dim_y):
            dot_area = draw_square(dots_ctr_0[i] - dot_size[i] / 2,
                                   dots_ctr_1[i] - dot_size[i] / 2,
                                   dot_size[i])
            add_flag = True
            for xd, da in enumerate(dot_area):
                for d in da:
                    if d < 0 or d >= self.dim_X[xd]:
                        add_flag = False
                    if d in dots_area_lists[xd]:
                        add_flag = False
            if add_flag:
                dots_area_lists[0].extend(dot_area[0])
                dots_area_lists[1].extend(dot_area[1])
                n_dots += 1
        base[dots_area_lists[0], dots_area_lists[1]] = 1
        return base, n_dots
