#!/usr/bin/env python
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any, TypeVar
from saqc.core.generic import FuncMixin
from saqc.core.variable import Variable

UNFLAGGED = -np.inf
T = TypeVar("T")

# Wording
# data : a single Series holding the actual data
# history : an instance of FlagsFrame holding all flagging information
# final-history (fflags): a single Series representing the resulting history for data
# raw (as attribute in internal classes) : holds the actual data and named
#       `raw` to avoid confusion, especially `history.raw` is less confusing
#       than `history.data` .


def _for_each(obj, func, *args, **kwargs):
    new = SaQC()
    for key, var in obj._vars.items():
        var = obj[key].copy()
        result = func(var, *args, **kwargs)
        new[key] = var if result is None else result  # cover inplace case
    return new


class SaQC(FuncMixin):
    def __init__(
        self,
        data: list[Variable | pd.Series]
        | pd.DataFrame
        | pd.Series
        | Variable
        | None = None,
    ):
        self._vars = dict()

    vars = property(lambda self: self._vars)

    def saqc_only_meth(self):
        pass

    def __getitem__(self, item):
        return self._vars[item]

    def __setitem__(self, key, value):
        if isinstance(key, str):
            if isinstance(value, pd.Series):
                value = Variable(value)
            if isinstance(value, Variable):
                self._vars[key] = value
                return
            raise TypeError(
                f"Expected value of type Series or Variable, not {type(value)}"
            )

        if isinstance(key, (list, pd.Index)):
            if not len(key) == len(value):
                raise ValueError(f"Got {len(key)} keys, but {len(value)} values.")
            _vars = dict(self._vars)  # shallow copy
            try:
                for k, v in zip(key, value):
                    self.__setitem__(k, v)
                _vars = self._vars
            finally:
                self._vars = _vars
            return
        raise TypeError(f"Expected key of type str, Index or list, not {type(value)}")

    def show(self):
        df = pd.DataFrame()
        for k, var in self._vars.items():
            var: Variable
            df[f"{k}-data"] = var.data
            df[f"{k}-fflags"] = var.flags
        print(df)

    def copy(self):
        new = SaQC()
        new._vars = {k: v.copy() for k, v in self._vars.items()}
        return new

    def _for_each(self, func, *args, **kwargs):
        new = SaQC()
        for key, var in self._vars.items():
            var = self[key].copy()
            result = func(var, *args, **kwargs)
            new[key] = var if result is None else result  # cover inplace case
        return new

    def dododo(self, arg0, arg1, kw0=None, kw1=None):
        return _for_each(self, Variable.dododo, arg0, arg1, kw0=kw0, kw1=kw1)
