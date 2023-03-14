#!/usr/bin/env python
from __future__ import annotations

from typing import Callable

import pandas as pd
import numpy as np
from sliceable_dict import TypedSliceDict

from qaqc.constants import UNFLAGGED
from qaqc.typing import T
from qaqc.core.variable import Variable

# Wording
# data : a single Series holding the actual data
# history : an instance of FlagsFrame holding all flagging information
# final-history (fflags): a single Series representing the resulting history for data
# raw (as attribute in internal classes) : holds the actual data and named
#       `raw` to avoid confusion, especially `history.raw` is less confusing
#       than `history.data` .


def _for_each(obj, func, *args, **kwargs):
    new = QaqcFrame()
    for key, var in obj._vars.items():
        var = obj[key].copy()
        result = func(var, *args, **kwargs)
        new[key] = var if result is None else result  # cover inplace case
    return new


class _Vars(TypedSliceDict):
    _key_types = (str,)
    _value_types = (Variable,)

    def _cast(self, key: str, value: pd.Series | Variable):
        if isinstance(value, pd.Series):
            value = Variable(value)
        return key, value


class QaqcFrame:
    @property
    def _constructor(self: QaqcFrame) -> type[QaqcFrame]:
        return type(self)

    def __init__(
        self,
        data: list[Variable | pd.Series]
        | dict[str, Variable]
        | pd.DataFrame
        | pd.Series
        | Variable
        | None = None,
    ):
        self._vars: _Vars = _Vars(data)

    def copy(self, deep: bool = True) -> QaqcFrame:
        cls = self.__class__
        if deep:
            return cls(data={k: v.copy(deep=True) for k, v in self._vars.items()})
        c = cls.__new__(cls)
        # Changes on existing variables (e.g. the masking by new flags)
        # will reflect back on the variables of the copy and vice-versa.
        c._vars = _Vars(self._vars)
        return c

    @property
    def columns(self) -> pd.Index:
        return pd.Index(self._vars.keys())

    def __getitem__(self, key) -> Variable | QaqcFrame:
        raw = self._vars.__getitem__(key)
        if isinstance(raw, _Vars):
            raw = self._constructor(raw).copy()
        return raw

    def __setitem__(self, key, value: Variable | pd.Series) -> None:
        self._vars.__setitem__(key, value)

    def show(self) -> None:
        df = pd.DataFrame()
        for k, var in self._vars.items():
            df[f"{k}-data"] = var.data
            df[f"{k}-flags"] = var.flags
        print(df)

    def _for_each(self, func: Callable[..., Variable], *args, **kwargs) -> QaqcFrame:
        new = QaqcFrame()
        for key, var in self._vars.items():
            var = self[key].copy()
            result = func(var, *args, **kwargs)
            new[key] = var if result is None else result  # cover inplace case
        return new

    def dododo(self, arg0, arg1, kw0=None, kw1=None):
        return _for_each(self, Variable.dododo, arg0, arg1, kw0=kw0, kw1=kw1)
