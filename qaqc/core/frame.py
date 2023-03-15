#!/usr/bin/env python
from __future__ import annotations

import warnings
from typing import Callable, Union, TypeVar

import pandas as pd
import numpy as np
from sliceable_dict import TypedSliceDict

from qaqc.constants import UNFLAGGED
from qaqc.typing import T, Columns, Axes
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

    def copy(self: _Vars, deep: bool = True) -> _Vars:
        cls = self.__class__
        if deep:
            data = {k: v.copy(deep=True) for k, v in self.items()}
        else:
            data = zip(self.keys(), self.values())
        return cls(data)


Series = TypeVar("Series", bound=pd.Series)
VarLike = Union[Variable, pd.Series]


def _find_faulty_DataFrame_param(data, index, columns):
    def fail(**kwargs) -> Exception | None:
        try:
            pd.DataFrame(data, **kwargs)
            return None
        except Exception as e:
            return e

    d_fail = fail()
    c_fail = fail(columns=columns)
    i_fail = fail(index=index)
    if d_fail and c_fail and i_fail:
        return None  # cannot construct obj with given params
    if i_fail:
        return type(i_fail)("faulty index")
    if c_fail:
        return type(c_fail)("faulty columns")
    return None


class QaqcFrame:
    _vars: _Vars

    @property
    def _constructor(self: QaqcFrame) -> type[QaqcFrame]:
        return type(self)

    def __init__(
        self,
        data: None
        | pd.DataFrame
        | QaqcFrame
        | dict[str, VarLike]
        | list[VarLike]
        | VarLike = None,
        columns: Columns | None = None,
        index: Axes | None = None,
        copy: bool = True,
    ):
        """
        Parameters
        ----------
        data :
            input data

        columns :
            Column labels to use for resulting frame when data does not have them,
            defaulting to Index("v0", "v1", ...). If data contains column labels,
            will perform column selection instead.

        index :
            Index to use for resulting frame when data or items in data does not have
            any. It defaults to RangeIndex if no indexing information part of input
            data and no index provided.

        copy :
        """
        if columns is not None:
            columns = pd.Index(columns)
            if isinstance(columns, pd.MultiIndex):
                raise TypeError("columns must not be a multi level index")
            if not columns.is_unique:
                raise ValueError("columns must have unique values")

        if data is None:
            data = {}
            copy = False
        elif isinstance(data, QaqcFrame):
            data = data._vars.copy(deep=copy)
            copy = False
        elif isinstance(data, pd.DataFrame):
            data = dict(data.copy(deep=copy))
            copy = False
        elif isinstance(data, (Variable, pd.Series)):
            data = [data.copy(deep=copy)]
            copy = False

        if isinstance(data, list):
            if columns is None:
                columns = "var" + pd.RangeIndex(len(data))
            elif len(data) != len(columns):
                raise ValueError(
                    f"data have {len(data)} values, "
                    f"but columns imply {len(columns)}"
                )
            data = dict(zip(columns, data))
            # no subset of columns required
            columns = None

        if not isinstance(data, dict):
            name = self.__class__.__name__
            try:
                tmp = dict(pd.DataFrame(data, index=index, columns=columns, copy=False))
            except Exception as e:
                e2 = _find_faulty_DataFrame_param(data, index, columns)
                if not e2:
                    raise TypeError(f"Cannot construct {name} from given inputs")
                raise type(e2)(str(e2)) from e
            data = tmp

        data = _Vars(data)

        # select a subset with columns
        if columns is not None:
            diff = columns.difference(data.keys())  # type: ignore
            if not diff.empty:
                warnings.warn("Not all values in columns are present in data")
            common = columns.intersection(data.keys())  # type: ignore
            data = data[common]

        if copy:
            data = data.copy(deep=True)
        self._vars = data

    def copy(self, deep: bool = True) -> QaqcFrame:
        cls = self.__class__
        if deep:
            return cls(self._vars, copy=True)
        c = cls.__new__(cls)
        # Changes on existing variables (e.g. the masking by new flags)
        # will reflect back on the variables of the copy and vice-versa.
        c._vars = self._vars.copy(deep=False)
        return c

    @property
    def columns(self) -> pd.Index:
        return pd.Index(self._vars.keys())

    def __getitem__(self, key) -> Variable | QaqcFrame:
        raw = self._vars.__getitem__(key)
        if isinstance(raw, _Vars):
            # using a shallow copy would be nice, but we would run into same
            # view-vs-copy problems as pandas. To detect a setting-on-view
            # would require too much effort (for now?)
            raw = self._constructor(raw, copy=True)
        return raw

    def __setitem__(self, key, value: Variable | pd.Series) -> None:
        self._vars.__setitem__(key, value)

    def __str__(self):
        return self.__repr__() + "\n"

    def __repr__(self):
        # index = shared
        columns = pd.MultiIndex.from_product([self.columns, ["data", "flags"]])
        df = pd.DataFrame(columns=columns)
        for k, var in self._vars.items():
            df[(k, "data")] = var.data
            df[(k, "flags")] = var.flags
        return repr(df)

    def to_pandas(self) -> pd.DataFrame:
        pass

    def _for_each(self, func: Callable[..., Variable], *args, **kwargs) -> QaqcFrame:
        new = QaqcFrame()
        for key, var in self._vars.items():
            var = self[key].copy()
            result = func(var, *args, **kwargs)
            new[key] = var if result is None else result  # cover inplace case
        return new

    # def dododo(self, arg0, arg1, kw0=None, kw1=None):
    #     return _for_each(self, Variable.flagna, arg0, arg1, kw0=kw0, kw1=kw1)


if __name__ == "__main__":
    from qaqc import QaqcFrame
    from qaqc._testing import dtindex

    # qc = QaqcFrame(np.arange(16).reshape(4, 4), index=dtindex(4), columns=list("abcd"))
    qc = QaqcFrame(slice(1), index=dtindex(4), columns=list("abcd"))
    print(qc)
    print(qc["a"])
