#!/usr/bin/env python
from __future__ import annotations

from abc import abstractmethod
import functools
import warnings
from typing import Callable, Union, TypeVar, Iterator, overload, Any, Iterable, Hashable

import pandas as pd
import numpy as np
from sliceable_dict import TypedSliceDict

from qaqc.typing import T, Cols, Idx, QaqcFrameT, FlagLike, NDArray, VarOrSer
from qaqc.core.variable import Variable
from qaqc.errors import ImplementationError
from qaqc.core.utils import construct_index


class IdxMixin:
    @abstractmethod
    def _get_indexes(self) -> list[pd.Index]:
        ...

    def union_index(self) -> pd.Index:
        indexes = self._get_indexes()
        if indexes:
            return functools.reduce(pd.Index.union, indexes)
        return pd.Index([])

    def shared_index(self) -> pd.Index:
        indexes = self._get_indexes()
        if indexes:
            return functools.reduce(pd.Index.intersection, indexes)
        return pd.Index([])


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
        if isinstance(value, Variable) and value._name is None:
            value._name = key
        return key, value

    def copy(self: _Vars, deep: bool = True) -> _Vars:
        cls = self.__class__
        if deep:
            data = {k: v.copy(deep=True) for k, v in self.items()}
        else:
            data = zip(self.keys(), self.values())  # type: ignore
        return cls(data)


class QaqcFrame(IdxMixin):
    _vars: _Vars

    @property
    def _constructor(self: QaqcFrame) -> type[QaqcFrame]:
        return type(self)

    def __init__(
        self,
        data: pd.DataFrame
        | QaqcFrame
        | NDArray
        | dict[str, VarOrSer]
        | list[VarOrSer]
        | VarOrSer
        | None = None,
        columns: Cols | None = None,
        index: Idx | None = None,
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
        columns = construct_index(columns, name="Columns", optional=True)

        raw: dict | _Vars
        if data is None:
            raw = {}
            copy = False
        elif isinstance(data, QaqcFrame):
            raw = data._vars
        elif isinstance(data, pd.DataFrame):
            raw = dict(data)
        elif isinstance(data, (Variable, pd.Series, list)):
            if not isinstance(data, list):
                data = [data]
            if columns is None:
                columns = "var" + pd.RangeIndex(len(data))  # type: ignore
            elif len(data) != len(columns):
                raise ValueError(
                    f"data have {len(data)} values, "
                    f"but columns imply {len(columns)}"
                )
            raw = dict(zip(columns, data))
            # no subset of columns required
            columns = None
        elif isinstance(data, dict):
            raw = data
        else:
            # We pass any other kind of data to DataFrame
            # to see if pandas can handle it...
            name = self.__class__.__name__
            try:
                raw = dict(pd.DataFrame(data, index=index, columns=columns, copy=False))
            except Exception as e:
                e2 = _find_faulty_DataFrame_param(data, index, columns)
                if not e2:
                    raise TypeError(
                        f"Cannot construct {name} from given inputs"
                    ) from None
                raise type(e2)(str(e2)) from e

        _vars = _Vars(raw)

        # select a subset with columns
        if columns is not None:
            keys = pd.Index(_vars.keys())
            diff = columns.difference(keys)
            if not diff.empty:
                warnings.warn(
                    "Not all the values of given columns are "
                    "present in data while selecting the subset",
                    stacklevel=2,
                )
            common = columns.intersection(keys)
            _vars = _vars[common]

        if copy:
            _vars = _vars.copy(deep=True)
        self._vars = _vars

    def copy(self, deep: bool = True) -> QaqcFrame:
        cls = self._constructor
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

    @overload
    def __getitem__(self, key: Hashable) -> Variable:
        ...

    @overload
    def __getitem__(self, key: Iterable) -> QaqcFrame:
        ...

    def __getitem__(self, key):
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
        df = self.to_pandas(how="outer", flat=False)
        if df.empty:
            return f"Empty {self.__class__.__name__}\nColumns: []"
        columns = pd.MultiIndex.from_product([self.columns, ["data", "flags", "|"]])
        df = df.reindex(columns=columns)
        df.loc[:, (slice(None), "|")] = "|"
        return repr(df)

    def to_pandas(self, how="outer", flat=False) -> pd.DataFrame:
        if how == "outer":
            index = self.union_index()
        elif how == "inner":
            index = self.shared_index()
        else:
            raise ValueError(f"{how=}")

        if flat:
            columns = None
        else:
            columns = pd.MultiIndex.from_product([self.columns, ["data", "flags"]])

        df = pd.DataFrame(index=index, columns=columns)
        for k, var in self._vars.items():
            if flat:
                df[f"{k}-data"] = var.data
                df[f"{k}-flags"] = var.flags
            else:
                df[(k, "data")] = var.data
                df[(k, "flags")] = var.flags
        return df

    def _get_indexes(self) -> list[pd.Index]:
        # needed for IdxMixin
        return list(map(lambda v: v.index, self._vars.values()))

    def _for_each(self, func: Callable[..., Variable], *args, **kwargs) -> QaqcFrame:
        # avoid some common mistake
        for kw in ["inplace", "subset"]:
            if kw in kwargs:
                raise ImplementationError(f"Do not pass {kw} to Variable methods")

        for key, var in self._vars.items():
            var = self[key].copy()
            result = func(var, *args, **kwargs)
            # this cover inplace, in case we want to
            # allow this feature at some point
            self[key] = var if result is None else result
        return self

    def flag_limits(
        self, lower=-np.inf, upper=np.inf, flag: FlagLike = 9, inplace=False
    ) -> QaqcFrame:
        result = self if inplace else self.copy()
        result = result._for_each(
            Variable.flag_limits, lower=lower, upper=upper, flag=flag
        )
        return result


def _find_faulty_DataFrame_param(data, index, columns):
    def fail(**kwargs) -> Exception | None:
        try:
            pd.DataFrame(data, **kwargs)
            return None
        except (TypeError, ValueError, IndexError) as e:
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


if __name__ == "__main__":
    from qaqc import QaqcFrame
    from qaqc._testing import dtindex

    qc = QaqcFrame(None, index=dtindex(4), columns=list("abcd"))
    print(qc)
    qc = QaqcFrame(np.arange(16).reshape(4, 4), index=dtindex(4), columns=list("abcd"))
    qc = QaqcFrame(qc, columns=list("abx"))
    qc["x"] = pd.Series(range(2, 8), dtindex(6))
    qc2 = qc.flag_limits(4, 10, inplace=False)
    print(qc2)
    qc["a"] = qc2["b"].dropna()
    print(qc)
    print(qc["a"])
    print(qc["a"]._ref)
