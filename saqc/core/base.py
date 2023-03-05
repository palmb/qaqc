#!/usr/bin/env python
from __future__ import annotations

from abc import abstractmethod
import pandas as pd
import numpy as np
from numpy.ma import MaskedArray

from saqc._typing import VariableT, final, MaskLike, T
from typing import Any
from saqc.core.flagsframe import FlagsFrame


class _Data:
    __slots__ = ("_raw", "_index")
    _raw: np.ma.MaskedArray
    _index: pd.Index

    def __init__(
        self,
        raw: Any,
        index: pd.Index | None = None,
        mask: bool | MaskLike = False,
        dtype=float,
        fill_value: float = np.nan,
    ):
        if isinstance(raw, pd.Series):
            if index is None:
                index = raw.index
            raw = raw.array

        if dtype is not float:
            raise NotImplementedError("Only float dtype is supported.")

        if isinstance(mask, pd.Series):
            mask = mask.array

        series = pd.Series(raw, index=index, dtype=dtype)
        self._raw = np.ma.MaskedArray(
            data=series.array, mask=mask, dtype=float, fill_value=fill_value, copy=True
        )
        series.index.name = None
        self._index = series.index

    def copy(self, deep=False) -> _Data:
        cls = self.__class__
        if deep:
            return cls(
                raw=self._raw,
                index=self.index,
                mask=self._raw.mask,
                fill_value=self.fill_value,
            )
        c = cls.__new__(cls)
        c._raw = self._raw
        c._index = self.index
        return c

    @property
    def mask(self) -> np.ndarray:
        return self._raw.mask
        # this return a view
        # modifications will reflect back to us
        # this enables Variable.mask[:3] = False
        # return pd.Series(self._raw.mask, index=self._index, dtype=bool, copy=False)

    @mask.setter
    def mask(self, value: bool | MaskLike):
        self._raw.mask = value

    @property
    def series(self) -> pd.Series:
        return pd.Series(self._raw.filled(), self._index, dtype=float, copy=True)

    @property
    def index(self) -> pd.Index:
        return self._index.copy()

    @index.setter
    def index(self, value: Any):
        index = pd.Index(value)
        assert len(index) == len(self._index) and not isinstance(index, pd.MultiIndex)
        self._index = index

    @property
    def fill_value(self) -> float:
        return self._raw.fill_value

    @fill_value.setter
    def fill_value(self, value: float):
        self._raw.fill_value = value


class BaseVariable:
    _data: _Data
    _flags: FlagsFrame

    @property
    @abstractmethod
    def _constructor(self: VariableT) -> type[VariableT]:
        ...
    
    def __init__(
        self,
        data,
        index: pd.Index | None = None,
        flags: FlagsFrame | pd.Series | None = None,
    ):
        if isinstance(data, type(self)):
            if flags is None:
                flags = data._flags
            index = data.index
            data = data._data._raw

        data = _Data(data, index)

        # if no flags are given we create an empty FlagsFrame
        # otherwise we create a FlagsFrame with an initial column.
        # The resulting FlagsFrame will at most have the initial
        # column, even if a multidim object is passed.
        flags = FlagsFrame(index=data.index, initial=flags)

        self._data = data
        self._flags = flags

    @property
    def index(self) -> pd.Index:
        return self._data.index

    @index.setter
    def index(self, value) -> None:
        self._data.index = value

    @property
    def data(self) -> pd.Series:
        return self._data.series

    @property
    def flags(self) -> FlagsFrame:
        return self._flags

    @property
    def mask(self) -> np.ndarray:
        return self._data.mask

    @mask.setter
    def mask(self, value):
        self._data.mask = value

    def flagged(self) -> pd.Series:
        """return a boolean series that indicates flagged values"""
        return self.flags.flagged()

    def copy(self: VariableT, deep: bool = True) -> VariableT:
        cls = self.__class__
        c = cls.__new__(cls)
        c._data = self._data.copy(deep)
        c._flags = self._flags.copy(deep)
        return c

    def equals(self, other: Any) -> bool:
        return (
            isinstance(other, type(self))
            and self.data.equals(other.data)
            and self.flags.equals(other.flags)
        )

    def __eq__(self, other) -> bool:
        # todo ??
        return self.equals(other)

    # ############################################################
    # Rendering
    # ############################################################

    def _df_to_render(self) -> pd.DataFrame:
        # newest first ?
        # df = self.flags.raw.loc[:, ::-1]
        df = self.flags.raw.loc[:]
        df.insert(0, "|", ["|"] * len(df.index))
        df.insert(0, "flags", self.flags.current().array)
        df.insert(0, "data", self.data.array)
        return df

    @final
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        if self.index.empty:
            return repr(self.data).replace("Series", cls_name)
        return repr(self._df_to_render()).replace("DataFrame", cls_name)

    def __str__(self):
        return self.__repr__() + "\n"

    @final
    def to_string(self, *args, **kwargs) -> str:
        if self.index.empty:
            df = self.data.to_frame("data")
        else:
            df = self._df_to_render()
        s = df.to_string(*args, **kwargs).replace("DataFrame", self.__class__.__name__)
        return s


if __name__ == '__main__':
    class Variable(BaseVariable):
        def __init__(self, data, index=None, flags=None):
            super().__init__(data=data, index=index, flags=flags)

        @property
        def _constructor(self: VariableT) -> type[VariableT]:
            return type(self)

    from saqc._testing import dtindex, N
    v = Variable([1,2,3,4], dtindex(4))
    print(v)
    v = Variable([1,2,3,4], dtindex(4), pd.Series([N, N, 99, N]))
    print(v)