#!/usr/bin/env python
from __future__ import annotations

from abc import abstractmethod
import pandas as pd
import numpy as np

from saqc._typing import VariableT, final, MaskLike, MaskerT
from typing import Any, overload, Iterable, Collection, Callable, Union
from saqc.core.flagsframe import FlagsFrame
from saqc.constants import UNFLAGGED
from saqc.errors import (
    MaskerError,
    MaskerResultError,
    MaskerExecutionError,
)


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
    def masked_series(self) -> pd.Series:
        return pd.Series(self._raw.filled(), self._index, dtype=float, copy=True)

    @property
    def orig_series(self) -> pd.Series:
        return pd.Series(self._raw.data, self._index, dtype=float, copy=True)

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

    def __len__(self):
        return len(self._raw)


class BaseVariable:
    _data: _Data
    _history: FlagsFrame
    masker: MaskerT | None

    # size of Variable
    # --------------------------
    # _history._raw   (pd.Dataframe)
    #   .index      pd.Index = datetime64(8) * length
    #   -> col      float(8) * length * #columns
    # _data._raw    (MaskedArray)
    #   .data       float(8) * length
    #   .mask       bool(1) * length
    # _data._index  is a reference to _flag._raw.index
    # --------------------------
    # (8 + 1 + 8 + 8) * length + (8 * length * columns)
    # 25B * length + 8B * length * columns
    # size(column) == size(data) == size(index)
    # Columns + Data + 2x Index = Columns+3
    # ==> 8B * length * (columns+3) + 1length [from the mask]

    # vs size of data-series + history-df
    # Columns + Data + 2x Index = Columns+3
    # ==> 8B * length * (columns+3)

    global_masker: MaskerT | None = None

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
                flags = data._history
            index = data.index
            data = data._data._raw

        data = _Data(data, index)

        # if no history are given we create an empty FlagsFrame
        # otherwise we create a FlagsFrame with an initial column.
        # The resulting FlagsFrame will at most have the initial
        # column, even if a multidim object is passed.
        flags = FlagsFrame(index=data.index, initial=flags)

        self._data = data
        self._history = flags
        self.masker = None
        self._optimize()

    def copy(self: VariableT, deep: bool = True) -> VariableT:
        self._update_mask()
        cls = self.__class__
        c = cls.__new__(cls)
        c._data = self._data.copy(deep)
        c._history = self._history.copy(deep)
        c.masker = self.masker
        return c._optimize()

    def _optimize(self):
        # use a single index
        self._data._index = self._history._raw.index

    @property
    def index(self) -> pd.Index:
        return self._data.index

    @index.setter
    def index(self, value) -> None:
        try:
            self._data.index = value
            self._history.index = value
        finally:
            # on error this also resets the index
            # of data with old flags.index
            self._optimize()

    @property
    def orig(self) -> pd.Series:
        return self._data.orig_series

    @property
    def data(self) -> pd.Series:
        self._update_mask()
        return self._data.masked_series

    @property
    def flags(self) -> pd.Series:
        return self._history._current()

    @property
    def history(self) -> FlagsFrame:
        return self._history

    def is_masked(self) -> pd.Series:
        """return a boolean series that indicates flagged values"""
        mask = self._get_mask()
        return pd.Series(mask, index=self.index, dtype=bool, copy=True)

    def is_flagged(self) -> pd.Series:
        """return a boolean series that indicates flagged values"""
        return self.flags > UNFLAGGED

    def is_unflagged(self) -> pd.Series:
        """return a boolean series that indicates unflagged values"""
        return self.flags == UNFLAGGED

    def template(self, fill_value=np.nan) -> pd.Series:
        return pd.Series(fill_value, index=self.index, dtype=float)

    # ############################################################
    # Comparisons and Arithmetics
    # ############################################################

    def equals(self, other: Any) -> bool:
        return (
            isinstance(other, type(self))
            and self.data.equals(other.data)
            and self.history.equals(other.history, current=False)
        )

    def __eq__(self, other) -> bool:
        return self.equals(other)

    def __getitem__(self, key):
        if key in ["data", "mask", "flags"]:
            return getattr(self, key)
        return self._history.__getitem__(key)

    def __setitem__(self, key, value):
        raise NotImplementedError("use 'append_flags' instead")

    def append_flags(self, value, meta: Any = None) -> BaseVariable:
        # todo: rename ??
        #   - append_flags(mask, value, meta)
        #   - append_flagseries(float-series, meta)
        #   same for history without flag prefix ??
        self._history.append(value, meta)
        return self

    def mask(self):
        pass

    def set_flags(self, value: pd.Series, **meta):
        self._history.append(value, **meta)
        pass

    # ############################################################
    # Masker and Masking
    # ############################################################

    def _get_mask(self) -> np.ndarray:
        # maybe cache, hash and cash $$$
        # with pd.util.hash_pandas_object() ??
        self._update_mask()
        return self._data.mask

    def _update_mask(self) -> None:
        mask = self._create_mask()
        self._data._raw.soften_mask()
        self._data._raw.mask = mask
        self._data._raw.harden_mask()

    def _create_mask(self) -> np.ndarray:
        masker = self.masker or self.global_masker
        if masker is None:
            return self.__default_masker()

        flags = self.flags.to_numpy()
        length = len(flags)
        try:
            result = masker(flags)
        except Exception as e:
            raise MaskerExecutionError(
                masker, f"Execution of user-set masker {masker} failed"
            ) from e

        try:
            self._check_masker_result(result, length)
        except Exception as e:
            raise MaskerResultError(
                masker,
                f"Unexpected result of the masker-function {masker.__name__}\n"
                f"\tMost probably due to an incorrect implementation:\n"
                f"\t" + str(e),
            ) from None

        return result

    @staticmethod
    def _check_masker_result(result: np.ndarray, length: int) -> None:
        if not isinstance(result, np.ndarray):
            raise TypeError(
                f"Result has wrong type. Expected {np.ndarray}, but got {type(result)}"
            )
        if not result.dtype == bool:
            raise ValueError(
                f"Result has wrong dtype. Expected {np.dtype(bool)!r}, "
                f"but result has {result.dtype!r} dtype"
            )
        if len(result) != length:
            raise ValueError(
                f"Result has wrong length.  Expected {length}, "
                f"but got {len(result)} values"
            )

    @final
    def __default_masker(self) -> np.ndarray:
        return self.flags.to_numpy() > UNFLAGGED

    # ############################################################
    # Rendering
    # ############################################################

    @final
    def __str__(self):
        return self.__repr__() + "\n"

    @final
    def __render_frame(self, df: pd.DataFrame) -> str:
        if not self.index.empty:
            if 'f0' in df:
                n = df.columns.get_loc('f0')
                df.insert(n, "|", ["|"] * len(df.index))  # noqa
        return repr(df).replace("DataFrame", self.__class__.__name__)

    @final
    def __repr__(self) -> str:
        try:
            df = self.to_pandas()
        except MaskerError as e:
            # now we create a frame without calling masker
            df = self.history.to_pandas()
            df.insert(0, "flags", self.flags.array)
            df.insert(0, "**ORIG**", self._data._raw.data)
            string = self.__render_frame(df)
            raise RuntimeError(
                f"\n{string}\nUser-set masker failed for repr(), "
                f"see the prior exception above"
            ) from e
        return self.__render_frame(df)

    @final
    def to_string(self, *args, **kwargs) -> str:
        df = self.to_pandas()
        if not self.index.empty:
            if 'f0' in df:
                n = df.columns.get_loc('f0')
                df.insert(n, "|", ["|"] * len(df.index))  # noqa
        s = df.to_string(*args, **kwargs).replace("DataFrame", self.__class__.__name__)
        return s

    def memory_usage(self, index: bool = True, deep: bool = False) -> pd.Series:
        flags_df = self._history._raw
        data = self._data._raw.data
        mask = self._data._raw.mask
        flags_index = flags_df.index
        data_index = self._data._index
        result = pd.Series(dtype=int)
        if index:
            n = flags_index.memory_usage(deep=deep)
            if data_index is not flags_index:
                n += data_index.memory_usage(deep=deep)
            result["Index"] = n
        result["history"] = flags_df.memory_usage(index=False, deep=deep).sum()
        result["data"] = data.itemsize * data.size
        result["mask"] = mask.itemsize * mask.size
        return result

    def to_pandas(self) -> pd.DataFrame:
        df = self.history.to_pandas()
        df.insert(0, "flags", self.history._current().array)
        # to prevent calling masker multiple times
        # we use private methods.
        df.insert(0, "mask", self._get_mask())
        df.insert(0, "data", self._data._raw.filled())
        return df


if __name__ == "__main__":

    class Variable(BaseVariable):
        def __init__(self, data, index=None, flags=None):
            super().__init__(data=data, index=index, flags=flags)

        @property
        def _constructor(self: VariableT) -> type[VariableT]:
            return type(self)

    from saqc._testing import dtindex, N

    v = Variable([1, 2, 3, 4], dtindex(4))
    print(v)
    v = Variable([1, 2, 3, 999999.0], dtindex(4), pd.Series([N, N, 99, N]))
    v.history.append([N, 55, 55, N])
    v.history.append([N, 99, N, N])
    print(v)
    runtime_fail = lambda x: 1 / 0
    result_fail = lambda x: x.astype(str)
    v.masker = result_fail
    print(v)
    v2 = Variable([])
    v2.history.append([], meta="la")

    print(v.memory_usage(deep=True))
    v._optimize()
    print("totoal:", v.memory_usage(deep=True).sum())
    print("pandas:")
    df = v.to_pandas().astype(float)
    df["mask"] = df["mask"].astype(bool)
    df.drop("flags", axis=1, inplace=True)
    print(df)
    print(df.memory_usage(deep=True))
    print(df.memory_usage(deep=True).sum())

    print()
    print(v.to_string())
