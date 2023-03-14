#!/usr/bin/env python
from __future__ import annotations

from abc import abstractmethod
import pandas as pd
import numpy as np

from qaqc.typing import (
    final,
    MaskT,
    MaskerT,
    CondT,
    ListLike,
    FlagLike,
    VariableT,
)
from typing import Any, overload, TYPE_CHECKING, NoReturn, cast

from qaqc.core import utils
from qaqc.core.flagsframe import FlagsFrame, Meta
from qaqc.constants import UNFLAGGED
from qaqc.errors import (
    MaskerError,
    MaskerResultError,
    MaskerExecutionError,
)

__all__ = [
    "BaseVariable",
]


# This class is private because it is completely
# shadowed by methods and properties in BaseVariable.
# A user should not access or create _Data, nor
# should have need for that.
class _Data:
    __slots__ = ("_raw", "_index")
    _raw: np.ma.MaskedArray
    _index: pd.Index

    def __init__(
        self,
        raw: Any,
        index: pd.Index | None = None,
        mask: bool | MaskT = False,
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
            mask = mask.to_numpy()

        series = pd.Series(raw, index=index, dtype=dtype)
        self._raw = np.ma.MaskedArray(
            data=series.array, mask=mask, dtype=float, fill_value=fill_value, copy=True
        )
        self._index = series.index.copy()

    def copy(self, deep=False) -> _Data:
        cls = self.__class__
        if deep:
            return cls(
                raw=self._raw.data,
                index=self.index,
                mask=self._raw.mask,
                fill_value=self.fill_value,
            )
        c = cls.__new__(cls)
        # Setting the very same obj here is ok, because
        # _raw.data and _index are basically immutable
        # by the public api, especially if accessed via
        # Variable. Nevertheless, changes on _raw.mask
        # (e.g. if new flags are set in Variable) will
        # reflect back on the copy and vice-versa.
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
    def mask(self, value: bool | MaskT):
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
    # class variable
    global_masker: MaskerT | None = None

    # type hints
    _data: _Data
    _history: FlagsFrame
    masker: MaskerT | None

    @property
    @abstractmethod
    def _constructor(self: VariableT) -> type[VariableT]:
        ...

    def __init__(
        self,
        data,
        flags: FlagsFrame | pd.Series | None = None,
        index: pd.Index | None = None,
    ):
        if isinstance(data, type(self)):
            if flags is None:
                flags = data._history
            if index is None:
                index = data.index
            data = data._data._raw

        if isinstance(index, pd.Series):
            raise TypeError(
                "Using a pd.Series as index is not allowed, "
                "use pd.Series.values instead"
            )

        data = _Data(data, index)

        # if no history are given we create an empty FlagsFrame,
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
        c._optimize()
        return c

    def _optimize(self):
        # share a single index
        self._data._index = self._history._raw.index

    _restore_index = _optimize  # alias for now

    @property
    def index(self) -> pd.Index:
        return self._history.index

    @index.setter
    def index(self, value) -> None:
        try:
            self._data.index = value
            self._history.index = value
        finally:
            self._restore_index()

    @property
    def orig(self) -> pd.Series:
        """Returns a copy of the underlying data (no masking)"""
        return self._data.orig_series

    @property
    def data(self) -> pd.Series:
        """Returns a copy of the underlying data (masked by flags)"""
        self._update_mask()
        return self._data.masked_series

    @property
    def flags(self) -> pd.Series:
        """Returns a series of currently set flags"""
        return self._history.current()

    @property
    def meta(self) -> Meta:
        return self._history.meta  # mutable shallow copy

    @meta.setter
    def meta(self, value: pd.Series | Meta):
        self._history.meta = value  # type: ignore

    # ############################################################
    # basic functions
    # ############################################################

    def get_history(self) -> FlagsFrame:
        return self._history.copy()

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

    def equals(self, other: Any, compare_history=False) -> bool:
        return (
            isinstance(other, type(self))
            and self.data.equals(other.data)
            and (
                self._history.equals(other._history)
                if compare_history
                else self.flags.equals(other.flags)
            )
        )

    def __eq__(self, other) -> bool:
        # compares data==data and flags==flags
        return self.equals(other, compare_history=False)

    def __getitem__(self, key) -> pd.Series:
        if key in ["data", "flags"]:
            return getattr(self, key)
        if key == "mask":
            return self.is_masked()
        if key in self._history:
            return self._history.__getitem__(key)
        raise KeyError(key)

    def __setitem__(self, key, value) -> NoReturn:
        raise NotImplementedError("use 'set_flags()' instead")

    @overload
    def set_flags(
        self: VariableT, value: FlagLike, cond: CondT | None = None, **meta
    ) -> VariableT:
        ...

    @overload
    def set_flags(
        self: VariableT, value: ListLike, cond: None = None, **meta
    ) -> VariableT:
        ...

    def set_flags(self, value, cond=None, **meta):
        # allowed:
        #   cond=None, value=listlike
        #   cond=bool-listlike, value=scalar
        if cond is None and pd.api.types.is_list_like(value):
            self._history.append(value, **meta)
        elif utils.is_indexer(cond) and isinstance(value, (int, float)):
            self._history.append_conditional(value, cond, **meta)
        else:
            raise TypeError(
                "'value' must be float and 'cond' a pd.Index / boolean indexer, "
                "OR value must be a list-like of floats and cond must be None."
                f"type cond: {type(cond)}, type value: {type(value)}"
            )
        return self

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
            if "f0" in df:
                n = df.columns.get_loc("f0")
                df.insert(n, "|", ["|"] * len(df.index))  # type: ignore
        return repr(df).replace("DataFrame", self.__class__.__name__)

    @final
    def __repr__(self) -> str:
        try:
            df = self.to_pandas()
        except MaskerError as e:
            # now we create a frame without calling masker
            df = self._history.to_pandas()
            df.insert(0, "flags", self.flags.array)  # type: ignore
            df.insert(0, "**ORIG**", self._data._raw.data)
            string = self.__render_frame(df)
            raise RuntimeError(
                f"\n{string}\nUser-set masker failed for repr(), "
                f"see the prior exception above"
            ) from e

        string = self.__render_frame(df)
        meta = str(self.meta._render_short())
        return string + "\n" + meta

    @final
    def to_string(self, *args, show_meta=True, **kwargs) -> str:
        df = self.to_pandas()
        if not self.index.empty:
            if "f0" in df:
                n = df.columns.get_loc("f0")
                df.insert(n, "|", ["|"] * len(df.index))  # type: ignore
        s = df.to_string(*args, **kwargs).replace(  # type: ignore
            "DataFrame", self.__class__.__name__
        )
        if show_meta:
            s += "\n" + str(self.meta._render_short())
        return s

    def memory_usage(self, index: bool = True, deep: bool = False) -> pd.Series:
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

        # for comparison:
        # size of a plain data-series and a simple history-df
        # Columns + Data + 2x Index = Columns+3
        # ==> 8B * length * (columns+3)

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
        df = self._history.to_pandas()
        df.insert(0, "flags", self._history.current().array)  # type: ignore
        # to prevent calling masker multiple times
        # we use private methods and attributes.
        df.insert(0, "mask", self._get_mask())
        df.insert(0, "data", self._data._raw.filled())
        return df


if __name__ == "__main__":

    class TestVariable(BaseVariable):
        def __init__(self, data, index=None, flags=None):
            super().__init__(data=data, index=index, flags=flags)

        @property
        def _constructor(self: TestVariable) -> type[TestVariable]:
            return type(self)

    from qaqc._testing import dtindex, N

    v = TestVariable([1, 2, 3, 4], dtindex(4))
    print(v)
    v = TestVariable([1, 2, 3, 999999.0], dtindex(4), pd.Series([N, N, 99, N]))
    print(v)
    v.set_flags([N, 55, 55, N])
    print(v)
    v.set_flags([N, 99, N, N])
    print(v)
    runtime_fail = lambda x: 1 / 0
    result_fail = lambda x: x.astype(str)
    # v.masker = result_fail
    print(v)
    v2 = TestVariable([])
    v2.set_flags([], meta="la")

    print(v.memory_usage(deep=True))
    v._optimize()
    print("totoal:", v.memory_usage(deep=True).sum())
    print("pandas:")
    df = v.to_pandas().astype(float)
    df.drop("flags", axis=1, inplace=True)
    df.drop("mask", axis=1, inplace=True)
    print(df)
    print(df.memory_usage(deep=True))
    print(df.memory_usage(deep=True).sum())
    v.set_flags(99.0, np.array([1, "la"], dtype=str))

    print()
    print(v.to_string())
