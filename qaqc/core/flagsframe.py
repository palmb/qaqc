#!/usr/bin/env python
from __future__ import annotations

import pandas as pd
import numpy as np
from copy import deepcopy
from abc import abstractmethod
from qaqc.typing import SupportsIndex, final, Idx, FlagsFrameT, PandasLike, SerOrDf
from qaqc.constants import SET_UNFLAGGED
from qaqc.core.utils import construct_index, repr_extended, check_index
from typing import Any, NoReturn, Generic, TypeVar
from qaqc.core.ops import OpsMixin

__all__ = ["Meta", "FlagsFrame"]


class Meta:
    __slots__ = ("_raw",)
    _raw: pd.Series

    def __init__(self):
        self._raw = pd.Series(dtype=object)

    def copy(self, deep: bool = True) -> Meta:
        new = self.__class__()
        if deep:
            # raw is a pd.Series with object dtype,
            # and all elements are dicts. This requires
            # recursive copying, but pandas don't do that.
            raw = deepcopy(self._raw)
        else:
            raw = self._raw.copy(deep=False)
        new._raw = raw
        return new

    def to_pandas(self) -> pd.Series:
        return self.copy(deep=True)._raw

    def explode(self) -> pd.DataFrame:
        meta = self.to_pandas()
        meta[meta.isna()] = ({},)
        return pd.DataFrame(meta.to_list())

    @property
    def index(self) -> pd.Index:
        return self._raw.index

    def __len__(self) -> int:
        return len(self._raw)

    def __contains__(self, key) -> bool:
        return self._raw.__contains__(key)

    def __getitem__(self, key) -> dict:
        return self._raw.__getitem__(key)

    def __setitem__(self, key: str, value: dict | None):
        if not isinstance(key, str):
            raise TypeError(f"Key must be a str, not {type(key)}")
        if value is not None and not isinstance(value, dict):
            raise TypeError(f"Value must be a dict or None, not {type(value)}")
        self._raw.__setitem__(key, value)

    @final
    def __str__(self):
        return self.__repr__() + "\n"

    @final
    def __repr__(self):
        if self._raw.empty:
            return "Meta([])"
        return repr(self._raw)

    def _render_short(self, wrap=None):
        return f"Meta: {repr_extended(dict(self._raw), kwdicts=True, wrap=wrap)}"


class BaseFlagsFrame(Generic[SerOrDf]):
    __slots__ = ("_meta", "_raw")
    _meta: Meta
    _raw: SerOrDf

    def __init__(self, *args, **kwargs):
        pass

    @property
    def _constructor(self: BaseFlagsFrame) -> type[BaseFlagsFrame]:
        return type(self)

    @abstractmethod
    def copy(self: FlagsFrameT, deep=True) -> FlagsFrameT:
        ...

    def to_pandas(self) -> SerOrDf:
        return self._raw.copy()

    @property
    def index(self) -> pd.Index:
        return self._raw.index

    @index.setter
    def index(self, value) -> None:
        self._raw.index = value

    @property
    def meta(self) -> Meta:
        # we use a shallow copy, so the user might
        # modify the data in each row, but deletions
        # and appends won't reflect back on us.
        return self._meta.copy(deep=False)

    @meta.setter
    def meta(self, value: pd.Series | Meta):
        if isinstance(value, Meta):
            meta = value
        else:
            assert isinstance(value, pd.Series)
            meta = Meta()
            for k, e in value.items():
                meta[k] = e  # type: ignore
        assert value.index.equals(self.meta.index)
        self._meta = meta

    def __len__(self):
        if len(self._raw.axes) == 1:
            return len(self._raw.axes[0])  # index
        return len(self._raw.axes[1])  # columns

    def __contains__(self, key) -> bool:
        return self._raw.__contains__(key)

    def __getitem__(self, key):
        return self._raw.__getitem__(key).copy()

    def __setitem__(self, key, value):
        raise AttributeError("use 'append' instead")

    def is_flagged(self) -> SerOrDf:
        return self._raw.notna() & (self._raw != SET_UNFLAGGED)

    def is_unflagged(self) -> SerOrDf:
        return self._raw.isna() | (self._raw == SET_UNFLAGGED)

    @abstractmethod
    def current(self, raw: bool = True) -> pd.Series:
        ...

    @final
    def __str__(self):
        return self.__repr__() + "\n"


class CompressedFlags(BaseFlagsFrame[pd.Series], OpsMixin):
    __slots__ = ("_keys",)
    _raw: pd.Series
    _keys: pd.Series
    _meta: Meta

    @property
    def _constructor(self: CompressedFlags) -> type[CompressedFlags]:
        return type(self)

    def __init__(self, flags: pd.Series, keys: pd.Series, meta: Meta):
        super().__init__()
        assert len(keys) == len(flags)
        self._raw = flags.copy()
        self._keys = keys.copy()
        self._meta = meta.copy()

    def copy(self: CompressedFlags, deep=True) -> CompressedFlags:
        cls = self._constructor
        c = cls.__new__(cls)
        c._raw = self._raw.copy(deep=deep)
        c._meta = self._meta.copy(deep=deep)
        c._keys = self._keys.copy(deep=deep)
        return c

    @property
    def keys(self) -> pd.Series:
        return self._keys.copy()

    @property
    def meta(self) -> Meta:
        return super().meta

    @meta.setter
    def meta(self, value: Any) -> NoReturn:
        raise AttributeError("Cant set 'meta' on a CompressedFlags")

    def current(self, raw: bool = True) -> pd.Series:
        result = self._raw.copy()
        if not raw:
            result = result.replace(SET_UNFLAGGED, np.nan)
        return result

    def compress_further(self):
        new = self.copy(deep=True)
        new._raw.dropna(inplace=True)
        new._keys.dropna(inplace=True)
        # remove unused meta entries
        uniques = new._keys.unique()
        m = new._meta._raw.index.isin(uniques)
        new._meta._raw = new._meta._raw[m]
        return new

    @final
    def __repr__(self) -> str:
        s: pd.Series = self.to_pandas()
        df = s.to_frame(name="flags")
        df["keys"] = self._keys.array
        data = repr(df).replace("DataFrame", self.__class__.__name__)
        meta = repr(self._meta)
        return data + "\n" + meta


class FlagsFrame(BaseFlagsFrame[pd.DataFrame], OpsMixin):
    __slots__ = ()
    _raw: pd.DataFrame
    _meta: Meta

    """
    Formerly known as History
    - existing data cannot be mutated
    - data (columns) can be appended
    - created empty from a pd.Index
    - comparable eq, ne, le, ge, lt, gt return pd.Dataframes
    """

    @property
    def _constructor(self: FlagsFrame) -> type[FlagsFrame]:
        return type(self)

    def __init__(self, initial: Any | None = None, index: Idx | None = None) -> None:
        # late import to avoid cycles
        super().__init__()
        from qaqc.core.variable import Variable

        idx: pd.Index
        f0: pd.Series | None

        index = construct_index(index, name="Index", optional=True)

        if initial is None:
            if index is None:
                raise ValueError("One of index and initial must not be None")
            f0 = None
            idx = index
        elif isinstance(initial, pd.Index):  # special case
            f0 = None
            idx = initial
        else:
            if isinstance(initial, Variable):
                initial = initial.get_history()
            if isinstance(initial, FlagsFrame):
                initial = initial.current()
            if isinstance(initial, pd.DataFrame):
                initial = initial.squeeze(axis=1)
            if isinstance(initial, pd.Series):
                f0 = initial
            else:
                raise TypeError(type(initial))
            idx = f0.index

        # a given index overwrites any extracted index
        if index is not None:
            idx = index
        idx = check_index(idx, name="index")

        self._raw = pd.DataFrame(index=idx.copy(), dtype=float)
        self._meta = Meta()

        if f0 is not None:
            self.append(initial, initial=True)

    def copy(self: FlagsFrame, deep=True) -> FlagsFrame:
        new = self._constructor(index=self._raw.index)
        new._raw = self._raw.copy(deep=deep)
        new._meta = self._meta.copy(deep=deep)
        return new

    @property
    def columns(self) -> pd.Index:
        return self._raw.columns

    def current(self, raw: bool = True) -> pd.Series:
        if self._raw.empty:
            result = pd.Series(data=np.nan, index=self.index, dtype=float)
        else:
            result = self._raw.ffill(axis=1).iloc[:, -1]
        if not raw:
            result = result.replace(SET_UNFLAGGED, np.nan)
        return result

    def append_conditional(self, flag: float | int, cond, **meta) -> FlagsFrame:
        new = pd.Series(np.nan, index=self.index, dtype=float)
        new[cond] = float(flag)
        return self.append(new, **meta)

    def append(self, value, **meta) -> FlagsFrame:
        value = pd.Series(value, dtype=float)
        assert len(value) == len(self.index)
        col = f"f{len(self)}"
        self._raw[col] = value.array
        self._meta[col] = meta or None
        return self

    # ############################################################
    # Comparison
    # ############################################################

    def equals(self, other: FlagsFrame) -> bool:
        return isinstance(other, type(self)) and self._raw.equals(other._raw)

    def _cmp_method(self, other: Any, op) -> pd.DataFrame:
        if isinstance(other, self.__class__):
            other = other._raw
        df: pd.DataFrame = self._raw
        if isinstance(other, pd.DataFrame):
            if not df.columns.equals(other.columns):
                raise ValueError("Columns does not match")
            if not df.index.equals(other.index):
                raise ValueError("Index does not match")
        return op(df, other)

    # ############################################################
    # Rendering
    # ############################################################

    @final
    def __repr__(self) -> str:
        data = repr(self.to_pandas()).replace("DataFrame", self.__class__.__name__)
        meta = repr(self._meta)
        return data + "\n" + meta

    @final
    def to_string(self, *args, show_meta=True, **kwargs) -> str:
        meta = f"\n{repr(self._meta)}" if show_meta else ""
        return (
            self.to_pandas()
            .to_string(*args, **kwargs)
            .replace("DataFrame", self.__class__.__name__)
        ) + meta

    def compress(self) -> CompressedFlags:
        """Reduce flags to a pd.Series of meta keys()"""
        raw = self._raw.copy(deep=True)
        meta = self._meta.copy(deep=True)

        for fn in raw.columns:
            # we ignore SET_UNFLAGGED here,
            # because it holds info, namely a meta
            flagged = raw[fn].notna()
            if not flagged.any():
                meta._raw.pop(fn)
            raw.loc[flagged, fn] = fn

        flags = self.current(raw=True)
        keys = raw.ffill(axis=1).iloc[:, -1]
        return CompressedFlags(flags, keys, meta)


if __name__ == "__main__":
    from qaqc._testing import dtindex, N

    s = pd.Series([1, 2, 3], index=dtindex(3))
    ff = FlagsFrame(pd.Index([]))
    print(ff)
    ff = FlagsFrame(s.index)
    ff.append([N, N, N], someSerArg=s, someOther=None)
    print(ff)
    ff.append(s)
    ff.append(s.values)
    ff2 = ff.copy()
    ff2.append([N, N, N])
    print(ff)
    print(ff2)
    print(ff2 == ff2)
    ff2.append([N, 99, N])
    print(ff2)
    print(ff == ff2)
    print(FlagsFrame(ff2))
