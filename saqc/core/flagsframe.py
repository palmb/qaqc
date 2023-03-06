#!/usr/bin/env python
from __future__ import annotations

import pandas as pd
import numpy as np
from saqc.core.generic import compose
from copy import deepcopy
from saqc._typing import SupportsIndex, final
from saqc.constants import UNFLAGGED
import saqc.core.utils as utils
from typing import Any

# from pandas.core.arraylike import OpsMixin


class FlagsFrame:
    __slots__ = ("_raw", "_meta")
    _raw: pd.DataFrame
    _meta: pd.Series

    """
    Formerly known as History
    - existing data cannot be mutated
    - data (columns) can be appended
    - created empty from a pd.Index
    - comparable eq, ne, le, ge, lt, gt return pd.Dataframes
    """

    def __init__(self, index: Any = None, initial: SupportsIndex | None = None) -> None:
        if index is None:
            if hasattr(initial, "index") and isinstance(initial.index, pd.Index):
                index = initial.index
            else:
                raise TypeError(
                    "If no 'index' is given, 'initial' must " "have a pandas.Index."
                )

        # special case:  FlagsFrame(flags_frame)
        elif isinstance(index, type(self)):
            initial = index._current()
            index = index.index
        else:
            try:
                index = pd.Index(index)
            except TypeError:
                raise TypeError(f"Cannot create index for from {type(index)}")

        index.name = None

        if isinstance(initial, type(self)):
            initial = initial._current()

        self._raw = pd.DataFrame(index=index, dtype=float)
        self._meta = pd.Series(dtype=object)

        if initial is not None:
            self.append(initial, "init")

    def copy(self, deep=True) -> FlagsFrame:
        new = self.__class__(self.index)
        new._raw = self._raw.copy(deep)
        if deep:
            # meta is a pd.Series with object dtype,
            # and we allow dicts to be stored there.
            # This requires recursive copying, but
            # pandas don't do that.
            meta = deepcopy(self._meta)
        else:
            meta = self._meta.copy(False)
        new._meta = meta
        return new

    def to_pandas(self) -> pd.DataFrame:
        return self._raw.copy()

    @property
    def index(self) -> pd.Index:
        return self._raw.index

    @index.setter
    def index(self, value) -> None:
        self._raw.index = value

    @property
    def columns(self) -> pd.Index:
        return self._raw.columns

    @property
    def meta(self) -> pd.Series:
        # we use a shallow copy, so the user might
        # modify the data in each row, but deletions
        # and appends won't reflect back on us.
        return self._meta.copy(deep=False)

    def __getitem__(self, key):
        return self._raw.__getitem__(key).copy()

    def __setitem__(self, key, value):
        raise NotImplementedError("use 'append' instead")

    def _current(self) -> pd.Series:
        if self._raw.empty:
            result = pd.Series(data=np.nan, index=self.index, dtype=float)
        else:
            result = self._raw.ffill(axis=1).iloc[:, -1]
        return result.fillna(UNFLAGGED)

    def is_flagged(self) -> pd.DataFrame:
        return self._raw > UNFLAGGED

    def is_unflagged(self) -> pd.DataFrame:
        return self._raw == UNFLAGGED

    def append_with_mask(self, mask, flag: float | int, meta: Any = None) -> FlagsFrame:
        new = pd.Series(np.nan, index=self.index, dtype=float)
        new[mask] = float(flag)
        return self.append(new, meta)

    def append(self, value, **meta) -> FlagsFrame:
        value = pd.Series(value, dtype=float)
        assert len(value) == len(self.index)
        col = f"f{len(self)}"
        self._raw[col] = value.array
        self._meta[col] = meta or None
        return self

    def equals(self, other: FlagsFrame) -> bool:
        return isinstance(other, type(self)) and self._raw.equals(other._raw)

    def __len__(self):
        return len(self._raw.columns)

    @staticmethod
    def __bool_cmp__(funcname):
        def cmp(self, other: FlagsFrame) -> pd.Series:
            if isinstance(other, self.__class__):
                other = other._raw
            df: pd.DataFrame = self._raw
            if isinstance(other, pd.DataFrame):
                if not df.columns.equals(other.columns):
                    raise ValueError("Columns does not match")
                if not df.index.equals(other.index):
                    raise ValueError("Index does not match")
            return getattr(df, funcname)(other)

        return cmp

    __contains__ = compose("_raw", "__contains__")
    __iter__ = compose("_raw", "__iter__")
    __eq__ = __bool_cmp__("__ge__")
    __ne__ = __bool_cmp__("__ge__")
    __gt__ = __bool_cmp__("__ge__")
    __ge__ = __bool_cmp__("__ge__")
    __lt__ = __bool_cmp__("__lt__")
    __le__ = __bool_cmp__("__le__")

    # ############################################################
    # Rendering
    # ############################################################

    @final
    def __str__(self):
        return self.__repr__() + "\n"

    @final
    def __repr__(self) -> str:
        string = repr(self.to_pandas()).replace("DataFrame", self.__class__.__name__)
        meta = dict(self.meta)
        string += f"\nMeta: {utils.repr_extended(meta, kwdicts=True)}"
        return string

    @final
    def to_string(self, *args, show_meta=True, **kwargs) -> str:
        return (
            self.to_pandas()
            .to_string(*args, **kwargs)
            .replace("DataFrame", self.__class__.__name__)
        )


if __name__ == "__main__":
    from saqc._testing import dtindex, N

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
    print(ff == ff2)
    ff2.append([N, 99, N])
    print(ff2)
    print(ff == ff2)
    print(FlagsFrame(ff2))
