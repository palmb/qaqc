#!/usr/bin/env python
from __future__ import annotations

import pandas as pd
import numpy as np
from saqc.core.generic import compose
from copy import deepcopy
from saqc._typing import SupportsIndex
from saqc.constants import UNFLAGGED
from typing import Any
# from pandas.core.arraylike import OpsMixin


class FlagsFrame:

    __slots__ = ('_raw', "_meta")
    _raw: pd.DataFrame
    _meta: list[dict[str, Any]]

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
                    "If no 'index' is given, 'initial' must "
                    "have a pandas.Index."
                )

        # special case:  FlagsFrame(flags_frame)
        elif isinstance(index, type(self)):
            initial = index.current()
            index = index.index
        else:
            try:
                index = pd.Index(index)
            except TypeError:
                raise TypeError(f"Cannot create index for from {type(index)}")

        index.name = None

        if isinstance(initial, type(self)):
            initial = initial.current()

        self._raw = pd.DataFrame(index=index, dtype=float)
        self._meta = []

        if initial is not None:
            self.append(initial, dict(source='init'))

    @property
    def raw(self) -> pd.DataFrame:
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
    def meta(self) -> list[dict[str, Any]]:
        # we use a shallow copy, so user can modify the data
        # within meta, but cannot remove from or append to meta
        return list(self._meta)

    def _validate(self) -> None:
        assert len(self._raw.columns) == len(self.meta)

    def copy(self, deep=True) -> FlagsFrame:
        cfunc = deepcopy if deep else list
        new = self.__class__(self.index)
        new._raw = self._raw.copy(deep)
        new._meta = cfunc(self.meta)
        return new

    def current(self) -> pd.Series:
        if self._raw.empty:
            result = pd.Series(data=np.nan, index=self.index, dtype=float)
        else:
            result = self.raw.ffill(axis=1).iloc[:, -1]
        return result.fillna(UNFLAGGED)

    def flagged(self) -> pd.Series:
        return self.current() > UNFLAGGED

    def template(self, fill_value=np.nan) -> pd.Series:
        return pd.Series(fill_value, index=self.index, dtype=float)

    def append_with_mask(
        self, mask, flag: float | int, meta: dict | None = None
    ) -> FlagsFrame:
        new = self.template()
        new[mask] = float(flag)
        self.append(new, meta)
        return self

    def append(self, value, meta: dict | None = None) -> FlagsFrame:
        value = pd.Series(value, dtype=float)
        assert len(value) == len(self.index)
        self._raw[f"f{len(self)}"] = value.array
        self._meta.append(meta or dict())
        return self

    def equals(self, other: FlagsFrame) -> bool:
        return isinstance(other, self.__class__) and self._raw.equals(other._raw)

    def __len__(self):
        return len(self._raw.columns)

    @staticmethod
    def __bool_cmp__(funcname):
        def cmp(self, other: FlagsFrame) -> pd.Series:
            if isinstance(other, self.__class__):
                other = other.current()
            current = self.current()
            return getattr(current, funcname)(other)

        return cmp

    __contains__ = compose("_raw", "__contains__")
    __iter__ = compose("_raw", "__iter__")
    __eq__ = __bool_cmp__("__ge__")
    __ne__ = __bool_cmp__("__ge__")
    __gt__ = __bool_cmp__("__ge__")
    __ge__ = __bool_cmp__("__ge__")
    __lt__ = __bool_cmp__("__lt__")
    __le__ = __bool_cmp__("__le__")

    def __repr__(self):
        return repr(self._df_to_render()).replace("DataFrame", "FlagsFrame") + "\n"

    def _df_to_render(self) -> pd.DataFrame:
        # newest first ?
        # df = self._raw.loc[:, ::-1]
        df = self._raw.loc[:]
        df.insert(0, "|", ["|"] * len(df.index))
        df.insert(0, "current", self.current().array)
        return df

    def to_string(self, *args, **kwargs) -> str:
        return self._df_to_render().to_string(*args, **kwargs)


if __name__ == "__main__":
    from saqc._testing import dtindex, N

    s = pd.Series([1, 2, 3], index=dtindex(3))
    ff = FlagsFrame(s.index)
    ff.append([N, N, N])
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
