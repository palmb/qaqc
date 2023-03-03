#!/usr/bin/env python
from __future__ import annotations

import pandas as pd
import numpy as np
from saqc.core.generic import compose
from copy import deepcopy
from saqc.types import Any, SupportsIndex
from saqc.constants import UNFLAGGED
from typing import Any, Callable, Dict, List, Tuple


class FlagsFrame:

    """
    Formerly known as History
    - existing data cannot be mutated
    - data (columns) can be appended
    - created empty from a pd.Index
    - comparable eq, ne, le, ge, lt, gt return pd.Dataframes
    """

    def __init__(self, flags_or_index: pd.Index | FlagsFrame) -> None:
        if isinstance(flags_or_index, self.__class__):
            values = flags_or_index.current().array
            index = flags_or_index.index
        else:
            values = None
            index = flags_or_index
        self._raw = pd.DataFrame(values, index=index, dtype=float)
        self._meta = []

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
        new = FlagsFrame(None)
        new._raw = self._raw.copy(deep)
        new._meta = deepcopy(self._meta) if deep else list(self._meta)
        return new

    def current(self) -> pd.Series:
        if self._raw.empty:
            result = pd.Series(data=np.nan, index=self.index, dtype=float)
        else:
            result = self.raw.ffill(axis=1).iloc[:, -1]
        return result.fillna(UNFLAGGED)

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
        def cmp(self, other: FlagsFrame) -> pd.DataFrame:
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
        return repr(self._raw).replace("DataFrame", "FlagsFrame") + "\n"


if __name__ == "__main__":
    ff = FlagsFrame.init_like(pd.Series([1, 2, 3]))
    print(ff >= 2)
    ff2 = ff.copy()
    ff2.append([2, 2, 2])
    print(ff == ff2)
