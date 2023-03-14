#!/usr/bin/env python
from __future__ import annotations

import pandas as pd
import numpy as np
from copy import deepcopy
from qaqc._typing import SupportsIndex, final
from qaqc.constants import UNFLAGGED
import qaqc.core.utils as utils
from typing import Any
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
        return f"Meta: {utils.repr_extended(dict(self._raw), kwdicts=True, wrap=wrap)}"


class FlagsFrame(OpsMixin):
    __slots__ = ("_raw", "_meta")
    _raw: pd.DataFrame
    _meta: Meta

    """
    Formerly known as History
    - existing data cannot be mutated
    - data (columns) can be appended
    - created empty from a pd.Index
    - comparable eq, ne, le, ge, lt, gt return pd.Dataframes
    """

    def __init__(self, index: Any = None, initial: SupportsIndex | None = None) -> None:
        if index is None:
            if (
                initial is not None
                and hasattr(initial, "index")
                and isinstance(initial.index, pd.Index)
            ):
                index = initial.index
            else:
                raise TypeError(
                    "If no 'index' is given, 'initial' must have a pandas.Index."
                )
        # special case:  FlagsFrame(flags_frame)
        elif isinstance(index, type(self)):
            initial = index.current()
            index = index.index
        elif not isinstance(index, pd.Index):
            index = pd.Index(index)

        if isinstance(initial, type(self)):
            initial = initial.current()

        self._raw = pd.DataFrame(index=index.copy(), dtype=float)
        self._meta = Meta()

        if initial is not None:
            self.append(initial, initial=True)

    def copy(self, deep=True) -> FlagsFrame:
        new = self.__class__(self.index)
        new._raw = self._raw.copy(deep)
        new._meta = self._meta.copy(deep)
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
        assert value.index.equals(self.columns)
        self._meta = meta

    def __len__(self):
        return len(self._raw.columns)

    def __contains__(self, key) -> bool:
        return self._raw.__contains__(key)

    def __getitem__(self, key):
        return self._raw.__getitem__(key).copy()

    def __setitem__(self, key, value):
        raise NotImplementedError("use 'append' instead")

    def current(self) -> pd.Series:
        if self._raw.empty:
            result = pd.Series(data=np.nan, index=self.index, dtype=float)
        else:
            result = self._raw.ffill(axis=1).iloc[:, -1]
        return result.fillna(UNFLAGGED)

    def is_flagged(self) -> pd.DataFrame:
        return self._raw > UNFLAGGED

    def is_unflagged(self) -> pd.DataFrame:
        return self._raw == UNFLAGGED

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
    def __str__(self):
        return self.__repr__() + "\n"

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
