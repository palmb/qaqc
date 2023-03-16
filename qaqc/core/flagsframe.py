#!/usr/bin/env python
from __future__ import annotations

import pandas as pd
import numpy as np
from copy import deepcopy
from qaqc.typing import SupportsIndex, final, Idx
from qaqc.constants import SET_UNFLAGGED
from qaqc.core.utils import construct_index, repr_extended, check_index
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
        return f"Meta: {repr_extended(dict(self._raw), kwdicts=True, wrap=wrap)}"


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

    @property
    def _constructor(self: FlagsFrame) -> type[FlagsFrame]:
        return type(self)

    def __init__(self, initial: Any | None = None, index: Idx | None = None) -> None:
        # late import to avoid cycles
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

    def copy(self, deep=True) -> FlagsFrame:
        new = self._constructor(index=self.index)
        new._raw = self._raw.copy(deep=deep)
        new._meta = self._meta.copy(deep=deep)
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
        return result.replace(SET_UNFLAGGED, np.nan)

    def is_flagged(self) -> pd.DataFrame:
        return self._raw.notna() & (self._raw > SET_UNFLAGGED)

    def is_unflagged(self) -> pd.DataFrame:
        return self._raw.isna() | (self._raw == SET_UNFLAGGED)

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

    def get_meta_mapping(self, drop_unflagged: bool = False):
        """mapping for current flags to meta.

        This can be used to find the meta information for current flags.

        Parameters
        ----------

        drop_unflagged : bool, default False
            Reduce mapping to flagged data

        Returns
        -------
        mapping: pd.Series
            A series mapping flags index to meta key
        """
        mapping, *_ = self._get_meta_mapping(reduce=drop_unflagged)
        if mapping is None:
            mapping = pd.Series(np.nan, index=self.index, dtype=str)
        if drop_unflagged:
            mapping.dropna(inplace=True)
        return mapping

    def _get_meta_mapping(
        self,
        reduce: bool = True,
    ) -> tuple[pd.Series, Meta, str | None] | tuple[None, None, None]:
        """mapping for current flags to meta.

        This can be used to find the meta information for current flags.

        None

        Parameters
        ----------
        reduce : bool, default False
            Reduce mapping and meta to relevant info only

        Returns
        -------
        mapping: pd.Series or None
            A series mapping flags index to meta key, or None if no flags present
        meta: Meta or None
            A reduced meta, holding only information that are mapped to, or None
            if no flags present
        last_column: str or None
            Last column of the current History, or None if history is empty

        """
        df = self.to_pandas()
        meta = self.meta.copy(deep=True)

        last = None
        for fn in df.columns:
            last = fn
            # we ignore SET_UNFLAGGED here because it holds info
            flagged = df[fn].notna()
            if reduce and not flagged.any():
                meta._raw.pop(fn)
            df.loc[flagged, fn] = fn

        if df.empty:
            return None, None, last

        mapping = df.ffill(axis=1).iloc[:, -1]
        if reduce:
            mapping.dropna(inplace=True)

        return mapping, meta, last


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
