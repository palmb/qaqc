#!/usr/bin/env python
from __future__ import annotations

import weakref
from typing import Any, Sequence

import pandas as pd
import numpy as np
import abc
from saqc.constants import UNFLAGGED
from saqc.core.flagsframe import FlagsFrame
from saqc.core.generic import FuncMixin
from saqc.core.masking import MaskingContextManager
from saqc.types import T


class _MaskingMixin(abc.ABC):
    def __init__(self, *args, **kwargs):
        # context manager for use with `with`-statement
        # we hold a list to allow nested `with`-statements
        self._cms: weakref.WeakSet[MaskingContextManager] = weakref.WeakSet()
        # context manager for direct calls to `mask` and `unmask`
        self._cm: MaskingContextManager | None = None

    @abc.abstractmethod
    def copy(self: T) -> T:
        ...

    def __finalize__(self: T, obj) -> T:
        # self is new, obj is old
        if obj._cm is not None:
            self._cm = obj._cm.copy(self)

        self._cms = obj._cms.copy()
        for cm in obj._cms:
            cm.register(self)
        return self

    @property
    def is_masked(self) -> bool:
        # todo: check if weakref.WeakSet works as expected in
        #   any case
        return not (self._cm is None or len(self._cms) == 0)

    # ############################################################
    # with statement context manager
    # ############################################################

    @property
    def selecting(self) -> MaskingContextManager:
        new = self.copy()
        cm = MaskingContextManager(new, invert=True)
        new._cms.add(cm)
        return cm

    @property
    def masking(self) -> MaskingContextManager:
        new = self.copy()
        cm = MaskingContextManager(new)
        new._cms.add(cm)
        return cm

    # ############################################################
    # method interface
    # ############################################################

    def _mask_or_select(self: T, mask=None, invert=False) -> T:
        if self._cm is not None:
            raise RuntimeError(
                "Data is already masked or selected. For mixing or nesting "
                "masks and/or selections use the contextmanager `masking` "
                "and `selecting` with the `with` statement. "
            )
        self._cm = MaskingContextManager(self, invert=invert)
        return self._cm.mask(mask)

    def _undo(self: T, action: str) -> T:
        if self._cm is None:
            raise RuntimeError(f"Data is not {action}ed.")
        assert self._cm._obj is self
        self._cm.unmask()
        self._cm = None  # destroy reference
        return self

    def mask(self: T, mask=None, copy=False) -> T:
        obj = self.copy() if copy else self
        return obj._mask_or_select(mask=mask)

    def select(self: T, sel=None, copy=False) -> T:
        obj = self.copy() if copy else self
        return obj._mask_or_select(mask=sel, invert=True)

    def unmask(self: T, copy=False) -> T:
        obj = self.copy() if copy else self
        return obj._undo("mask")

    def deselect(self: T, copy=False) -> T:
        obj = self.copy() if copy else self
        return obj._undo("select")


class Series:
    def __init__(
        self,
        data: Any | None = None,
        index: pd.Index | None = None,
    ):
        data = pd.Series(data, index=index, dtype=float)
        self._raw: np.ndarray = np.array(data, dtype=float)
        self._index: pd.Index = data.index

    @property
    def raw(self) -> np.ndarray:
        return self._raw

    @property
    def index(self) -> pd.Index:
        return self._index

    @index.setter
    def index(self, value):
        index = pd.Index(value)
        if isinstance(index, pd.MultiIndex):
            raise ValueError("MultiIndex is not allowed here.")
        if not len(index) == len(self._index):
            raise ValueError(
                f"Length mismatch: Expected axis "
                f"has {len(self._index)} elements, "
                f"new values have {len(index)} elements"
            )
        if not index.is_unique:
            raise ValueError("Index must not have duplicates")
        self._index = index

    # def __getitem__(self, item):
    #     self._raw.__getitem__(item)
    #
    # def __setitem__(self, key, value):
    #     self._raw.__setitem__(key, value)

    def to_pandas(self, copy=True) -> pd.Series:
        return pd.Series(data=self._raw, index=self._index, dtype=float, copy=copy)

    def __repr__(self):
        if not hasattr(self, '_index'):
            return super().__repr__()  # for errors within init
        return repr(self.to_pandas(copy=False))


class MaskedSeries(Series):
    def __init__(
        self,
        data: Any | None = None,
        index: pd.Index | None = None,
        mask: Sequence | None = None
    ):
        super().__init__(data=data, index=index)
        self._raw: np.ma.MaskedArray = np.ma.array(self._raw, mask=mask, hard_mask=True)

    @property
    def mask(self):
        return self._raw.mask

    @mask.setter
    def mask(self, value):
        if len(self.mask) != value:
            raise ValueError(
                f"Length mismatch: Expected array "
                f"has {len(self._index)} elements, "
                f"given mask have {len(value)} elements"
            )
        self._raw.mask = value

# Series: base for data storage
# Variable has data-series and flags-series
#   flags always as long as data
#   can reindex and stuff
# MaskedVariable has masked-data-series and flags series
#   flags always as long as data
#   only non index changing operations

class Variable(Series):
    pass


class Variable2(FuncMixin, _MaskingMixin):
    def __init__(
        self,
        data: pd.Series,
        flags: FlagsFrame | None = None,
        attrs: dict | None = None,
    ):
        super().__init__()
        self._data: pd.Series = data
        if flags is None:
            flags = FlagsFrame(
                data=pd.DataFrame(np.nan, index=data.index, columns=[0], dtype=float),
                meta=[dict()],
            )
        self._flags: FlagsFrame = flags
        self.attrs: dict = attrs or dict()
        assert isinstance(self._data, pd.Series)
        assert isinstance(self._flags, FlagsFrame)
        assert isinstance(self.attrs, dict)
        self._validate()

    data = property(lambda self: self._data)
    flags = property(lambda self: self._flags)
    index = property(lambda self: self._data.index)

    def var_only_meth(self):
        pass

    def copy(self) -> Variable:
        return Variable(
            data=self._data.copy(), flags=self._flags.copy(), attrs=dict(self.attrs)
        ).__finalize__(self)

    def _validate(self):
        assert self._data.index.equals(self._flags.index)

    def flagged(self) -> pd.Series:
        return self.flags.current() > UNFLAGGED  # noqa

    def __repr__(self) -> str:
        df = self._data.to_frame("data")
        df["final-flags"] = self.flags.current()
        return repr(df) + "\n"

    def dododo(self, arg0, arg1, kw0=None, kw1=None):
        self.data[::2] = -self.data[::2]
        return self

if __name__ == '__main__':
    s = Series([1,2,3])
    print(s)