#!/usr/bin/env python
from __future__ import annotations

import numpy as np
import weakref
import abc

from typing import TYPE_CHECKING, TypeVar
if TYPE_CHECKING:
    from saqc.core.variable import Variable

T = TypeVar('T')


class MaskingContextManager:
    def __init__(self, obj: Variable, invert=False):
        self._obj: Variable = obj
        self._invert = invert
        self._orig = None
        self._mask = None
        self._kwargs = {}
        # We need to keep track of copies that are made
        # in our context. On exit of our context we call
        # unmasking with each created copy.
        self._refs = weakref.WeakSet()

    # arguments to `with obj.masking(mask=...) as masked: ...`
    def __call__(self, mask=None, **kwargs) -> MaskingContextManager:
        self._mask = self._obj.flagged() if mask is None else mask
        if self._invert:
            self._mask = ~self._mask
        self._kwargs = kwargs
        return self

    def __enter__(self) -> Variable:
        self._orig = self._obj.data.copy()
        self._obj.data[self._mask] = np.nan
        return self._obj

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._unmask(self._obj)
        self._unmask_copies()
        return None

    def _unmask_copies(self):
        for _ in range(len(self._refs)):
            obj = self._refs.pop()
            self._unmask(obj)

    def _unmask(self, obj):
        obj.data[self._mask] = self._orig[self._mask]
        return obj

    # ############################################################
    # Public API
    # ############################################################

    def mask(self, mask=None) -> Variable:
        """Store and reset data at mask until unmask is called."""
        return self.__call__(mask).__enter__()

    def unmask(self) -> Variable:
        """Restore original data to previously masked locations."""
        return self._unmask(self._obj)

    def register(self, obj: Variable):
        """register a new object for unmasking on exit."""
        assert obj.data.index.equals(self._orig.index)
        self._refs.add(obj)

    def copy(self, obj) -> MaskingContextManager:
        assert obj.data.index.equals(self._orig.index)
        new = MaskingContextManager(obj, invert=self._invert)
        new._orig = self._orig
        new._mask = self._mask
        new._kwargs = self._kwargs
        # do not copy self.refs
        return new


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




