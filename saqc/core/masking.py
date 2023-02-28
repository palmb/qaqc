#!/usr/bin/env python
from __future__ import annotations

import numpy as np
import weakref

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from saqc.core.variable import Variable


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


