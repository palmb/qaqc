#!/usr/bin/env python
from __future__ import annotations

import weakref
from abc import ABC
from typing import Any, Sequence, cast, Callable

import pandas as pd
import numpy as np
import abc
from saqc.constants import UNFLAGGED
from saqc.core.flagsframe import FlagsFrame
from saqc.core.generic import VariableABC
from saqc.core.univariate import UnivariatDataFunc, UnivariatFlagFunc
from saqc.types import Self


def _cast_inplace(obj, klass):
    _type = type(obj)
    if _type is klass:
        pass
    if _type is Variable:
        if klass is MaskedVariable:
            obj.__class__ = klass
            obj._orig = None
            obj._mask = None
    elif _type is MaskedVariable:
        if klass is Variable:
            obj.__class__ = klass
            delattr(obj, '_orig')
            delattr(obj, '_mask')
    else:
        raise NotImplementedError(f"Cannot cast from {_type} to {klass}")
    return cast(klass, obj)


class VariableBase(VariableABC):
    def __init__(
            self, data, index: pd.Index | None = None, flags: FlagsFrame | None = None
    ):
        if isinstance(data, self.__class__):  # fastpath
            self._data = data._data.copy()
            self._flags = FlagsFrame(data.flags.current())
            return

        if index is not None:
            index = index.copy()
            index.name = None
        self._data = pd.Series(data, index=index, dtype=float)

        # the FlagsFrame either is empty (if None is passed) or
        # have a single column if another FlagsFrame is passed,
        # then the current flags are used.
        if flags is None:
            flags = self.data.index
        self._flags: FlagsFrame = FlagsFrame(flags)

    @classmethod
    def _from_other(cls, other: VariableBase) -> Variable:
        inst = cls(None)
        inst._data = other._data
        inst._flags = other._flags
        return cast(Variable, inst)

    @property
    def index(self):
        return self._data.index

    @index.setter
    def index(self, value) -> None:
        self._data.index = value

    @property
    def data(self) -> pd.Series:
        return self._data.copy()

    @property
    def flags(self) -> FlagsFrame:
        return self._flags

    def is_flagged(self) -> pd.Series:
        """return a boolean series that indicates flagged values"""
        return self.flags.is_flagged()

    def copy(self, deep=True):
        c = self.__class__(None)
        c._data = self._data.copy(deep)
        c._flags = self._flags.copy(deep)
        return c

    def equals(self, other) -> bool:
        return (
                isinstance(other, type(self))
                and self.data.equals(other.data)
                and self.flags.equals(other.flags)
        )

    def __eq__(self, other) -> bool:
        return self.equals(other)

    def __repr__(self):
        return repr(self._df_to_render()).replace("DataFrame", "Variable") + "\n"

    def _df_to_render(self) -> pd.DataFrame:
        # newest first ?
        # df = self.flags.raw.loc[:, ::-1]
        df = self.flags.raw.loc[:]
        df.insert(0, "|", ["|"] * len(df.index))
        df.insert(0, "flags", self.flags.current().array)
        df.insert(0, "data", self.data.array)
        return df

    def to_string(self, *args, **kwargs) -> str:
        return self._df_to_render().to_string(*args, **kwargs)

    def mask_data(self, mask=None, inplace=False) -> MaskedVariable:
        if mask is None:
            mask = self.is_flagged()
        inst: Variable = self if inplace else self.copy()
        inst: MaskedVariable = _cast_inplace(inst, MaskedVariable)
        inst._mask_data(mask)
        return inst

    @property
    def is_masked(self):
        return False


class Variable(VariableBase, UnivariatFlagFunc):

    @property
    def _constructor(self: Variable) -> type[Variable]:
        return self.__class__


class MaskedVariable(Variable):

    def __init__(
            self,
            data,
            index: pd.Index | None = None,
            flags: FlagsFrame | None = None,
            mask: Any = None,
    ):
        super().__init__(data, index, flags)
        self._orig = None
        self._mask = None
        if mask is not None:
            self._mask_data(mask)

    @classmethod
    def _from_other(cls, other: VariableBase, mask=None) -> MaskedVariable:
        inst = super()._from_other(other)
        inst._orig = None
        inst._mask = None
        return cast(MaskedVariable, inst)

    def _mask_data(self, mask) -> MaskedVariable:
        """mask data inplace."""
        mask = pd.Series(mask)  # cast and ensure shallow copy
        assert len(mask) == len(self.index)
        self._orig = self._data.copy()
        self._mask = mask
        self._data[mask] = np.nan
        return self

    @property
    def is_masked(self):
        if self._mask is None:
            return False
        return self.mask.any()

    @property
    def mask(self) -> pd.Series:
        if self._mask is None:
            return pd.Series(False, index=self.index, dtype=bool)
        return self._mask.copy()

    def mask_data(self, mask=None, **kwargs) -> MaskedVariable:
        pass

    def unmask_data(self):
        data = self._orig
        mask = self._mask
        data[mask] = self._data[mask]
        self._data = data
        return self


if __name__ == "__main__":
    from saqc._testing import dtindex, N

    m = MaskedVariable(None)

    v = Variable([1, 2, 3, 4], index=dtindex(4))
    v = v.flag_limits(2)
    v.flags.append(pd.Series([N, 5, 5, N]))
    print(v)
