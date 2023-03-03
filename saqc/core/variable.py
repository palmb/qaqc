#!/usr/bin/env python
from __future__ import annotations

import weakref
from abc import ABC
from typing import Any, Sequence, cast, Callable, NoReturn, final

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
            delattr(obj, "_orig")
            delattr(obj, "_mask")
    else:
        raise NotImplementedError(f"Cannot cast from {_type} to {klass}")
    return cast(klass, obj)


class VariableBase(VariableABC, ABC):
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

    def flagged(self) -> pd.Series:
        """return a boolean series that indicates flagged values"""
        return self.flags.flagged()

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

    # ############################################################
    # Masking
    # ############################################################

    def mask_data(self, mask=None, inplace=False) -> MaskedVariable:
        if mask is None:
            mask = self.flagged()
        else:
            mask = pd.Series(mask, index=self.index, dtype=bool)
        inst: Variable = self if inplace else self.copy()
        inst: MaskedVariable = _cast_inplace(inst, MaskedVariable)
        inst._mask_data(mask)
        return inst

    def unmask_data(self, inplace=False) -> Variable:
        return self if inplace else self.copy()

    @property
    def is_masked(self):
        return False

    # ############################################################
    # Rendering
    # ############################################################

    @final
    def __repr__(self):
        return (
            repr(self._df_to_render()).replace("DataFrame", self.__class__.__name__)
            + "\n"
        )

    def _df_to_render(self) -> pd.DataFrame:
        # newest first ?
        # df = self.flags.raw.loc[:, ::-1]
        df = self.flags.raw.loc[:]
        df.insert(0, "|", ["|"] * len(df.index))
        df.insert(0, "flags", self.flags.current().array)
        df.insert(0, "data", self.data.array)
        return df

    @final
    def to_string(self, *args, **kwargs) -> str:
        return (
            self._df_to_render()
            .replace("DataFrame", self.__class__.__name__)
            .to_string(*args, **kwargs)
        )


class Variable(VariableBase, UnivariatFlagFunc):
    @property
    def _constructor(self: Variable) -> type[Variable]:
        return self.__class__


class MaskedVariable(Variable):
    @property
    def _constructor(self: MaskedVariable) -> type[MaskedVariable]:
        return self.__class__

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

    def copy(self, deep=True) -> MaskedVariable:
        c = cast(MaskedVariable, super().copy(deep))
        if self._orig is not None:
            c._orig = self._orig.copy(deep)
        if self._mask is not None:
            c._mask = self._mask.copy(deep)
        return c

    @property
    def is_masked(self):
        return self._mask is not None and self.mask.any()

    @property
    def mask(self) -> pd.Series:
        if self._mask is None:
            return pd.Series(False, index=self.index, dtype=bool)
        return self._mask.copy()

    @mask.setter
    def mask(self, value):
        mask = pd.Series(value, index=self.index, dtype=bool)
        self._unmask_data()
        self._mask_data(mask)

    def _mask_data(self, mask) -> MaskedVariable:
        """mask data inplace."""
        mask = pd.Series(mask)  # cast and ensure shallow copy
        assert len(mask) == len(self.index)
        self._orig = self._data.copy()
        self._mask = mask
        self._data[mask] = np.nan
        return self

    def _unmask_data(self) -> MaskedVariable:
        self._data = self._orig
        self._mask = None
        self._orig = None
        return self

    def mask_data(self, mask=None, inplace=False) -> MaskedVariable:
        if mask is None:
            mask = self.flagged()
        else:
            mask = pd.Series(mask, index=self.index, dtype=bool)
        c = self.copy()
        c._unmask_data()
        c._mask_data(mask)
        if inplace:
            self._orig = c._orig
            self._mask = c._mask
            self._data = c._data
            c = self
        return c

    def unmask_data(self, inplace=True) -> Variable:
        obj: MaskedVariable = self if inplace else self.copy()
        obj._unmask_data()
        obj: Variable = _cast_inplace(obj, Variable)
        return obj

    # ############################################################
    # Rendering
    # ############################################################

    def _df_to_render(self) -> pd.DataFrame:
        df = super()._df_to_render()
        df.insert(1, "mask", self.mask.array)
        return df


if __name__ == "__main__":
    from saqc._testing import dtindex, N

    print(Variable(None))
    print(MaskedVariable(None))

    v = Variable([1, 2, 3, 4], index=dtindex(4))
    v = v.flag_limits(2)
    v.flags.append(pd.Series([N, 5, 5, N]))
    v = v.mask_data(inplace=True)
    print(v)
    new = Variable(v)
    print(new)
    v = v.mask_data(mask=True, inplace=True)
    print(v)
    v._data[:] = 99
    print(v)
    v.unmask_data(inplace=True)
    print(v)
