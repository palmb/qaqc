#!/usr/bin/env python
from __future__ import annotations

import weakref
from abc import ABC
from typing import Any, Sequence

import pandas as pd
import numpy as np
import abc
from saqc.constants import UNFLAGGED
from saqc.core.flagsframe import FlagsFrame
from saqc.core.generic import VariableABC
from saqc.core.masking import MaskingContextManager
from saqc.types import Self


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

    def copy(self, deep=True):
        return self.__class__(data=self._data.copy(deep), flags=self._flags.copy(deep))

    def equals(self, other) -> bool:
        return (
            isinstance(other, type(self))
            and self.data.equals(other.data)
            and self.flags.equals(other.flags)
        )

    def __eq__(self, other) -> bool:
        return self.equals(other)

    def __repr__(self) -> str:
        return repr(self._df_to_render()) + "\n"

    def _df_to_render(self) -> pd.DataFrame:
        df = self.flags.raw.loc[:, ::-1]
        df.insert(0, "|", ["|"] * len(df.index))
        df.insert(0, "flags", self.flags.current().array)
        df.insert(0, "data", self.data.array)
        return df

    def to_string(self, *args, **kwargs) -> str:
        return self._df_to_render().to_string(*args, **kwargs)


class UnivariatFunctions(VariableABC, ABC):

    def flag_limits(self: Variable, lower=-np.inf, upper=np.inf, flag=9) -> Variable:
        result = self.copy(deep=False)
        mask = (result.data < lower) | (result.data > upper)
        result.flags.append_with_mask(mask, flag)
        return result

    def flag_something(self: Variable, flag=99) -> Variable:
        result = self.copy(deep=False)
        sample = result.data.sample(frac=0.3)
        new = result.flags.template()
        new[sample.index] = flag
        result.flags.append(new)
        return result

    def clip(self: Variable, lower, upper, flag=-88) -> Variable:
        # with data manipulation we must crate new Variable, not a copy
        result = self.flag_limits(lower, upper).copy()
        result.data.clip(lower, upper, inplace=True)
        return Variable(result)

    def flagna(self, flag=999):
        result = self.copy(False)
        meta = dict(source='flagna')
        result.flags.append_with_mask(result.data.isna(), flag, meta)
        return result

    def interpolate(self: Variable, flag=None) -> Variable:
        # with data manipulation we must crate
        # new Variable, not a copy
        meta = dict(source='interpolate')
        if flag is not None:
            flags = self.flagna().flags
        else:
            flags = self.flags
        data = self.data.interpolate()
        return Variable(data, flags)

    def reindex(self: Variable, index=None, method=None) -> Variable:
        data = self.data.reindex(index, method=method)
        return Variable(data)

    def flag_by_condition(self, func, flag=99):
        # func ==> ``lambda v: v.data != 3``
        new = self.copy()
        mask = func(self.data)
        new.flags.append_with_mask(mask, flag, dict(source='flag_by_condition'))
        return new


class Variable(VariableBase, UnivariatFunctions):
    pass


# Series: base for data storage
# Variable has data-series and flags-series
#   flags always as long as data
#   can reindex and stuff
# MaskedVariable has masked-data-series and flags series
#   flags always as long as data
#   only non index changing operations


class Variable2:
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

    def hide_data(self, mask=None):
        if mask is None:
            mask = self.flags.current() > 0
        else:
            mask = pd.Series(mask)
        self._orig = self._data.copy()
        self._mask = mask
        self._data[mask] = np.nan
        return self

    def unhide_data(self):
        data = self._orig
        mask = self._mask
        data[mask] = self._data[mask]
        self._data = data
        return self

    def foo666(self):
        self.hide_data()
        self.data[:] = 666
        self.unhide_data()
        return self

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


if __name__ == "__main__":
    N = np.nan
    v = Variable()
    v = v.flag_limits()
    v.flags.append(pd.Series([N, 5, 5, N]))
    v.foo666()
    print(v)
