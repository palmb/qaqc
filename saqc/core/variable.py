#!/usr/bin/env python
from __future__ import annotations

import warnings

import pandas as pd
import numpy as np
import inspect

from saqc.core.base import BaseVariable
from saqc._typing import FlagLike
from types import BuiltinFunctionType
import saqc.core.utils as utils

# todo: Feature: #tags
# todo: Feature: translation
# todo: Func-obj, name, file, lineno
#   meta["src"] = self.flag_limits   the function itself ??
#   func to evaluate funcs in meta (aka callable_to_string)
#       -> "flag_limits in /path/to/file.py, line n"


def _parent(level=2):
    # get parent function
    stack = inspect.stack(level)
    func: callable = sum
    return func

class Variable(BaseVariable):  # noqa
    @property
    def _constructor(self: Variable) -> type[Variable]:
        return type(self)

    # ############################################################
    # Flagging
    # ############################################################

    def flag_limits(self, lower=-np.inf, upper=np.inf, flag: FlagLike = 9) -> Variable:
        mask = (self.data < lower) | (self.data > upper)
        return self.copy().set_flags(flag, mask, src=self.flag_limits)

    def flag_random(self, frac=0.3, flag: FlagLike = 99) -> Variable:
        sample = self.data.dropna().sample(frac=frac)
        return self.copy().set_flags(flag, sample.index, src="flag_random")

    def flagna(self, flag: FlagLike = 999) -> Variable:
        # no masking desired !
        isna = self.orig.isna()
        return self.copy().set_flags(flag, isna, src="flagna")

    def replace_flag(self, old: FlagLike, new: FlagLike) -> Variable:
        mask = self.flags == old
        return self.copy().set_flags(new, mask, src="replace_flag")

    def flag_generic(self, func: callable, raw=False, flag: FlagLike = 99) -> Variable:
        # func ==> `lambda v: v.data != 3`
        if not callable(func):
            raise TypeError('func must be a callable')

        meta = dict(func_name=func.__name__)
        if hasattr(func, '__code__'):
            file = func.__code__.co_filename
            line = func.__code__.co_firstlineno
            meta.update(func=f'File "{file}, line {line}"')
        elif isinstance(func, BuiltinFunctionType):
            meta.update(func='buildin')

        data = self.data
        if raw:
            data = data.to_numpy()
        mask = func(data)

        if pd.api.types.is_iterator(mask):
            # eg. lambda data: (d==99 for d in data)
            mask = list(mask)

        if not utils.is_listlike(mask) or not utils.is_boolean_indexer(mask):
            raise TypeError(
                "Expected result of given function to be a boolean list-like."
            )

        return self.copy().set_flags(flag, mask, src='flag_generic', **meta)

    # ############################################################
    # Corrections
    # ############################################################
    #  Functions that alter data.
    # - must return a new Variable
    # - may set new fframe on the new Variable
    # - may use the existing old Flags squeezed to a pd.Series
    #  (see `FlagsFrame.current`) as the initial fframe for the
    #  new Variable
    # ############################################################

    # def clip(self, lower, upper, flag=-88) -> UnivariateMixin:
    #     # keep index
    #     # alter data
    #     # initial fframe: squeezed old
    #     result = self.flag_limits(lower, upper, flag=flag).copy()
    #     result.data.clip(lower, upper, inplace=True)
    #     return self._constructor(result)
    #
    # def interpolate(self, flag=None) -> UnivariateMixin:
    #     # keep index
    #     # alter data
    #     # initial fframe: squeezed old
    #     if flag is not None:
    #         flags = self.flagna(flag).history
    #     else:
    #         flags = self.history
    #     data = self.data.interpolate()
    #     return self._constructor(data, flags)
    #
    # def reindex(self, index=None, method=None) -> UnivariateMixin:
    #     # - set new index
    #     # - reset all fframe
    #     data = self.data.reindex(index, method=method)
    #     return self._constructor(data)
    #


if __name__ == "__main__":
    from saqc._testing import dtindex, N
    from saqc.core.variable import Variable

    def do1(data):
        return data == 3

    do2 = lambda x: pd.Series(False, index=x.index)

    print(Variable(None))

    v = Variable([1, 2, 3, 4], index=dtindex(4))
    v = v.flag_limits(2)
    print(v)
    v = v.set_flags(pd.Series([N, 5, 5, N]))
    v = v.flag_generic(do1)
    v = v.flag_generic(do2)
    print(v.meta.to_pandas().to_string())
