#!/usr/bin/env python
from __future__ import annotations

from typing import Callable, Union, Any

import pandas as pd
import numpy as np

from qaqc import UNFLAGGED
from qaqc.core.base import BaseVariable
from qaqc.typing import FlagLike
import qaqc.core.utils as utils
from qaqc.core.utils import get_caller as this

__all__ = ["Variable"]

# todo: Feature: #tags
# todo: Feature: translation


class Variable(BaseVariable):  # noqa
    @property
    def _constructor(self: Variable) -> type[Variable]:
        return type(self)

    # ############################################################
    # tools
    # ############################################################

    def flag_nothing(self, *args, **kwargs) -> Variable:
        """dummy function nothing"""
        return self.copy()

    def replace_flag(self, old: FlagLike, new: FlagLike) -> Variable:
        mask = self.flags == old
        return self.copy().set_flags(new, mask, func=this())

    def clear_flags(self) -> Variable:
        return self.copy().set_flags(UNFLAGGED, self.index, func=this())

    def flag_unflagged(self, flag: FlagLike) -> Variable:
        return self.copy().set_flags(flag, self.is_unflagged(), func=this())

    # ############################################################
    # Flagging
    # ############################################################

    def flag_limits(self, lower=-np.inf, upper=np.inf, flag: FlagLike = 9) -> Variable:
        mask = (self.data < lower) | (self.data > upper)
        return self.copy().set_flags(flag, mask, func=this())

    def flag_random(self, frac=0.3, flag: FlagLike = 99) -> Variable:
        sample = self.data.dropna().sample(frac=frac)
        return self.copy().set_flags(flag, sample.index, func=this())

    def flagna(self, flag: FlagLike = 999) -> Variable:
        # no masking desired !
        isna = self.orig.isna()
        return self.copy().set_flags(flag, isna, func=this())

    def flag_generic(
        self,
        # func: Callable[[FloatSeries], BoolSeries] | Callable[[FloatArray], BoolArray],
        func: Callable[..., pd.Series | np.ndarray],
        raw: bool = False,
        flag: FlagLike = 99,
    ) -> Variable:
        # func ==> `lambda v: v.data != 3`

        func_info = utils.get_func_location(func)

        data = self.data
        if raw:
            data = data.to_numpy()  # type: ignore
        mask = func(data)

        if pd.api.types.is_iterator(mask):
            # eg. lambda data: (d==99 for d in data)
            mask = list(mask)  # type: ignore

        if not utils.is_listlike(mask) or not utils.is_boolean_indexer(mask):
            raise TypeError(
                "Expected result of given function to be a boolean list-like."
            )

        return self.copy().set_flags(flag, mask, func=this(), user_func=func_info)

    # ############################################################
    # Corrections
    # ############################################################
    #  Functions that alter data.
    # - must return a new Variable
    # - may set new flags on the new Variable
    # - may use the existing old History squeezed to a pd.Series
    #  (see `FlagsFrame.current`) as the initial flags for the
    #  new Variable
    # ############################################################

    def clip(self, lower, upper, flag: FlagLike = -88) -> Variable:
        # todo: meta
        flags = self.flag_limits(lower, upper, flag=flag).flags
        data = self.data.clip(lower, upper)
        return self._constructor(data, flags, index=self.index)

    def interpolate(self, flag: FlagLike | None = None) -> Variable:
        # todo: meta
        if flag is not None:
            flags = self.flagna(flag).flags
        else:
            flags = self.flags
        data = self.data.interpolate()
        return self._constructor(data, flags)

    def reindex(self, index=None, method=None) -> Variable:
        # todo: meta
        data = self.data.reindex(index, method=method)
        return self._constructor(data)


if __name__ == "__main__":
    from qaqc._testing import dtindex, N
    from qaqc.core.variable import Variable

    def do1(data):
        return data == 3

    do2 = lambda x: pd.Series(False, index=x.index)

    print(Variable(None))

    v = Variable([1, 2, 3, 4, N, N, 9], index=dtindex(7))
    print(v)
    v = v.flag_limits(2)
    print(v)
    v = v.flag_generic(do1)
    v = v.flag_generic(do2)
    v = v.flagna()
    print(v)
    print(v.meta.to_pandas().to_string())
