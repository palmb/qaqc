#!/usr/bin/env python
from __future__ import annotations
import pandas as pd
import numpy as np
from saqc.types import VarOrQcT
from saqc.errors import ImplementationError

from typing import TYPE_CHECKING
from typing import Callable

if TYPE_CHECKING:
    from saqc import Variable, SaQC

from functools import wraps
from saqc.types import F, T


# def add_to_SaQC(func: T) -> T:
#     @wraps(func)
#     def decorator(self: VarOrQcT, *args, **kwargs) -> VarOrQcT:
#         return func(self, *args, **kwargs)
#
#     return decorator


class FuncMixin:
    # This class make methods available on Variable and on SaQC.
    # If a method is called on a Variable it may work inplace,
    # but in any case must return a Variable instance (can be
    # modified self).
    # If the method is called on a SaQC object, the method is
    # implicitly called on each variable within the SaQC object
    # and the result is returned in a new SaQC instance.
    #
    # Note that methods which only works on Variable instances
    # or only on SaQC instances must be implemented in the
    # respected class body and does not belong here.
    _to_dispatch = ["flagUpper", "flagFoo", "setAll"]

    def setAll(self, val):
        self._data[:] = val
        return self

    def flagFoo(self: VarOrQcT) -> VarOrQcT:
        return self

    def flagUpper(self: VarOrQcT, thresh=-np.inf, flag=99.0) -> VarOrQcT:
        mask = self.data >= thresh
        s = pd.Series(np.nan, index=self.index, dtype=float)
        s[mask] = flag
        self.flags.append(s)
        return self

    def __dispatch__(self, func: F) -> F:
        from saqc import SaQC

        @wraps(func)
        def for_each(*args, **kwargs):
            new = SaQC()
            for key, var in self._vars.items():
                var = self[key].copy()
                result = func(var, *args, **kwargs)
                new[key] = var if result is None else result  # cover inplace case
            return new

        return for_each

    def __getattribute__(self, item: str):
        from saqc import Variable, SaQC

        if item in FuncMixin._to_dispatch:  # recursion protector
            if isinstance(self, SaQC):
                return FuncMixin.__dispatch__(self, getattr(Variable, item))
        return super().__getattribute__(item)


for _name in [n for n in dir(FuncMixin) if not n.startswith("_")]:
    if _name not in FuncMixin._to_dispatch:
        raise ImplementationError(
            f"(public) api methods of FuncMixin.{_name} must be in "
            f"FuncMixin._to_dispatch or belong to a different class."
        )
