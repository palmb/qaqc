#!/usr/bin/env python
from __future__ import annotations

# import warnings
# import weakref
# from abc import ABC, abstractmethod
# from typing import Any, Sequence, cast, Callable, NoReturn, final
#
# import pandas as pd
# import numpy as np
# import abc
# from saqc.constants import UNFLAGGED
# from saqc.core.flagsframe import FlagsFrame
from saqc.core.base import BaseVariable
from saqc._typing import VariableT
#
# # todo: modular Features classes last becomes Variable
# # todo: Feature: #tags
# # todo: Feature: translation
# # todo: Feature: OpsMixin (see pd.core.arraylike)
#
#
#
#
# import saqc.core.univariate as _univ  # noqa
#
#
class Variable(BaseVariable):  # noqa
    @property
    def _constructor(self: VariableT) -> type[VariableT]:
        return self.__class__
#
#
#
#
# class MaskedVariableOld(Variable):
#     # todo: add this to BaseVariable ??? to simplify ??
#     def __init__(
#         self,
#         data,
#         index: pd.Index | None = None,
#         flags: FlagsFrame | None = None,
#         mask: Any = None,
#     ):
#         super().__init__(data, index, flags)
#         self._orig = None
#         self._mask = None
#         if mask is not None:
#             self._mask_data(mask)
#
#     def copy(self, deep=True) -> MaskedVariable:
#         c = super().copy(deep)
#         if self._orig is not None:
#             c._orig = self._orig.copy(deep)
#         if self._mask is not None:
#             c._mask = self._mask.copy(deep)
#         return cast(MaskedVariable, c)
#
#     @property
#     def is_masked(self) -> bool:
#         return self._mask is not None and self.mask.any()
#
#     @property
#     def mask(self) -> pd.Series:
#         if self._mask is None:
#             return pd.Series(False, index=self.index, dtype=bool)
#         return self._mask.copy()
#
#     @mask.setter
#     def mask(self, value):
#         mask = pd.Series(value, index=self.index, dtype=bool)
#         self._unmask_data()
#         self._mask_data(mask)
#
#     def _mask_data(self, mask) -> MaskedVariable:
#         """mask data inplace."""
#         mask = pd.Series(mask)  # cast and ensure shallow copy
#         assert len(mask) == len(self.index)
#         self._orig = self._data.copy()
#         self._mask = mask
#         self._data[mask] = np.nan
#         return self
#
#     def _unmask_data(self) -> MaskedVariable:
#         self._data = self._orig
#         self._mask = None
#         self._orig = None
#         return self
#
#     def mask_data(self, mask=None, inplace=False) -> MaskedVariable:
#         if mask is None:
#             mask = self.flagged()
#         else:
#             mask = pd.Series(mask, index=self.index, dtype=bool)
#         c = self.copy()
#         c._unmask_data()
#         c._mask_data(mask)
#         if inplace:
#             self._orig = c._orig
#             self._mask = c._mask
#             self._data = c._data
#             c = self
#         return c
#
#     def unmask_data(self, inplace=True) -> Variable:
#         obj: MaskedVariable = self if inplace else self.copy()
#         obj._unmask_data()
#         obj: Variable = _cast_inplace(obj, Variable)
#         return obj
#
#     # ############################################################
#     # Rendering
#     # ############################################################
#
#     def _df_to_render(self) -> pd.DataFrame:
#         df = super()._df_to_render()
#         df.insert(1, "mask", self.mask.array)
#         return df
#
#
# def _cast_inplace(obj: BaseVariable, klass) -> MaskedVariable | Variable:
#     _type = type(obj)
#     if _type is klass:
#         pass
#     if _type is Variable:
#         if klass is MaskedVariable:
#             obj.__class__ = klass
#             obj._orig = None
#             obj._mask = None
#     elif _type is MaskedVariable:
#         if klass is Variable:
#             obj.__class__ = klass
#             delattr(obj, "_orig")
#             delattr(obj, "_mask")
#     else:
#         raise NotImplementedError(f"Cannot cast from {_type} to {klass}")
#     return cast(klass, obj)
#
#
# if __name__ == "__main__":
#     from saqc._testing import dtindex, N
#     from saqc.core.variable import Variable
#
#     print(Variable(None))
#     print(MaskedVariable(None))
#
#     v = Variable([1, 2, 3, 4], index=dtindex(4))
#     v = v.flag_limits(2)
#     v.flags.append(pd.Series([N, 5, 5, N]))
#     v = v.mask_data(inplace=True)
#     print(v)
#     new = Variable(v)
#     print(new)
#     v = v.mask_data(mask=True, inplace=True)
#     print(v)
#     v._data[:] = 99
#     print(v)
#     v.unmask_data(inplace=True)
#     print(v)
