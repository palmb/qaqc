#!/usr/bin/env python
from __future__ import annotations

import warnings
from abc import ABC

import numpy as np
import pandas as pd
from typing import Any, cast
from saqc._typing import Scalar, VariableT
from saqc.core.base import BaseVariable
from saqc.core.flagsframe import FlagsFrame


class _Mask:
    def __init__(
            self,
            data: pd.Series | None = None,
            mask: pd.Series | bool | list[bool] | None = None,
            fill_value: float = np.nan,
    ):
        if data is None and mask is not None:
            raise ValueError(
                "Cannot create from mask only. data must not "
                "be None or both of mask and data must be None."
            )
        if isinstance(data, pd.Series):
            data = data.copy()
        if mask is not None:
            mask = pd.Series(mask, index=data.index, dtype=bool)
        self.data = data
        self.mask = mask
        self.fill_value = fill_value

    def copy(self, deep):
        c = _Mask()
        if self.data is not None:
            c.data = self.data.copy(deep)
        if self.mask is not None:
            c.mask = self.mask.copy(deep)
        c.fill_value = self.fill_value
        return c

    @property
    def _has_mask(self) -> bool:
        return self.mask is not None

    @property
    def has_data(self) -> bool:
        return self.data is not None

    def __bool__(self) -> bool:
        return self.has_data

    @property
    def is_masked(self) -> bool:
        return self.has_data and self._has_mask and self.mask.any()

    def get_mask(self) -> pd.Series | None:
        if self._has_mask:
            return self.mask.copy()
        return None

    def get_masked_data(self) -> pd.Series:
        if not self.has_data:
            raise RuntimeError("No data to mask")
        data = self.data.copy()
        if self.is_masked:
            data[self.mask] = self.fill_value
        return data

    def get_original_data(self) -> pd.Series:
        return self.data


class MaskingMixin(BaseVariable, ABC):
    def __init__(self, *args, mask=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._mask: _Mask = _Mask()
        if mask is not None:
            self._mask_data(mask)

    def copy(self: MaskingMixin, deep: bool = True) -> MaskingMixin:
        c = super().copy(deep)
        c._mask = self._mask.copy(deep)
        return cast(MaskingMixin, c)

    @property
    def is_masked(self) -> bool:
        return self._mask.is_masked

    @property
    def data_mask(self) -> pd.Series | None:
        return self._mask.get_mask()

    def _mask_data(self, mask) -> MaskingMixin:
        self._mask = _Mask(self._data, mask)
        self._data = self._mask.get_masked_data()
        return self

    def mask_data(self, mask=None, inplace: bool = False) -> MaskingMixin:
        obj = self if inplace else self.copy()
        if obj.is_masked:
            warnings.warn("Variable is already masked, the old mask will be discarded")
            obj = obj.unmask_data(inplace=inplace)
        if mask is None:
            mask = obj.flagged()
        return obj._mask_data(mask)

    def unmask_data(self, inplace: bool = False) -> MaskingMixin:
        obj = self if inplace else self.copy()
        if obj.is_masked:
            obj._data = obj._mask.get_original_data()
            obj._mask = _Mask()
        return obj

    def _df_to_render(self) -> pd.DataFrame:
        df = super()._df_to_render()
        if self.is_masked:
            df.insert(1, "mask", self._mask.get_masked_data())
        return df


if __name__ == '__main__':

    class Variable(MaskingMixin):
        def __init__(self, data, index=None, flags=None, mask=None):
            super().__init__(data=data, index=index, flags=flags, mask=mask)

        @property
        def _constructor(self: VariableT) -> type[VariableT]:
            return type(self)

    mv = Variable([1,3,4])
    mv = mv.copy()
