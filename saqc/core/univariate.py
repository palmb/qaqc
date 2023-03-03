#!/usr/bin/env python
from __future__ import annotations

import numpy as np
from saqc.core.variable import BaseVariable
from abc import ABC
from saqc._typing import VariableT


class UnivariateMixin(BaseVariable, ABC):

    # ############################################################
    # Flagging
    # ############################################################

    def flag_limits(self: VariableT, lower=-np.inf, upper=np.inf, flag=9) -> VariableT:
        result = self.mask_data()
        mask = (result.data < lower) | (result.data > upper)
        result.flags.append_with_mask(mask, flag)
        return result.unmask_data()

    def flag_something(self: VariableT, flag=99) -> VariableT:
        result = self.mask_data()
        sample = result.data.dropna().sample(frac=0.3)
        new = result.flags.template()
        new[sample.index] = flag
        result.flags.append(new)
        return result.unmask_data()

    def flagna(self: VariableT, flag=999) -> VariableT:
        # no masking desired !
        result = self.copy()
        meta = dict(source="flagna")
        result.flags.append_with_mask(result.data.isna(), flag, meta)
        return result

    def replace_flag(self: VariableT, old, new) -> VariableT:
        # no masking needed
        result = self.copy()
        mask = self.flags.current() == old
        result.flags.append_with_mask(mask, new, dict(source="replace_flag"))
        return result

    def flag_generic(self: VariableT, func, flag=99) -> VariableT:
        # func ==> ``lambda v: v.data != 3``
        new = self.copy()
        mask = func(self.data)
        new.flags.append_with_mask(mask, flag, dict(source="flag_generic"))
        return new

    # ############################################################
    # Corrections
    # ############################################################
    #  Functions that alter data.
    # - must return a new Variable
    # - may set new flags on the new Variable
    # - may use the existing old Flags squeezed to a pd.Series
    #  (see `FlagsFrame.current`) as the initial flags for the
    #  new Variable
    # ############################################################

    def clip(self: VariableT, lower, upper, flag=-88) -> VariableT:
        # keep index
        # alter data
        # initial flags: squeezed old
        result = self.flag_limits(lower, upper, flag=flag).copy()
        result.data.clip(lower, upper, inplace=True)
        return self._constructor(result)

    def interpolate(self: VariableT, flag=None) -> VariableT:
        # keep index
        # alter data
        # initial flags: squeezed old
        meta = dict(source="interpolate")
        if flag is not None:
            flags = self.flagna(flag).flags
        else:
            flags = self.flags
        data = self.data.interpolate()
        return self._constructor(data, flags)

    def reindex(self: VariableT, index=None, method=None) -> VariableT:
        # - set new index
        # - reset all flags
        data = self.data.reindex(index, method=method)
        return self._constructor(data)

