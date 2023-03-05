#!/usr/bin/env python
from __future__ import annotations

import numpy as np
from saqc.core.base import BaseVariable
from abc import ABC


class UnivariateMixin(BaseVariable, ABC):

    # ############################################################
    # Flagging
    # ############################################################

    def flag_limits(self, lower=-np.inf, upper=np.inf, flag=9) -> BaseVariable:
        result = self.copy().mask_data()
        mask = (result.data < lower) | (result.data > upper)
        result.flags.append_with_mask(mask, flag, meta="flag_limits")
        return result.unmask_data()

    def flag_something(self, flag=99) -> UnivariateMixin:
        result = self.copy().mask_data()
        sample = result.data.dropna().sample(frac=0.3)
        new = result.flags.template()
        new[sample.index] = flag
        result.flags.append(new, "flag_something")
        return result.unmask_data()

    def flagna(self, flag=999) -> UnivariateMixin:
        # no masking desired !
        result = self.copy()
        result.flags.append_with_mask(result.data.isna(), flag, "flagna")
        return result

    def replace_flag(self, old, new) -> UnivariateMixin:
        # no masking needed
        result = self.copy()
        mask = self.flags.current() == old
        result.flags.append_with_mask(mask, new, "replace_flag")
        return result

    def flag_generic(self, func, flag=99) -> UnivariateMixin:
        # func ==> ``lambda v: v.data != 3``
        new = self.copy()
        mask = func(self.data)
        new.flags.append_with_mask(mask, flag, "flag_generic")
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

    def clip(self, lower, upper, flag=-88) -> UnivariateMixin:
        # keep index
        # alter data
        # initial flags: squeezed old
        result = self.flag_limits(lower, upper, flag=flag).copy()
        result.data.clip(lower, upper, inplace=True)
        return self._constructor(result)

    def interpolate(self, flag=None) -> UnivariateMixin:
        # keep index
        # alter data
        # initial flags: squeezed old
        if flag is not None:
            flags = self.flagna(flag).flags
        else:
            flags = self.flags
        data = self.data.interpolate()
        return self._constructor(data, flags)

    def reindex(self, index=None, method=None) -> UnivariateMixin:
        # - set new index
        # - reset all flags
        data = self.data.reindex(index, method=method)
        return self._constructor(data)

