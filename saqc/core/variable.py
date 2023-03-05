#!/usr/bin/env python
from __future__ import annotations

import pandas as pd

from saqc.core.base import BaseVariable
from saqc.core.univariate import UnivariateMixin
from saqc._typing import VariableT

# todo: modular Features classes last becomes Variable
# todo: Feature: #tags
# todo: Feature: translation
# todo: Feature: OpsMixin (see pd.core.arraylike)


class Variable(UnivariateMixin, BaseVariable):  # noqa
    @property
    def _constructor(self: VariableT) -> type[VariableT]:
        return self.__class__


if __name__ == "__main__":
    from saqc._testing import dtindex, N
    from saqc.core.variable import Variable
    from saqc.core.flagsframe import FlagsFrame

    print(Variable(None))

    v = Variable([1, 2, 3, 4], index=dtindex(4))
    v = v.flag_limits(2)
    v.flags.append(pd.Series([N, 5, 5, N]))
    v.mask_data()
    print(v)
    new = Variable(v)
    print(new)
    v.mask = True
    v._data._raw.harden_mask()
    print(v)
    v._data._raw[:] = 99
    v._data: "_Data"
    v._data.mask
    v.data: pd.Series
    v.flags: FlagsFrame
    v.flags.raw: pd.DataFrame

    v.data: "DataSeries"
    v.flags: FlagsFrame
    v.data.series: pd.Series
    v.flags.frame: pd.DataFrame
    print(v)
    v.unmask_data()
    v.mask[1] = True
    print(v)
