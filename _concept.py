from __future__ import annotations

from typing import overload

import numpy as np
import pandas as pd
from fancy_collections import DictOfPandas

# - : original value
# n : created by function ...
# i : created by interpolation ...
# c : corrected by function
# a : aggregated from values ...
# d : derived from other variable ...
# s : shifted value

# f : flagged by function ...


# Wording
# =======
# data          the actual data
# index         an index to access the data in a convenient way
# flags         internal float flags
# labels        translated flags
# cass Variable structure to hold data and meta-data (incl. flags)
# cass Frame    structure to hold variables (formerly known as SaQC)
# masking       simple and consisting way of ignoring data values (by mask or flags)
# selecting     inverse of masking
# grouping      simple and consisting way of call multiple functions
# flag-policy   user-settable policy for aggregate flags (horizontal)
# flag-function     function to set flags
# fill-function  function fill NaN values
# data-corrections  function to alter data values or the shape of data
# univariat         flagging or processing of a single Variable
# multivariat       flagging or processing of a multiple Variables at once
#
# DETAILS
# =======
#
# function types details
# ----------------------
# flag-functions  (e.g. flagging of limits, constants, spikes, ...)
#   short: sets flags
#   result: same Variable (or copy)
#   - does not alter index
#   - does not alter data
#   - might set new flags
#   - extend history of flags
#
# fill-functions  (interpolations, fillna, ...)
#   short: fill NaNs
#   result: same Variable (or copy)
#   - fix nans
#   - does not alter index
#   - does not alter existing data
#   - does not alter flags of existing data
#   - might set new flags for new data
#   - extend history of flags
#
# data-corrections  (corrections, arithmetics, reindexing, shifting, ...)
#   short: new data -> new Variable -> new History
#   result: new Variable
#   - might alter index
#   - might alter data
#   - might fill nans
#   - might set flags
#   - flag history is not preserved
#   - initial flags of new Variable can be the current flags (squeezed History)
#     of the derived variable, but the flags history is NOT preserved
#
# univariate vs. multivariate details
# -----------------------------------
# univariate functions
#   are implemented as methods in Variable.
#   Optionally a wrapper in Frame is implemented, which call the
#   method for each Variable in the Frame. The wrapper should have
#   an keyword named `subset`, which optionally select a subset of
#   Variables to process.
#
# multivariate functions
#   are implemented as methods in Frame. The function is free to choose
#   how to select some Variables or to work on all present. Nevertheless
#   the method should have an keyword named `subset`, which optionally
#   pre-select a subset of Variables to process.
#
# Other details
# -------------
# no need for `target` because of get/setitem
#
# Specifications
# ==============
#
# ------------------------------------
# Variable (hold data and flags)
# ------------------------------------
# Variable(pd.Series)
# Variable.data  -> pd.Series? np.ndarray?
# Variable.flags -> pd.Series? np.ndarray?
# Variable.index -> pd.Index
# Variable.index = pd.Index
# Variable.flag_function(...) -> Variable (or copy)
# Variable.fill_function(...) -> Variable (or copy)
# Variable.data_correction(...) -> new Variable (not a copy)
#
#
# --------------
# Frame (formerly known as SaQC)
# --------------
# Frame(a=Variable, b=Variable)
# Frame[str] -> Variable
# Frame[str] = Variable | pd.Series
# Frame[list] -> new Frame
# Frame[list] = Frame  (must have same length in columns)
# Frame.columns -> pd.Index
# repr(Frame) -> nice output
# Frame.flag_function(..., subset=None) -> new Frame
# Frame.fill_function(..., subset=None) -> new Frame
# Frame.data_correction(..., subset=None) -> new Frame

class Frame:
    def __init__(self, **kwargs):
        assert all(isinstance(v, Variable) for v in kwargs.values())
        self._vars = kwargs

    @overload
    def __getitem__(self, item: str) -> Variable:
        ...

    @overload
    def __getitem__(self, item: list[str] | pd.Index) -> Frame:
        ...

    def __getitem__(self, item):
        if isinstance(item, (list, pd.Index)):
            return Frame(**{k: v for k, v in self._vars.items() if k in item})
        return self._vars.__getitem__(item)

    def __setitem__(self, key: str, value: Variable):
        assert isinstance(key, str)
        if not isinstance(value, Variable):
            value = Variable(value)
        self._vars.__setitem__(key, value)

    def __delitem__(self, key):
        self._vars.__delitem__(key)

    def __len__(self):
        return len(self._vars)

    def columns(self):
        return pd.Index(self._vars.keys())

    def __repr__(self):
        return repr(DictOfPandas({k: v.data for k, v in self._vars.items()}))


class Variable:
    def __init__(self, data=None, flags=None, meta=None):
        data = pd.Series(data, dtype=float)
        self.data = data
        if flags is None:
            flags = pd.Series(np.nan, index=data.index, dtype=float)
        self.flags = flags
        if meta is None:
            meta = pd.Series("", index=data.index, dtype=str)
        self.meta = meta

    def copy(self):
        return Variable(self.data.copy(), self.flags.copy())

    def flag_limits(self, lower=-np.inf, upper=np.inf, flag=9):
        # - index:  keep
        # - data:   keep
        # - flags:  might add new
        new = self.copy()
        new.flags[(self.data < lower) | (self.data > upper)] = float(flag)
        return new

    def flag_something(self):
        return self.copy()

    def clip(self, lower, upper, flag=9):
        # - index:  keep
        # - data:   altered
        # - flags:  must set flags for altered data
        data = self.data.clip(lower, upper)
        new = Variable(data, self.flags.copy())
        new.flags[(self.data < lower) | (self.data > upper)] = float(flag)
        return Variable(data)

    def interpolate(self, flag=None):
        # - index:  keep
        # - data:   altered (fill nans)
        # - flags:  must set flags for altered data
        flags = self.flags.copy()
        meta = self.meta.copy()
        meta[self.data.isna()] = "interpolated"
        if flag is not None:
            flags[self.data.isna()] = flag
        data = self.data.interpolate()
        return Variable(data, flags, meta)

    def reindex(self, index=None, method="shift"):
        # - index:  new
        # - data:   new (derived from old)
        # - flags:  new (fresh start)
        ["linear", "shift"]
        return Variable()

    def __repr__(self):
        df = pd.DataFrame(dict(data=self.data, flags=self.flags, info=self.meta))
        return repr(df) + "\n"


def masking(mask):
    pass


if __name__ == "__main__":

    v0 = Variable([1, 2, np.nan, np.nan, 10])
    v0.reindex(pd.Index)

    f = Frame()
    f["a"] = v0
    a = f["a"]
    del f['a']
    f['a'] = a
    print(f)
    index = pd.date_range("2000", freq='1y', periods=10)
    s = pd.Series(range(10), index)
    print(s[:'2005'])
    index.get_indexer

