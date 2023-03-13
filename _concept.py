from __future__ import annotations

import functools
import typing
from collections.abc import MutableMapping
from typing import overload, Iterator

import numpy as np
import pandas as pd
from fancy_collections import DictOfPandas
from pandas._typing import F

# - : original value
# n : created by function ...
# i : filled by interpolation ...
# c : corrected by function
# a : aggregated from values ...
# d : derived from other variable ...
# s : shifted value
# f : flagged by function ...

# Variante A:
# data is read only
# new data must be passed into a new Variable
#
#
# Variante B:
# data can only be altered if NaN
# data cannot alter in shape
# a possible hack: flag data, mask it, it becomes Nan, modify it

# todo: basics
#   a Variable has data, quality-labels, an index and meta information.
#       data    read-only array of values
#       labels  mutable array of values
#       flags   mutable multidim array of floats (internal use only)
#       meta    mutable multidim array of strings
# e.g.
#             data labels  flags     [long]    [lat]
# 2000-01-01   0.0           NaN  335.33.22  19.3.44
# 2000-01-02   1.0    BAD   99.0  335.33.22  19.3.44
# 2000-01-03   2.0  DOUBT   66.0  335.33.22  19.3.44
# 2000-01-04   NaN           NaN  335.33.22  19.3.44
# 2000-01-05   4.0    BAD   99.0  335.33.22  19.3.44
#
#
#   data is immutable
#   the index is just another way to access data (it can be set arbitrary)
#   every data operation result in a new Variable

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


# -------
# Masking
# -------
# Variable.orig always show all data
# Variable.data is masked by default according to user settable function

# -----
# Usage A
# -----
# v.data    -> pd.Series    R   masked data
# v.orig    -> pd.Series    R   original data
# v.flags  -> FlagsFrame   R

# -----
# Usage B
# -----
# v.data    -> pd.Series    R    masked data
# v.orig    -> pd.Series    R   original data
# v.flags   -> pd.Series    R   current flags (aggregated frame)
# v.raw     -> Accessor     R
# v.raw.flags  -> FlagsFrame
# v.raw.data    -> FlagsFrame

# -----
# Usage C  (current)
# -----
# v.data    -> pd.Series        masked data
# v.orig    -> pd.Series        original data
# v.flags   -> pd.Series        current flags (aggregated frame)
# v.fframe  -> pd.FlagsFrame    flags frame
# v.fframe.current()    -> pd.Series      same as v.flags
# v.fframe.df           -> pd.DataFrame   raw flags frame (f0, f1, f2, ...)

# -----
# Usage D
# -----
# v.data    -> pd.Series        masked data
# v.orig    -> pd.Series        original data
# v.flags   -> pd.Series        current flags (aggregated frame)
# v.history -> pd.FlagsFrame    flags frame
# v.history.agg()    -> pd.Series      same as v.flags
# v.history.df       -> pd.DataFrame   raw flags frame (f0, f1, f2, ...)
class Frame(MutableMapping):
    def __init__(self, **kwargs):
        self._vars = {
            k: v if isinstance(v, Variable) else Variable(v) for k, v in kwargs.items()
        }

    # ############################################################
    # Mutable Mapping
    # ############################################################

    @overload
    def __getitem__(self, item: str) -> Variable:
        ...

    @overload
    def __getitem__(self, item: list[str] | pd.Index) -> Frame:
        ...

    def __getitem__(self, item):
        if isinstance(item, (list, pd.Index)):
            return self.__class__(**{k: v for k, v in self._vars.items() if k in item})
        return self._vars.__getitem__(item)

    def __setitem__(self, key: str, value: Variable | pd.Series):
        if isinstance(value, pd.Series):
            value = Variable(value)
        assert isinstance(key, str)
        assert isinstance(value, Variable)
        self._vars.__setitem__(key, value)

    def __delitem__(self, key):
        self._vars.__delitem__(key)

    def __len__(self):
        return len(self._vars)

    def __iter__(self) -> Iterator[str]:
        return self._vars.__iter__()

    def __contains__(self, item):
        return self._vars.__contains__(item)

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and len(self) == len(other)
            and all(self[l] == other[r] for l, r in zip(self, other))
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def keys(self):
        return self._vars.keys()

    def items(self):
        return self._vars.items()

    def values(self):
        return self._vars.values()

    def get(self, *key):
        return self._vars.get(*key)

    def pop(self, *key):
        return self._vars.pop(*key)

    def popitem(self):
        return self._vars.popitem()

    def clear(self):
        return self._vars.clear()

    def update(self, other, **kwargs):
        return self._vars.update(other, **kwargs)

    def setdefault(self, *args):
        return self._vars.setdefault(*args)

    # ############################################################
    # Other
    # ############################################################

    @property
    def columns(self):
        return pd.Index(self._vars.keys())

    def __repr__(self):
        return repr(DictOfPandas({k: v.data for k, v in self._vars.items()}))

    def interpolate(self, *args, **kwargs):
        return self

    def to_grid(self, *args, **kwargs):
        return self

    def flag_by_condition(self, func, subset=None, flag=99):
        columns = self.columns if subset is None else pd.Index(subset)
        new = self.copy()
        for key in columns:
            mask = func(self.copy())
            new[key].flags[mask] = flag
        return new

    def copy(self):
        return Frame(**{k: v.copy() for k, v in self._vars.items()})


class Variable:
    def __init__(self, data=None, index=None, flags=None, meta=None): # noqa
        data = pd.Series(data, index=index, dtype=float)
        self.data = data
        if flags is None:
            flags = pd.Series(np.nan, index=data.index, dtype=float)
        self.flags = flags
        if meta is None:
            meta = pd.Series("", index=data.index, dtype=str)
        self.meta = meta

    @property
    def index(self):
        return self.data.index

    def copy(self):
        return Variable(data=self.data.copy(), flags=self.flags.copy())

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self.data.equals(other.data)
            and self.flags.equals(other.flags)
        )

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

    def reindex(self, index=None, method=None):
        # - index:  new
        # - data:   new (derived from old)
        # - flags:  new (fresh start)
        data = self.data.reindex(index, method=method)
        return Variable(data=data)

    def __repr__(self):
        df = pd.DataFrame(dict(data=self.data, flags=self.flags, info=self.meta))
        return repr(df) + "\n"

    def flag_by_condition(self, func, flag=99):
        new = self.copy()
        mask = func(self.data)
        new.flags[mask] = flag
        return new


if __name__ == "__main__":
    N = np.nan
    index = pd.date_range("2000", periods=20, freq="1d")
    d = Variable([1, 2, N, 3, N, 99, 6], index=index[:7])
    bat = Variable(
        [10, 10.1, 10.2, N, N, 9.3, 8.2, 8.0, 5.4, 5.4, N, N, 9.1], index=index[2:15]
    )
    df = Frame(d=d, bat=bat)
    print(df.keys())
    exit(99)
    df = df.interpolate().to_grid(infere_freq=True)
    df["bat(orig)"] = df["bat"].copy()
    df["bat"] = df["bat"].reindex(df["d"].index, method="ffill")
    df["d"] = df["d"].flag_generic(lambda x: df["bat"].data < 10.2)
    df = df.flag_generic(
        lambda x: x["bat"].data < 10.1, flag=55, subset=["d", "bat"]
    )
    print(df)
    print(df["d"])
    print(df["bat"])
