#!/usr/bin/env python
from __future__ import annotations

import pandas as pd
import numpy as np

import weakref
from abc import abstractmethod
from typing import Any, overload, TYPE_CHECKING, NoReturn, cast, Callable, Union, Any

from qaqc.typing import (
    final,
    MaskT,
    MaskerT,
    CondT,
    ListLike,
    FlagLike,
    VariableT,
)

from qaqc.core.utils import (
    get_caller as this,
    is_indexer,
    is_listlike,
    is_iterable,
    is_boolean_indexer,
    get_func_location,
    construct_index,
)
from qaqc.constants import SET_UNFLAGGED
from qaqc.core.flagsframe import FlagsFrame, Meta
from qaqc.errors import (
    MaskerError,
    MaskerResultError,
    MaskerExecutionError,
)

__all__ = ["Variable"]

# todo: Feature: #tags
# todo: Feature: translation


# This class is private because it is completely
# shadowed by methods and properties in BaseVariable.
# A user should not access or create _Data, nor
# should have need for that.
class _Data:
    __slots__ = ("_raw", "_index")
    _raw: np.ma.MaskedArray
    _index: pd.Index

    def __init__(
        self,
        raw,
        index: pd.Index | None = None,
        mask: bool | MaskT = False,
        dtype=float,
        fill_value: float = np.nan,
    ):
        copy = True
        if isinstance(raw, pd.Series):
            if index is None:
                index = raw.index
            raw = raw.to_numpy(copy=True)
            copy = False
        elif index is None:
            raise TypeError("If data is not a pandas.Series, an index must be given")

        index = construct_index(index, name="Index", errors="raise")

        if dtype is not float:
            raise NotImplementedError("Only float dtype is supported.")

        if isinstance(mask, pd.Series):
            mask = mask.to_numpy(copy=True)

        ser = pd.Series(raw, index=index, dtype=float, copy=copy)
        self._raw = np.ma.MaskedArray(
            data=ser.to_numpy(copy=False),
            mask=mask,
            dtype=float,
            fill_value=fill_value,
            copy=False,
        )
        self._index = ser.index.copy()

    def copy(self, deep=False) -> _Data:
        cls = self.__class__
        if deep:
            return cls(
                raw=self._raw.data,
                index=self.index,
                mask=self._raw.mask,
                fill_value=self.fill_value,
            )
        c = cls.__new__(cls)
        # Setting the very same obj here is ok, because
        # _raw.data and _index are basically immutable
        # by the public api, especially if accessed via
        # Variable. Nevertheless, changes on _raw.mask
        # (e.g. if new flags are set in Variable) will
        # reflect back on the copy and vice-versa.
        c._raw = self._raw
        c._index = self.index
        return c

    @property
    def mask(self) -> np.ndarray:
        # This returns a view and modifications will reflect
        # back to us. This is intended, because it allows
        # modifications on the mask from within Variable, to
        # the user this is completely shadowed by Variable
        return self._raw.mask

    @mask.setter
    def mask(self, value: bool | MaskT):
        self._raw.mask = value

    @property
    def masked_series(self) -> pd.Series:
        return pd.Series(self._raw.filled(), self._index, dtype=float, copy=True)

    @property
    def orig_series(self) -> pd.Series:
        return pd.Series(self._raw.data, self._index, dtype=float, copy=True)

    @property
    def index(self) -> pd.Index:
        return self._index.copy()

    @index.setter
    def index(self, value: Any):
        index = pd.Index(value)
        assert len(index) == len(self._index) and not isinstance(index, pd.MultiIndex)
        self._index = index

    @property
    def fill_value(self) -> float:
        return self._raw.fill_value

    @fill_value.setter
    def fill_value(self, value: float):
        self._raw.fill_value = value

    def __len__(self):
        return len(self._raw)


class BaseVariable:
    # class variables
    global_masker: MaskerT | None = None

    # type hints
    _name: str | None
    _data: _Data
    _history: FlagsFrame
    masker: MaskerT | None

    @property
    def _constructor(self: VariableT) -> type[VariableT]:
        return type(self)

    def derive(
        self: VariableT, data, flags, other: VariableT | None, **meta
    ) -> VariableT:
        """
        A: derive from self (other=None), init from data/flags, index must not match
        B: derive and init from other, index *must* match
        """
        cls = type(self)

        if other is None:
            ref = self
            init = flags
            compressed = ref._history.compress()
            append = None
        else:
            ref = other
            compressed = ref._history.compress()
            init = compressed.current()
            append = flags

        if data is None:
            data = ref.orig

        f0_meta = dict(derived=compressed.compress_further())
        # ensure the given meta is present,
        # even if not extra flags are appended
        if append is None:
            f0_meta = {**meta, **f0_meta}
            meta = {}  # consumed

        derived = cls(data, init)
        derived._history._meta["f0"] = f0_meta  # override or create
        derived._name = ref._name
        derived.masker = ref.masker
        if append is not None:
            derived.set_flags(append, None, **meta)
        return derived

    def __init__(
        self,
        data,
        flags: FlagsFrame | pd.Series | None = None,
        index: pd.Index | None = None,
        ref: BaseVariable | None = None,
    ):
        if isinstance(data, BaseVariable):
            if flags is None:
                flags = data._history
            if index is None:
                index = data.index
            # if ref is None:
            #     ref = data
            data = data._data._raw

        if isinstance(index, pd.Series):
            raise TypeError(
                "Using a pd.Series as index is not allowed, "
                "use pd.Series.values instead"
            )

        if index is None and data is None:
            raise ValueError("Either data or index must be given")

        # handles both, data and index casting and errors
        data = _Data(data, index)

        # If no history are given we create an empty FlagsFrame,
        # otherwise we create a FlagsFrame with an initial column.
        # The resulting FlagsFrame will at most have the initial
        # column, even if a multidim object is passed.
        flags = FlagsFrame(index=data.index, initial=flags)

        self._data = data
        self._history = flags
        # self._ref = None if ref is None else ref._create_ref()
        self._name = None
        self.masker = None
        self._optimize()

    def copy(self: VariableT, deep: bool = True) -> VariableT:
        self._update_mask()
        cls = self._constructor
        c = cls.__new__(cls)
        c._data = self._data.copy(deep)
        c._history = self._history.copy(deep)
        c._name = self._name
        # c._ref = self._ref
        c.masker = self.masker
        c._optimize()
        return c

    def _optimize(self):
        # share a single index
        self._data._index = self._history._raw.index

    _restore_index = _optimize  # alias for now

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def index(self) -> pd.Index:
        return self._history.index

    @index.setter
    def index(self, value) -> None:
        try:
            self._data.index = value
            self._history.index = value
        finally:
            self._restore_index()

    @property
    def orig(self) -> pd.Series:
        """Returns a copy of the underlying data (no masking)"""
        return self._data.orig_series

    @property
    def data(self) -> pd.Series:
        """Returns a copy of the underlying data (masked by flags)"""
        self._update_mask()
        return self._data.masked_series

    @property
    def flags(self) -> pd.Series:
        """Returns a series of currently set flags"""
        return self._history.current(raw=False)

    @property
    def meta(self) -> Meta:
        return self._history.meta  # mutable shallow copy

    @meta.setter
    def meta(self, value: pd.Series | Meta):
        self._history.meta = value  # type: ignore

    # ########################################################################
    # Some basics
    # ########################################################################

    def get_history(self) -> FlagsFrame:
        return self._history.copy()

    def is_masked(self) -> pd.Series:
        """return a boolean series that indicates flagged values"""
        mask = self._get_mask()
        return pd.Series(mask, index=self.index, dtype=bool, copy=True)

    def is_flagged(self) -> pd.Series:
        """return a boolean series that indicates flagged values"""
        return self.flags.notna()

    def is_unflagged(self) -> pd.Series:
        """return a boolean series that indicates unflagged values"""
        return self.flags.isna()

    def template(self, fill_value=np.nan) -> pd.Series:
        return pd.Series(fill_value, index=self.index, dtype=float)

    # ########################################################################
    # Comparisons and Arithmetics
    # ########################################################################
    # todo: may also use OpsMixin ?

    def equals(self, other: Any, compare_history=False) -> bool:
        return (
            isinstance(other, type(self))
            and self.data.equals(other.data)
            and (
                self._history.equals(other._history)
                if compare_history
                else self.flags.equals(other.flags)
            )
        )

    def __eq__(self, other) -> bool:
        # compares data==data and flags==flags
        return self.equals(other, compare_history=False)

    def __getitem__(self, key) -> pd.Series:
        if key in ["data", "flags"]:
            return getattr(self, key)
        if key == "mask":
            return self.is_masked()
        if key in self._history:
            return self._history.__getitem__(key)
        raise KeyError(key)

    def __setitem__(self, key, value) -> NoReturn:
        raise NotImplementedError("use 'set_flags()' instead")

    @overload
    def set_flags(
        self: VariableT, value: FlagLike, cond: CondT | None = None, **meta
    ) -> VariableT:
        ...

    @overload
    def set_flags(
        self: VariableT, value: ListLike, cond: None = None, **meta
    ) -> VariableT:
        ...

    def set_flags(self, value, cond=None, **meta):
        # allowed:
        #   cond=None, value=listlike
        #   cond=bool-listlike, value=scalar
        if cond is None and pd.api.types.is_list_like(value):
            self._history.append(value, **meta)
        elif is_indexer(cond) and isinstance(value, (int, float)):
            self._history.append_conditional(value, cond, **meta)
        else:
            raise TypeError(
                "'value' must be float and 'cond' a pd.Index / boolean indexer, "
                "OR value must be a list-like of floats and cond must be None."
                f"type cond: {type(cond)}, type value: {type(value)}"
            )
        return self

    # ########################################################################
    # Masker and Masking
    # ########################################################################

    def _get_mask(self) -> np.ndarray:
        # maybe cache, hash and cash $$$
        # with pd.util.hash_pandas_object() ??
        self._update_mask()
        return self._data.mask

    def _update_mask(self) -> None:
        mask = self._create_mask()
        self._data._raw.soften_mask()
        self._data._raw.mask = mask

    def _create_mask(self) -> np.ndarray:
        masker = self.masker or self.global_masker
        if masker is None:
            return self.__default_masker()

        flags = self.flags.to_numpy()
        length = len(flags)
        try:
            result = masker(flags)
        except Exception as e:
            raise MaskerExecutionError(
                masker, f"Execution of user-set masker {masker} failed"
            ) from e

        try:
            self._check_masker_result(result, length)
        except Exception as e:
            raise MaskerResultError(
                masker,
                f"Unexpected result of the masker-function {masker.__name__}\n"
                f"\tMost probably due to an incorrect implementation:\n"
                f"\t" + str(e),
            ) from None

        return result

    @staticmethod
    def _check_masker_result(result: np.ndarray, length: int) -> None:
        if not isinstance(result, np.ndarray):
            raise TypeError(
                f"Result has wrong type. Expected {np.ndarray}, but got {type(result)}"
            )
        if not result.dtype == bool:
            raise ValueError(
                f"Result has wrong dtype. Expected {np.dtype(bool)!r}, "
                f"but result has {result.dtype!r} dtype"
            )
        if len(result) != length:
            raise ValueError(
                f"Result has wrong length.  Expected {length}, "
                f"but got {len(result)} values"
            )

    @final
    def __default_masker(self) -> np.ndarray:
        # this is faster than series.notna()
        return ~np.isnan(self.flags.to_numpy())

    # ########################################################################
    # Rendering
    # ########################################################################

    @final
    def __str__(self):
        return self.__repr__() + "\n"

    @final
    def __render_frame(self, df: pd.DataFrame) -> str:
        if not self.index.empty:
            if "f0" in df:
                n = df.columns.get_loc("f0")
                df.insert(n, "|", ["|"] * len(df.index))
        return repr(df).replace("DataFrame", self.__class__.__name__)

    @final
    def __repr__(self) -> str:
        try:
            df = self.to_pandas()
        except MaskerError as e:
            # now we create a frame without calling masker
            df = self._history.to_pandas()
            df.insert(0, "flags", self.flags.array)  # type: ignore
            df.insert(0, "**ORIG**", self._data._raw.data)
            string = self.__render_frame(df)
            raise RuntimeError(
                f"\n{string}\nUser-set masker failed for repr(), "
                f"see the prior exception above"
            ) from e

        string = self.__render_frame(df)
        meta = str(self.meta._render_short())
        return string + "\n" + meta

    @final
    def to_string(self, *args, show_meta=True, **kwargs) -> str:
        df = self.to_pandas()
        if not self.index.empty:
            if "f0" in df:
                n = df.columns.get_loc("f0")
                df.insert(n, "|", ["|"] * len(df.index))
        s = df.to_string(*args, **kwargs).replace("DataFrame", self.__class__.__name__)
        if show_meta:
            s += "\n" + str(self.meta._render_short())
        return s

    def memory_usage(self, index: bool = True, deep: bool = False) -> pd.Series:
        # size of Variable
        # --------------------------
        # _history._raw   (pd.Dataframe)
        #   .index      pd.Index = datetime64(8) * length
        #   -> col      float(8) * length * #columns
        # _data._raw    (MaskedArray)
        #   .data       float(8) * length
        #   .mask       bool(1) * length
        # _data._index  is a reference to _flag._raw.index
        # --------------------------
        # (8 + 1 + 8 + 8) * length + (8 * length * columns)
        # with size(column) == size(data) == size(index)
        #   => #columns + data + index = #columns+2
        # => 8B * length * (#columns+2) + 1B * length [from the mask]
        # => 8B * length * (columns+2.125)
        #
        # for comparison:
        # size of a plain data-series and a simple history-df
        # Columns + Data + 2x Index = Columns+3
        # => 8B * length * (columns+3)
        #
        # ==> Variable is x0.875 smaller than a plain Series+History

        flags_df = self._history._raw
        data = self._data._raw.data
        mask = self._data._raw.mask
        flags_index = flags_df.index
        data_index = self._data._index
        result = pd.Series(dtype=int)
        if index:
            n = flags_index.memory_usage(deep=deep)
            if data_index is not flags_index:
                n += data_index.memory_usage(deep=deep)
            result["Index"] = n
        result["history"] = flags_df.memory_usage(index=False, deep=deep).sum()
        result["data"] = data.itemsize * data.size
        result["mask"] = mask.itemsize * mask.size
        return result

    def to_pandas(self) -> pd.DataFrame:
        df = self._history.to_pandas()
        df.insert(0, "flags", self._history.current(raw=False).array)  # type: ignore
        # to prevent calling masker multiple times
        # we use private methods and attributes.
        df.insert(0, "mask", self._get_mask())
        df.insert(0, "data", self._data._raw.filled())
        return df


class ___REF__UNUSED__:
    __slots__ = ("_ref", "name", "meta", "mapping", "last_col")

    _ref: weakref.ReferenceType[BaseVariable]
    name: str | None
    meta: Meta | None
    mapping: pd.Series | None

    def __init__(self, obj: BaseVariable, meta, mapping, last_col):
        self._ref = weakref.ref(obj)
        self.name = obj.name
        self.meta = meta
        self.mapping = mapping
        self.last_col = last_col

    @property
    def _ref_alive(self) -> bool:
        return self._ref() is not None

    @property
    def _ref_var(self) -> BaseVariable | None:
        return self._ref()

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, type(self))
            and self.last_col == other.last_col
            and self.name == other.name
            and (
                self.mapping is None
                and other.mapping is None
                or self.mapping is not None
                and other.mapping is not None
                and self.mapping.equals(other.mapping)
            )
            and (
                self.meta is None
                and other.meta is None
                or self.meta is not None
                and other.meta is not None
                and self.meta._raw.equals(other.meta._raw)
            )
        )

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __repr__(self):
        meta = "meta"
        if self.meta is None:
            meta += ": None"
        else:
            meta += f"\n----\n{self.meta._raw}"
        mapping = "mapping"
        if self.mapping is None:
            mapping += ": None"
        else:
            mapping += f"\n-------\n{self.mapping}"
        return (
            f"Reference\n"
            f"=========\n"
            f"name: {self.name}\n"
            f"{meta}\n"
            f"{mapping}"
        )


# ############################################################################
# (public) Variable
# ############################################################################
# All following public methods must return a new Variable. Flagging functions
# should return a copy of the original variable with a single History column
# appended. They should also set the meta key `func`. For that a helper function,
# called `this()`, is provided. For examples see the already provided methods.
#
# Methods that alter the data, also called 'corrections', must return a brand-new
# Variable, in contrast to a copy of the original.
# ############################################################################


class Variable(BaseVariable):  # noqa
    # tools
    # =====

    def flag_nothing(self, *args, **kwargs) -> Variable:
        """dummy function nothing"""
        return self.copy()

    def replace_flag(self, old: FlagLike, new: FlagLike) -> Variable:
        mask = self.flags == old
        return self.copy().set_flags(new, mask, func=this())

    def clear_flags(self) -> Variable:
        return self.copy().set_flags(SET_UNFLAGGED, self.index, func=this())

    def flag_unflagged(self, flag: FlagLike) -> Variable:
        return self.copy().set_flags(flag, self.is_unflagged(), func=this())

    # ########################################################################
    # Flagging
    # ########################################################################

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

        func_info = get_func_location(func)

        data = self.data
        if raw:
            data = data.to_numpy()  # type: ignore
        mask = func(data)

        if pd.api.types.is_iterator(mask):
            # eg. lambda data: (d==99 for d in data)
            mask = list(mask)  # type: ignore

        if not is_listlike(mask) or not is_boolean_indexer(mask):
            raise TypeError(
                "Expected result of given function to be a boolean list-like."
            )

        return self.copy().set_flags(flag, mask, func=this(), user_func=func_info)

    # ########################################################################
    # Corrections
    # ########################################################################
    #  Functions that alter data.
    # - must return a *new* Variable (not a copy)
    # - may set new flags on the new Variable
    # - may use the existing old History squeezed to a pd.Series (see
    #  `FlagsFrame.current`) as the initial flags for the new Variable
    # ########################################################################
    def dropna(self):
        data = self.orig.dropna()
        flags = self.flags.reindex(data.index)
        return self.derive(data, flags, None, func=this())

    def clip(self, lower, upper, flag: FlagLike = -88) -> Variable:
        # todo: meta
        flags = self.flag_limits(lower, upper, flag=flag).flags
        data = self.data.clip(lower, upper)
        return self.derive(data, flags, self, func=this())

    def interpolate(self, flag: FlagLike | None = None) -> Variable:
        # todo: meta
        if flag is not None:
            flags = self.flagna(flag).flags
        else:
            flags = self.flags
        data = self.data.interpolate()
        return self.derive(data, flags, self, func=this())

    def reindex(self, index=None, method=None) -> Variable:
        # todo: meta
        index = construct_index(index, "index", optional=False)
        data = self.data.reindex(index, method=method)
        return self.derive(data, None, self, func=this())


if __name__ == "__main__":
    from qaqc._testing import dtindex, N
    from qaqc.core.variable import Variable

    v = Variable([1, 2, 3, 4, N, N, 9], index=dtindex(7))
    v = v.flag_limits(lower=2)
    v = v.flag_limits(upper=6)
    v2 = v.dropna()
    v3 = v.reindex(v2.index)
    print(v, v2, v3)
