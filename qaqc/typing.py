#!/usr/bin/env python
from __future__ import annotations


from typing import (
    Any,
    TypeVar,
    Callable,
    Union,
    TYPE_CHECKING,
    Collection,
    Sequence,
)

import numpy as np
import pandas as pd
from numpy.typing import NDArray


from qaqc.core.utils import FuncInfo  # noqa

from typing_extensions import final, Final  # noqa

if TYPE_CHECKING:
    from qaqc.core.flagsframe import FlagsFrame
    from qaqc.core.variable import Variable, BaseVariable
    from qaqc.core.frame import QaqcFrame
else:
    FlagsFrame = "FlagsFrame"
    BaseVariable = "BaseVariable"
    Variable = "Variable"
    QaqcFrame = "QaqcFrame"

# ############################################################
# Types that are supported with isinstance
# ############################################################
SupportsIndex = Union[pd.DataFrame, pd.Series, Variable, FlagsFrame]
SupportsColumns = Union[pd.DataFrame, FlagsFrame]

Axes = Union[pd.Index, NDArray, list, range]
Columns = Union[pd.Index, NDArray, list[str]]

Numeric = Union[int, float]  # we do not accept complex (yet?)
FlagLike = Numeric
Scalar = Union[Numeric, str, bool]
PandasLike = Union[pd.Series, pd.DataFrame]
ListLike = Union[pd.Series, list, NDArray]

# ############################################################
# SomeThingT  only for type checker not usable with isinstance
# ############################################################

# FloatSeries = TypeVar("FloatSeries", bound=pd.Series)
# FloatArray = TypeVar("FloatArray", bound=np.ndarray)
# BoolSeries = TypeVar("BoolSeries", bound=pd.Series)
# BoolArray = TypeVar("BoolArray", bound=np.ndarray)

# unfortunately Sequence is not overwrite subclasshook and so
# does not work reliable with isinstance
MaskT = Union[pd.Series, NDArray, Sequence[bool]]
CondT = Union[MaskT, pd.Index]  # add Callable[..., MaskT] at some point

MaskerT = Callable[[NDArray], NDArray]

# VariableT is stricter and ensures that the same subclass of Variable always is
# used. E.g. `def func(a: VariableT) -> variableT: ...` means that if a
# MaskedVariable is passed into a function, a MaskedVariable is always
# returned and if a Variable is passed in, a Variable is always returned.
VariableT = TypeVar("VariableT", bound=BaseVariable)
QaqcFrameT = TypeVar("QaqcFrameT", bound=QaqcFrame)

# Either Variable or SaQC not the Union of both,
# that mean tha a function defined like this `foo(a: VarOrQcT) -> VarOrQcT`
# returns an object of the same type as `a` is.
# It never takes a Variable and return a SaQC object.
VarOrQcT = TypeVar("VarOrQcT", Variable, QaqcFrame)

# to maintain type information across generic functions and parametrization
T = TypeVar("T")

# used in decorators to preserve the signature of the function it decorates
# see https://mypy.readthedocs.io/en/stable/generics.html#declaring-decorators
FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)
