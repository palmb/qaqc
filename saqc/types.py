#!/usr/bin/env python
from __future__ import annotations


from typing import Any, TypeVar, Callable, Union

try:
    from typing import Self
except ImportError:
    Self = TypeVar("Self")


import pandas as pd

# Either Variable or SaQC not the Union of both,
# that mean tha a function defined like this `foo(a: VarOrQcT) -> VarOrQcT`
# returns an object of the same type as `a` is.
# It never takes a Variable and return a SaQC object.
VarOrQcT = TypeVar("VarOrQcT", "Variable", "SaQC")

FlagsFrameT = TypeVar("FlagsFrameT", bound='FlagsFrame')
SupportsIndex = Union[pd.DataFrame, "Variable", FlagsFrameT]
SupportsColumns = Union[pd.DataFrame, FlagsFrameT]

# to maintain type information across generic functions and parametrization
T = TypeVar("T")

# used in decorators to preserve the signature of the function it decorates
# see https://mypy.readthedocs.io/en/stable/generics.html#declaring-decorators
FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)

