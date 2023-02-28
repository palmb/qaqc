#!/usr/bin/env python
from __future__ import annotations


from typing import Any, TypeVar, Callable

__all__ = [
    "Any",
    "VarOrQcT"
]

# Either Variable or SaQC not the Union of both,
# that mean tha a function defined like this `foo(a: VarOrQcT) -> VarOrQcT`
# returns an object of the same type as `a` is.
# It never takes a Variable and return a SaQC object.
VarOrQcT = TypeVar("VarOrQcT", "Variable", "SaQC")


# to maintain type information across generic functions and parametrization
T = TypeVar("T")

# used in decorators to preserve the signature of the function it decorates
# see https://mypy.readthedocs.io/en/stable/generics.html#declaring-decorators
FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)

