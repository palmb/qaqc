#!/usr/bin/env python
from __future__ import annotations
import operator

# copied from pandas.core.arraylike
# and pandas.core.roperator


def radd(left, right):
    return right + left


def rsub(left, right):
    return right - left


def rmul(left, right):
    return right * left


def rdiv(left, right):
    return right / left


def rtruediv(left, right):
    return right / left


def rfloordiv(left, right):
    return right // left


def rmod(left, right):
    # check if right is a string as % is the string
    # formatting operation; this is a TypeError
    # otherwise perform the op
    if isinstance(right, str):
        typ = type(left).__name__
        raise TypeError(f"{typ} cannot perform the operation mod")

    return right % left


def rdivmod(left, right):
    return divmod(right, left)


def rpow(left, right):
    return right**left


def rand_(left, right):
    return operator.and_(right, left)


def ror_(left, right):
    return operator.or_(right, left)


def rxor(left, right):
    return operator.xor(right, left)


class OpsMixin:
    # -------------------------------------------------------------
    # Comparisons

    def _cmp_method(self, other, op):
        return NotImplemented

    def __eq__(self, other):
        return self._cmp_method(other, operator.eq)

    def __ne__(self, other):
        return self._cmp_method(other, operator.ne)

    def __lt__(self, other):
        return self._cmp_method(other, operator.lt)

    def __le__(self, other):
        return self._cmp_method(other, operator.le)

    def __gt__(self, other):
        return self._cmp_method(other, operator.gt)

    def __ge__(self, other):
        return self._cmp_method(other, operator.ge)

    # -------------------------------------------------------------
    # Logical Methods

    def _logical_method(self, other, op):
        return NotImplemented

    def __and__(self, other):
        return self._logical_method(other, operator.and_)

    def __rand__(self, other):
        return self._logical_method(other, rand_)

    def __or__(self, other):
        return self._logical_method(other, operator.or_)

    def __ror__(self, other):
        return self._logical_method(other, ror_)

    def __xor__(self, other):
        return self._logical_method(other, operator.xor)

    def __rxor__(self, other):
        return self._logical_method(other, rxor)

    # -------------------------------------------------------------
    # Arithmetic Methods

    def _arith_method(self, other, op):
        return NotImplemented

    def __add__(self, other):
        return self._arith_method(other, operator.add)

    def __radd__(self, other):
        return self._arith_method(other, radd)

    def __sub__(self, other):
        return self._arith_method(other, operator.sub)

    def __rsub__(self, other):
        return self._arith_method(other, rsub)

    def __mul__(self, other):
        return self._arith_method(other, operator.mul)

    def __rmul__(self, other):
        return self._arith_method(other, rmul)

    def __truediv__(self, other):
        return self._arith_method(other, operator.truediv)

    def __rtruediv__(self, other):
        return self._arith_method(other, rtruediv)

    def __floordiv__(self, other):
        return self._arith_method(other, operator.floordiv)

    def __rfloordiv__(self, other):
        return self._arith_method(other, rfloordiv)

    def __mod__(self, other):
        return self._arith_method(other, operator.mod)

    def __rmod__(self, other):
        return self._arith_method(other, rmod)

    def __divmod__(self, other):
        return self._arith_method(other, divmod)

    def __rdivmod__(self, other):
        return self._arith_method(other, rdivmod)

    def __pow__(self, other):
        return self._arith_method(other, operator.pow)

    def __rpow__(self, other):
        return self._arith_method(other, rpow)
