#!/usr/bin/env python
from __future__ import annotations
import operator
import pandas.core.roperator as roperator


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
        return self._logical_method(other, roperator.rand_)

    def __or__(self, other):
        return self._logical_method(other, operator.or_)

    def __ror__(self, other):
        return self._logical_method(other, roperator.ror_)

    def __xor__(self, other):
        return self._logical_method(other, operator.xor)

    def __rxor__(self, other):
        return self._logical_method(other, roperator.rxor)

    # -------------------------------------------------------------
    # Arithmetic Methods

    def _arith_method(self, other, op):
        return NotImplemented

    def __add__(self, other):
        return self._arith_method(other, operator.add)

    def __radd__(self, other):
        return self._arith_method(other, roperator.radd)

    def __sub__(self, other):
        return self._arith_method(other, operator.sub)

    def __rsub__(self, other):
        return self._arith_method(other, roperator.rsub)

    def __mul__(self, other):
        return self._arith_method(other, operator.mul)

    def __rmul__(self, other):
        return self._arith_method(other, roperator.rmul)

    def __truediv__(self, other):
        return self._arith_method(other, operator.truediv)

    def __rtruediv__(self, other):
        return self._arith_method(other, roperator.rtruediv)

    def __floordiv__(self, other):
        return self._arith_method(other, operator.floordiv)

    def __rfloordiv__(self, other):
        return self._arith_method(other, roperator.rfloordiv)

    def __mod__(self, other):
        return self._arith_method(other, operator.mod)

    def __rmod__(self, other):
        return self._arith_method(other, roperator.rmod)

    def __divmod__(self, other):
        return self._arith_method(other, divmod)

    def __rdivmod__(self, other):
        return self._arith_method(other, roperator.rdivmod)

    def __pow__(self, other):
        return self._arith_method(other, operator.pow)

    def __rpow__(self, other):
        return self._arith_method(other, roperator.rpow)
