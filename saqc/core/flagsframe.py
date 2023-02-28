#!/usr/bin/env python
from __future__ import annotations

import pandas as pd
import numpy as np
from copy import deepcopy
from saqc.types import Any
from saqc.constants import UNFLAGGED


class FlagsFrame:
    def __init__(self, data: pd.DataFrame, meta: list[dict[str, Any]]):
        self._raw: pd.DataFrame = data
        self._meta: list[dict[str, Any]] = meta
        assert isinstance(data, pd.DataFrame)
        assert isinstance(meta, list) and all(isinstance(e, dict) for e in meta)
        self._validate()

    def _validate(self):
        assert len(self._raw.columns) == len(self.meta)

    raw = property(lambda self: self._raw, doc="raw flags")
    meta = property(lambda self: self._meta)
    index = property(lambda self: self._raw.index, doc="index of flags")

    def copy(self, deep=True):
        if deep is True:
            return FlagsFrame(self._raw.copy(), deepcopy(self._meta))
        return FlagsFrame(self._raw, self._meta)

    def __len__(self):
        return len(self._raw.columns)

    def __repr__(self):
        return repr(self._raw).replace("DataFrame", "FlagsFrame") + "\n"

    def current(self) -> pd.Series:
        if self.raw.empty:
            result = pd.Series(data=np.nan, index=self.raw.index, dtype=float)
        else:
            result = self.raw.ffill(axis=1).iloc[:, -1]
        return result.fillna(UNFLAGGED)

    def append(self, value: pd.Series, meta=None) -> None:
        assert isinstance(value, pd.Series)
        assert self.index.equals(value.index)
        self.raw[len(self)] = value
        self.meta.append(meta or dict())



