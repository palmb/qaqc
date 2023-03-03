#!/usr/bin/env python
from __future__ import annotations

import pandas as pd
import numpy as np
from copy import deepcopy
from saqc.types import Any, SupportsIndex
from saqc.constants import UNFLAGGED


class FlagsFrame:
    def __init__(self, data, meta: list[dict[str, Any]], index: pd.Index | None = None):
        self._raw: np.ndarray = np.array(data)
        self._index = index
        self._meta: list[dict[str, Any]] = meta

        # checks
        assert isinstance(data, np.ndarray)
        assert data.ndim == 2
        assert isinstance(index, pd.Index)
        assert index.is_unique
        assert len(index) == data.shape[0]
        assert isinstance(meta, list) and all(isinstance(e, dict) for e in meta)

        self._raw.flags.writeable = False

        # test
        assert isinstance(self.data, pd.DataFrame)

    @classmethod
    def init_like(cls, obj: SupportsIndex) -> FlagsFrame:
        """Another constructor."""
        return cls(
            data=np.full(shape=(len(obj.index), 1), fill_value=np.nan, dtype=float),
            index=obj.index,
            meta=[dict()],
        )

    @property
    def data(self) -> pd.DataFrame:
        return pd.DataFrame(data=self._raw, index=self._index, dtype=float)

    @property
    def index(self):
        return self._index

    def copy(self, deep=True):
        raw = self._raw
        meta = self._meta
        if deep is True:
            raw = raw.copy()
            meta = deepcopy(meta)
        return FlagsFrame(raw, meta=meta, index=self.index)

    def __len__(self):
        return self._raw.shape[1]

    def __repr__(self):
        return repr(self.data).replace("DataFrame", "FlagsFrame") + "\n"

    def current(self) -> pd.Series:
        if self._raw.shape[1]:
            result = pd.Series(data=np.nan, index=self.raw.index, dtype=float)
        else:
            result = self.raw.ffill(axis=1).iloc[:, -1]
        return result.fillna(UNFLAGGED)

    def append(self, value: pd.Series, meta=None) -> None:
        assert isinstance(value, pd.Series)
        assert self.index.equals(value.index)
        self.raw[len(self)] = value
        self.meta.append(meta or dict())
