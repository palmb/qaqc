#!/usr/bin/env python
from __future__ import annotations

import numpy as np
import pandas as pd

N = np.nan


def dtindex(periods: int, start="2000", freq="1D", end=None, jitter: str | None = None):
    return pd.date_range(start, end, periods, freq)


if __name__ == "__main__":
    i = dtindex(10)
    print(i)
