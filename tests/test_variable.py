#!/usr/bin/env python
from __future__ import annotations

import pandas as pd
import pytest
from qaqc import Variable


@pytest.mark.parametrize("data", [[1, 2, 3], pd.Series([0, 1]), range(4)])
def test_construct(data):
    v = Variable(data)
    assert isinstance(v, Variable)
    assert isinstance(v.index, pd.Index)
