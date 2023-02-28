#!/usr/bin/env python
from __future__ import annotations


import pandas as pd
from saqc import Variable, SaQC


if __name__ == "__main__":
    s = pd.Series(range(5), dtype=float)
    f = Variable(s)
    r = f.mask(s >= 2).setAll(77).unmask().select(s > 2).setAll(66).deselect()
    print(r, "\n\n")

    print("###### with statement ####", "\n\n")

    s = pd.Series(range(8), dtype=float)
    g = Variable(s)
    with g.selecting(s > 2) as masked:
        masked = masked.setAll(11)
        with masked.selecting(s >= 5) as masked_the_masked:
            r = masked_the_masked.setAll(99)
    print(r)

if __name__ == "__main__":
    n = 5
    index = pd.date_range("2000", freq="1d", periods=n)
    x = Variable(pd.Series(range(n), index=index, dtype=float))
    b = x.flagUpper(3, flag=33)

    qc = SaQC()
    qc["a"] = x
    qc["b"] = x.copy()
    qc = qc.flagUpper(3)
    qc = qc.flagFoo()
    qc = qc.dododo(None, None)
    qc.show()
    qc = qc.dododo(None, None)
    qc.show()

    # pd.DataFrame.replace()
    # pd.DataFrame.reindex
    # pd.Series.reindex()

    # a.flags.current()
    #
    # print("a")
    # print(a)
    # print("a.data")
    # print(a.data, end="\n\n")
    # print("a.flags")
    # print(a.flags)
