#!/usr/bin/env python
from __future__ import annotations
from typing import Callable
from saqc._typing import PandasT
import pandas as pd


def dict_to_keywords(
    dict_: dict, deep=False, brace=True, extended=True, **kwargs
) -> str:
    """
    Make a oneline repr (like keywords) from dict
    """
    kwargs = {**kwargs, "kwdicts": False}

    # needed for self-referential dicts (d=dict(); d['x'] = d)
    dicts = {id(dict_)}  # a set

    def _dict_to_keywords(obj: dict, to_repr: Callable, enclose: bool) -> str:
        s = ", ".join(f"{str(k)}={to_repr(v)}" for k, v in obj.items())
        return "{" + s + "}" if enclose else s

    def _repr(obj):
        if deep and isinstance(obj, dict):
            if id(obj) in dicts:
                return "{...}"
            dicts.add(id(obj))
            return _dict_to_keywords(obj, to_repr=_repr, enclose=True)
        return repr_extended(obj, **kwargs)

    return _dict_to_keywords(dict_, to_repr=_repr, enclose=brace)


def repr_extended(obj, wrap=70, maxlen=None, maxlen_items=None, kwdicts=True):
    if kwdicts and isinstance(obj, dict):
        return dict_to_keywords(obj, deep=True, brace=True)
    if isinstance(obj, pd.Series):
        return f"Series[{len(obj)} rows]"
    if isinstance(obj, pd.DataFrame):
        return f"DataFrame[{obj.shape[0]} rows x {obj.shape} columns]"
    return repr(obj)


if __name__ == "__main__":
    d = dict(a=dict(a=0, b=99, c=dict(a=dict(d=dict))), b=999999)

    r = dict_to_keywords(d)
    print(r)
    r = dict_to_keywords(d, deep=True)
    print(r)
    r = dict_to_keywords(d, brace=False)
    print(r)
    r = dict_to_keywords(d, brace=False, deep=True)
    print(r)

    d = dict(a=99, b=None)
    d["d"] = d
    d["r"] = dict(d=d)
    print(d)
    r = dict_to_keywords(d, deep=True)
    print(r)
