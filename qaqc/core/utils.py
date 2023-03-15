#!/usr/bin/env python
from __future__ import annotations

from typing import Callable, Any, overload
import pandas as pd
import textwrap
import collections.abc as abc
import inspect
from collections import namedtuple


FuncInfo = namedtuple("FuncInfo", "name file lineno")


def maybe_construct_Index(obj, errors='raise', optional=False) -> pd.Index | None:
    """ make pd.Index from obj or return obj if already an Index"""
    if isinstance(obj, pd.Index):
        return obj
    if obj is None and optional:
        return obj
    try:
        return pd.Index(obj)
    except (ValueError, TypeError) as e:
        if errors == 'raise':
            raise
        return None


def dict_to_keywords(
    dict_: dict, deep=False, brace=True, extended=True, **kwargs
) -> str:
    """
    Make a oneline repr (like keywords) from dict
    """
    kwargs = {**kwargs, "kwdicts": False}

    # needed for self-referential dicts (d=dict(); d['r'] = d)
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
        if extended:
            return repr_extended(obj, **kwargs)
        return repr(obj)

    return _dict_to_keywords(dict_, to_repr=_repr, enclose=brace)


def get_placeholder(obj, default: str = " [..]") -> str:
    if isinstance(obj, list):
        return " ..]"
    if isinstance(obj, str):
        return " [..]"
    if isinstance(obj, type):
        return " ..>"
    return default


def repr_extended(
    obj: Any,
    wrap: int | None = None,
    maxlen: int | None = None,
    maxlen_items: int | None = None,
    kwdicts: bool = True,
) -> str:
    rpr: str
    assert wrap is None or isinstance(wrap, int)

    if kwdicts and isinstance(obj, dict):
        rpr = dict_to_keywords(
            obj, deep=True, brace=True, extended=True, maxlen=maxlen_items
        )
    elif isinstance(obj, pd.Series):
        rpr = f"Series[{len(obj)} rows]"
    elif isinstance(obj, pd.DataFrame):
        rpr = f"DataFrame[{obj.shape[0]} rows x {obj.shape} columns]"
    elif isinstance(obj, FuncInfo):
        rpr = f"FuncInfo(name={obj.name}, ...)"
    else:
        rpr = repr(obj)
    if maxlen is not None:
        ph = get_placeholder(obj, default=" [..]")
        rpr = textwrap.shorten(rpr, maxlen, placeholder=ph)
    if wrap:
        rpr = textwrap.fill(rpr, wrap)
    return rpr


def is_iterable(obj: Any) -> bool:
    try:
        iter(obj)
    except Exception:
        return False
    return True


def is_listlike(obj: Any, allow_sets: bool = False) -> bool:
    return pd.api.types.is_list_like(obj, allow_sets=allow_sets)


def is_boolean_indexer(obj: abc.Iterable) -> bool:
    """
    Check whether `obj` is a valid boolean indexer.

    This assumes that obj is iterable !

    Parameters
    ----------
    obj : Any
        Only list-likes may be considered as valid
        boolean indexer. At least `obj` must be iterable.

    Returns
    -------
    bool
        Whether `obj` is valid boolean indexer.
    """
    return (
        hasattr(obj, "dtype")
        and obj.dtype == bool
        # (return first non-boolean element or return True) is True
        or next(filter(lambda e: not isinstance(e, bool), obj), True) is True
    )


def is_indexer(obj: Any) -> bool:
    return isinstance(obj, pd.Index) or is_iterable(obj) and is_boolean_indexer(obj)


def forall(obj, cond, args=()):
    return next(filter(lambda e: not cond(e, *args), obj), True) is True


def get_func_location(obj: Callable, errors: str = "raise") -> FuncInfo | None:
    """
    Return funcdef_name, funcdef_file, lineno
    errors: 'raise', 'ignore'
    """
    from inspect import isclass, isfunction, ismethod, isbuiltin

    def get_loc():
        if not callable(obj):
            raise TypeError("obj must be a callable")

        if isbuiltin(obj):
            name = obj.__qualname__
            file = "build-in function"
            line = ""
        elif isclass(obj):
            raise NotImplementedError(
                "Only callable instances, methods and "
                "functions are supported, not classes"
            )
        else:
            if ismethod(obj):
                name = obj.__qualname__
                code = obj.__func__.__code__
            elif isfunction(obj):  # include lambdas
                name = obj.__qualname__
                code = obj.__code__
            else:  # callable class instance
                name = obj.__qualname__
                code = obj.__call__.__func__.__code__
            file = code.co_filename
            line = code.co_firstlineno
        return FuncInfo(name, file, line)

    try:
        return get_loc()
    except Exception:
        if errors == "raise":
            raise
    return None


def get_caller(stacklevel=1):
    """get parent function"""
    stack = inspect.stack(0)
    if len(stack) > stacklevel:
        frame = stack[stacklevel]
        name, file, line = frame.function, frame.filename, frame.lineno
    else:
        name, file, line = "Unknown", "Unknown", 0
    return FuncInfo(name, file, line)


if __name__ == "__main__":
    d = dict(a=dict(a=0, b=99, c=dict(a=dict(d=dict))), b=999999)

    a = get_func_location(lambda x: x)

    # r = dict_to_keywords(d)
    # print(r)
    # r = dict_to_keywords(d, deep=True)
    # print(r)
    # r = dict_to_keywords(d, brace=False)
    # print(r)
    # r = dict_to_keywords(d, brace=False, deep=True)
    # print(r)
    #
    # d = dict(a=99, b=None)
    d["d"] = d
    d["r"] = dict(d=d)
    # r = dict_to_keywords(d, deep=True)
    # print(r)

    d["laalaalsalslsl"] = "lala " * 10
    d["li"] = list(range(300))

    print(d)
    print(repr_extended(d, maxlen_items=12))
