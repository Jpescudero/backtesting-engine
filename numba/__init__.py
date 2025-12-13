"""Lightweight numba stub used when the real dependency is unavailable."""
from __future__ import annotations

from typing import Any, Callable, TypeVar, overload

F = TypeVar("F", bound=Callable[..., Any])


@overload
def njit(func: F) -> F:
    ...


@overload
def njit(signature: Any | None = ..., **kwargs: Any) -> Callable[[F], F]:
    ...


def njit(func: Any | None = None, **kwargs: Any):  # type: ignore[override]
    """Fallback decorator that returns the original function unmodified."""

    def decorator(inner: F) -> F:
        return inner

    if callable(func):
        return func
    return decorator


__all__ = ["njit"]
