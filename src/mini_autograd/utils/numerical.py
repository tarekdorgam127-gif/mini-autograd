from __future__ import annotations

from mini_autograd.value import Value


def numerical_gradient(
    f,
    x: Value,
    h: float = 1e-6,
) -> float:
    original = x.data

    x.data = original + h
    y_plus = f().data

    x.data = original - h
    y_minus = f().data

    x.data = original

    return (y_plus - y_minus) / (2 * h)
