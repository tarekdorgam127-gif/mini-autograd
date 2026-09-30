from __future__ import annotations

import math
from collections.abc import Callable


class Value:
    def __init__(
        self,
        data: float,
        _children: tuple[Value, ...] = (),
        _op: str = "",
    ):
        self.data = float(data)
        self.grad = 0.0

        self._prev = set(_children)
        self._op = _op

        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data + other.data,
            (self, other),
            "+",
        )

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __radd__(self, other: float) -> Value:
        return self + other

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data * other.data,
            (self, other),
            "*",
        )

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other: float) -> Value:
        return self * other

    def __neg__(self) -> Value:
        return self * -1

    def __sub__(self, other: Value | float) -> Value:
        return self + (-other)

    def __rsub__(self, other: float) -> Value:
        return other + (-self)

    def __pow__(self, other: float) -> Value:
        if not isinstance(other, (int, float)):
            raise TypeError("Power must be an int or float")

        out = Value(
            self.data**other,
            (self,),
            f"**{other}",
        )

        def _backward() -> None:
            self.grad += other * self.data ** (other - 1) * out.grad

        out._backward = _backward

        return out

    def __truediv__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return self * (other**-1)

    def __rtruediv__(self, other: float) -> Value:
        return other * (self**-1)

    def backward(self) -> None:
        topo: list[Value] = []
        visited: set[Value] = set()

        def build_topological(node: Value) -> None:
            if node not in visited:
                visited.add(node)

                for child in node._prev:
                    build_topological(child)

                topo.append(node)

        build_topological(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()

    def tanh(self) -> Value:
        t = math.tanh(self.data)

        out = Value(
            t,
            (self,),
            "tanh",
        )

        def _backward() -> None:
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out

    def zero_grad(self) -> None:
        self.grad = 0.0
