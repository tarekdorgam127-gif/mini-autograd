import math

from mini_autograd.utils.numerical import numerical_gradient
from mini_autograd.value import Value


def test_addition():
    a = Value(2)
    b = Value(3)

    c = a + b

    assert c.data == 5


def test_multiplication():
    a = Value(2)
    b = Value(3)

    c = a * b

    assert c.data == 6


def test_backward():
    a = Value(2)
    b = Value(3)

    c = a * b

    c.backward()

    assert a.grad == 3
    assert b.grad == 2


def test_complex_expression():
    a = Value(2)
    b = Value(3)

    c = a * b
    d = c + a
    e = d**2

    e.backward()

    assert a.grad == 64
    assert b.grad == 32


def test_scalar_operations():
    x = Value(3)

    assert (x + 2).data == 5
    assert (2 + x).data == 5
    assert (x * 2).data == 6
    assert (2 * x).data == 6
    assert (x - 2).data == 1
    assert (2 - x).data == -1
    assert (x / 2).data == 1.5


def test_tanh_forward():
    x = Value(0)

    y = x.tanh()

    assert y.data == 0.0


def test_tanh_backward():
    x = Value(0)

    y = x.tanh()
    y.backward()

    assert math.isclose(x.grad, 1.0)


def test_tanh_gradient_against_numerical():
    x = Value(1.7)

    y = x.tanh()
    y.backward()

    numerical = numerical_gradient(
        lambda: x.tanh(),
        x,
    )

    assert math.isclose(
        x.grad,
        numerical,
        rel_tol=1e-5,
        abs_tol=1e-5,
    )
