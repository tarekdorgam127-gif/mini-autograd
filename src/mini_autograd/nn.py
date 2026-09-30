from __future__ import annotations

import random

from mini_autograd.value import Value


class Neuron:
    def __init__(self, nin: int):
        self.w = [Value(random.uniform(-1.0, 1.0)) for _ in range(nin)]

        self.b = Value(0.0)

    def __call__(self, x: list[Value]) -> Value:
        activation = self.b

        for wi, xi in zip(self.w, x):
            activation = activation + wi * xi

        return activation.tanh()

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer:
    def __init__(self, nin: int, nout: int):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x: list[Value]) -> list[Value]:
        return [neuron(x) for neuron in self.neurons]

    def parameters(self) -> list[Value]:
        return [
            parameter for neuron in self.neurons for parameter in neuron.parameters()
        ]


class MLP:
    def __init__(self, nin: int, nouts: list[int]):
        sizes = [nin] + nouts

        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(nouts))]

    def __call__(self, x: list[Value]) -> list[Value]:
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self) -> list[Value]:
        return [parameter for layer in self.layers for parameter in layer.parameters()]

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()

    def step(self, learning_rate: float) -> None:
        for parameter in self.parameters():
            parameter.data -= learning_rate * parameter.grad
