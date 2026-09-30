# Mini Autograd

A tiny automatic differentiation engine built from scratch in Python, inspired by [micrograd](https://github.com/karpathy/micrograd).

The project implements scalar reverse-mode automatic differentiation and uses the resulting engine to build and train a small multilayer perceptron (MLP) on the XOR problem.

## Features

* Scalar `Value` objects
* Computation graph construction
* Reverse-mode automatic differentiation
* Topological graph traversal for backpropagation
* Gradient accumulation
* Operator overloading
* Addition, subtraction, multiplication, division, and power
* `tanh` activation
* Numerical gradient checking
* `Neuron`, `Layer`, and `MLP` abstractions
* Gradient descent parameter updates
* Gradient reset with `zero_grad()`
* Automated tests with `pytest`
* Code quality checks with Ruff

## Architecture

The project is organized around two main components:

### `Value`

`Value` represents a scalar number together with:

* its numerical value: `data`
* its derivative: `grad`
* references to parent nodes
* the operation that created it
* a local backward function

For example:

```python
from mini_autograd import Value

a = Value(2.0)
b = Value(3.0)

c = a * b
c.backward()

print(a.grad)  # 3.0
print(b.grad)  # 2.0
```

### Neural Network

The neural-network layer is built on top of `Value`:

```text
Value
  ↓
Neuron
  ↓
Layer
  ↓
MLP
  ↓
Gradient Descent
```

The MLP can be trained using the automatic differentiation engine without relying on PyTorch, TensorFlow, or another autograd framework.

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv ml-env
source ml-env/bin/activate
```

Install the project:

```bash
python -m pip install -e .
```

Install development dependencies:

```bash
python -m pip install pytest ruff
```

## Usage

A basic automatic differentiation example:

```python
from mini_autograd import Value

x = Value(2.0)
y = Value(3.0)

z = x * y + x

z.backward()

print(z.data)
print(x.grad)
print(y.grad)
```

## Training the MLP

Run the XOR training example:

```bash
python examples/train.py
```

The network learns the XOR mapping:

```text
[0.0, 0.0] -> approximately 0
[0.0, 1.0] -> approximately 1
[1.0, 0.0] -> approximately 1
[1.0, 1.0] -> approximately 0
```

One successful run reached a final loss of approximately:

```text
0.0158
```

with predictions approximately:

```text
[0.0, 0.0] -> 0.0051
[0.0, 1.0] -> 0.9097
[1.0, 0.0] -> 0.9184
[1.0, 1.0] -> 0.0136
```

Because the model uses random initialization, exact results may vary between runs.

## Testing

Run the complete test suite:

```bash
pytest -v
```

Current test coverage includes:

* Basic arithmetic
* Scalar operations
* Backpropagation
* Complex computation graphs
* `tanh`
* Numerical gradient verification
* MLP forward pass
* MLP backward pass
* Gradient resetting
* Training loss reduction

The project currently passes:

```text
12 passed
```

## Code Quality

Check the code with Ruff:

```bash
ruff check .
ruff format --check .
```

## Mathematics

The engine implements reverse-mode automatic differentiation using the chain rule.

For multiplication:

```text
z = x * y

∂z/∂x = y
∂z/∂y = x
```

For `tanh`:

```text
z = tanh(x)

∂z/∂x = 1 - tanh(x)^2
```

During `backward()`, the computation graph is traversed in reverse topological order so that gradients can be propagated from the output back to the inputs.

## Project Structure

```text
mini_autograd/
├── src/
│   └── mini_autograd/
│       ├── __init__.py
│       ├── value.py
│       ├── nn.py
│       └── utils/
│           ├── __init__.py
│           └── numerical.py
├── tests/
│   ├── test_value.py
│   └── test_nn.py
├── examples/
│   └── train.py
├── pyproject.toml
├── README.md
└── .gitignore
```

## Limitations

This project intentionally focuses on simplicity and educational value.

Current limitations include:

* Scalar values only
* No tensor/vector operations
* No GPU support
* No batching abstraction
* Limited activation functions
* No optimizers beyond basic gradient descent
* No automatic graph visualization

## Future Improvements

Possible extensions include:

* Tensor support
* Additional activation functions
* More optimizers
* Graph visualization
* Better reproducibility controls
* More neural-network layers
* More comprehensive gradient tests
* Benchmarking against other implementations

## Motivation

The goal of this project is to understand what automatic differentiation and neural-network training are doing internally rather than treating autograd as a black box.

The implementation demonstrates how computation graphs, local derivatives, the chain rule, and gradient descent fit together to train a neural network from scratch.
