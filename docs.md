# Architecture

## Overview

Mini Autograd is a scalar reverse-mode automatic differentiation engine
with a minimal neural-network abstraction.

## Core Abstraction

The central class is `Value`.

Each `Value` stores:

- `data`: scalar numerical value
- `grad`: accumulated derivative
- `_prev`: parent nodes in the computation graph
- `_op`: operation that created the node
- `_backward`: local backward function

## Computation Graph

Every arithmetic operation creates a new `Value` and stores its
dependencies.

Example:

a = Value(2)
b = Value(3)
c = a * b

Graph:

a ──┐
    ×── c
b ──┘

## Backpropagation

`Value.backward()` first builds a topological ordering of the graph.

The graph is then traversed in reverse order.

Each node executes its local `_backward()` function.

Gradients are accumulated using:

    parent.grad += local_gradient * output.grad

This implements the chain rule.

## Neural Network

The neural-network layer consists of:

Value
  ↓
Neuron
  ↓
Layer
  ↓
MLP

Each neuron computes:

    tanh(sum(w_i * x_i) + b)

## Optimization

Training uses vanilla gradient descent:

    parameter -= learning_rate * gradient

## Gradient Verification

Analytical gradients are checked against numerical finite differences:

    f'(x) ≈ (f(x+h) - f(x-h)) / (2h)

This provides an independent correctness check for the autograd engine.

## Design Goals

The project prioritizes:

- simplicity
- explicit computation graphs
- understandable backpropagation
- testability
- educational value