from mini_autograd.nn import MLP
from mini_autograd.value import Value


def test_mlp_forward():
    model = MLP(3, [4, 4, 1])

    x = [
        Value(1.0),
        Value(2.0),
        Value(3.0),
    ]

    output = model(x)

    assert len(output) == 1
    assert isinstance(output[0], Value)


def test_mlp_backward():
    model = MLP(3, [4, 4, 1])

    x = [
        Value(1.0),
        Value(2.0),
        Value(3.0),
    ]

    output = model(x)
    loss = output[0] ** 2

    loss.backward()

    parameters = model.parameters()

    assert len(parameters) > 0
    assert any(parameter.grad != 0 for parameter in parameters)


def test_zero_grad():
    model = MLP(3, [4, 1])

    x = [
        Value(1.0),
        Value(2.0),
        Value(3.0),
    ]

    output = model(x)
    loss = output[0] ** 2

    loss.backward()

    model.zero_grad()

    assert all(parameter.grad == 0 for parameter in model.parameters())


def test_training_reduces_loss():
    model = MLP(2, [4, 1])

    dataset = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ]

    def calculate_loss() -> float:
        loss = Value(0.0)

        for inputs, target in dataset:
            x = [Value(value) for value in inputs]
            prediction = model(x)[0]
            loss = loss + (prediction - target) ** 2

        return loss.data

    initial_loss = calculate_loss()

    for _ in range(200):
        model.zero_grad()

        loss = Value(0.0)

        for inputs, target in dataset:
            x = [Value(value) for value in inputs]
            prediction = model(x)[0]
            loss = loss + (prediction - target) ** 2

        loss.backward()

        for parameter in model.parameters():
            parameter.data -= 0.05 * parameter.grad

    final_loss = calculate_loss()

    assert final_loss < initial_loss
