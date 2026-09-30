from mini_autograd.nn import MLP
from mini_autograd.value import Value


def main() -> None:
    model = MLP(2, [4, 4, 1])

    dataset = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ]

    learning_rate = 0.001
    epochs = 4000

    for epoch in range(epochs):
        predictions = []

        # Forward
        for inputs, target in dataset:
            x = [Value(value) for value in inputs]
            prediction = model(x)[0]

            predictions.append((prediction, target))

        # Loss
        loss = Value(0.0)

        for prediction, target in predictions:
            error = prediction - target
            loss = loss + error**2

        # Reset gradients
        model.zero_grad()

        # Backward
        loss.backward()
        model.step(learning_rate)
        # Gradient descent
        for parameter in model.parameters():
            parameter.data -= learning_rate * parameter.grad

        if epoch % 100 == 0:
            print(f"epoch={epoch:4d} loss={loss.data:.6f}")

    print("\nFinal predictions:")

    for inputs, target in dataset:
        x = [Value(value) for value in inputs]
        prediction = model(x)[0]

        print(f"{inputs} -> {prediction.data:.4f} (target={target})")


if __name__ == "__main__":
    main()
