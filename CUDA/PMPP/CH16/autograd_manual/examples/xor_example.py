import numpy as np
from nn.layers.activations import ReLU, Sigmoid
from nn.layers.linear import Linear
from nn.loss import MSELoss
from nn.model import Sequential
from nn.optim import SGD


def train_model(
    model, criterion, optimizer, x, y, epochs=1000, batch_size=32, print_every=100
):
    """
    Train a neural network model on the provided data.

    Args:
        model: The neural network model to train
        criterion: Loss function
        optimizer: Optimization algorithm
        x: Input features
        y: Target values
        epochs: Number of training epochs
        batch_size: Size of mini-batches
        print_every: How often to print training progress

    Returns:
        model: The trained model
        losses: List of losses during training
    """
    losses = []
    num_samples = len(x)
    num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division

    for epoch in range(epochs):
        epoch_loss = 0.0

        # Shuffle data at the beginning of each epoch
        indices = np.random.permutation(num_samples)
        x_shuffled = x[indices]
        y_shuffled = y[indices]

        # Process mini-batches
        for i in range(num_batches):
            # Get mini-batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)

            batch_x = x_shuffled[start_idx:end_idx]
            batch_y = y_shuffled[start_idx:end_idx]

            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # Backward pass
            grad_output = criterion.backward()
            model.backward(grad_output)

            # Update weights
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss

        # Calculate average loss for the epoch
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        # Print progress
        if (epoch + 1) % print_every == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

    return model, losses


def xor_example():
    # Generate some simple data
    input_size = 2
    hidden_size = 4
    output_size = 1
    num_samples = 100

    # Create XOR-like data
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)

    y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    # Repeat data to create more samples
    x = np.tile(x, (num_samples // 4 + 1, 1))[:num_samples]
    y = np.tile(y, (num_samples // 4 + 1, 1))[:num_samples]

    # Add some noise to make it more realistic
    x += np.random.normal(0, 0.05, x.shape).astype(np.float32)

    # Create a simple model with named layers
    model = Sequential(
        [
            Linear(input_size, hidden_size, name="layer1"),
            ReLU(),
            Linear(hidden_size, hidden_size, name="layer2"),
            ReLU(),
            Linear(hidden_size, output_size, name="layer3"),
            Sigmoid(),
        ]
    )

    # Prepare loss function and optimizer
    criterion = MSELoss()
    optimizer = SGD(model, learning_rate=0.1, momentum=0.9)

    # Train the model
    print("Starting training...")
    trained_model, losses = train_model(model, criterion, optimizer, x, y, epochs=1000)

    print("\nFinal predictions:")
    test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)

    predictions = model(test_inputs)
    print("Input => Output (Expected)")
    for i in range(4):
        print(f"{test_inputs[i].tolist()} => {predictions[i][0]:.4f} ({y[i%4][0]})")

    print("\nTraining complete!")
    print(f"Final loss: {losses[-1]}")
