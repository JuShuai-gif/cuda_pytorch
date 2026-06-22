import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from nn.layers.activations import Flatten, ReLU
from nn.layers.conv import Conv2D
from nn.layers.linear import Linear
from nn.layers.pooling import MaxPooling2D
from nn.loss import CrossEntropyLoss
from nn.model import Sequential
from nn.optim import Adam


# MNIST CNN Example function
def mnist_cnn_example():
    print("Loading MNIST data using torchvision...")

    # Define a transform to normalize the data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert images to tensor and scales to [0,1]
        ]
    )

    # Download and load the MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Create dataloaders (we'll use them for batching)
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    print(
        f"Loaded {len(train_dataset)} training samples and {len(test_dataset)} test samples"
    )

    # Define the CNN architecture for MNIST
    model = Sequential(
        [
            # First convolutional layer
            Conv2D(
                in_channels=1, out_channels=16, kernel_size=3, padding=1, name="conv1"
            ),
            ReLU(),
            MaxPooling2D(kernel_size=2, stride=2, name="pool1"),  # Output: 16x14x14
            # Second convolutional layer
            Conv2D(
                in_channels=16, out_channels=32, kernel_size=3, padding=1, name="conv2"
            ),
            ReLU(),
            MaxPooling2D(kernel_size=2, stride=2, name="pool2"),  # Output: 32x7x7
            # Flatten layer
            Flatten(),
            # Fully connected layers
            Linear(32 * 7 * 7, 128, name="fc1"),
            ReLU(),
            Linear(128, 10, name="fc2"),  # 10 output classes for digits 0-9
        ]
    )

    # Create loss function and optimizer
    criterion = CrossEntropyLoss()
    optimizer = Adam(model, learning_rate=0.001)

    # Training settings
    num_epochs = 3
    num_batches = len(train_loader)
    print_every = num_batches // 5  # Print 5 times per epoch

    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        # Process mini-batches
        for i, (images, labels) in enumerate(train_loader):
            # Convert PyTorch tensors to numpy arrays (NCHW format)
            batch_images = images.numpy()
            batch_labels = labels.numpy()

            # Forward pass
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)

            # Backward pass
            grad_output = criterion.backward()
            model.backward(grad_output)

            # Update weights
            optimizer.step()

            # Calculate accuracy
            predictions = np.argmax(outputs, axis=1)
            correct += (predictions == batch_labels).sum()
            total += batch_labels.size
            total_loss += loss

            # Print status
            if (i + 1) % print_every == 0:
                avg_loss = total_loss / (i + 1)
                accuracy = correct / total * 100
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{num_batches}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
                )

        # Print epoch results
        avg_loss = total_loss / num_batches
        accuracy = correct / total * 100
        print(
            f"Epoch [{epoch+1}/{num_epochs}] completed, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )

    # Test the model
    print("\nTesting the model...")
    correct = 0
    total = 0
    for images, labels in test_loader:
        # Convert to numpy arrays
        test_images = images.numpy()
        test_labels = labels.numpy()

        # Forward pass
        outputs = model(test_images)

        # Calculate accuracy
        predictions = np.argmax(outputs, axis=1)
        correct += (predictions == test_labels).sum()
        total += test_labels.size

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

    # Display some example predictions
    print("\nShowing some example predictions...")

    # Get a batch of test samples
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    # Convert to numpy and get predictions
    test_images = images.numpy()
    test_labels = labels.numpy()
    outputs = model(test_images)
    predictions = np.argmax(outputs, axis=1)

    # Plot a few examples
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()

    for i in range(10):
        # Plot the image
        axes[i].imshow(test_images[i, 0], cmap="gray")
        axes[i].set_title(f"Pred: {predictions[i]}, True: {test_labels[i]}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("mnist_predictions.png")
    plt.show()

    return model
