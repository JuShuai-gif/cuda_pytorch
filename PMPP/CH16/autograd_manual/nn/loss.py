import numpy as np


# Softmax function for predictions
def softmax(x):
    # Subtract max for numerical stability
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class CrossEntropyLoss:
    def __init__(self):
        self.prediction = None
        self.target = None
        self.softmax_output = None

    def forward(self, pred, target):
        batch_size = pred.shape[0]
        self.prediction = pred
        self.target = target

        # Calculate softmax for numerical stability
        pred_shifted = pred - np.max(pred, axis=1, keepdims=True)
        exp_pred = np.exp(pred_shifted)
        self.softmax_output = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)

        # Cross entropy loss
        loss = 0.0
        for i in range(batch_size):
            loss -= np.log(self.softmax_output[i, target[i]] + 1e-10)

        return loss / batch_size

    def backward(self):
        batch_size = self.prediction.shape[0]

        # Initialize gradient with softmax values
        grad = self.softmax_output.copy()

        # Subtract 1 from the correct class
        for i in range(batch_size):
            grad[i, self.target[i]] -= 1

        # Normalize by batch size
        grad /= batch_size

        return grad

    def __call__(self, pred, target):
        return self.forward(pred, target)


class MSELoss:
    def __init__(self):
        self.prediction = None
        self.target = None

    def forward(self, pred, target):
        self.prediction = pred
        self.target = target
        return np.mean((pred - target) ** 2)

    def backward(self):
        batch_size = self.prediction.shape[0]
        return 2.0 * (self.prediction - self.target) / batch_size

    def __call__(self, pred, target):
        return self.forward(pred, target)
