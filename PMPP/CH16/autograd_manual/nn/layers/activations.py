import numpy as np
from nn.layers.base import Layer


# Flatten Layer
class Flatten(Layer):
    def __init__(self):
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)

    def parameters(self):
        return []

    def get_gradients(self):
        return []


class Sigmoid(Layer):
    def __init__(self):
        self.output = None

    def forward(self, x):
        # Compute sigmoid with numerical stability
        self.output = np.zeros_like(x)
        mask = x >= 0
        self.output[mask] = 1 / (1 + np.exp(-x[mask]))
        exp_x = np.exp(x[~mask])
        self.output[~mask] = exp_x / (1 + exp_x)
        return self.output

    def backward(self, grad_output):
        # Compute gradient: dL/dx = dL/dy * dy/dx
        # For sigmoid: dy/dx = y * (1 - y)
        grad_input = grad_output * self.output * (1 - self.output)
        return grad_input

    def parameters(self):
        return []

    def get_gradients(self):
        return []


# ReLU Layer
class ReLU(Layer):
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        # dL/dx = dL/doutput * d(output)/dx
        # For ReLU: d(ReLU(x))/dx = 1 if x > 0, 0 if x <= 0
        # So: dL/dx = dL/doutput when x > 0, and 0 when x <= 0
        grad_input = grad_output.copy()
        grad_input[self.input < 0] = 0
        return grad_input

    def parameters(self):
        return []

    def get_gradients(self):
        return []
