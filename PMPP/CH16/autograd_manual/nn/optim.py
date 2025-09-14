import numpy as np


class Adam:
    def __init__(
        self, model, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

        # Initialize moments
        self.m = {}
        self.v = {}
        for param, name in model.parameters():
            self.m[name] = np.zeros_like(param)
            self.v[name] = np.zeros_like(param)

    def step(self):
        self.t += 1

        # Get current gradients and parameters
        gradients = {name: grad for grad, name in self.model.get_gradients()}

        # Update parameters with their gradients
        for param, param_name in self.model.parameters():
            if param_name in gradients and gradients[param_name] is not None:
                # Update biased moment estimates
                self.m[param_name] = (
                    self.beta1 * self.m[param_name]
                    + (1 - self.beta1) * gradients[param_name]
                )
                self.v[param_name] = self.beta2 * self.v[param_name] + (
                    1 - self.beta2
                ) * (gradients[param_name] ** 2)

                # Compute bias-corrected moment estimates
                m_corrected = self.m[param_name] / (1 - self.beta1**self.t)
                v_corrected = self.v[param_name] / (1 - self.beta2**self.t)

                # Update parameter
                param -= (
                    self.learning_rate
                    * m_corrected
                    / (np.sqrt(v_corrected) + self.epsilon)
                )


class SGD:
    def __init__(self, model, learning_rate=0.01, momentum=0.9):
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}

        # Initialize velocities for each parameter
        for param, name in model.parameters():
            self.velocities[name] = np.zeros_like(param)

    def step(self):
        # Get current gradients and parameters
        gradients = {name: grad for grad, name in self.model.get_gradients()}

        # Update parameters with their gradients
        for param, param_name in self.model.parameters():
            if param_name in gradients and gradients[param_name] is not None:
                # Update velocity with momentum
                self.velocities[param_name] = (
                    self.momentum * self.velocities[param_name]
                    + self.learning_rate * gradients[param_name]
                )

                # Update parameter
                param -= self.velocities[param_name]
