import torch
from torch.autograd import Function


class LinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            # dL/dx = dL/doutput * d(output)/dx = dL/doutput * W
            grad_input = grad_output.matmul(weight)

        if ctx.needs_input_grad[1]:
            # dL/dW = dL/doutput * d(output)/dW = X^T * dL/doutput
            grad_weight = grad_output.t().matmul(input)

        if bias is not None and ctx.needs_input_grad[2]:
            # dL/db = sum(dL/doutput)
            # Sum over batch dimension since bias is added to each sample
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


class SigmoidFunction(Function):
    @staticmethod
    def forward(ctx, input):
        output = 1.0 / (1.0 + torch.exp(-input))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (output,) = ctx.saved_tensors

        # dL/dx = dL/doutput * d(output)/dx
        # For sigmoid: d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
        # So: dL/dx = dL/doutput * sigmoid(x) * (1 - sigmoid(x))
        grad_input = output * (1 - output) * grad_output

        return grad_input


class Linear:
    def __init__(self, in_features, out_features):
        self.weight = torch.randn(out_features, in_features, requires_grad=True)
        self.bias = torch.zeros(out_features, requires_grad=True)
        self.parameters = [self.weight, self.bias]

    def __call__(self, x):
        return LinearFunction.apply(x, self.weight, self.bias)


class Sigmoid:
    def __call__(self, x):
        return SigmoidFunction.apply(x)


class MSELoss:
    def __call__(self, pred, target):
        return ((pred - target) ** 2).mean()


class SimpleNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.linear1 = Linear(input_size, hidden_size)
        self.sigmoid = Sigmoid()
        self.linear2 = Linear(hidden_size, output_size)

        # Collect all parameters for the optimizer
        self.parameters = self.linear1.parameters + self.linear2.parameters

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        return x

    def __call__(self, x):
        return self.forward(x)


class SGD:
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()

    def step(self):
        for param in self.parameters:
            if param.grad is not None:
                param.data -= self.learning_rate * param.grad


def train_model(model, criterion, optimizer, x_data, y_data, epochs=100):
    for epoch in range(epochs):
        # Forward pass
        outputs = model(x_data)
        loss = criterion(outputs, y_data)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model


def main():
    input_size = 10
    hidden_size = 20
    output_size = 2
    num_samples = 100

    x = torch.randn(num_samples, input_size)
    true_w = torch.randn(input_size)
    y = torch.sigmoid(x.matmul(true_w)).unsqueeze(1)

    # Initialize the model, loss, and optimizer
    model = SimpleNetwork(input_size, hidden_size, output_size)
    criterion = MSELoss()
    optimizer = SGD(model.parameters, learning_rate=0.01)

    _ = train_model(model, criterion, optimizer, x, y, epochs=100)

    print("Training complete!")


if __name__ == "__main__":
    main()
