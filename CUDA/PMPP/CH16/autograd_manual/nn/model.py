# Sequential Model
class Sequential:
    def __init__(self, layers=None):
        self.layers = layers if layers is not None else []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def get_gradients(self):
        grads = []
        for layer in self.layers:
            if hasattr(layer, "get_gradients"):
                grads.extend(layer.get_gradients())
        return grads
