#include <torch/torch.h>
#include <iostream>
// Define the neural network structure
struct Net : torch::nn::Module {
    Net() {
        // Input layer to hidden layer
        fc1 = register_module("fc1", torch::nn::Linear(784, 128));
        // Hidden layer to output layer
        fc2 = register_module("fc2", torch::nn::Linear(128, 10));
    }
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
        x = fc2->forward(x);
        return torch::log_softmax(x, /*dim=*/1);
    }
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

int main() {
    // Create an instance of the network
    auto net = std::make_shared<Net>();
    // Create a dummy input tensor
    auto input = torch::randn({64, 1, 28, 28});
    // Forward pass
    auto output = net->forward(input);
    std::cout << "Output size: " << output.sizes() << std::endl;
    return 0;
}