#include <torch/torch.h>
#include <iostream>
int main() {
    // Create a 3x3 tensor with random values between 0 and 1
    torch::Tensor X = torch::rand({3, 3});
    // Add a constant value to every element
    torch::Tensor Y = X + 5;
    // Apply ReLU activation
    torch::Tensor Z = torch::relu(Y);
    // Print all results
    std::cout << "Original tensor X:\n"
              << X << "\n\n";
    std::cout << "After adding 5 (Y):\n"
              << Y << "\n\n";
    std::cout << "After ReLU activation (Z):\n"
              << Z << std::endl;
    return 0;
}