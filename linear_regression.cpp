#include <iostream>
#include <iomanip>
#include "torch/torch.h"


int main() {
    std:: cout << "Linear Regression Example." << std::endl;

    // hyperparameters
    const int input_size = 1;
    const int output_size = 1;
    const int training_epoch = 200;
    const double learning_rate = 0.001;

    // model and optimizer
    torch::nn::Linear model(input_size, output_size);
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate));

    // toy data
    torch::Tensor x = torch::randint(0, 10, {15, 1});
    torch::Tensor y = x * 2 + 4;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Start Training..." << std::endl;

    // Train the model
    for (size_t epoch = 0; epoch != training_epoch; ++epoch) {
        // Forward pass
        auto output = model(x);
        auto loss = torch::nn::functional::mse_loss(output, y);

        // Backward pass and optimize
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if ((epoch + 1) % 5 == 0) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << training_epoch <<
                "], Loss: " << loss.item<double>() << "\n";
        }
    }

    std::cout << "Training finished!\n";
}