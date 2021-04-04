#include <iostream>
#include <iomanip>
#include <string>

#include "torch/torch.h"
#include "mlp.h"

int main() {
    std::cout << "Multi-layer Perceptron." << std::endl;

    // Device
    bool cuda_is_avaliable = torch::cuda::is_available();
    torch::Device device(cuda_is_avaliable? torch::kCUDA: torch::kCPU);
    std::cout << "Training on " << (cuda_is_avaliable? "GPU": "CPU") << std::endl;

    // Hyper Parameters
    const size_t batch_size = 16;
    const size_t num_class = 10;
    const size_t input_size = 784;
    const size_t hidden_size = 64;
    const size_t total_epoch = 20;
    const double learning_rate = 0.0001;

    const std::string data_path = "../../dataset/mnist";

    // Build Dataset
    auto train_dataset = torch::data::datasets::MNIST(data_path)
    .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
    .map(torch::data::transforms::Stack<>());
    auto train_size = train_dataset.size().value();

    auto test_dataset = torch::data::datasets::MNIST(data_path, torch::data::datasets::MNIST::Mode::kTest)
    .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
    .map(torch::data::transforms::Stack<>());
    auto test_size = test_dataset.size().value();

    // Make Dataloader
    auto train_dataloader = torch::data::make_data_loader(
        std::move(train_dataset),
        batch_size
    );
    auto test_dataloader = torch::data::make_data_loader(
        std::move(test_dataset),
        batch_size
    );

    // Make Model
    MLP model(input_size, hidden_size, num_class);
    model->to(device);

    // Make Optimizer
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate));

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    // Train model
    std::cout << "Training...\n";
    for (size_t epoch = 0; epoch < total_epoch; ++epoch) {
        // Initialize running metrics
        double running_loss = 0.0;
        size_t num_correct = 0;

        // Batch training
        for (auto& batch: *train_dataloader) {
            auto input = batch.data.view({batch_size, -1}).to(device);
            auto target = batch.target.to(device);

            auto output = model->forward(input);
            auto loss = torch::nn::functional::cross_entropy(output, target);

            running_loss += loss.item<double>();
            auto predict = output.argmax(-1);
            num_correct += (predict == target).sum().item<int64_t>();

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }

        auto sample_mean_loss = running_loss / train_size;
        auto accuracy = static_cast<double>(num_correct) / train_size;

        std::cout << "Epoch [" << (epoch + 1) << "/" << total_epoch << "], Trainset - Loss: "
            << sample_mean_loss << ", Accuracy: " << accuracy << '\n';
    }
    std::cout << "Training finished!\n\n";


    // Test the model
    model->eval();
    torch::NoGradGuard no_grad;

    double running_loss = 0.0;
    size_t num_correct = 0;

    for (const auto& batch : *test_dataloader) {
        auto data = batch.data.view({batch_size, -1}).to(device);
        auto target = batch.target.to(device);

        auto output = model->forward(data);

        auto loss = torch::nn::functional::cross_entropy(output, target);

        running_loss += loss.item<double>();

        auto prediction = output.argmax(1);

        num_correct += prediction.eq(target).sum().item<int64_t>();
    }

    std::cout << "Testing finished!\n";

    auto test_accuracy = static_cast<double>(num_correct) / test_size;
    auto test_sample_mean_loss = running_loss / test_size;

    std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';

    return 0;
}