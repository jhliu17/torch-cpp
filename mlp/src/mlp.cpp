#include "mlp.h"
#include "torch/torch.h"


MLPImpl::MLPImpl(int64_t input_size, int64_t hidden_size, int64_t output_size) 
    : fc1(input_size, hidden_size), fc2(hidden_size, output_size) {
    register_module("fc1", fc1);
    register_module("fc2", fc2);
}


torch::Tensor MLPImpl::forward(torch::Tensor input) {
    auto output = fc1->forward(input);
    output = torch::nn::functional::relu(output);
    output = fc2->forward(output);
    return output;
}