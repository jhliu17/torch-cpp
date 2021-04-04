#pragma once

#include "torch/torch.h"

class MLPImpl: public torch::nn::Module {
    public:
        MLPImpl(int64_t input_size, int64_t hidden_size, int64_t output_size);
        torch::Tensor forward(torch::Tensor input);

    private:
        torch::nn::Linear fc1;
        torch::nn::Linear fc2;
};

TORCH_MODULE(MLP);
