#pragma once

#include <tuple>
#include "torch/torch.h"


class RNNLMImpl: public torch::nn::Module {
    public:
        RNNLMImpl(int64_t vocab_size, int64_t embed_size, int64_t hidden_size);
        std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> forward(
            torch::Tensor input, 
            std::tuple<torch::Tensor, torch::Tensor> hidden
        );

    private:
        torch::nn::Embedding embed_layer;
        torch::nn::LSTM lstm;
        torch::nn::Linear linear;
};

TORCH_MODULE(RNNLM);
