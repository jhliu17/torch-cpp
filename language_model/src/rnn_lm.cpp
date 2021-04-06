#include  "rnn_lm.h"


RNNLMImpl::RNNLMImpl(int64_t vocab_size, int64_t embed_size, int64_t hidden_size)
    : embed_layer(vocab_size, embed_size), 
      lstm(torch::nn::LSTMOptions(embed_size, hidden_size).num_layers(1).batch_first(true)),
      linear(hidden_size, vocab_size) {
    register_module("embed_layer", embed_layer);
    register_module("lstm", lstm);
    register_module("linear", linear);
}

std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> RNNLMImpl::forward(
            torch::Tensor input, 
            std::tuple<torch::Tensor, torch::Tensor> hidden) {
    
    torch::Tensor output;
    std::tuple<torch::Tensor, torch::Tensor> next_hidden;

    auto embed = embed_layer->forward(input);
    std::tie(output, next_hidden) = lstm->forward(embed, hidden);

    output = linear->forward(output);
    output = torch::nn::functional::log_softmax(output, torch::nn::functional::LogSoftmaxFuncOptions(-1));
    return std::make_tuple(output, next_hidden);
}