#include <iostream>
#include <iomanip>
#include <string>
#include <tuple>
#include <fstream>

#include "torch/torch.h"
#include "corpus.h"
#include "rnn_lm.h"
#include "tqdm.h"

using data_utils::Corpus;
using data_utils::EOS;
using torch::indexing::Slice;
using torch::indexing::None;

#define DEBUG false


int main() {
    std::cout << "RNN Language Model." << std::endl;

    // Device
    bool cuda_is_avaliable = torch::cuda::is_available();
    torch::Device device(cuda_is_avaliable? torch::kCUDA: torch::kCPU);
    std::cout << "Training on " << (cuda_is_avaliable? "GPU": "CPU") << std::endl;

    // Hyper Parameters
    const size_t batch_size = 64;
    const size_t embed_size = 128;
    const size_t hidden_size = 256;
    const size_t total_epoch = 10;
    const size_t sequence_length = 64;
    const size_t generate_number = 100;

    const double learning_rate = 0.001;
    const double clip_gradient = 5;

    const std::string corpus_path = "../../dataset/penntreebank/ptb.train.txt";
    const std::string output_path = "./language_model_sample.txt";

    // Build Corpus
    Corpus corpus(corpus_path);
    torch::Tensor index_data = corpus.get_batch_data(batch_size);
    if (DEBUG) index_data = index_data.index({Slice(), Slice(None, 60)});
    int64_t vocab_size = static_cast<int64_t>(corpus.get_dict().size());

    // Make Model
    RNNLM model(vocab_size, embed_size, hidden_size);
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
        double running_perplexity = 0.0;
        size_t num_batch = 0;

        // Intitial hidden state
        torch::Tensor h = torch::zeros({1, batch_size, hidden_size}, torch::TensorOptions().requires_grad(false)).to(device);
        torch::Tensor c = torch::zeros({1, batch_size, hidden_size}, torch::TensorOptions().requires_grad(false)).to(device);
        torch::Tensor output;
        
        // Make progress bar
        tqdm bar;
        bar.disable_colors();

        // Batch training
        int64_t batch_iter_num = index_data.size(-1) - sequence_length;
        for (int64_t i = 0; i < batch_iter_num; i += sequence_length) {
            auto input = index_data.index({Slice(), Slice(i, i + sequence_length)}).to(device);
            auto target = index_data.index({Slice(), Slice(i + 1, i + 1 + sequence_length)}).to(device);

            std::forward_as_tuple(output, std::tie(h, c)) = model->forward(input, std::make_tuple(h, c));
            h.detach_();
            c.detach_();
            auto loss = torch::nn::functional::nll_loss(
                output.reshape({-1, vocab_size}), 
                target.reshape(-1)
            );

            // update metrics
            running_loss += loss.item<double>();
            running_perplexity += torch::exp(loss).item<double>();
            ++num_batch;

            optimizer.zero_grad();
            loss.backward();
            torch::nn::utils::clip_grad_value_(model->parameters(), clip_gradient);
            optimizer.step();

            // tqdm
            bar.progress(i, batch_iter_num);
        }
        bar.finish();

        auto epoch_running_loss = running_loss / num_batch;
        auto epoch_running_perplexity = running_perplexity / num_batch;
        std::cout << "Epoch [" << (epoch + 1) << "/" << total_epoch << "], Trainset - Loss: "
            << epoch_running_loss << ", Perplexity: " << epoch_running_perplexity << '\n';
    }
    std::cout << "Training finished!\n\n";


    // Test the model
    model->eval();
    torch::NoGradGuard no_grad;
    std::ofstream sample_file(output_path);

    // Generation
    std::cout << "Generating samples...\n";
    for (size_t i = 0; i < generate_number; ++i) {
        // Initialize input 
        torch::Tensor output;
        torch::Tensor propability = torch::ones(vocab_size);
        torch::Tensor input_token = torch::multinomial(propability, 1).unsqueeze(0).to(device);
        torch::Tensor h = torch::zeros({1, 1, hidden_size}, torch::TensorOptions().requires_grad(false)).to(device);
        torch::Tensor c = torch::zeros({1, 1, hidden_size}, torch::TensorOptions().requires_grad(false)).to(device);
    
        int generate_length = 1;
        bool generate_finish = false;
        sample_file << corpus.get_dict().id2word(input_token.item<int64_t>()) << " ";

        while (!generate_finish) {
            // sample token
            std::forward_as_tuple(output, std::tie(h, c)) = model->forward(input_token, std::make_tuple(h, c));
            propability = output.exp().squeeze_(1);
            input_token = propability.multinomial(1);

            // output word
            auto output_word = corpus.get_dict().id2word(input_token.item<int64_t>());

            // at the end?
            if (generate_length > sequence_length || output_word == EOS) generate_finish = true;

            // output to file
            std::string word = generate_finish? ".\n": output_word + " ";
            sample_file << word;

            ++generate_length;
        }
    }

    std::cout << "Finished generating samples!\nSaved output to " << output_path << "\n";

    return 0;
}