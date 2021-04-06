#include <fstream>
#include <sstream>
#include <vector>
#include <exception>

#include "corpus.h"

namespace data_utils {
    torch::Tensor Corpus::get_batch_data(int64_t batch_size) {
        std::ifstream file(_path);

        if (file) {
            std::string line, word;
            std::vector<int64_t> index;

            while (std::getline(file, line)) {
                std::istringstream ss(line);
                while (ss >> word) {
                    index.push_back(_dict.add_word(word));
                }
                index.push_back(_dict.add_word(EOS));
            }

            int64_t batch_num = index.size() / batch_size;
            return torch::from_blob(
                index.data(),
                {batch_size, batch_num},
                torch::TensorOptions().dtype(torch::kInt64)
            ).clone();
        } else {
            throw std::runtime_error("Could not read file at path: " + _path);
        }
    }
}