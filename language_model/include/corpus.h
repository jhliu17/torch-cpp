#pragma once

#include <string>
#include "torch/torch.h"
#include "dictionary.h"

namespace data_utils {
    class Corpus {
        public:
            explicit Corpus(const std::string& path): _path(path) {};
            const Dictionary& get_dict() const {return _dict;};
            torch::Tensor get_batch_data(int64_t batch_size);
        
        private:
            std::string _path;
            Dictionary _dict;
    };
}