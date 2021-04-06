#pragma once

#include <string>
#include <vector>
#include <unordered_map>


namespace data_utils {
    const std::string UNK = "<unk>";
    const std::string BOS = "<bos>";
    const std::string EOS = "<eos>";
    const std::string PAD = "<pad>";

    class Dictionary {
        public:
            size_t size() const {return _id2word.size();};
            bool exist_word(const std::string& word) const;
            size_t word2id(const std::string& word) const;
            std::string id2word(size_t id) const;
            size_t add_word(const std::string& word);
        
        private:
            std::vector<std::string> _id2word = {UNK, BOS, EOS, PAD};
            std::unordered_map<std::string, size_t> _word2id  = {
                {UNK, 0},
                {BOS, 1},
                {EOS, 2},
                {PAD, 3},
            };
    };
}