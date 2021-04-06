#include "dictionary.h"


namespace data_utils {
    bool Dictionary::exist_word(const std::string& word) const {
        auto it = _word2id.find(word);
        return it != _word2id.end();
    }

    size_t Dictionary::word2id(const std::string& word) const {
        if (exist_word(word)) return _word2id.at(word);
        else return _word2id.at(UNK);
    }

    std::string Dictionary::id2word(size_t id) const {
        if (id < size()) return _id2word[id];
        else return UNK;
    }

    size_t Dictionary::add_word(const std::string& word) {
        if (exist_word(word)) return word2id(word);

        _id2word.push_back(word);
        size_t id = _id2word.size() - 1;
        _word2id.insert({word, id});
        return id;
    }
}