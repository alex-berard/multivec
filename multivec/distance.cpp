#include "monolingual.hpp"

/**
 * @brief Compute cosine similarity between word1 and word2. Throw runtime_error if 
 * those words are unknown. For the score to be in [0,1], the weights need to be normalized beforehand.
 */
float MonolingualModel::similarity(const string& word1, const string& word2, int policy) const {
    if (word1 == word2) {
        return 1.0;
    } else {
        auto it = vocabulary.find(word1);
        if (it == vocabulary.end()) {
          return 0.0;
        }
        it = vocabulary.find(word2);
        if (it == vocabulary.end()) {
          return 0.0;
        }
        vec v1 = wordVec(word1, policy);
        vec v2 = wordVec(word2, policy);
        return v1.dot(v2) / (v1.norm() * v2.norm());
    }
}

float MonolingualModel::distance(const string& word1, const string& word2, int policy) const {
    return 1 - similarity(word1, word2, policy);
}

float MonolingualModel::similarityNgrams(const string& seq1, const string& seq2, int policy) const {
    auto words1 = split(seq1);
    auto words2 = split(seq2);

    if (words2.size() != words2.size()) {
        throw runtime_error("input sequences don't have the same size");
    }
    
    float res = 0;
    int n = 0;
    for (size_t i = 0; i < words1.size(); ++i) {
        try {
            res += similarity(words1[i], words2[i], policy);
            n += 1;
        }
        catch (runtime_error) {}
    }
    
    if (n == 0) {
        throw runtime_error("N-gram size of zero !");
    } else {
        return res / n;
    }
}

float MonolingualModel::similaritySentence(const string& seq1, const string& seq2, int policy) const {
    auto words1 = split(seq1);
    auto words2 = split(seq2);

    float res = 0;
    int n = 0;
    auto l_word1 = wordVec(words1[0], policy);
    auto l_word2 = wordVec(words2[0], policy);
    auto sentence1 = l_word1;
    auto sentence2 = l_word2;
    for (size_t i = 1; i < words1.size(); ++i) {
        try {
            for (size_t j = 0; j < sentence1.size(); ++j) {
                l_word1 = wordVec(words1[i], policy);
                sentence1[j]+=l_word1[j];
            }
        }
        catch (runtime_error) {}
    }
    for (size_t i = 1; i < words2.size(); ++i) {
        try {
            for (size_t j = 0; j < sentence2.size(); ++j) {
                l_word2 = wordVec(words2[i], policy);
                sentence2[j]+=words2[i][j];
            }
        }
        catch (runtime_error) {}
    }
    for (size_t i = 0; i < sentence1.size(); ++i) {
        if (sentence1[i] != 0.0 && sentence2[i] !=0.0) {
          n = 1;
          break;
        }
    }
    if (n == 0) { 
      return 0.0;
    }
    return sentence1.dot(sentence2) / (sentence1.norm() * sentence2.norm());
}

void normalizeWeights(mat& weights) {
    if (weights.empty()) {
        return;
    }
    
    int dim = weights[0].size();
    
    vec min_values = weights[0];
    vec max_values = weights[0];
    for (size_t i = 1; i < weights.size(); ++i) {
        for (size_t j = 0; j < dim; ++j) {
            min_values[j] = min(min_values[j], weights[i][j]);
            max_values[j] = max(max_values[j], weights[i][j]);
        }
    }
    
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < dim; ++j) {
            if (max_values[j] != min_values[j]) {
                weights[i][j] = (weights[i][j] - min_values[j]) / (max_values[j] - min_values[j]);
            }
        }
    }
}

void MonolingualModel::normalizeWeights() {
    ::normalizeWeights(input_weights);
    ::normalizeWeights(output_weights);
    ::normalizeWeights(output_weights_hs);
    ::normalizeWeights(sent_weights);
}