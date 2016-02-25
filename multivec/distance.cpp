#include "monolingual.hpp"

/**
 * @brief Compute cosine similarity between word1 and word2.
 * For the score to be in [0,1], the weights need to be normalized beforehand.
 * Return 0 if word1 or word2 is unknown.
 */
float MonolingualModel::similarity(const string& word1, const string& word2, int policy) const {
    auto it1 = vocabulary.find(word1);
    auto it2 = vocabulary.find(word2);

    if (it1 == vocabulary.end() || it2 == vocabulary.end()) {
        return 0.0;
    } else if (it1->second.index == it2->second.index) {
        return 1.0;
    } else {
        vec v1 = wordVec(it1->second.index, policy);
        vec v2 = wordVec(it2->second.index, policy);
        return cosineSimilarity(v1, v2);
    }
}

float MonolingualModel::distance(const string& word1, const string& word2, int policy) const {
    return 1 - similarity(word1, word2, policy);
}


static bool comp(const pair<string, float>& p1, const pair<string, float>& p2) {
    return p1.second > p2.second;
}

/**
 * @brief Return an ordered list of the `n` closest words to `word` according to cosine similarity.
 */
vector<pair<string, float>> MonolingualModel::closest(const string& word, int n, int policy) const {
    vector<pair<string, float>> res;
    auto it = vocabulary.find(word);

    if (it == vocabulary.end()) {
        cerr << "OOV word" << endl;
        return res;
    }

    int index = it->second.index;
    vec v1 = wordVec(index, policy);

    for (auto it = vocabulary.begin(); it != vocabulary.end(); ++it) {
        if (it->second.index != index) {
            vec v2 = wordVec(it->second.index, policy);
            res.push_back({it->second.word, cosineSimilarity(v1, v2)});
        }
    }

    std::partial_sort(res.begin(), res.begin() + n, res.end(), comp);
    if (res.size() > n) res.resize(n);
    return res;
}

vector<pair<string, float>> MonolingualModel::closest(const vec& v, int n, int policy) const {
    vector<pair<string, float>> res;

    for (auto it = vocabulary.begin(); it != vocabulary.end(); ++it) {
        vec v2 = wordVec(it->second.index, policy);
        res.push_back({it->second.word, cosineSimilarity(v, v2)});
    }

    std::partial_sort(res.begin(), res.begin() + n, res.end(), comp);
    if (res.size() > n) res.resize(n);
    return res;
}

/**
 * @brief Return sorted list of `words` according to their similarity to `word`.
 */
vector<pair<string, float>> MonolingualModel::closest(const string& word, const vector<string>& words, int policy) const {
    vector<pair<string, float>> res;
    auto it = vocabulary.find(word);

    if (it == vocabulary.end()) {
        cerr << "OOV word" << endl;
        return res;
    }

    int index = it->second.index;
    vec v1 = wordVec(index, policy);

    for (auto it = words.begin(); it != words.end(); ++it) {
        auto node_it = vocabulary.find(*it);
        if (node_it != vocabulary.end()) {
            vec v2 = wordVec(node_it->second.index, policy);
            res.push_back({node_it->second.word, cosineSimilarity(v1, v2)});
        }
    }

    std::sort(res.begin(), res.end(), comp);
    return res;
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
        throw runtime_error("all word pairs are unknown (OOV)");
    } else {
        return res / n;
    }
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
