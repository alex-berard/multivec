#pragma once
#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <thread>
#include <assert.h>
#include <iomanip> // setprecision, setw, left
#include <chrono>
#include <iterator>
#include <random>
#include <climits>
#include <mutex>
#include "vec.hpp"

using namespace std;
using namespace std::chrono;

const float MAX_EXP = 6;
const int UNIGRAM_TABLE_SIZE = 1e8; // size of the frequency table
const int EXP_TABLE_SIZE = 1000;

typedef Vec vec;
typedef vector<vec> mat;

inline float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

inline float cosine_similarity(const vec &v1, const vec &v2) {
    return v1.dot(v2) / (v1.norm() * v2.norm());
}

inline string lower(string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

inline vector<string> split(const string& sequence) {
    vector<string> words;
    istringstream iss(sequence);
    string word;

    while (iss >> word) {
        words.push_back(word);
    }

    return words;
}

inline void check(bool predicate, const string& message) {
    if (not predicate) {
        throw runtime_error(message);
    }
}

inline void check_is_open(ifstream& infile, const string& filename) {
    check(infile.is_open(), "couldn't open file " + filename);
}

inline void check_is_open(ofstream& outfile, const string& filename) {
    check(outfile.is_open(), "couldn't open file " + filename);
}

inline void check_is_non_empty(ifstream& infile, const string& filename) {
    check(infile.peek() != std::ifstream::traits_type::eof(),
          "training file " + filename + " is empty");
}

namespace multivec {
    /**
     * @brief Custom random generator. std::rand is thread-safe but very slow with multiple threads.
     * https://en.wikipedia.org/wiki/Linear_congruential_generator
     *
     * @return next random number
     */
    inline unsigned long long rand(unsigned long long max) {
        thread_local std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<unsigned long long> dis(0, max - 1);
        return dis(gen);
    }

    inline float randf() {
        thread_local std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<float> dis(0, 1.0);
        return dis(gen);
    }
    
    extern std::mutex print_mutex;
}

/**
 * @brief Node of a Huffman binary tree, used for the hierarchical softmax algorithm.
 */
struct HuffmanNode {
    static const HuffmanNode UNK; // node for out-of-vocabulary words

    string word;

    vector<int> code; // Huffman code of this node: path from root to leaf (0 for left, 1 for right)
    vector<int> parents; // indices of the parent nodes

    HuffmanNode* left; // used for constructing the tree, useless afterwards
    HuffmanNode* right;

    int index;
    int count;

    bool is_leaf;
    bool is_unk;

    HuffmanNode() : index(-1), is_unk(true) {}

    HuffmanNode(int index, const string& word) :
            word(word), index(index), count(1), is_leaf(true), is_unk(false)
    {}

    HuffmanNode(int index, HuffmanNode* left, HuffmanNode* right) :
            left(left), right(right), index(index), count(left->count + right->count), is_leaf(false), is_unk(false)
    {}

    bool operator==(const HuffmanNode& node) const {
        return index == node.index;
    }

    bool operator!=(const HuffmanNode& node) const {
        return !(operator==(node));
    }

    static bool comp(const HuffmanNode* v1, const HuffmanNode* v2) {
        return (v1->count) > (v2->count);
    }
};

struct Config {
    float alpha;
    int dimension; // size of the embeddings
    int min_count; // minimum count of each word in the training file to be included in the vocabulary
    int iterations; // number of training epochs
    int window_size;
    int threads;
    float subsampling;
    bool verbose; // print additional information
    bool hierarchical_softmax;
    bool skip_gram; // set to true to use skip-gram model instead of CBOW
    int negative; // number of negative samples used for the negative sampling training algorithm
    bool sent_vector; // includes sentence vectors in the training
    bool no_average; // no context averaging in CBOW
    bool concat; // concat vectors in CBOW instead of summing them

    Config() :
        alpha(0.05),
        dimension(100),
        min_count(5),
        iterations(5),
        window_size(5),
        threads(4),
        subsampling(1e-03),
        verbose(false), // not serialized
        hierarchical_softmax(false),
        skip_gram(false),
        negative(5),
        sent_vector(false),
        no_average(false),
        concat(false)
        {}

    virtual void print() const {
        std::cout << std::boolalpha; // to print false/true instead of 0/1
        std::cout << "dimension:   " << dimension << std::endl;
        std::cout << "window size: " << window_size << std::endl;
        std::cout << "min count:   " << min_count << std::endl;
        std::cout << "alpha:       " << alpha << std::endl;
        std::cout << "iterations:  " << iterations << std::endl;
        std::cout << "threads:     " << threads << std::endl;
        std::cout << "subsampling: " << subsampling << std::endl;
        std::cout << "skip-gram:   " << skip_gram << std::endl;
        std::cout << "HS:          " << hierarchical_softmax << std::endl;
        std::cout << "negative:    " << negative << std::endl;
        std::cout << "sent vector: " << sent_vector << std::endl;
        std::cout << "no average:  " << no_average << std::endl;
        std::cout << "concat:      " << concat << std::endl;
    }
};

struct BilingualConfig : Config {
    float beta;
    BilingualConfig() : beta(1.0f) {}
    void print() const {
        Config::print();
        std::cout << "beta:        " << beta << std::endl;
    }
};
