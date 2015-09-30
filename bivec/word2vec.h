#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <thread>
#include <assert.h>

using namespace std;

vector<string> split(const string& sentence);

const float MAX_EXP = 6;
const int UNIGRAM_TABLE_SIZE = 1e8; // approximate size of the frequency table

inline float sigmoid(float x) {
    assert(x > -MAX_EXP && x < MAX_EXP); // we don't want NaN values hanging around
    return 1 / (1 + exp(-x));
}

typedef vector<float> vec;
typedef vector<vec> mat;

struct HuffmanNode {
    static const HuffmanNode UNK; // node for OOV words, initialized in bivec.cpp

    string word;

    vector<int> code;
    vector<int> parents;

    HuffmanNode* left;
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
};

struct Config {
    float starting_alpha;
    int dimension;
    int min_count;
    int max_iterations;
    int window_size;
    int n_threads;
    float sampling;
    bool debug;
    bool negative_sampling; // default is hierarchical softmax
    bool skip_gram; // default is cbow
    int negative;

    Config() : starting_alpha(0.05), dimension(100), min_count(5), max_iterations(5), window_size(5),
        n_threads(4), sampling(1e-03), debug(true), negative_sampling(false), skip_gram(false), negative(5) {}

    void print() const {
        std::cout << "dimension:   " << dimension << std::endl;
        std::cout << "window size: " << window_size << std::endl;
        std::cout << "min count:   " << min_count << std::endl;
        std::cout << "alpha:       " << starting_alpha << std::endl;
        std::cout << "iterations:  " << max_iterations << std::endl;
        std::cout << "threads:     " << n_threads << std::endl;
        std::cout << "subsampling: " << sampling << std::endl;
        std::cout << "CBOW:        " << !skip_gram << std::endl;
        std::cout << "HS:          " << !negative_sampling << std::endl;
        std::cout << "negative:    " << negative << std::endl;
    }
};

class MonolingualModel
{
    friend class BilingualModel;

private:
    mat syn0; // input weights
    mat syn1; // output weights
    long long train_words; // total number of words in training file (used to compute word frequencies)
    long long total_word_count;
    float alpha;
    Config config;
    unordered_map<string, HuffmanNode> vocabulary;
    vector<HuffmanNode*> unigram_table;

    void addWordToVocab(const string& word);
    void reduceVocab();
    void createBinaryTree();
    void assignCodes(HuffmanNode* node, vector<int> code, vector<int> parents) const;
    void initUnigramTable();

    HuffmanNode* getRandomHuffmanNode(); // uses the unigram frequency table to sample a random node

    vector<HuffmanNode> getNodes(const string& sentence) const;
    void subsample(vector<HuffmanNode>& node) const;

    void readVocab(const string& training_file);
    void initNet();

    void trainChunk(const string& training_file, const vector<long long>& chunks, int chunk_id);
    int trainSentence(const string& sent);

    void trainWord(const vector<HuffmanNode>& nodes, int word_pos);
    void trainWordCBOW(const vector<HuffmanNode>& nodes, int word_pos);
    void trainWordSkipGram(const vector<HuffmanNode>& nodes, int word_pos);

    vec hierarchicalUpdate(const HuffmanNode& node, const vec& hidden, float alpha, bool update = true);
    vec negSamplingUpdate(const HuffmanNode& node, const vec& hidden, float alpha, bool update = true);

    HuffmanNode loadWord(ifstream& infile);
    void saveWord(ofstream& outfile, const HuffmanNode& node) const;

    vector<long long> static chunkify(const string& filename, int n_chunks);

public:
    MonolingualModel() : train_words(0), total_word_count(0) {} // model with default configuration
    MonolingualModel(Config config) : train_words(0), total_word_count(0), config(config) {}

    vec wordVec(const string& word) const; // word embedding
    vec sentVec(const string& sentence); // paragraph vector (Le & Mikolov)

    void train(const string& training_file); // training from scratch (resets vocabulary and weights)

    void saveEmbeddings(const string& filename) const; // save the word embeddings in a word2vec compatible format
    void load(const string& filename); // loads the entire model, but not the configuration (except dimension parameter)
    void save(const string& filename) const; // save the entire model, but not the configuration
};
