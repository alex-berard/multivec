#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <thread>
#include <assert.h>
#include <iomanip> // setprecision
#include <boost/serialization/serialization.hpp>
#include <chrono>
#include <random>

using namespace std;
using namespace std::chrono;

vector<string> split(const string& sentence);

const float MAX_EXP = 6;
const int UNIGRAM_TABLE_SIZE = 1e8; // size of the frequency table
//const int EXP_TABLE_SIZE = 1000;

/*
static vector<float> initExpTable() {
    auto exp_table = vector<float>(EXP_TABLE_SIZE);
    for (int i = 0; i < EXP_TABLE_SIZE; i++) {
        float x = exp((i / static_cast<float>(EXP_TABLE_SIZE) * 2 - 1) * MAX_EXP); // Precompute the exp() table
        exp_table[i] = x / (x + 1);  // Precompute f(x) = x / (x + 1)
    }
    return exp_table;
}
*/

inline float sigmoid(float x) {
    assert(x > -MAX_EXP && x < MAX_EXP);

    //static auto exp_table = initExpTable(); // C++ static magic
    //return exp_table[static_cast<int>((x + MAX_EXP) * EXP_TABLE_SIZE / MAX_EXP / 2)];
    return 1 / (1 + exp(-x));
}

inline string lower(string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

typedef vector<float> vec;
typedef vector<vec> mat;

struct HuffmanNode {
    static const HuffmanNode UNK; // node for out-of-vocabulary words

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

    static bool comp(const HuffmanNode* v1, const HuffmanNode* v2) {
        return (v1->count) > (v2->count);
    }
};

struct Config {
    float starting_alpha;
    int dimension;
    int min_count;
    int max_iterations;
    int window_size;
    int n_threads;
    float subsampling;
    bool verbose;
    bool hierarchical_softmax;
    bool skip_gram;
    int negative;

    Config() :
        starting_alpha(0.05),
        dimension(100),
        min_count(5),
        max_iterations(5),
        window_size(5),
        n_threads(4),
        subsampling(1e-03),
        verbose(false),
        hierarchical_softmax(false),
        skip_gram(false),
        negative(5)
        {}

    void print() const {
        std::cout << "Word2vec++" << endl;
        std::cout << "dimension:   " << dimension << std::endl;
        std::cout << "window size: " << window_size << std::endl;
        std::cout << "min count:   " << min_count << std::endl;
        std::cout << "alpha:       " << starting_alpha << std::endl;
        std::cout << "iterations:  " << max_iterations << std::endl;
        std::cout << "threads:     " << n_threads << std::endl;
        std::cout << "subsampling: " << subsampling << std::endl;
        std::cout << "skip-gram:   " << skip_gram << std::endl;
        std::cout << "HS:          " << hierarchical_softmax << std::endl;
        std::cout << "negative:    " << negative << std::endl;
    }
};

class MonolingualModel
{
    friend class BilingualModel;
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive& ar, const unsigned int version) {
        ar & config & syn0 & syn1 & train_words & vocabulary;
    }

private:
    mat syn0; // input weights, TODO: syn1neg + more explicit names
    mat syn1; // output weights
    long long train_words; // total number of words in training file (used to compute word frequencies)
    long long total_word_count;
    float alpha;
    Config config;
    map<string, HuffmanNode> vocabulary;
    vector<HuffmanNode*> unigram_table;
    static __thread unsigned long long next_random;

    static unsigned long long rand() {
    //static int rand() {
        //static thread_local std::mt19937 random_generator;
        //std::uniform_int_distribution<int> distribution(0, RAND_MAX);
        //return distribution(random_generator);

        next_random = next_random * static_cast<unsigned long long>(25214903917) + 11;
        return next_random;
    }
    static float randf() {
        //return  MonolingualModel::rand() / static_cast<float>(RAND_MAX);
        return  (MonolingualModel::rand() & 0xFFFF) / 65536.0f;
    }

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
    
    bool wordInVocab(const string& word) const; // Test whether the word is in the vocabulary

    void train(const string& training_file); // training from scratch (resets vocabulary and weights)

    void saveEmbeddings(const string& filename) const; // saves the word embeddings in the word2vec binary format

    void load(const string& filename); // loads the entire model
    void save(const string& filename) const; // saves the entire model
    void loadW2V(const string& filename); // loads the entire model
    float getSimilarity(const string& word1, const string& word2);
    float getDistance(const string& word1, const string& word2);

    void computeAccuracy(istream& infile, int max_vocabulary_size = 0) const;
};
