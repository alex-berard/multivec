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
#include <iterator>
//#include "mat.hpp"
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>


using namespace std;
using namespace std::chrono;

vector<string> split(const string& sentence);

const float MAX_EXP = 6;
const int UNIGRAM_TABLE_SIZE = 1e8; // size of the frequency table

typedef vector<float> vec;
typedef vector<vec> mat;

inline float sigmoid(float x) {
    assert(x > -MAX_EXP && x < MAX_EXP);
    return 1 / (1 + exp(-x));
}

inline string lower(string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

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
    bool is_sent_id;

    HuffmanNode() : index(-1), is_unk(true), is_sent_id(false) {}

    HuffmanNode(int index, const string& word, bool is_sent_id = false) :
            word(word), index(index), count(1), is_leaf(true), is_unk(false), is_sent_id(is_sent_id)
    {}

    HuffmanNode(int index, HuffmanNode* left, HuffmanNode* right) :
            left(left), right(right), index(index), count(left->count + right->count), is_leaf(false), is_unk(false), is_sent_id(false)
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
    bool sent_ids;

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
        negative(5),
        sent_ids(false)
        {}

    void print() const {
        std::cout << std::boolalpha; // to print false/true instead of 0/1
        std::cout << "Word2vec++"    << std::endl;
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
        std::cout << "sent ids:    " << sent_ids << std::endl;
    }
};

class MonolingualModel
{
    friend class BilingualModel;
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive& ar, const unsigned int version) {
        ar & config & input_weights & output_weights & output_weights_hs & train_words & vocabulary;
    }

private:
    mat input_weights;
    mat output_weights; // output weights for negative sampling
    mat output_weights_hs; // output weights for hierarchical softmax

    long long train_words; // total number of words in training file (used to compute word frequencies)
    long long total_word_count;
    float alpha;
    Config config;
    map<string, HuffmanNode> vocabulary;
    vector<HuffmanNode*> unigram_table;

    static unsigned long long rand() {
        static unsigned long long next_random(time(NULL));
        next_random = next_random * static_cast<unsigned long long>(25214903917) + 11; // possible race conditions, but who cares...
        return next_random >> 16;
    }
    static float randf() {
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

    vector<long long> static chunkify(const string& filename, int n_chunks);

public:
    MonolingualModel() : train_words(0), total_word_count(0) {} // model with default configuration
    MonolingualModel(Config config) : train_words(0), total_word_count(0), config(config) {}

    vec wordVec(const string& word) const; // word embedding
    vec sentVec(const string& sentence); // paragraph vector (Le & Mikolov)

    void train(const string& training_file); // training from scratch (resets vocabulary and weights)

    void saveEmbeddings(const string& filename) const; // saves the word embeddings in the word2vec binary format
    void saveEmbeddingsTxt(const string& filename) const; // saves the word embeddings in the word2vec text format

    void load(const string& filename); // loads the entire model
    void save(const string& filename) const; // saves the entire model

    void computeAccuracy(istream& infile, int max_vocabulary_size = 0) const;
};
