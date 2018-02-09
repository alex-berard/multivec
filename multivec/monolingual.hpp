#pragma once
#include "utils.hpp"

class MonolingualModel
{
    friend class BilingualModel;
    friend void save(ofstream& outfile, const MonolingualModel& model);
    friend void load(ifstream& infile, MonolingualModel& model);

private:
    Config* const config;

    mat input_weights;
    mat output_weights;
    mat sent_weights;

    long long vocab_word_count; // property of vocabulary (sum of all word counts)

    // training file stats (properties of this training instance)
    long long training_words; // total number of words in training file (used for progress estimation)
    long long training_lines;
    // training state
    long long words_processed;
    
    unordered_map<string, HuffmanNode> vocabulary;
    vector<HuffmanNode*> unigram_table;

    void add_word_to_vocab(const string& word);
    void reduce_vocab();
    void create_binary_tree();
    void assign_codes(HuffmanNode* node, vector<int> code, vector<int> parents) const;
    void init_unigram_table();

    HuffmanNode* get_random_huffman_node(); // uses the unigram frequency table to sample a random node

    vector<HuffmanNode> get_nodes(const string& sentence) const;
    vector<HuffmanNode> get_sorted_vocab() const;
    void subsample(vector<HuffmanNode>& node) const;

    void read_vocab(const string& training_file);
    void init_net();
    void init_sent_weights();

    void train_chunk(const string& training_file, const vector<long long>& chunks, int chunk_id);

    int train_sentence(const string& sent, vec* sent_vec, float alpha);
    void train_word(const vector<HuffmanNode>& nodes, int word_pos, vec* sent_vec, float alpha, bool update = true);
    void train_word_DBOW(const vector<HuffmanNode>& nodes, int word_pos, vec* sent_vec, float alpha, bool update = true);
    void train_word_CBOW(const vector<HuffmanNode>& nodes, int word_pos, vec* sent_vec, float alpha, bool update = true);
    void train_word_skip_gram(const vector<HuffmanNode>& nodes, int word_pos, vec* sent_vec, float alpha, bool update = true);
    vec hierarchical_update(const HuffmanNode& node, const vec& hidden, float alpha, bool update = true);
    vec neg_sampling_update(const HuffmanNode& node, const vec& hidden, float alpha, bool update = true);

    vector<long long> chunkify(const string& filename, int n_chunks);
    vec word_vec(int index, int policy) const;

public:
    MonolingualModel(Config* config) : config(config) {}  // prefer this constructor

    vec word_vec(const string& word, int policy = 0) const; // word embedding
    vec sent_vec(const string& sentence); // paragraph vector (Le & Mikolov)
    
    void sent_vectors(const string &input_file);

    void train(const string& training_file, bool initialize = true); // training from scratch (resets vocabulary and weights)

    void save_vectors_bin(const string &filename, int policy = 0, bool norm = false) const; // saves word embeddings in the word2vec binary format
    void save_vectors(const string &filename, int policy = 0, bool norm = false) const; // saves word embeddings in the word2vec text format
    void save_sent_vectors(const string &filename, bool norm = false) const;
    void load(const string& filename); // loads the entire model
    void save(const string& filename) const; // saves the entire model

    float similarity(const string& word1, const string& word2, int policy = 0) const; // cosine similarity
    float distance(const string& word1, const string& word2, int policy = 0) const; // 1 - cosine similarity
    float similarity_ngrams(const string& seq1, const string& seq2, int policy = 0) const; // similarity between two sequences of same size
    float similarity_bag_of_words(const string& seq1, const string& seq2, int policy = 0) const; // similarity between two variable-size sequences
    // similarity between two variable-size sequences taking into account part-of-speech tags and inverse document frequencies of terms in the sequences
    float similarity_syntax(const string& seq1, const string& seq2, const string& tags1, const string& tags2,
                            const vector<float>& idf1, const vector<float>& idf2, float alpha = 0.0, int policy = 0) const;
    float soft_word_error_rate(const string& hyp, const string& ref, int policy = 0) const; // soft Word Error Rate

    vector<pair<string, float>> trg_closest(const string& src_word, int n = 10, int policy = 0) const; // n closest words to given word
    vector<pair<string, float>> src_closest(const string& trg_word, int n = 10, int policy = 0) const;

    int get_dimension() const { return config->dimension; };

    vector<pair<string, float>> closest(const string& word, int n = 10, int policy = 0) const; // n closest words to given word
    vector<pair<string, float>> closest(const string& word, const vector<string>& words, int policy = 0) const;
    vector<pair<string, float>> closest(const vec& v, int n = 10, int policy = 0) const;

    vector<pair<string, int>> get_word_counts() const; // get words with their counts
};
