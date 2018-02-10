#pragma once
#include "monolingual.hpp"

using namespace std;

class BilingualModel
{
    friend void save(ofstream& outfile, const BilingualModel& model);
    friend void load(ifstream& infile, BilingualModel& model);

private:
    // Configuration of the model (monolingual models have the same configuration)
    BilingualConfig* const config;

    mat mapping;
    long long words_processed; // number of words processed so far
    float alpha;
    vector<vector<int>> alignments;

    void train_chunk(const string& src_file,
                     const string& trg_file,
                     const vector<long long>& src_chunks,
                     const vector<long long>& trg_chunks,
                     int thread_id);

    void read_alignments(const string& align_file);
    vector<int> get_alignment(const vector<int>& src_nodes, const vector<int>& trg_nodes, int sent_id);

    int train_sentence(const string& trg_sent, const string& src_sent, int sent_id);

    void train_word(MonolingualModel& src_params, MonolingualModel& trg_params,
        const vector<int>& src_nodes, const vector<int>& trg_nodes,
        int src_pos, int trg_pos, float alpha);

    void train_word_CBOW(MonolingualModel&, MonolingualModel&,
        const vector<int>&, const vector<int>&,
        int, int, float);

    void train_word_skip_gram(MonolingualModel&, MonolingualModel&,
        const vector<int>&, const vector<int>&,
        int, int, float);

public:
    // A bilingual model is comprised of two monolingual models
    MonolingualModel src_model;
    MonolingualModel trg_model;

    // prefer this constructor
    BilingualModel(BilingualConfig* config) : config(config), src_model(config), trg_model(config) {}

    void train(const string& src_file, const string& trg_file, const string& align_file, bool initialize = true);
    void load(const string& filename);
    void save(const string& filename) const;

    float similarity(const string& src_word, const string& trg_word, int policy = 0) const; // cosine similarity
    float distance(const string& src_word, const string& trg_word, int policy = 0) const; // 1 - cosine similarity
    float similarity_ngrams(const string& src_seq, const string& trg_seq, int policy = 0) const; // similarity between two sequences of same size
    float similarity_bag_of_words(const string& src_seq, const string& trg_seq, int policy = 0) const; // similarity between two variable-size sequences
    // similarity between two variable-size sequences taking into account part-of-speech tags and inverse document frequencies of terms in the sequences
    float similarity_syntax(const string& src_seq, const string& trg_seq, const string& src_tags, const string& trg_tags,
                                     const vector<float>& src_idf, const vector<float>& trg_idf, float alpha = 0.0, int policy = 0) const;
    
    vector<pair<string, float>> trg_closest(const string& src_word, int n = 10, int policy = 0) const; // n closest words to given word
    vector<pair<string, float>> src_closest(const string& trg_word, int n = 10, int policy = 0) const;
    
    vector<pair<string, string>> dictionary_induction(int src_count = 0, int trg_count = 0, int policy = 0) const;
    vector<pair<string, string>> dictionary_induction(const vector<string>& src_vocab, const vector<string>& trg_vocab, int policy = 0) const;
    
    void learn_mapping(const vector<pair<string, string>>& dict);
};
