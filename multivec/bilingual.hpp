#pragma once
#include "monolingual.hpp"

using namespace std;

class BilingualModel
{
    friend void save(ofstream& outfile, const BilingualModel& model);
    friend void load(ifstream& infile, BilingualModel& model);

private:
    // Configuration of the model (monolingual models have the same configuration)
    BilingualConfig config;

    long long total_word_count; // number of words processed so far
    float alpha;

    void trainChunk(const string& src_file,
                    const string& trg_file,
                    const vector<long long>& src_chunks,
                    const vector<long long>& trg_chunks,
                    int thread_id);

    // TODO: unsupervised alignment (GIZA)
    vector<int> uniformAlignment(const vector<HuffmanNode>& src_nodes, const vector<HuffmanNode>& trg_nodes);

    int trainSentence(const string& trg_sent, const string& src_sent);

    void trainWord(MonolingualModel& src_params, MonolingualModel& trg_params,
        const vector<HuffmanNode>& src_nodes, const vector<HuffmanNode>& trg_nodes,
        int src_pos, int trg_pos, float alpha);

    void trainWordCBOW(MonolingualModel&, MonolingualModel&,
        const vector<HuffmanNode>&, const vector<HuffmanNode>&,
        int, int, float);

    void trainWordSkipGram(MonolingualModel&, MonolingualModel&,
        const vector<HuffmanNode>&, const vector<HuffmanNode>&,
        int, int, float);

public:
    // A bilingual model is comprised of two monolingual models
    MonolingualModel src_model;
    MonolingualModel trg_model;

    // TODO: ensure that src_model.config and trg_model.config stay in sync with config
    BilingualModel() : total_word_count(0), src_model(config), trg_model(config) {}
    BilingualModel(BilingualConfig config) : config(config), total_word_count(0),
        src_model(config), trg_model(config) {}

    void train(const string& src_file, const string& trg_file);
    void load(const string& filename);
    void save(const string& filename) const;

    float similarity(const string& wordSrc, const string& wordTgt, int policy = 0) const; // cosine similarity
    float distance(const string& wordSrc, const string& wordTgt, int policy = 0) const; // 1 - cosine similarity
    float similarityNgrams(const string& seqSrc, const string& seqTgt, int policy = 0) const; // similarity between two sequences of same size
    float similaritySentence(const string& seqSrc, const string& seqTgt, int policy = 0) const; // similarity between two variable-size sequences
    float softEditDistance(const string& seqSrc, const string& seqTgt, int policy = 0) const; // soft Levenshtein distance

    vector<pair<string, float>> closest(const string& word, int n = 50, int policy = 0) const; // n closest words to given word
    vector<pair<string, float>> closest(const string& word, const vector<string>& words, int policy = 0) const;
    vector<pair<string, float>> closest(const vec& v, int n, int policy = 0) const;
};
