#pragma once

#include "multivec-mono.hpp"

using namespace std;

struct BilingualConfig : Config {
    float bi_weight;
    BilingualConfig() : bi_weight(1.0f) {}
    void print() const {
        Config::print();
        std::cout << "bi-weight:   " << bi_weight << std::endl;
    }
};

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
};
