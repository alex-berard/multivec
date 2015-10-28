#pragma once

#include "multivec-bi.hpp"

template<typename T>
inline void save(ofstream& outfile, const vector<T>& v) {
    outfile << v.size() << endl;
    for (auto it = v.begin(); it != v.end(); ++it) {
        outfile << *it << endl;
    }
}

template<typename T>
inline void save(ofstream& outfile, const vector<vector<T>>& v) {
    outfile << v.size() << endl;
    for (auto it = v.begin(); it != v.end(); ++it) {
        save(outfile, *it);
    }
}

inline void save(ofstream& outfile, const Config& cfg) {
    outfile << cfg.starting_alpha << endl
            << cfg.dimension << endl
            << cfg.min_count << endl
            << cfg.max_iterations << endl
            << cfg.window_size << endl
            << cfg.n_threads << endl
            << cfg.subsampling << endl
            << cfg.hierarchical_softmax << endl
            << cfg.skip_gram << endl
            << cfg.negative << endl
            << cfg.sent_vector << endl;
}

inline void save(ofstream& outfile, const BilingualConfig& cfg) {
    save(outfile, reinterpret_cast<const Config&>(cfg));
    outfile << cfg.bi_weight << endl;
}

inline void save(ofstream& outfile, const HuffmanNode& node) {
    outfile << node.word << endl
            << node.code.size() << endl
            << node.parents.size() << endl
            << node.index << endl
            << node.count << endl;

    save(outfile, node.code);
    save(outfile, node.parents);
}

inline void save(ofstream& outfile, const MonolingualModel& model) {
    save(outfile, model.config);

    outfile << model.vocabulary.size() << endl;

    for (auto it = model.vocabulary.begin(); it != model.vocabulary.end(); ++it) {
        save(outfile, it->second);
    }

    save(outfile, model.input_weights);
    save(outfile, model.output_weights);
    save(outfile, model.output_weights_hs);
    save(outfile, model.sent_weights);
}

inline void save(ofstream& outfile, const BilingualModel& model) {
    save(outfile, model.config);
    save(outfile, model.src_model);
    save(outfile, model.trg_model);
}

template<typename T>
inline void load(ifstream& infile, vector<T>& v) {
    int size = 0;
    v.clear();
    infile >> size;
    T x;
    for (int i = 0; i < size; ++i) {
        infile >> x;
        v.push_back(x);
    }
}

template<typename T>
inline void load(ifstream& infile, vector<vector<T>>& v) {
    int size = 0;
    v.clear();
    infile >> size;
    vector<T> x;
    for (int i = 0; i < size; ++i) {
        load(infile, x);
        v.push_back(x);
    }
}

inline void load(ifstream& infile, Config& cfg) {
    infile >> cfg.starting_alpha
           >> cfg.dimension
           >> cfg.min_count
           >> cfg.max_iterations
           >> cfg.window_size
           >> cfg.n_threads
           >> cfg.subsampling
           >> cfg.hierarchical_softmax
           >> cfg.skip_gram
           >> cfg.negative
           >> cfg.sent_vector;
}

inline void load(ifstream& infile, BilingualConfig& cfg) {
    load(infile, reinterpret_cast<Config&>(cfg));
    infile >> cfg.bi_weight;
}

inline void load(ifstream& infile, HuffmanNode& node) {
    int code_size = 0;
    int parent_size = 0;

    infile >> node.word
           >> code_size
           >> parent_size
           >> node.index
           >> node.count;

    node.code.clear();
    node.parents.clear();

    load(infile, node.code);
    load(infile, node.parents);
}

inline void load(ifstream& infile, MonolingualModel& model) {
    load(infile, model.config);

    int vocabulary_size = 0;
    infile >> vocabulary_size;
    model.vocabulary.clear();
    HuffmanNode node(0, "");

    for (int i = 0; i < vocabulary_size; ++i) {
        load(infile, node);
        model.vocabulary.insert({node.word, node});
    }

    load(infile, model.input_weights);
    load(infile, model.output_weights);
    load(infile, model.output_weights_hs);
    load(infile, model.sent_weights);
}

inline void load(ifstream& infile, BilingualModel& model) {
    load(infile, model.config);
    load(infile, model.src_model);
    load(infile, model.trg_model);
}
