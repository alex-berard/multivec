#pragma once
#include "bilingual.hpp"

template<typename T>
inline void save(ofstream& outfile, T x) {
    // for basic types (int, bool, float, etc.)
    outfile.write(reinterpret_cast<const char*>(&x), sizeof(x));
}

template<typename T>
inline void load(ifstream& infile, T& x) {
    infile.read(reinterpret_cast<char*>(&x), sizeof(x));
}

inline void save(ofstream& outfile, const string& s) {
    save(outfile, s.size());
    outfile.write(s.data(), s.size());
}

inline void load(ifstream& infile, string& s) {
    size_t size = 0;
    load(infile, size);
    char* temp = new char[size + 1];
    temp[size] = '\0';
    infile.read(temp, size);
    s = temp;
    delete [] temp;
}

template<typename T>
inline void save(ofstream& outfile, const std::vector<T>& v) {
    save(outfile, v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        save(outfile, v[i]);
    }
}

template<typename T>
inline void load(ifstream& infile, std::vector<T>& v) {
    size_t size = 0;
    v.clear();
    load(infile, size);
    for (size_t i = 0; i < size; ++i) {
        T x;
        load(infile, x);
        v.push_back(x);
    }
}

inline void save(ofstream& outfile, const vec& v) {
    save(outfile, v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        save(outfile, v[i]);
    }
}

inline void load(ifstream& infile, vec& v) {
    size_t size = 0;
    load(infile, size);
    v = vec(size);
    for (size_t i = 0; i < size; ++i) {
        load(infile, v[i]);
    }
}

inline void save(ofstream& outfile, const Config& cfg) {
    save(outfile, cfg.learning_rate);
    save(outfile, cfg.dimension);
    save(outfile, cfg.min_count);
    save(outfile, cfg.iterations);
    save(outfile, cfg.window_size);
    save(outfile, cfg.threads);
    save(outfile, cfg.subsampling);
    save(outfile, cfg.hierarchical_softmax);
    save(outfile, cfg.threads);
    save(outfile, cfg.skip_gram);
    save(outfile, cfg.negative);
    save(outfile, cfg.sent_vector);
}

inline void load(ifstream& infile, Config& cfg) {
   load(infile, cfg.learning_rate);
   load(infile, cfg.dimension);
   load(infile, cfg.min_count);
   load(infile, cfg.iterations);
   load(infile, cfg.window_size);
   load(infile, cfg.threads);
   load(infile, cfg.subsampling);
   load(infile, cfg.hierarchical_softmax);
   load(infile, cfg.threads);
   load(infile, cfg.skip_gram);
   load(infile, cfg.negative);
   load(infile, cfg.sent_vector);
}

inline void save(ofstream& outfile, const BilingualConfig& cfg) {
    save(outfile, reinterpret_cast<const Config&>(cfg));
    save(outfile, cfg.beta);
}

inline void load(ifstream& infile, BilingualConfig& cfg) {
    load(infile, reinterpret_cast<Config&>(cfg));
    load(infile, cfg.beta);
}

inline void save(ofstream& outfile, const HuffmanNode& node) {
    save(outfile, node.index);
    save(outfile, node.count);
    save(outfile, node.word);
    save(outfile, node.code);
    save(outfile, node.parents);
}

inline void load(ifstream& infile, HuffmanNode& node) {
    load(infile, node.index);
    load(infile, node.count);
    load(infile, node.word);
    load(infile, node.code);
    load(infile, node.parents);
}

inline void save(ofstream& outfile, const MonolingualModel& model) {
    save(outfile, *model.config);
    save(outfile, model.vocabulary.size());

    // transform into map to save in lexicographical order (for consistency)
    map<string, HuffmanNode> voc_ordered(model.vocabulary.begin(), model.vocabulary.end());
    for (auto it = voc_ordered.begin(); it != voc_ordered.end(); ++it) {
        save(outfile, it->second);
    }

    save(outfile, model.input_weights);
    save(outfile, model.output_weights);
    save(outfile, model.output_weights_hs);
    save(outfile, model.sent_weights);
}

inline void load(ifstream& infile, MonolingualModel& model) {
    load(infile, *model.config);

    size_t vocabulary_size = 0;
    load(infile, vocabulary_size);
    model.vocabulary.clear();

    for (size_t i = 0; i < vocabulary_size; ++i) {
        HuffmanNode node(0, ""); // empty constructor creates UNK node
        load(infile, node);
        model.vocabulary.insert({node.word, node});
    }

    load(infile, model.input_weights);
    load(infile, model.output_weights);
    load(infile, model.output_weights_hs);
    load(infile, model.sent_weights);
}

inline void save(ofstream& outfile, const BilingualModel& model) {
    save(outfile, *model.config);
    save(outfile, model.src_model);
    save(outfile, model.trg_model);
}

inline void load(ifstream& infile, BilingualModel& model) {
    load(infile, *model.config);
    load(infile, model.src_model);
    load(infile, model.trg_model);
}
