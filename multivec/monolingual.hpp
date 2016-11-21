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
    mat output_weights; // output weights for negative sampling
    mat output_weights_hs; // output weights for hierarchical softmax
    mat sent_weights;

    long long vocab_word_count; // property of vocabulary (sum of all word counts)

    // training file stats (properties of this training instance)
    long long training_words; // total number of words in training file (used for progress estimation)
    long long training_lines;
    // training state
    long long words_processed;
    float alpha;

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
    void initSentWeights();

    void trainChunk(const string& training_file, const vector<long long>& chunks, int chunk_id);

    int trainSentence(const string& sent, int sent_id);
    void trainWord(const vector<HuffmanNode>& nodes, int word_pos, int sent_id);
    void trainWordCBOW(const vector<HuffmanNode>& nodes, int word_pos, int sent_id);
    void trainWordSkipGram(const vector<HuffmanNode>& nodes, int word_pos, int sent_id);

    vec hierarchicalUpdate(const HuffmanNode& node, const vec& hidden, float alpha, bool update = true);
    vec negSamplingUpdate(const HuffmanNode& node, const vec& hidden, float alpha, bool update = true);

    vector<long long> chunkify(const string& filename, int n_chunks);
    vec wordVec(int index, int policy) const;

public:
    MonolingualModel(Config* config) : config(config) {}  // prefer this constructor

    vec wordVec(const string& word, int policy = 0) const; // word embedding
    vec sentVec(const string& sentence); // paragraph vector (Le & Mikolov), TODO: custom alpha and iterations
    void sentVec(istream& infile); // compute paragraph vector for all lines in a stream

    void train(const string& training_file, bool initialize = true); // training from scratch (resets vocabulary and weights)

    void saveVectorsBin(const string &filename, int policy = 0) const; // saves word embeddings in the word2vec binary format
    void saveVectors(const string &filename, int policy = 0) const; // saves word embeddings in the word2vec text format
    void saveSentVectors(const string &filename) const;
    void load(const string& filename); // loads the entire model
    void save(const string& filename) const; // saves the entire model

    void normalizeWeights(); // normalize all weights between 0 and 1

    float similarity(const string& word1, const string& word2, int policy = 0) const; // cosine similarity
    float distance(const string& word1, const string& word2, int policy = 0) const; // 1 - cosine similarity
    float similarityNgrams(const string& seq1, const string& seq2, int policy = 0) const; // similarity between two sequences of same size
    float similaritySentence(const string& seq1, const string& seq2, int policy = 0) const; // similarity between two variable-size sequences
    // similarity between two variable-size sequences taking into account part-of-speech tags and inverse document frequencies of terms in the sequences
    float similaritySentenceSyntax(const string& seq1, const string& seq2, const string& tags1, const string& tags2,
                                   const vector<float>& idf1, const vector<float>& idf2, float alpha = 0.0, int policy = 0) const;
    float softWER(const string& hyp, const string& ref, int policy = 0) const; // soft Word Error Rate

    vector<pair<string, float>> trg_closest(const string& src_word, int n = 10, int policy = 0) const; // n closest words to given word
    vector<pair<string, float>> src_closest(const string& trg_word, int n = 10, int policy = 0) const;

    int getDimension() const { return config->dimension; };

    vector<pair<string, float>> closest(const string& word, int n = 10, int policy = 0) const; // n closest words to given word
    vector<pair<string, float>> closest(const string& word, const vector<string>& words, int policy = 0) const;
    vector<pair<string, float>> closest(const vec& v, int n = 10, int policy = 0) const;

    vector<pair<string, int>> getWords() const; // get words with their counts
    
    void analogicalReasoning(const string& filename, int max_voc = 0, int policy = 0) const;
};
