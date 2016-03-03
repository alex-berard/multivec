#include "monolingual.hpp"
#include "serialization.hpp"

const HuffmanNode HuffmanNode::UNK;

void MonolingualModel::addWordToVocab(const string& word) {
    auto it = vocabulary.find(word);

    if (it != vocabulary.end()) {
        it->second.count++;
    } else {
        HuffmanNode node(static_cast<int>(vocabulary.size()), word);
        vocabulary.insert({word, node});
    }
}

void MonolingualModel::reduceVocab() {
    int i = 0;
    for (auto it = vocabulary.begin(); it != vocabulary.end(); ) {
        if ((it->second.count) < config.min_count) {
            vocabulary.erase(it++);
        } else {
            it++->second.index = i++; // reassign indices in [0, vocabulary size - 1)
        }
    }
}

void MonolingualModel::readVocab(const string& training_file) {
    ifstream infile(training_file);

    if (!infile.is_open()) {
        throw runtime_error("couldn't open file " + training_file);
    }

    vocabulary.clear();

    string word;
    while (infile >> word) {
        addWordToVocab(word);
    }

    if (config.verbose)
        cout << "Vocabulary size: " << vocabulary.size() << endl;

    reduceVocab();

    if (config.verbose)
        cout << "Reduced vocabulary size: " << vocabulary.size() << endl;

    training_words = 0; // count number of words in the training file
    for (auto it = vocabulary.begin(); it != vocabulary.end(); ++it) {
        training_words += it->second.count;
    }

    createBinaryTree();
    initUnigramTable();
}

void MonolingualModel::createBinaryTree() {
    vector<HuffmanNode*> heap;
    vector<HuffmanNode> parent_nodes;
    parent_nodes.reserve(vocabulary.size());

    for (auto it = vocabulary.begin(); it != vocabulary.end(); ++it) {
        heap.push_back(&it->second);
    }

    sort(heap.begin(), heap.end(), HuffmanNode::comp);

    for (int i = 0; heap.size() > 1; i++) {
        HuffmanNode* left = heap.back();
        heap.pop_back();

        HuffmanNode* right = heap.back();
        heap.pop_back();

        parent_nodes.push_back({i, left, right});

        HuffmanNode* parent = &parent_nodes.back();
        auto it = lower_bound(heap.begin(), heap.end(), parent, HuffmanNode::comp);
        heap.insert(it, parent);
    }

    assignCodes(heap.front(), {}, {});
}

void MonolingualModel::assignCodes(HuffmanNode* node, vector<int> code, vector<int> parents) const {
    if (node->is_leaf) {
        node->code = code;
        node->parents = parents;
    } else {
        parents.push_back(node->index);
        vector<int> code_left(code);
        code_left.push_back(0);
        vector<int> code_right(code);
        code_right.push_back(1);

        assignCodes(node->left, code_left, parents);
        assignCodes(node->right, code_right, parents);
    }
}

void MonolingualModel::initUnigramTable() {
    unigram_table.clear();

    float power = 0.75; // weird word2vec tweak ('normal' value would be 1.0)
    float total_count = 0.0;
    for (auto it = vocabulary.begin(); it != vocabulary.end(); ++it) {
        total_count += pow(it->second.count, power);
    }

    for (auto it = vocabulary.begin(); it != vocabulary.end(); ++it) {
        float f = pow(it->second.count, power) / total_count;

        int d = static_cast<int>(f * UNIGRAM_TABLE_SIZE);
        for (int i = 0; i < d; ++i) {
            unigram_table.push_back(&it->second);
        }
    }
}

HuffmanNode* MonolingualModel::getRandomHuffmanNode() {
    auto index = multivec::rand() % unigram_table.size();
    return unigram_table[index];
}

void MonolingualModel::initNet() {
    int v = static_cast<int>(vocabulary.size());
    int d = config.dimension;

    input_weights = mat(v, vec(d));

    for (size_t row = 0; row < v; ++row) {
        for (size_t col = 0; col < d; ++col) {
            input_weights[row][col] = (multivec::randf() - 0.5f) / d;
        }
    }

    output_weights_hs = mat(v, vec(d));
    output_weights = mat(v, vec(d));
}

void MonolingualModel::initSentWeights() {
    int d = config.dimension;
    sent_weights = mat(training_lines, vec(d));

    for (size_t row = 0; row < training_lines; ++row) {
        for (size_t col = 0; col < d; ++col) {
            sent_weights[row][col] = (multivec::randf() - 0.5f) / d;
        }
    }
}

vector<HuffmanNode> MonolingualModel::getNodes(const string& sentence) const {
    vector<HuffmanNode> nodes;
    istringstream iss(sentence);
    string word;

    while (iss >> word) {
        auto it = vocabulary.find(word);
        HuffmanNode node = HuffmanNode::UNK;

        if (it != vocabulary.end()) {
            node = it->second;
        }

        nodes.push_back(node);
    }

    return nodes;
}

/**
 * @brief Discard random nodes according to their frequency. The more frequent a word is, the more
 * likely it is to be discarded. Discarded nodes are replaced by UNK token.
 */
void MonolingualModel::subsample(vector<HuffmanNode>& nodes) const {
    for (auto it = nodes.begin(); it != nodes.end(); ++it) {
        auto node = *it;
        float f = static_cast<float>(node.count) / training_words; // frequency of this word
        float p = 1 - (1 + sqrt(f / config.subsampling)) * config.subsampling / f; // word2vec formula

        if (p >= multivec::randf()) {
            *it = HuffmanNode::UNK;
        }
    }
}

void MonolingualModel::saveVectorsBin(const string &filename, int policy) const {
    if (config.verbose)
        cout << "Saving embeddings in binary format to " << filename << endl;

    ofstream outfile(filename, ios::binary | ios::out);

    if (!outfile.is_open()) {
        throw runtime_error("couldn't open file " + filename);
    }

    outfile << vocabulary.size() << " " << config.dimension << endl;

    for (auto it = vocabulary.begin(); it != vocabulary.end(); ++it) {
        string word = string(it->second.word);
        word.push_back(' ');
        vec embedding = wordVec(it->second.index, policy);

        outfile.write(word.c_str(), word.size());
        outfile.write(reinterpret_cast<const char*>(embedding.data()), sizeof(float) * config.dimension);
        outfile << endl;
    }
}

void MonolingualModel::saveVectors(const string &filename, int policy) const {
    if (config.verbose)
        cout << "Saving embeddings in text format to " << filename << endl;

    ofstream outfile(filename, ios::binary | ios::out);

    if (!outfile.is_open()) {
        throw runtime_error("couldn't open file " + filename);
    }

    outfile << vocabulary.size() << " " << config.dimension << endl;

    for (auto it = vocabulary.begin(); it != vocabulary.end(); ++it) {
        outfile << it->second.word << " ";
        vec embedding = wordVec(it->second.index, policy);
        for (int c = 0; c < config.dimension; ++c) {
            outfile << embedding[c] << " ";
        }
        outfile << endl;
    }
}

void MonolingualModel::saveSentVectors(const string &filename) const {
    if (config.verbose)
        cout << "Saving sentence vectors in text format to " << filename << endl;

    ofstream outfile(filename, ios::binary | ios::out);

    if (!outfile.is_open()) {
        throw runtime_error("couldn't open file " + filename);
    }

    for (auto it = sent_weights.begin(); it != sent_weights.end(); ++it) {
        vec embedding = *it;
        for (int c = 0; c < config.dimension; ++c) {
            outfile << embedding[c] << " ";
        }
        outfile << endl;
    }
}

void MonolingualModel::load(const string& filename) {
    if (config.verbose)
        cout << "Loading model" << endl;

    ifstream infile(filename);

    if (!infile.is_open()) {
        throw runtime_error("couldn't open file " + filename);
    }

    ::load(infile, *this);
    initUnigramTable();
}

void MonolingualModel::save(const string& filename) const {
    if (config.verbose)
        cout << "Saving model" << endl;

    ofstream outfile(filename);

    if (!outfile.is_open()) {
        throw runtime_error("couldn't open file " + filename);
    }

    ::save(outfile, *this);
}

vec MonolingualModel::wordVec(int index, int policy) const {
    if (policy == 1 && config.negative > 0) // concat input and output
    {
        int d = config.dimension;
        vec res(d * 2);
        for (int c = 0; c < d; ++c) res[c] = input_weights[index][c];
        for (int c = 0; c < d; ++c) res[d + c] = output_weights[index][c];
        return res;
    }
    else if (policy == 2 && config.negative > 0) // sum input and output
    {
        return input_weights[index] + output_weights[index];
    }
    else if (policy == 3 && config.negative > 0) // only output weights
    {
        return output_weights[index];
    }
    else // only input weights
    {
        return input_weights[index];
    }
}

/**
 * @brief Return weight vector corresponding to the given word.
 *
 * @param word
 * @param policy defines which weights to return.
 * 0 (default): input weights only,
 * 1: concatenation of input and output weights,
 * 2: sum of input and output weights,
 * 3: output weights only.
 * @return vec
 */
vec MonolingualModel::wordVec(const string& word, int policy) const {
    auto it = vocabulary.find(word);

    if (it == vocabulary.end()) {
        throw runtime_error("out of vocabulary");
    } else {
        return wordVec(it->second.index, policy);
    }
}

void MonolingualModel::sentVec(istream& input, int policy) {
    string line;
    while(getline(input, line)) {
        vec embedding(config.dimension, 0);
        try {
            embedding = sentVec(line, policy);
        } catch (runtime_error) {
            // in case of error (empty sentence, or all words are OOV), print a vector of zeros
        };

        for (int c = 0; c < config.dimension; ++c) {
            cout << embedding[c] << " ";
        }
    }
}

vec MonolingualModel::sentVec(const string& sentence, int policy) {
    // TODO: integrate this in the normal training procedure
    int dimension = config.dimension;
    float alpha = config.starting_alpha;  // TODO: decreasing learning rate

    auto nodes = getNodes(sentence);  // no subsampling here
    nodes.erase(
        remove(nodes.begin(), nodes.end(), HuffmanNode::UNK),
        nodes.end()); // remove UNK tokens

    if (nodes.empty())
        throw runtime_error("too short sentence, or OOV words");

    vec sent_vec(dimension, 0);

    for (int k = 0; k < config.max_iterations; ++k) {
        for (int word_pos = 0; word_pos < nodes.size(); ++word_pos) {
            vec hidden(dimension, 0);
            HuffmanNode cur_node = nodes[word_pos];

            int this_window_size = 1 + multivec::rand() % config.window_size;
            int count = 0;

            for (int pos = word_pos - this_window_size; pos <= word_pos + this_window_size; ++pos) {
                if (pos < 0 || pos >= nodes.size() || pos == word_pos) continue;
                hidden += input_weights[nodes[pos].index];
                ++count;
            }

            if (count == 0) continue;
            hidden = (hidden + sent_vec) / (count + 1); // TODO this or (hidden / count) + sent_vec?

            vec error(dimension, 0);
            if (config.hierarchical_softmax) {
                error += hierarchicalUpdate(cur_node, hidden, alpha, false);
            }
            if (config.negative > 0) {
                error += negSamplingUpdate(cur_node, hidden, alpha, false);
            }

            sent_vec += error;
        }
    }

    return sent_vec;
}

void MonolingualModel::train(const string& training_file) {
    cout << "MultiVec-mono" << endl;
    config.print();
    cout << "Training file: " << training_file << endl;

    if (!config.freeze) {
        // reads vocab and counts words
        readVocab(training_file); // TODO: incremental training
        if (config.verbose)
            cout << "Total number of words: " << training_words << endl;
    }

    words_processed = 0;
    alpha = config.starting_alpha;

    // read file to find out the beginning of each chunk
    auto chunks = chunkify(training_file, config.n_threads); // also counts the number of lines
    if (!config.freeze)
        initNet();
    if (config.sent_vector)
        initSentWeights();

    high_resolution_clock::time_point start = high_resolution_clock::now();
    if (config.n_threads == 1) {
        trainChunk(training_file, chunks, 0);
    } else {
        vector<thread> threads;

        for (int i = 0; i < config.n_threads; ++i) {
            threads.push_back(thread(&MonolingualModel::trainChunk, this,
                training_file, chunks, i));
        }

        for (auto it = threads.begin(); it != threads.end(); ++it) {
            it->join();
        }
    }
    high_resolution_clock::time_point end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();

    if (config.verbose)
        cout << endl;

    cout << "Training time: " << static_cast<float>(duration) / 1000000 << endl;
}

/**
 * @brief Divide a given file into chunks with the same number of lines each
 * TODO: avoid reading training file several times
 *
 * @param filename path of the file
 * @param n_chunks number of chunks
 * @return starting position (in bytes) of each chunk
 */
vector<long long> MonolingualModel::chunkify(const string& filename, int n_chunks) {
    ifstream infile(filename);

    if (!infile.is_open()) {
        throw runtime_error("couldn't open file " + filename);
    }

    vector<long long> chunks;
    vector<long long> line_positions;

    string line;
    do {
        line_positions.push_back(infile.tellg());
    } while (getline(infile, line));

    training_lines = line_positions.size() - 1;
    int chunk_size = line_positions.size() / n_chunks;  // number of lines in each chunk

    for (int i = 0; i < n_chunks; i++) {
        long long chunk_start = line_positions[i * chunk_size];
        chunks.push_back(chunk_start);
    }

    return chunks;
}

void MonolingualModel::trainChunk(const string& training_file,
                                  const vector<long long>& chunks,
                                  int chunk_id) {
    ifstream infile(training_file);
    float starting_alpha = config.starting_alpha;
    int max_iterations = config.max_iterations;

    if (!infile.is_open()) {
        throw runtime_error("couldn't open file " + training_file);
    }

    for (int k = 0; k < max_iterations; ++k) {
        int word_count = 0, last_count = 0;

        infile.clear();
        infile.seekg(chunks[chunk_id], infile.beg);

        int chunk_size = training_lines / chunks.size();
        int sent_id = chunk_id * chunk_size;

        string sent;
        while (getline(infile, sent)) {
            word_count += trainSentence(sent, sent_id++); // asynchronous update (possible race conditions)

            // update learning rate
            if (word_count - last_count > 10000) {
                words_processed += word_count - last_count; // asynchronous update
                last_count = word_count;

                if (!config.freeze) { // FIXME
                    // decreasing learning rate
                    alpha = starting_alpha * (1 - static_cast<float>(words_processed) / (max_iterations * training_words));
                    alpha = max(alpha, starting_alpha * 0.0001f);

                    if (config.verbose) {
                        printf("\rAlpha: %f  Progress: %.2f%%", alpha, 100.0 * words_processed /
                                        (max_iterations * training_words));
                        fflush(stdout);
                    }
                }
            }

            // stop when reaching the end of a chunk
            if (chunk_id < chunks.size() - 1 && infile.tellg() >= chunks[chunk_id + 1])
                break;
        }

        words_processed += word_count - last_count;
    }
}

int MonolingualModel::trainSentence(const string& sent, int sent_id) {
    auto nodes = getNodes(sent);  // same size as sent, OOV words are replaced by <UNK>

    // counts the number of words that are in the vocabulary
    int words = nodes.size() - count(nodes.begin(), nodes.end(), HuffmanNode::UNK);

    if (config.subsampling > 0) {
        subsample(nodes); // puts <UNK> tokens in place of the discarded tokens
    }

    if (nodes.empty()) {
        return words;
    }

    // remove <UNK> tokens
    nodes.erase(
        remove(nodes.begin(), nodes.end(), HuffmanNode::UNK),
        nodes.end());

    // Monolingual training
    for (int pos = 0; pos < nodes.size(); ++pos) {
        trainWord(nodes, pos, sent_id);
    }

    return words; // returns the number of words processed, for progress estimation
}

void MonolingualModel::trainWord(const vector<HuffmanNode>& nodes, int word_pos, int sent_id) {
    if (config.skip_gram) {
        trainWordSkipGram(nodes, word_pos, sent_id);
    } else {
        trainWordCBOW(nodes, word_pos, sent_id);
    }
}

void MonolingualModel::trainWordCBOW(const vector<HuffmanNode>& nodes, int word_pos, int sent_id) {
    int dimension = config.dimension;
    vec hidden(dimension, 0);
    HuffmanNode cur_node = nodes[word_pos];

    int this_window_size = 1 + multivec::rand() % config.window_size; // reduced window
    int count = 0;

    for (int pos = word_pos - this_window_size; pos <= word_pos + this_window_size; ++pos) {
        if (pos < 0 || pos >= nodes.size() || pos == word_pos) continue;
        hidden += input_weights[nodes[pos].index];
        ++count;
    }

    if (config.sent_vector) {
        hidden += sent_weights[sent_id];
        ++count;
    }

    if (count == 0) return;
    hidden /= count;

    vec error(dimension, 0);
    if (config.hierarchical_softmax) {
        error += hierarchicalUpdate(cur_node, hidden, alpha);
    }
    if (config.negative > 0) {
        error += negSamplingUpdate(cur_node, hidden, alpha);
    }

    // update input weights
    for (int pos = word_pos - this_window_size; pos <= word_pos + this_window_size; ++pos) {
        if (config.freeze) break;
        if (pos < 0 || pos >= nodes.size() || pos == word_pos) continue;
        input_weights[nodes[pos].index] += error;
    }

    if (config.sent_vector) {
        sent_weights[sent_id] += error;
    }
}

void MonolingualModel::trainWordSkipGram(const vector<HuffmanNode>& nodes, int word_pos, int sent_id) {
    int dimension = config.dimension;
    HuffmanNode input_word = nodes[word_pos]; // use this word to predict surrounding words

    int this_window_size = 1 + multivec::rand() % config.window_size;

    for (int pos = word_pos - this_window_size; pos <= word_pos + this_window_size; ++pos) {
        int p = pos;
        if (p == word_pos) continue;
        if (p < 0 || p >= nodes.size()) continue;
        HuffmanNode output_word = nodes[p];

        vec error(dimension, 0);
        if (config.hierarchical_softmax) {
            error += hierarchicalUpdate(output_word, input_weights[input_word.index], alpha);
        }
        if (config.negative > 0) {
            error += negSamplingUpdate(output_word, input_weights[input_word.index], alpha);
        }

        if (!config.freeze) {
            input_weights[input_word.index] += error;
        }
    }
}

vec MonolingualModel::negSamplingUpdate(const HuffmanNode& node, const vec& hidden, float alpha, bool update) {
    int dimension = config.dimension;
    vec temp(dimension, 0);

    for (int d = 0; d < config.negative + 1; ++d) {
        int label;
        const HuffmanNode* target;

        if (d == 0) { // 1 positive example
            target = &node;
            label = 1;
        } else { // n negative examples
            target = getRandomHuffmanNode();
            if (*target == node) continue;
            label = 0;
        }

        float x = hidden.dot(output_weights[target->index]);

        float pred;
        if (x >= MAX_EXP) {
            pred = 1;
        } else if (x <= -MAX_EXP) {
            pred = 0;
        } else {
            pred = sigmoid(x);
        }
        float error = alpha * (label - pred);

        temp += error * output_weights[target->index];

        if (update && !config.freeze) {
            output_weights[target->index] += error * hidden;
        }
    }

    return temp;
}

vec MonolingualModel::hierarchicalUpdate(const HuffmanNode& node, const vec& hidden,
        float alpha, bool update) {
    int dimension = config.dimension;
    vec temp(dimension, 0);

    for (int j = 0; j < node.code.size(); ++j) {
        int parent_index = node.parents[j];
        float x = hidden.dot(output_weights_hs[parent_index]);

        if (x <= -MAX_EXP || x >= MAX_EXP) {
            continue;
        }

        float pred = sigmoid(x);
        float error = -alpha * (pred - node.code[j]);

        temp += error * output_weights_hs[parent_index];

        if (update && !config.freeze) {
            output_weights_hs[parent_index] += error * hidden;
        }
    }

    return temp;
}

vector<pair<string, int>> MonolingualModel::getWords() const {
    vector<pair<string, int>> res;

    for (auto it = vocabulary.begin(); it != vocabulary.end(); ++it) {
        res.push_back({it->second.word, it->second.count});
    }

    return res;
}
