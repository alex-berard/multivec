#include "monolingual.hpp"
#include "serialization.hpp"

const HuffmanNode HuffmanNode::UNK;
std::mutex multivec::print_mutex;

void MonolingualModel::add_word_to_vocab(const string& word) {
    auto it = vocabulary.find(word);

    if (it != vocabulary.end()) {
        it->second.count++;
    } else {
        HuffmanNode node(static_cast<int>(vocabulary.size()), word);
        vocabulary.insert({word, node});
    }
}

void MonolingualModel::reduce_vocab() {
    int i = 0;
    for (auto it = vocabulary.begin(); it != vocabulary.end(); ) {
        if ((it->second.count) < config->min_count) {
            vocabulary.erase(it++);
        } else {
            it++->second.index = i++; // reassign indices in [0, vocabulary size - 1)
        }
    }
}

void MonolingualModel::read_vocab(const string& training_file) {
    ifstream infile(training_file);

    check_is_open(infile, training_file);
    check_is_non_empty(infile, training_file);

    vocabulary.clear();

    string word;
    while (infile >> word) {
        add_word_to_vocab(word);
    }

    if (config->verbose)
        std::cout << "Vocabulary size: " << vocabulary.size() << std::endl;

    reduce_vocab();

    if (config->verbose)
        std::cout << "Reduced vocabulary size: " << vocabulary.size() << std::endl;

    create_binary_tree();
    init_unigram_table();
}

void MonolingualModel::create_binary_tree() {
    vector<HuffmanNode*> heap;
    vector<HuffmanNode> parent_nodes;
    parent_nodes.reserve(vocabulary.size());

    for (auto it = vocabulary.begin(); it != vocabulary.end(); ++it) {
        heap.push_back(&it->second);
    }

    std::sort(heap.begin(), heap.end(), HuffmanNode::comp);

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

    assign_codes(heap.front(), {}, {});
}

void MonolingualModel::assign_codes(HuffmanNode* node, vector<int> code, vector<int> parents) const {
    if (node->is_leaf) {
        node->code = code;
        node->parents = parents;
    } else {
        parents.push_back(node->index);
        vector<int> code_left(code);
        code_left.push_back(0);
        vector<int> code_right(code);
        code_right.push_back(1);

        assign_codes(node->left, code_left, parents);
        assign_codes(node->right, code_right, parents);
    }
}

void MonolingualModel::init_unigram_table() {
    unigram_table.clear();
    vocab_word_count = 0;
    
    float power = 0.75; // weird word2vec tweak ('normal' value would be 1.0)
    float total_count = 0.0;
    for (auto it = vocabulary.begin(); it != vocabulary.end(); ++it) {
        vocab_word_count += it->second.count;
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

HuffmanNode* MonolingualModel::get_random_huffman_node() {
    auto index = multivec::rand(unigram_table.size());
    return unigram_table[index];
}

void MonolingualModel::init_net() {
    int v = static_cast<int>(vocabulary.size());
    int d = config->dimension;

    input_weights = mat(v, vec(d));

    for (size_t row = 0; row < v; ++row) {
        for (size_t col = 0; col < d; ++col) {
            input_weights[row][col] = (multivec::randf() - 0.5f) / d;
        }
    }

    output_weights = mat(v, vec(d));
}

void MonolingualModel::init_sent_weights() {
    int d = config->dimension;
    sent_weights = mat(training_lines, vec(d));

    for (size_t row = 0; row < training_lines; ++row) {
        for (size_t col = 0; col < d; ++col) {
            sent_weights[row][col] = (multivec::randf() - 0.5f) / d;
        }
    }
}

vector<HuffmanNode> MonolingualModel::get_nodes(const string& sentence) const {
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

vector<HuffmanNode> MonolingualModel::get_sorted_vocab() const {
    vector<HuffmanNode> sorted_vocab;
    for (auto it = vocabulary.begin(); it != vocabulary.end(); ++it) {
        sorted_vocab.push_back(it->second);
    }
    
    auto cmp = [](HuffmanNode a, HuffmanNode b) {
        return a.count > b.count or a.count == b.count and a.word.compare(b.word) <= 0;
    };
    
    std::sort(sorted_vocab.begin(), sorted_vocab.end(), cmp);
    return sorted_vocab;
}


/**
 * @brief Discard random nodes according to their frequency. The more frequent a word is, the more
 * likely it is to be discarded. Discarded nodes are replaced by UNK token.
 */
void MonolingualModel::subsample(vector<HuffmanNode>& nodes) const {
    for (auto it = nodes.begin(); it != nodes.end(); ++it) {
        auto node = *it;
        float f = static_cast<float>(node.count) / vocab_word_count; // frequency of this word
        float p = 1 - (1 + sqrt(f / config->subsampling)) * config->subsampling / f; // word2vec formula

        if (p >= multivec::randf()) {
            *it = HuffmanNode::UNK;
        }
    }
}

void MonolingualModel::save_vectors_bin(const string &filename, int policy, bool norm) const {
    if (config->verbose)
        std::cout << "Saving embeddings in binary format to " << filename << std::endl;

    ofstream outfile(filename, ios::binary | ios::out);
    check_is_open(outfile, filename);

    outfile << vocabulary.size() << " " << config->dimension << endl;

    auto sorted_vocab = get_sorted_vocab();
    
    for (auto it = sorted_vocab.begin(); it != sorted_vocab.end(); ++it) {
        string word = it->word;
        word.push_back(' ');
        vec embedding = word_vec(it->index, policy);
        if (norm) {
            embedding /= embedding.norm();
        }

        outfile.write(word.c_str(), word.size());
        outfile.write(reinterpret_cast<const char*>(embedding.data()), sizeof(float) * config->dimension);
        outfile << endl;
    }
}

void MonolingualModel::save_vectors(const string &filename, int policy, bool norm) const {
    if (config->verbose)
        std::cout << "Saving embeddings in text format to " << filename << std::endl;

    ofstream outfile(filename, ios::binary | ios::out);
    check_is_open(outfile, filename);

    outfile << vocabulary.size() << " " << config->dimension << endl;

    auto sorted_vocab = get_sorted_vocab();
    
    for (auto it = sorted_vocab.begin(); it != sorted_vocab.end(); ++it) {
        outfile << it->word << " ";
        vec embedding = word_vec(it->index, policy);
        if (norm) {
            embedding /= embedding.norm();
        }
        
        for (int c = 0; c < config->dimension; ++c) {
            outfile << embedding[c] << " ";
        }
        outfile << endl;
    }
}

void MonolingualModel::save_sent_vectors(const string &filename, bool norm) const {
    if (config->verbose)
        std::cout << "Saving sentence vectors in text format to " << filename << std::endl;

    ofstream outfile(filename, ios::binary | ios::out);
    check_is_open(outfile, filename);

    for (auto it = sent_weights.begin(); it != sent_weights.end(); ++it) {
        vec embedding = *it;
        if (norm) {
            embedding /= embedding.norm();
        }
        
        for (int c = 0; c < config->dimension; ++c) {
            outfile << embedding[c] << " ";
        }
        outfile << endl;
    }
}

void MonolingualModel::load(const string& filename) {
    if (config->verbose)
        std::cout << "Loading model from " << filename << std::endl;

    ifstream infile(filename);
    check_is_open(infile, filename);

    ::load(infile, *this);
    init_unigram_table();
    if (config->verbose)
        std::cout << "Vocabulary size: " << vocabulary.size() << std::endl;
}

void MonolingualModel::save(const string& filename) const {
    if (config->verbose)
        std::cout << "Saving model as " << filename << std::endl;

    ofstream outfile(filename);
    check_is_open(outfile, filename);

    ::save(outfile, *this);
}

vec MonolingualModel::word_vec(int index, int policy) const {
    if (config->hierarchical_softmax) // only input weights
    {
        return input_weights[index];
    }
    else if (policy == 1) // concat input and output
    {
        int d = config->dimension;
        vec res(d * 2);
        for (int c = 0; c < d; ++c) res[c] = input_weights[index][c];
        for (int c = 0; c < d; ++c) res[d + c] = output_weights[index][c];
        return res;
    }
    else if (policy == 2) // sum input and output
    {
        return input_weights[index] + output_weights[index];
    }
    else if (policy == 3) // only output weights
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
vec MonolingualModel::word_vec(const string& word, int policy) const {
    auto it = vocabulary.find(word);

    if (it == vocabulary.end()) {
        throw runtime_error("out of vocabulary");
    } else {
        return word_vec(it->second.index, policy);
    }
}

void MonolingualModel::sent_vectors(const string &input_file) {
    ifstream infile(input_file, ios::binary | ios::out);
    check_is_open(infile, input_file);

    string line;
    sent_weights.clear(); // TODO: random initialization?
    while(getline(infile, line)) {
        vec embedding(config->dimension, 0);
        try {
            embedding = sent_vec(line);
        } catch (runtime_error) {
            // in case of error (empty sentence, or all words are OOV), print a vector of zeros
        };
        
        sent_weights.push_back(embedding);
    }
}

/**
 * @brief Online paragraph vector on a given sentence. The parameters
 * of the model are frozen, while gradient descent is performed on this
 * single sentence. For batch paragraph vector, use the normal training
 * procedure with config->sent_vec set to true.
 *
 * @param sentence
 * @return sent_vec
 */
vec MonolingualModel::sent_vec(const string& sentence) {
    auto nodes = get_nodes(sentence);  // no subsampling here (TODO)
    nodes.erase(
        remove(nodes.begin(), nodes.end(), HuffmanNode::UNK),
        nodes.end()); // remove UNK tokens

    if (nodes.empty())
        throw runtime_error("too short sentence, or OOV words");

    vec sent_vec(config->dimension, 0);  // TODO: random initialization?
    for (int k = 0; k < config->iterations; ++k) {
        float alpha = config->alpha * (1 - static_cast<float>(k) / config->iterations);
        
        for (int pos = 0; pos < nodes.size(); ++pos) {
            train_word(nodes, pos, &sent_vec, alpha, false);
        }
    }

    return sent_vec;
}


/**
 * @brief Train model using given text file. Training is performed in parallel (each
 * thread reads one chunk of the file). Learning rate decays to zero.
 * Before calling this method, you need to call initialize or load, to initialize
 * the model parameters (vocabulary, unigram table, weights, etc.)
 *
 * @param training_file path of the training file (text file with one sentence per line)
 * @param initialize initialize the parameters of the model (vocabulary, unigram table,
 * weights). This parameter should be true, unless you are loading an existing model.
 **/
void MonolingualModel::train(const string& training_file, bool initialize) {
    std::cout << "Training file: " << training_file << std::endl;

    if (initialize) {
        if (config->verbose)
            std::cout << "Creating new model" << std::endl;

        // reads vocab and initializes unigram table
        read_vocab(training_file);
        init_net();
    } else if (vocab_word_count == 0) {
        throw runtime_error("the model needs to be initialized before training");
    }

    // TODO: also serialize training state
    words_processed = 0;

    // read file to find out the beginning of each chunk
    // also counts the number of lines and words
    auto chunks = chunkify(training_file, config->threads);

    if (config->verbose)
        std::cout << "Number of lines: " << training_lines
                  << ", words: " << training_words << std::endl;

    if (config->sent_vector)
        // no incremental training for paragraph vector
        init_sent_weights();

    high_resolution_clock::time_point start = high_resolution_clock::now();
    if (config->threads == 1) {
        train_chunk(training_file, chunks, 0);
    } else {
        vector<thread> threads;

        for (int i = 0; i < config->threads; ++i) {
            threads.push_back(thread(&MonolingualModel::train_chunk, this,
                training_file, chunks, i));
        }

        for (auto it = threads.begin(); it != threads.end(); ++it) {
            it->join();
        }
    }
    high_resolution_clock::time_point end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();

    if (config->verbose)
        std::cout << std::endl;

    std::cout << "Training time: " << static_cast<float>(duration) / 1000000 << std::endl;
}

/**
 * @brief Divide a given file into chunks with the same number of lines each
 *
 * @param filename path of the file
 * @param n_chunks number of chunks
 * @return starting position (in bytes) of each chunk
 */
vector<long long> MonolingualModel::chunkify(const string& filename, int n_chunks) {
    ifstream infile(filename);
    check_is_open(infile, filename);
    check_is_non_empty(infile, filename);

    vector<long long> chunks;
    vector<pair<long long, long long>> positions;
    long long words = 0;
    long long pos = 0;
    string line;
    while (getline(infile, line)) {
        words += split(line).size();            // total number of words up to (including) this line
        positions.push_back({pos, words}); // starting position of this line
        pos = infile.tellg();
    }

    training_lines = positions.size();
    training_words = words;
    int chunk_size = training_words / n_chunks;  // number of lines in each chunk

    int chunk_id = 0;
    for (auto it = positions.begin(); it != positions.end() and chunk_id < n_chunks; ++it) {
        if (it->second >= chunk_size * chunk_id) {
            chunks.push_back(it->first);
            chunk_id++;
        }
    }
    return chunks;
}

void MonolingualModel::train_chunk(const string& training_file, const vector<long long>& chunks, int chunk_id) {
    ifstream infile(training_file);
    float starting_alpha = config->alpha;
    float alpha = starting_alpha;
    int max_iterations = config->iterations;

    check_is_open(infile, training_file);
    check_is_non_empty(infile, training_file);
    
    for (int k = 0; k < max_iterations; ++k) {
        int word_count = 0;

        infile.clear();
        infile.seekg(chunks[chunk_id], infile.beg);

        int sent_id = chunk_id * (training_lines / chunks.size());

        string sent;
        while (getline(infile, sent)) {
            vec* sent_vec = config->sent_vector ? &sent_weights[sent_id] : nullptr;
            word_count += train_sentence(sent, sent_vec, alpha);  // asynchronous update (possible race conditions)
            sent_id++;
                
            // update learning rate
            if (word_count >= 10000) {
                std::lock_guard<std::mutex> guard(multivec::print_mutex);
                words_processed += word_count;
                word_count = 0;

                // decreasing learning rate
                alpha = starting_alpha * (1 - static_cast<float>(words_processed) / (max_iterations * training_words));
                if (alpha < starting_alpha * 0.0001f) {
                    alpha = starting_alpha * 0.0001f;
                }

                if (config->verbose) {
                    printf("\rAlpha: %f  Progress: %.2f%%", static_cast<float>(alpha), 100.0 * words_processed /
                                    (max_iterations * training_words));
                    fflush(stdout);
                }
            }

            // stop when reaching the end of a chunk
            if (chunk_id < chunks.size() - 1 && infile.tellg() >= chunks[chunk_id + 1])
                break;
        }

        words_processed += word_count;
    }
}

int MonolingualModel::train_sentence(const string& sent, vec* sent_vec, float alpha) {
    auto nodes = get_nodes(sent);  // same size as sent, OOV words are replaced by <UNK>
    // counts the number of words that are in the vocabulary
    // int words = nodes.size() - count(nodes.begin(), nodes.end(), HuffmanNode::UNK);
    int words = nodes.size();

    if (config->subsampling > 0) {  // disable with sent vector?
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
        train_word(nodes, pos, sent_vec, alpha);
    }

    return words; // returns the number of words processed, for progress estimation
}

void MonolingualModel::train_word(const vector<HuffmanNode>& nodes, int word_pos, vec* sent_vec, float alpha, bool update) {
    if (config->skip_gram and sent_vec != nullptr) {  // DBOW model
        train_word_DBOW(nodes, word_pos, sent_vec, alpha, update);
    } else if (config->skip_gram) {
        train_word_skip_gram(nodes, word_pos, sent_vec, alpha, update);
    } else {
        train_word_CBOW(nodes, word_pos, sent_vec, alpha, update);
    }
}

void MonolingualModel::train_word_CBOW(const vector<HuffmanNode>& nodes, int word_pos, vec* sent_vec, float alpha, bool update) {
    vec hidden(config->dimension, 0);
    const HuffmanNode& cur_node = nodes[word_pos];

    int this_window_size = 1 + multivec::rand(config->window_size); // reduced window
    int count = 0;

    for (int pos = word_pos - this_window_size; pos <= word_pos + this_window_size; ++pos) {
        if (pos < 0 || pos >= nodes.size() || pos == word_pos) continue;
        hidden += input_weights[nodes[pos].index];
        ++count;
    }

    if (sent_vec != nullptr) {
        hidden += *sent_vec;
        ++count;
    }

    if (count == 0) return;
    
    if (config->no_average) {
        count = 1;
    }
    
    hidden /= count;

    vec error(config->dimension, 0);
    if (config->hierarchical_softmax) {
        error += hierarchical_update(cur_node, hidden, alpha, update);
    } else if (config->negative > 0) {
        error += neg_sampling_update(cur_node, hidden, alpha, update);
    }
    
    // update input weights
    if (update) {
        for (int pos = word_pos - this_window_size; pos <= word_pos + this_window_size; ++pos) {
            if (pos < 0 || pos >= nodes.size() || pos == word_pos) continue;
            input_weights[nodes[pos].index] += error / count;
        }
    }

    if (sent_vec != nullptr) {
        *sent_vec += error / count;
    }
}

void MonolingualModel::train_word_DBOW(const vector<HuffmanNode>& nodes, int word_pos, vec* sent_vec, float alpha, bool update) {
    const HuffmanNode& output_word = nodes[word_pos];

    vec error(config->dimension, 0);
    if (config->hierarchical_softmax) {
        error += hierarchical_update(output_word, *sent_vec, alpha, update);
    } else if (config->negative > 0) {
        error += neg_sampling_update(output_word, *sent_vec, alpha, update);
    }

    *sent_vec += error;
}

void MonolingualModel::train_word_skip_gram(const vector<HuffmanNode>& nodes, int word_pos, vec* sent_vec, float alpha, bool update) {
    const HuffmanNode& input_word = nodes[word_pos]; // use this word to predict surrounding words

    int this_window_size = 1 + multivec::rand(config->window_size);

    for (int pos = word_pos - this_window_size; pos <= word_pos + this_window_size; ++pos) {
        int p = pos;
        if (p == word_pos) continue;
        if (p < 0 || p >= nodes.size()) continue;
        const HuffmanNode& output_word = nodes[p];
        
        vec error(config->dimension, 0);
        if (config->hierarchical_softmax) {
            error += hierarchical_update(output_word, input_weights[input_word.index], alpha, update);
        } else  if (config->negative > 0) {
            error += neg_sampling_update(output_word, input_weights[input_word.index], alpha, update);
        }

        if (update) {
            input_weights[input_word.index] += error;
        }
    }
}

vec MonolingualModel::neg_sampling_update(const HuffmanNode& node, const vec& hidden, float alpha, bool update) {
    vec temp(config->dimension, 0);

    for (int d = 0; d < config->negative + 1; ++d) {
        int label;
        const HuffmanNode* target;

        if (d == 0) { // 1 positive example
            target = &node;
            label = 1;
        } else { // n negative examples
            target = get_random_huffman_node();
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

        if (update) {
            output_weights[target->index] += error * hidden;
        }
    }

    return temp;
}

vec MonolingualModel::hierarchical_update(const HuffmanNode& node, const vec& hidden, float alpha, bool update) {
    vec temp(config->dimension, 0);

    for (int j = 0; j < node.code.size(); ++j) {
        int parent_index = node.parents[j];
        float x = hidden.dot(output_weights[parent_index]);

        if (x <= -MAX_EXP || x >= MAX_EXP) {
            continue;
        }

        float pred = sigmoid(x);
        float error = -alpha * (pred - node.code[j]);

        temp += error * output_weights[parent_index];

        if (update) {
            output_weights[parent_index] += error * hidden;
        }
    }

    return temp;
}

vector<pair<string, int>> MonolingualModel::get_word_counts() const {
    vector<pair<string, int>> res;
    auto sorted_vocab = get_sorted_vocab();

    for (auto it = sorted_vocab.begin(); it != sorted_vocab.end(); ++it) {
        res.push_back({it->word, it->count});
    }

    return res;
}
