#include "word2vec.h"

vector<string> split(const string& sentence) {
    istringstream iss(sentence);
    vector<string> words;
    string word;

    while (iss >> word) {
        words.push_back(word);
    }

    return words;
}

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
    train_words = 0;

    int i = 0;
    for (auto it = vocabulary.begin(); it != vocabulary.end(); ) {
        if ((it->second.count) < config.min_count) {
            vocabulary.erase(it++);
        } else {
            train_words += it->second.count;
            it++->second.index = i++; // reassign indices in [0, vocabulary size - 1)
        }
    }
}

void MonolingualModel::readVocab(const string& training_file) {
    ifstream infile(training_file);

    if (!infile.is_open()) {
        throw "couldn't open file " + training_file;
    }

    vocabulary.clear();

    string word;
    while (infile >> word) {
        addWordToVocab(word);
    }

    reduceVocab();
    createBinaryTree();
    initUnigramTable();
}

bool compHuffmanNodes(const HuffmanNode* v1, const HuffmanNode* v2) {
    return (v1->count) > (v2->count);
}

void MonolingualModel::createBinaryTree() {
    vector<HuffmanNode*> heap;
    vector<HuffmanNode> parent_nodes;
    parent_nodes.reserve(vocabulary.size());

    for (auto it = vocabulary.begin(); it != vocabulary.end(); ++it) {
        heap.push_back(&it->second);
    }

    sort(heap.begin(), heap.end(), compHuffmanNodes);

    for (int i = 0; heap.size() > 1; i++) {
        HuffmanNode* left = heap.back();
        heap.pop_back();

        HuffmanNode* right = heap.back();
        heap.pop_back();

        parent_nodes.push_back({i, left, right});

        HuffmanNode* parent = &parent_nodes.back();
        auto it = lower_bound(heap.begin(), heap.end(), parent, compHuffmanNodes);
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

    for (auto it = vocabulary.begin(); it != vocabulary.end(); ++it) {
        int count = it->second.count;
        float f = static_cast<float>(count) / train_words; // frequency of this word

        int d = static_cast<int>(f * UNIGRAM_TABLE_SIZE);
        for (int i = 0; i < d; ++i) {
            unigram_table.push_back(&it->second);
        }
    }
}

HuffmanNode* MonolingualModel::getRandomHuffmanNode() {
    int index = rand() % unigram_table.size();
    return unigram_table[index];
}

void MonolingualModel::initNet() {
    int v = static_cast<int>(vocabulary.size());
    int d = config.dimension;

    syn0 = mat(v, vec(d));

    for (int row = 0; row < v; ++row) {
        for (int col = 0; col < d; ++col) {
            syn0[row][col] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) / d;
        }
    }

    syn1 = mat(v, vec(d));
}

vector<HuffmanNode> MonolingualModel::getNodes(const string& sentence) const {
    vector<HuffmanNode> nodes;
    auto words = split(sentence);

    for (auto word = words.begin(); word != words.end(); ++word) {
        auto it = vocabulary.find(*word);
        HuffmanNode node = HuffmanNode::UNK;

        if (it != vocabulary.end()) {
            node = it->second;
        }

        nodes.push_back(node);
    }

    return nodes;
}

void MonolingualModel::subsample(vector<HuffmanNode>& nodes) const {
    for (auto it = nodes.begin(); it != nodes.end(); ++it) {
        auto node = *it;
        float f = static_cast<float>(node.count) / train_words; // frequency of this word
        //float p = 1 - sqrt(config.sampling / f); // formula used in the word2vec paper
        float p = 1 - (1 + sqrt(f / config.sampling)) * config.sampling / f; // formula used in word2vec
        float r = static_cast<float>(rand()) / RAND_MAX;

        // the higher the frequency the most likely to be discarded (p can be less than 0)
        if (p >= r) {
            *it = HuffmanNode::UNK;
        }
    }
}

void MonolingualModel::saveEmbeddings(const string& filename) const {
    ofstream outfile(filename, ios::binary | ios::out);

    if (!outfile.is_open()) {
        throw "couldn't open file " + filename;
    }

    outfile << vocabulary.size() << " " << config.dimension << endl;

    for (auto it = vocabulary.begin(); it != vocabulary.end(); ++it) {
        string word = string(it->second.word);
        word.push_back(' ');
        int index = it->second.index;

        outfile.write(word.c_str(), word.size());
        outfile.write(reinterpret_cast<const char*>(syn0[index].data()), sizeof(float) * config.dimension);
        outfile << endl;
    }
}

HuffmanNode MonolingualModel::loadWord(ifstream& infile) {
    int arr[5];
    char buffer[1024]; // fixed-size, boo!

    infile.read(reinterpret_cast<char*>(arr), sizeof(int) * 5);

    int index = arr[0];
    int count = arr[1];
    int word_size = arr[2];
    int parent_size = arr[3];
    int code_size = arr[4];

    infile.read(buffer, word_size);

    HuffmanNode node(index, string(buffer));
    node.count = count;

    int* buffer_ptr = reinterpret_cast<int*>(buffer);

    infile.read(buffer, sizeof(int) * parent_size);
    node.parents = vector<int>(buffer_ptr, buffer_ptr + parent_size);

    infile.read(buffer, sizeof(int) * code_size);
    node.code = vector<int>(buffer_ptr, buffer_ptr + code_size);

    return node;
}

void MonolingualModel::saveWord(ofstream& outfile, const HuffmanNode& node) const {
    int arr[] = {node.index,
                 node.count,
                 static_cast<int>(node.word.size() + 1), // include the '\0' character
                 static_cast<int>(node.parents.size()),
                 static_cast<int>(node.code.size())};

    outfile.write(reinterpret_cast<char*>(arr), sizeof(int) * 5);
    outfile.write(node.word.c_str(), node.word.size() + 1);

    outfile.write(reinterpret_cast<const char*>(node.parents.data()), sizeof(int) * node.parents.size());
    outfile.write(reinterpret_cast<const char*>(node.code.data()), sizeof(int) * node.code.size());
}

void MonolingualModel::load(const string& filename) {
    ifstream infile(filename, ios::binary);

    if (!infile.is_open()) {
        throw "couldn't open file " + filename;
    }

    int vocab_size;

    infile.read(reinterpret_cast<char*>(&vocab_size), sizeof(int));
    infile.read(reinterpret_cast<char*>(&train_words), sizeof(long long));
    infile.read(reinterpret_cast<char*>(&config.dimension), sizeof(int));

    vocabulary.clear();

    for (int i = 0; i < vocab_size; ++i) {
        HuffmanNode node = loadWord(infile);
        vocabulary.insert({node.word, node});
    }

    initUnigramTable();

    syn0.resize(vocab_size);
    for (auto it = syn0.begin(); it < syn0.end(); ++it) {
        it->resize(config.dimension);
        infile.read(reinterpret_cast<char*>(it->data()), sizeof(float) * config.dimension);
    }

    syn1.resize(vocab_size);
    for (auto it = syn1.begin(); it < syn1.end(); ++it) {
        it->resize(config.dimension);
        infile.read(reinterpret_cast<char*>(it->data()), sizeof(float) * config.dimension);
    }
}

void MonolingualModel::save(const string& filename) const {
    ofstream outfile(filename, ios::binary | ios::out);

    if (!outfile.is_open()) {
        throw "couldn't open file " + filename;
    }

    int vocab_size = vocabulary.size();

    outfile.write(reinterpret_cast<char*>(&vocab_size), sizeof(int));
    outfile.write(reinterpret_cast<const char*>(&train_words), sizeof(long long));
    outfile.write(reinterpret_cast<const char*>(&config.dimension), sizeof(int));

    for (auto it = vocabulary.begin(); it != vocabulary.end(); ++it) {
        saveWord(outfile, it->second);
    }

    for (auto it = syn0.begin(); it < syn0.end(); ++it) {
        outfile.write(reinterpret_cast<const char*>(it->data()), sizeof(float) * config.dimension);
    }

    for (auto it = syn1.begin(); it < syn1.end(); ++it) {
        outfile.write(reinterpret_cast<const char*>(it->data()), sizeof(float) * config.dimension);
    }
}

vec MonolingualModel::wordVec(const string& word) const {
    auto it = vocabulary.find(word);

    if (it == vocabulary.end()) {
        throw "out of vocabulary";
    } else {
        return syn0[it->second.index];
    }
}

vec MonolingualModel::sentVec(const string& sentence) {
    int dimension = config.dimension;
    float alpha = config.starting_alpha;  // TODO: decreasing learning rate

    auto nodes = getNodes(sentence);  // no subsampling here
    nodes.erase(
        remove(nodes.begin(), nodes.end(), HuffmanNode::UNK),
        nodes.end()); // remove UNK tokens

    if (nodes.empty())
        throw "too short sentence, or OOV words";

    vec sent_vec(dimension, 0);

    for (int k = 0; k < config.max_iterations; ++k) {
        for (int word_pos = 0; word_pos < nodes.size(); ++word_pos) {
            vec hidden(dimension, 0);
            HuffmanNode cur_node = nodes[word_pos];

            int this_window_size = 1 + rand() % config.window_size;
            int count = 0;

            for (int pos = word_pos - this_window_size; pos <= word_pos + this_window_size; ++pos) {
                if (pos < 0 || pos >= nodes.size() || pos == word_pos) continue;
                for (int c = 0; c < dimension; ++c) {
                    hidden[c] += syn0[nodes[pos].index][c];
                }
                ++count;
            }

            if (count == 0) continue;
            for (int c = 0; c < dimension; ++c) {
                hidden[c] += sent_vec[c];
            }

            for (int c = 0; c < dimension; ++c) {
                hidden[c] /= count + 1; //TODO this or (hidden / count) + sent_vec?
            }

            vec temp_vec;
            if (config.negative_sampling) {
                temp_vec = hierarchicalUpdate(cur_node, hidden, alpha, false);
            } else {
                temp_vec = negSamplingUpdate(cur_node, hidden, alpha, false);
            }
            
            for (int c = 0; c < dimension; ++c) {
                sent_vec[c] += temp_vec[c];
            }
        }
    }

    return sent_vec;
}

void MonolingualModel::train(const string& training_file) {
    if (config.debug)
        config.print();

    if (config.debug)
        cout << "Reading vocabulary" << endl;

    train_words = 0;
    readVocab(training_file); // TODO: no automatic call to readVocab and initNet (allows for incremental training)
    initNet();
    total_word_count = 0;
    alpha = config.starting_alpha;

    // read files to find out the beginning of each chunk
    auto chunks = chunkify(training_file, config.n_threads);

    if (config.debug)
        cout << "Starting training" << endl;

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

    if (config.debug)
        cout << endl << "Finished training" << endl;
}

vector<long long> MonolingualModel::chunkify(const string& filename, int n_chunks) {
    ifstream infile(filename);

    if (!infile.is_open()) {
        throw "couldn't open file " + filename;
    }

    vector<long long> chunks;
    vector<long long> line_positions;

    string line;
    do {
        line_positions.push_back(infile.tellg());
    } while (getline(infile, line));

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
        throw "couldn't open file " + training_file;
    }

    for (int k = 0; k < max_iterations; ++k) {
        int word_count = 0, last_count = 0;

        infile.clear();
        infile.seekg(chunks[chunk_id], infile.beg);

        string sent;
        while (getline(infile, sent)) {
            word_count += trainSentence(sent); // asynchronous update (possible race conditions)

            // update learning rate
            if (word_count - last_count > 10000) {
                total_word_count += word_count - last_count; // asynchronous update
                last_count = word_count;

                alpha = starting_alpha * (1 - static_cast<float>(total_word_count) / (max_iterations * train_words));
                alpha = std::max(alpha, starting_alpha * 0.0001f);

                if (config.debug) {
                    printf("\rAlpha: %f  Progress: %.2f%%", alpha, 100.0 * total_word_count /
                                    (max_iterations * train_words));
                    fflush(stdout);
                }
            }

            // stop when reaching the end of a chunk
            if (chunk_id < chunks.size() - 1 && infile.tellg() >= chunks[chunk_id + 1])
                break;
        }

        total_word_count += word_count - last_count;
    }
}

int MonolingualModel::trainSentence(const string& sent) {
    auto nodes = getNodes(sent);  // same size as sent, OOV words are replaced by <UNK>

    // counts the number of words that are in the vocabulary
    int words = nodes.size() - count(nodes.begin(), nodes.end(), HuffmanNode::UNK);

    if (config.sampling > 0) {
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
        trainWord(nodes, pos);
    }

    return words; // returns the number of words processed, for progress estimation
}

void MonolingualModel::trainWord(const vector<HuffmanNode>& nodes, int word_pos) {
    if (config.skip_gram) {
        trainWordSkipGram(nodes, word_pos);
    } else {
        trainWordCBOW(nodes, word_pos);
    }
}

void MonolingualModel::trainWordCBOW(const vector<HuffmanNode>& nodes, int word_pos) {
    int dimension = config.dimension;
    vec hidden(dimension, 0);
    HuffmanNode cur_node = nodes[word_pos];

    int this_window_size = 1 + rand() % config.window_size;
    int count = 0;

    for (int pos = word_pos - this_window_size; pos <= word_pos + this_window_size; ++pos) {
        if (pos < 0 || pos >= nodes.size() || pos == word_pos) continue;
        for (int c = 0; c < dimension; ++c) {
            hidden[c] += syn0[nodes[pos].index][c];
        }
        ++count;
    }

    if (count == 0) return;
    for (int c = 0; c < dimension; ++c) {
        hidden[c] /= count;
    }

    vec error(dimension, 0);
    if (config.negative_sampling) {
        error = negSamplingUpdate(cur_node, hidden, alpha);
    } else {
        error = hierarchicalUpdate(cur_node, hidden, alpha);
    }

    // update input weights
    for (int pos = word_pos - this_window_size; pos <= word_pos + this_window_size; ++pos) {
        if (pos < 0 || pos >= nodes.size() || pos == word_pos) continue;
        for (int c = 0; c < dimension; ++c) {
            syn0[nodes[pos].index][c] += error[c];
        }
    }
}

void MonolingualModel::trainWordSkipGram(const vector<HuffmanNode>& nodes, int word_pos) {
    HuffmanNode input_word = nodes[word_pos]; // use this word to predict surrounding words

    int this_window_size = 1 + rand() % config.window_size;

    for (int pos = word_pos - this_window_size; pos <= word_pos + this_window_size; ++pos) {
        if (pos < 0 || pos >= nodes.size() || pos == word_pos) continue;
        HuffmanNode output_word = nodes[pos];

        vec error(config.dimension, 0);
        if (config.negative_sampling) {
            error = negSamplingUpdate(output_word, syn0[input_word.index], alpha);
        } else {
            error = hierarchicalUpdate(output_word, syn0[input_word.index], alpha);
        }

        for (int c = 0; c < config.dimension; ++c) {
            syn0[input_word.index][c] += error[c];
        }
    }
}

vec MonolingualModel::negSamplingUpdate(const HuffmanNode& node, const vec& hidden,
        float alpha, bool update) {
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

        float x = 0;
        for (int c = 0; c < dimension; ++c) {
            x += hidden[c] * syn1[target->index][c];
        }

        float pred;
        if (x >= MAX_EXP) {
            pred = 1;
        } else if (x <= -MAX_EXP) {
            pred = 0;
        } else {
            pred = sigmoid(x);
        }
        float error = alpha * (label - pred);

        for (int c = 0; c < dimension; ++c) {
            temp[c] += error * syn1[target->index][c];
        }
        if (update) {
            for (int c = 0; c < dimension; ++c) {
                syn1[target->index][c] += error * hidden[c];
            }
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
        float x = 0;
        for (int c = 0; c < dimension; ++c) {
            x += hidden[c] * syn1[parent_index][c];
        }

        if (x <= -MAX_EXP || x >= MAX_EXP) {
            continue;
        }

        float pred = sigmoid(x);
        float error = -alpha * (pred - node.code[j]);

        for (int c = 0; c < dimension; ++c) {
            temp[c] += error * syn1[parent_index][c];
        }
        if (update) {
            for (int c = 0; c < dimension; ++c) {
                syn1[parent_index][c] += error * hidden[c];
            }
        }
    }

    return temp;
}