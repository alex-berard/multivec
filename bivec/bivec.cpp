#include "bivec.h"
#include "serialization.h"

void BilingualModel::train(const string& src_file, const string& trg_file) {
    if (config.verbose)
        cout << "Reading vocabulary" << endl;

    src_model.readVocab(src_file);
    trg_model.readVocab(trg_file);
    src_model.initNet();
    trg_model.initNet();
    total_word_count = 0;
    alpha = config.starting_alpha;

    // read files to find out the beginning of each chunk
    auto src_chunks = MonolingualModel::chunkify(src_file, config.n_threads);
    auto trg_chunks = MonolingualModel::chunkify(trg_file, config.n_threads);

    if (config.verbose)
        cout << "Starting training" << endl;

    vector<thread> threads;

    for (int i = 0; i < config.n_threads; ++i) {
        threads.push_back(thread(&BilingualModel::trainChunk, this,
            src_file, trg_file, src_chunks, trg_chunks, i));
    }

    for (auto it = threads.begin(); it != threads.end(); ++it) {
        it->join();
    }

    if (config.verbose)
        cout << endl << "Finished training" << endl;
}

void BilingualModel::trainChunk(const string& src_file,
                                const string& trg_file,
                                const vector<long long>& src_chunks,
                                const vector<long long>& trg_chunks,
                                int chunk_id) {
    ifstream src_infile(src_file);
    ifstream trg_infile(trg_file);
    float starting_alpha = config.starting_alpha;
    int max_iterations = config.max_iterations;
    long long train_words = src_model.train_words + trg_model.train_words;

    if (!src_infile.is_open()) {
        throw "couldn't open file " + src_file;
    }

    if (!trg_infile.is_open()) {
        throw "couldn't open file " + trg_file;
    }

    for (int k = 0; k < max_iterations; ++k) {
        int word_count = 0, last_count = 0;

        src_infile.clear();
        trg_infile.clear();
        src_infile.seekg(src_chunks[chunk_id], src_infile.beg);
        trg_infile.seekg(trg_chunks[chunk_id], trg_infile.beg);

        string src_sent, trg_sent;
        while (getline(src_infile, src_sent) && getline(trg_infile, trg_sent)) {
            word_count += trainSentence(src_sent, trg_sent);

            // update learning rate
            if (word_count - last_count > 10000) {
                total_word_count += word_count - last_count; // asynchronous update
                last_count = word_count;

                alpha = starting_alpha * (1 - static_cast<float>(total_word_count) / (max_iterations * train_words));
                alpha = std::max(alpha, starting_alpha * 0.0001f);

                if (config.verbose) {
                    printf("\rAlpha: %f  Progress: %.2f%%", alpha, 100.0 * total_word_count /
                                    (max_iterations * train_words));
                    fflush(stdout);
                }
            }

            // stop when reaching the end of a chunk
            if (chunk_id < src_chunks.size() - 1 && src_infile.tellg() >= src_chunks[chunk_id + 1])
                break;
        }

        total_word_count += word_count - last_count;
    }
}

vector<int> BilingualModel::uniformAlignment(const vector<HuffmanNode>& src_nodes,
                                             const vector<HuffmanNode>& trg_nodes) {
    // TODO: add GIZA alignment
    vector<int> alignment; // index = position in src_nodes, value = position in trg_nodes (or -1)

    vector<int> trg_mapping; // maps positions in trg_sent to positions in trg_nodes (or -1)
    int k = 0;
    for (auto it = trg_nodes.begin(); it != trg_nodes.end(); ++it) {
        trg_mapping.push_back(*it == HuffmanNode::UNK ? -1 : k++);
    }

    for (int i = 0; i < src_nodes.size(); ++i) {
        int j = i * trg_nodes.size() / src_nodes.size();

        if (src_nodes[i] != HuffmanNode::UNK) {
            alignment.push_back(trg_mapping[j]);
        }
    }

    return alignment;
}

int BilingualModel::trainSentence(const string& src_sent, const string& trg_sent) {
    auto src_nodes = src_model.getNodes(src_sent);  // same size as src_sent, OOV words are replaced by <UNK>
    auto trg_nodes = trg_model.getNodes(trg_sent);

    // counts the number of words that are in the vocabulary
    int words = 0;
    words += src_nodes.size() - count(src_nodes.begin(), src_nodes.end(), HuffmanNode::UNK);
    words += trg_nodes.size() - count(trg_nodes.begin(), trg_nodes.end(), HuffmanNode::UNK);

    if (config.subsampling > 0) {
        src_model.subsample(src_nodes); // puts <UNK> tokens in place of the discarded tokens
        trg_model.subsample(trg_nodes);
    }

    if (src_nodes.empty() || trg_nodes.empty()) {
        return words;
    }

    // The <UNK> tokens are necessary to perform the alignment (the nodes vector should have the same size
    // as the original sentence)
    auto alignment = uniformAlignment(src_nodes, trg_nodes);

    // remove <UNK> tokens
    src_nodes.erase(
        remove(src_nodes.begin(), src_nodes.end(), HuffmanNode::UNK),
        src_nodes.end());
    trg_nodes.erase(
        remove(trg_nodes.begin(), trg_nodes.end(), HuffmanNode::UNK),
        trg_nodes.end());

    // Monolingual training
    for (int src_pos = 0; src_pos < src_nodes.size(); ++src_pos) {
        trainWord(src_model, src_model, src_nodes, src_nodes, src_pos, src_pos, alpha);
    }

    for (int trg_pos = 0; trg_pos < trg_nodes.size(); ++trg_pos) {
        trainWord(trg_model, trg_model, trg_nodes, trg_nodes, trg_pos, trg_pos, alpha);
    }

    if (config.bi_weight == 0)
        return words;

    // Bilingual training
    for (int src_pos = 0; src_pos < src_nodes.size(); ++src_pos) {
        // 1-1 mapping between src_nodes and trg_nodes
        int trg_pos = alignment[src_pos];

        if (trg_pos != -1) { // target word isn't OOV
            trainWord(src_model, trg_model, src_nodes, trg_nodes, src_pos, trg_pos, alpha * config.bi_weight);
            trainWord(trg_model, src_model, trg_nodes, src_nodes, trg_pos, src_pos, alpha * config.bi_weight);
        }
    }

    return words; // returns the number of words processed, for progress estimation
}

void BilingualModel::trainWord(MonolingualModel& src_model, MonolingualModel& trg_model,
                               const vector<HuffmanNode>& src_nodes, const vector<HuffmanNode>& trg_nodes,
                               int src_pos, int trg_pos, float alpha) {

    if (config.skip_gram) {
        return trainWordSkipGram(src_model, trg_model, src_nodes, trg_nodes, src_pos, trg_pos, alpha);
    } else {
        return trainWordCBOW(src_model, trg_model, src_nodes, trg_nodes, src_pos, trg_pos, alpha);
    }
}

void BilingualModel::trainWordCBOW(MonolingualModel& src_model, MonolingualModel& trg_model,
                                   const vector<HuffmanNode>& src_nodes, const vector<HuffmanNode>& trg_nodes,
                                   int src_pos, int trg_pos, float alpha) {
    // Trains the model by predicting a source node from its aligned context in the target sentence.
    // This function can be used in the reverse direction just by reversing the arguments. Likewise,
    // for monolingual training, use the same values for source and target.

    // 'src_pos' is the position in the source sentence of the current node to predict
    // 'trg_pos' is the position of the corresponding node in the target sentence
    int dimension = config.dimension;
    vec hidden(dimension, 0);
    HuffmanNode cur_node = src_nodes[src_pos];

    int this_window_size = 1 + MonolingualModel::rand() % config.window_size;
    int count = 0;

    for (int pos = trg_pos - this_window_size; pos <= trg_pos + this_window_size; ++pos) {
        if (pos < 0 || pos >= trg_nodes.size() || pos == trg_pos) continue;
        for (int c = 0; c < dimension; ++c) {
          hidden[c] += trg_model.syn0[trg_nodes[pos].index][c];
        }
        ++count;
    }

    if (count == 0) return;
    for (int c = 0; c < dimension; ++c) {
        hidden[c] /= count;
    }

    vec error; // compute error & update output weights
    if (config.hierarchical_softmax) {
        error = src_model.hierarchicalUpdate(cur_node, hidden, alpha);
    } else {
        error = src_model.negSamplingUpdate(cur_node, hidden, alpha);
    }

    // Update input weights
    for (int pos = trg_pos - this_window_size; pos <= trg_pos + this_window_size; ++pos) {
        if (pos < 0 || pos >= trg_nodes.size() || pos == trg_pos) continue;
        for (int c = 0; c < dimension; ++c) {
            trg_model.syn0[trg_nodes[pos].index][c] += error[c];
        }
    }
}

void BilingualModel::trainWordSkipGram(MonolingualModel& src_model, MonolingualModel& trg_model,
                                       const vector<HuffmanNode>& src_nodes, const vector<HuffmanNode>& trg_nodes,
                                       int src_pos, int trg_pos, float alpha) {
    HuffmanNode input_word = src_nodes[src_pos];

    int this_window_size = 1 + MonolingualModel::rand() % config.window_size;

    for (int pos = trg_pos - this_window_size; pos <= trg_pos + this_window_size; ++pos) {
        if (pos < 0 || pos >= trg_nodes.size() || pos == trg_pos) continue;
        HuffmanNode output_word = trg_nodes[pos];

        vec error;
        if (config.hierarchical_softmax) {
            error = trg_model.hierarchicalUpdate(output_word, src_model.syn0[input_word.index], alpha);
        } else {
            error = trg_model.negSamplingUpdate(output_word, src_model.syn0[input_word.index], alpha);
        }

        for (int c = 0; c < config.dimension; ++c) {
            src_model.syn0[input_word.index][c] += error[c];
        }
    }
}

void BilingualModel::load(const string& filename) {
    ifstream infile(filename);

    if (!infile.is_open()) {
        throw runtime_error("couldn't open file " + filename);
    }

    if (config.verbose)
        cout << "Loading model" << endl;

    boost::archive::text_iarchive ia(infile);
    ia >> *this;
}

void BilingualModel::save(const string& filename) const {
    ofstream outfile(filename);

    if (!outfile.is_open()) {
        throw runtime_error("couldn't open file " + filename);
    }

    if (config.verbose)
        cout << "Saving model" << endl;

    boost::archive::text_oarchive oa(outfile);
    oa << *this;
}
