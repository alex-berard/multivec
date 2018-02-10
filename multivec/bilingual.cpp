#include "bilingual.hpp"
#include "serialization.hpp"

void BilingualModel::train(const string& src_file, const string& trg_file, const string& align_file, bool initialize) {
    std::cout << "Training files: " << src_file << ", " << trg_file << std::endl;

    if (initialize) {
        if (config->verbose)
            std::cout << "Creating new model" << std::endl;

        src_model.read_vocab(src_file);
        trg_model.read_vocab(trg_file);
        src_model.init_net();
        trg_model.init_net();
    } else {
        // TODO: check that initialization is fine
    }

    words_processed = 0;
    alpha = config->alpha;

    // read files to find out the beginning of each chunk
    auto src_chunks = src_model.chunkify(src_file, config->threads);
    auto trg_chunks = trg_model.chunkify(trg_file, config->threads);

    check(src_model.training_lines == trg_model.training_lines, "not a parallel corpus");
    if (not align_file.empty()) {
        read_alignments(align_file);
    }
    
    high_resolution_clock::time_point start = high_resolution_clock::now();
    if (config->threads == 1) {
        train_chunk(src_file, trg_file, src_chunks, trg_chunks, 0);
    } else {
        vector<thread> threads;

        for (int i = 0; i < config->threads; ++i) {
            threads.push_back(thread(&BilingualModel::train_chunk, this,
                src_file, trg_file, src_chunks, trg_chunks, i));
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

void BilingualModel::train_chunk(const string& src_file,
                                const string& trg_file,
                                const vector<long long>& src_chunks,
                                const vector<long long>& trg_chunks,
                                int chunk_id) {
    ifstream src_infile(src_file);
    ifstream trg_infile(trg_file);

    check_is_open(src_infile, src_file);
    check_is_open(trg_infile, trg_file);
    check_is_non_empty(src_infile, src_file);
    check_is_non_empty(trg_infile, trg_file);
    
    float starting_alpha = config->alpha;
    int max_iterations = config->iterations;
    long long training_words = src_model.training_words + trg_model.training_words;

    for (int k = 0; k < max_iterations; ++k) {
        int word_count = 0, last_count = 0;

        src_infile.clear();
        trg_infile.clear();
        src_infile.seekg(src_chunks[chunk_id], src_infile.beg);
        trg_infile.seekg(trg_chunks[chunk_id], trg_infile.beg);

        string src_sent, trg_sent;
        int sent_id = chunk_id * (src_model.training_lines / src_chunks.size());
        
        while (getline(src_infile, src_sent) && getline(trg_infile, trg_sent)) {
            word_count += train_sentence(src_sent, trg_sent, sent_id);
            sent_id++;

            // update learning rate
            if (word_count - last_count > 10000) {
                std::lock_guard<std::mutex> guard(multivec::print_mutex);
                words_processed += word_count - last_count;
                last_count = word_count;

                alpha = starting_alpha * (1 - static_cast<float>(words_processed) / (max_iterations * training_words));
                alpha = std::max(alpha, starting_alpha * 0.0001f);

                if (config->verbose) {
                    printf("\rAlpha: %f  Progress: %.2f%%", alpha, 100.0 * words_processed /
                                    (max_iterations * training_words));
                    fflush(stdout);
                }
            }

            // stop when reaching the end of a chunk
            if (chunk_id < src_chunks.size() - 1 && src_infile.tellg() >= src_chunks[chunk_id + 1])
                break;
        }

        words_processed += word_count - last_count;
    }
}

vector<int> BilingualModel::get_alignment(const vector<int>& src_nodes,
                                          const vector<int>& trg_nodes,
                                          int sent_id) {
    if (not alignments.empty()) {
        auto alignment = alignments[sent_id];
        check(alignment.size() <= src_nodes.size(), "bad alignment for line " + to_string(sent_id));
        auto max = max_element(alignment.begin(), alignment.end());
        if (max != alignment.end())
            check(*max < trg_nodes.size(), "bad alignment for line " + to_string(sent_id));
    }
    
    vector<int> alignment; // index = position in src_nodes, value = position in trg_nodes (or -1)
    vector<int> trg_mapping; // maps positions in trg_sent to positions in trg_nodes (or -1)
    int k = 0;
    for (int i = 0; i < trg_nodes.size(); ++i) {
        trg_mapping.push_back(trg_nodes[i] == -1 ? -1 : k++);
    }
    
    for (int i = 0; i < src_nodes.size(); ++i) {
        int j = -1;
        
        if (alignments.empty()) {
            j = i * trg_nodes.size() / src_nodes.size();  // uniform alignment
        } else if (alignments[sent_id].size() > i) {
            j = alignments[sent_id][i];
        }

        if (src_nodes[i] != -1) {
            if (j != -1) {
                j = trg_mapping[j];
            }
            
            alignment.push_back(j);
        }
    }

    return alignment;
}

void BilingualModel::read_alignments(const string& align_file) {
    ifstream align_infile(align_file);
    check_is_open(align_infile, align_file);
    
    alignments.clear();
    string line;
    while (getline(align_infile, line)) {
        if (line.find_first_not_of(' ') == line.npos) {  // if line is empty
            alignments.push_back(vector<int>());
            continue;
        }
        
        auto tokens = split(line);
        vector<int> alignment(tokens.size(), -1);
        for (auto it = tokens.begin(); it != tokens.end(); ++it) {
            string token = *it;
            int pos = token.find_first_of('-');
            int i = stoi(token.substr(0, pos));
            int j = stoi(token.substr(pos + 1, token.size() - pos - 1));
            
            if (i >= alignment.size()) {
                alignment.resize(i + 1, -1);
            }
            
            alignment[i] = j;
        }
        alignments.push_back(alignment);
    }
    
    check(alignments.size() == src_model.training_lines, "wrong number of lines inside " + align_file);
}

int BilingualModel::train_sentence(const string& src_sent, const string& trg_sent, int sent_id) {
    auto src_nodes = src_model.get_nodes(src_sent);  // same size as src_sent, OOV words are replaced by <UNK>
    auto trg_nodes = trg_model.get_nodes(trg_sent);

    // counts the number of words that are in the vocabulary_index
    int words = 0;
    words += src_nodes.size() - count(src_nodes.begin(), src_nodes.end(), -1);
    words += trg_nodes.size() - count(trg_nodes.begin(), trg_nodes.end(), -1);

    if (config->subsampling > 0) {
        src_model.subsample(src_nodes); // puts <UNK> tokens in place of the discarded tokens
        trg_model.subsample(trg_nodes);
    }
    
    if (src_nodes.empty() || trg_nodes.empty()) {
        return words;
    }

    // The <UNK> tokens are necessary to perform the alignment (the nodes vector should have the same size
    // as the original sentence)
    vector<int> alignment = get_alignment(src_nodes, trg_nodes, sent_id);

    // remove <UNK> tokens
    src_nodes.erase(
        std::remove(src_nodes.begin(), src_nodes.end(), -1),
        src_nodes.end());
    trg_nodes.erase(
        std::remove(trg_nodes.begin(), trg_nodes.end(), -1),
        trg_nodes.end());

    // Monolingual training
    for (int src_pos = 0; src_pos < src_nodes.size(); ++src_pos) {
        train_word(src_model, src_model, src_nodes, src_nodes, src_pos, src_pos, alpha);
    }

    for (int trg_pos = 0; trg_pos < trg_nodes.size(); ++trg_pos) {
        train_word(trg_model, trg_model, trg_nodes, trg_nodes, trg_pos, trg_pos, alpha);
    }

    if (config->beta == 0)
        return words;

    // Bilingual training
    for (int src_pos = 0; src_pos < src_nodes.size(); ++src_pos) {
        // 1-1 mapping between src_nodes and trg_nodes
        if (src_pos >= alignment.size()) {
            continue;
        }
        int trg_pos = alignment[src_pos];
        
        if (trg_pos != -1) { // target word isn't OOV
            train_word(src_model, trg_model, src_nodes, trg_nodes, src_pos, trg_pos, alpha * config->beta);
            train_word(trg_model, src_model, trg_nodes, src_nodes, trg_pos, src_pos, alpha * config->beta);
        }
    }

    return words; // returns the number of words processed (for progress estimation)
}

void BilingualModel::train_word(MonolingualModel& src_model, MonolingualModel& trg_model,
                                const vector<int>& src_nodes, const vector<int>& trg_nodes,
                                int src_pos, int trg_pos, float alpha) {
    if (config->skip_gram) {
        train_word_skip_gram(src_model, trg_model, src_nodes, trg_nodes, src_pos, trg_pos, alpha);
    } else {
        train_word_CBOW(src_model, trg_model, src_nodes, trg_nodes, src_pos, trg_pos, alpha);
    }
}

void BilingualModel::train_word_CBOW(MonolingualModel& src_model, MonolingualModel& trg_model,
                                     const vector<int>& src_nodes, const vector<int>& trg_nodes,
                                     int src_pos, int trg_pos, float alpha) {
    // Trains the model by predicting a source node from its aligned context in the target sentence.
    // This function can be used in the reverse direction just by reversing the arguments. Likewise,
    // for monolingual training, use the same values for source and target.

    // 'src_pos' is the position in the source sentence of the current node to predict
    // 'trg_pos' is the position of the corresponding node in the target sentence
    int dimension = config->dimension;
    vec hidden(dimension, 0);
    int cur_node = src_nodes[src_pos];

    int this_window_size = 1 + multivec::rand(config->window_size);
    int count = 0;

    for (int pos = trg_pos - this_window_size; pos <= trg_pos + this_window_size; ++pos) {
        if (pos < 0 || pos >= trg_nodes.size() || pos == trg_pos) continue;
        hidden += trg_model.input_weights[trg_nodes[pos]];
        ++count;
    }

    if (count == 0) return;
    hidden /= count;

    vec error(dimension, 0); // compute error & update output weights
    if (config->hierarchical_softmax) {
        error += src_model.hierarchical_update(cur_node, hidden, alpha);
    }
    if (config->negative > 0) {
        error += src_model.neg_sampling_update(cur_node, hidden, alpha);
    }

    // Update input weights
    for (int pos = trg_pos - this_window_size; pos <= trg_pos + this_window_size; ++pos) {
        if (pos < 0 || pos >= trg_nodes.size() || pos == trg_pos) continue;
        trg_model.input_weights[trg_nodes[pos]] += error;
    }
}

void BilingualModel::train_word_skip_gram(MonolingualModel& src_model, MonolingualModel& trg_model,
                                          const vector<int>& src_nodes, const vector<int>& trg_nodes,
                                          int src_pos, int trg_pos, float alpha) {
    int input_word = src_nodes[src_pos];

    int this_window_size = 1 + multivec::rand(config->window_size);

    for (int pos = trg_pos - this_window_size; pos <= trg_pos + this_window_size; ++pos) {
        if (pos < 0 || pos >= trg_nodes.size() || pos == trg_pos) continue;
        int output_word = trg_nodes[pos];

        vec error(config->dimension, 0);
        if (config->hierarchical_softmax) {
            error += trg_model.hierarchical_update(output_word, src_model.input_weights[input_word], alpha);
        }
        if (config->negative > 0) {
            error += trg_model.neg_sampling_update(output_word, src_model.input_weights[input_word], alpha);
        }

        src_model.input_weights[input_word] += error;
    }
}

void BilingualModel::load(const string& filename) {
    if (config->verbose)
        std::cout << "Loading model from " << filename << std::endl;

    ifstream infile(filename);
    check_is_open(infile, filename);

    ::load(infile, *this);
    src_model.init_unigram_table();
    trg_model.init_unigram_table();
}

void BilingualModel::save(const string& filename) const {
    if (config->verbose)
        std::cout << "Saving model as " << filename << std::endl;

    ofstream outfile(filename);
    check_is_open(outfile, filename);

    ::save(outfile, *this);
}
