#include "multivec-mono.hpp"

void evaluateTopic(const string& topic, const vector<string>& lines,
    const map<string, vec>& embeddings, pair<int, int>* res) {

    int total = 0, correct = 0;

    for (auto line_it = lines.begin(); line_it != lines.end(); ++line_it) {
        vector<string> words = split(*line_it);
        transform(words.begin(), words.end(), words.begin(), lower);

        auto it1 = embeddings.find(words[0]);
        auto it2 = embeddings.find(words[1]);
        auto it3 = embeddings.find(words[2]);
        auto it4 = embeddings.find(words[3]);

        // skip the query if one of the four words is absent from the vocabulary
        if (it1 == embeddings.end() || it2 == embeddings.end() ||
            it3 == embeddings.end() || it4 == embeddings.end()) {
            continue;
        }

        // w4 = w2 - w1 + w3 (find w4)
        vec v = it2->second;
        for (int c = 0; c < v.size(); ++c) {
            v[c] += it3->second[c] - it1->second[c];
        }

        // find the closest word
        float similarity = 0;
        string closest_word;

        for (auto it = embeddings.begin(); it != embeddings.end(); ++it) {
            // cheating...
            if (it->first == words[0] || it->first == words[1] || it->first == words[2]) {
                continue;
            }

            float sim = 0;
            for (int c = 0; c < v.size(); ++c) {
                sim += it->second[c] * v[c];
            }

            if (sim >= similarity) {
                closest_word = it->first;
                similarity = sim;
            }
        }

        if (closest_word == words[3]) {
            ++correct;
        }
        ++total;
    }

    //return pair<int, int>(correct, total);
    *res = pair<int, int>(correct, total);
}

void computeAccuracy(istream& infile, map<string, vec>& embeddings, int max_vocabulary_size, bool verbose)
{
    // normalize
    for (auto it = embeddings.begin(); it != embeddings.end(); ++it) {
        float norm = 0;
        for (int c = 0; c < it->second.size(); ++c) {
            float x = it->second[c];
            norm += x * x;
        }
        norm = sqrt(norm);

        for (int c = 0; c < it->second.size(); ++c) {
            it->second[c] /= norm;
        }
    }

    map<string, vector<string>> topics;

    string line, topic;
    while (getline(infile, line)) {
        if (line[0] == ':') {
            topic = string(line.begin() + 2, line.end());
            topics.insert({topic, vector<string>()});
        } else {
            topics[topic].push_back(line);
        }
    }

    vector<pair<int, int>> results(topics.size());
    vector<thread> threads;
    int i = 0;
    for (auto it = topics.begin(); it != topics.end(); ++it, ++i) {
        threads.push_back(thread(evaluateTopic, it->first, it->second, embeddings, &results[i]));
    }

    /*
    vector<thread> threads;

    for (int i = 0; i < config.n_threads; ++i) {
        threads.push_back(thread(&MonolingualModel::trainChunk, this,
            training_file, chunks, i));
    }
    */

    for (auto it = threads.begin(); it != threads.end(); ++it) {
        it->join();
    }

    int correct = 0, total = 0, questions = 0;
    int gram_correct = 0, gram_total = 0;

    auto res_it = results.begin();
    auto topic_it = topics.begin();
    for (; topic_it != topics.end() && res_it != results.end() ; ++topic_it, ++res_it) {
        pair<int, int> res = *res_it;
        //pair<int, int> res = evaluateTopic(topic_it->first, topic_it->second, embeddings);

        string topic = topic_it->first;
        int topic_correct = res.first, topic_total = res.second;
        int topic_questions = topic_it->second.size();

        correct += topic_correct;
        total += topic_total;
        questions += topic_questions;

        if (topic.find("gram") == 0) {
            gram_correct += topic_correct;
            gram_total += topic_total;
        }

        if (verbose)
            cout << topic << ":\n\taccuracy: " << setprecision(3) << 100.0 * topic_correct / topic_total << "%\n";
    }

    cout << "Total accuracy: " << setprecision(3) << 100.0 * correct / total << "%\n";
    cout << "Syntactic accuracy: " << setprecision(3) << 100.0 * gram_correct / gram_total << "%, "
         << "Semantic accuracy: " << setprecision(3) << 100.0 * (correct - gram_correct) / (total - gram_total) << "%\n";
    cout << "Questions seen: " << total << "/" << questions << ", "
         << setprecision(3) << 100.0 * total / questions << "%\n";
}

void MonolingualModel::computeAccuracy(istream& infile, int max_vocabulary_size) const {
    if (config.verbose)
        cout << "Starting evaluation" << endl;

    vector<const HuffmanNode*> nodes;

    for (auto it = vocabulary.begin(); it != vocabulary.end(); ++it) {
        nodes.push_back(&it->second);
    }

    // keep only the most frequent words (kind of biased...)
    sort(nodes.begin(), nodes.end(), HuffmanNode::comp);
    if (max_vocabulary_size > 0 && nodes.size() > max_vocabulary_size) {
        nodes.erase(nodes.begin() + max_vocabulary_size, nodes.end());
    }

    map<string, vec> embeddings;
    for (auto it = nodes.begin(); it != nodes.end(); ++it) {
        const HuffmanNode* node = *it;
        vec v = input_weights[node->index];
        embeddings.insert({lower(node->word), v}); // lowercase
    }

    ::computeAccuracy(infile, embeddings, max_vocabulary_size, config.verbose);
}
