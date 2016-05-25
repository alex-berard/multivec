#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <thread>
#include <algorithm>
#include <iomanip>
#include <math.h>

using namespace std;
typedef vector<float> vec;

vector<string> split(const string& sentence) {
    istringstream iss(sentence);
    vector<string> words;
    string word;

    while (iss >> word) {
        words.push_back(word);
    }

    return words;
}

inline string lower(string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

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

void computeAccuracy(istream& infile, map<string, vec>& embeddings, bool verbose)
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

    for (auto it = threads.begin(); it != threads.end(); ++it) {
        it->join();
    }

    int correct = 0, total = 0, questions = 0;
    int gram_correct = 0, gram_total = 0;

    auto res_it = results.begin();
    auto topic_it = topics.begin();
    for (; topic_it != topics.end() && res_it != results.end() ; ++topic_it, ++res_it) {
        pair<int, int> res = *res_it;

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

void computeAccuracy(const string& model_filename, istream& infile, long long max_vocabulary_size) {
    map<string, vec> embeddings;
    ifstream model_file(model_filename);

    if (!model_file.is_open()) {
        throw runtime_error("couldn't open file " + model_filename);
    }

    string line;
    long long words, size;
    model_file >> words >> size;    
    getline(model_file, line);

    cout << "Vocabulary size: " << words << endl;
    cout << "Embeddings size: " << size << endl;

    if (max_vocabulary_size > 0) {
        words = min(max_vocabulary_size, words);
    }

    for (size_t i = 0; i < words; ++i) {
        vec v(size);
        int j = 0;

        getline(model_file, line);
        vector<string> tokens = split(line);
        string word = tokens.front();

        for (size_t j = 0; j < size; ++j) {
            v[j] = std::stof(tokens[j + 1]);
        }
        
        embeddings.insert({word, v});
    }

    computeAccuracy(infile, embeddings, true);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " MODEL_FILE VOC_SIZE < QUESTIONS" << endl;
        return 0;
    }
    computeAccuracy(string(argv[1]), std::cin, atoi(argv[2]));
}
