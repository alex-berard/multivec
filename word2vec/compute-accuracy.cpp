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
    const map<string, vec>& embeddings, pair<int, int>* res, bool cos_mul, bool cased) {

    int total = 0, correct = 0;

    for (auto line_it = lines.begin(); line_it != lines.end(); ++line_it) {
        vector<string> words = split(*line_it);
        if (not cased) {
            transform(words.begin(), words.end(), words.begin(), lower);
        }
        
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
            if (it->first == words[0] || it->first == words[1] || it->first == words[2]) {
                continue;
            }

            float sim = 0, d1 = 0, d2 = 0, d3 = 0;
            for (int c = 0; c < it->second.size(); ++c) {
                d1 += it->second[c] * it1->second[c];
                d2 += it->second[c] * it2->second[c];
                d3 += it->second[c] * it3->second[c];
            }
            
            if (cos_mul) {
                d1 = (d1 + 1) / 2;
                d2 = (d2 + 1) / 2;
                d3 = (d3 + 1) / 2;
                sim = d3 * d2 / (d1 + 0.001);
            } else {
                sim = d3 + d2 - d1;
            }
            
            if (sim >= similarity) {
                closest_word = it->first;
                similarity = sim;
            }
        }

        if (cased and closest_word == words[3] or not cased and lower(closest_word) == words[3]) {
            ++correct;
        }
        ++total;
    }

    *res = pair<int, int>(correct, total);
}

void computeAccuracy(istream& infile, map<string, vec>& embeddings, bool verbose, bool cos_mul, bool cased)
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
        threads.push_back(thread(evaluateTopic, it->first, it->second, embeddings, &results[i], cos_mul, cased));
    }

    for (auto it = threads.begin(); it != threads.end(); ++it) {
        it->join();
    }

    int correct = 0, total = 0, questions = 0;
    int gram_correct = 0, gram_total = 0;
    float gram_score = 0, sem_score = 0;
    int gram_topics = 0, sem_topics = 0;

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

        if (topic_total == 0)
            continue;
        
        if (topic.find("gram") == 0) {
            gram_correct += topic_correct;
            gram_total += topic_total;
            gram_topics++;
            gram_score += 1.0 * topic_correct / topic_total;
        } else {
            sem_topics++;
            sem_score += 1.0 * topic_correct / topic_total;
        }

        if (verbose)
            cout << topic << ":\n\taccuracy: " << setprecision(3) << 100.0 * topic_correct / topic_total << "% ("
                 <<  topic_correct << "/" << topic_total << ")" << endl;
    }

    // balanced scores
    gram_score = 100.0 * gram_score / gram_topics;
    sem_score = 100.0 * sem_score / sem_topics;
    float score = (gram_score + sem_score) / 2;
   
    int sem_total = total - gram_total;
    int sem_correct = correct - gram_correct;
    
    float gram_acc = 100.0 * gram_correct / gram_total;
    float sem_acc = 100.0 * sem_correct / sem_total;
    
    cout << "Total accuracy:     " << setprecision(3) << 100.0 * correct / total << "% (" << correct << "/" << total << ")" << endl;
    cout << "Syntactic accuracy: " << setprecision(3) << gram_acc << "% (" << gram_correct << "/" << gram_total << ")" << endl;
    cout << "Semantic accuracy:  " << setprecision(3) << sem_acc << "% (" << sem_correct << "/" << sem_total << ")" << endl;
    cout << "Balanced score: " << setprecision(3) << score << "% (Syntactic: " << gram_score << "%, Semantic: " << sem_score << "%)" << endl;
    cout << "Questions seen: " << total << "/" << questions << ", " << setprecision(3) << 100.0 * total / questions << "%" << endl;
}

map<string, vec> read_embeddings(istream& model_file, int voc_size) {
    map<string, vec> embeddings;

    string line;
    int words = 0;
    int size = 0;
    
    while(getline(model_file, line)) {
        if (voc_size > 0 and words >= voc_size)
            break;
        
        vector<string> tokens = split(line);
        
        if (words == 0 and tokens.size() == 2) {  // reading header
            continue;
        }
        words++;
        
        string word = tokens.front();
        vector<float> data;
        
        for (auto it = tokens.begin() + 1; it != tokens.end(); ++it) {
            data.push_back(std::stof(*it));
        }
        
        if (size == 0) {
            size = data.size();
        } else if (size != data.size()) {
            cerr << "warning: inconsistent vector sizes: " << size << ", " << data.size() << endl;
        }
        
        vec v(data);
        embeddings.insert({word, v});
    }
    
    cout << "Vocabulary size: " << words << endl;
    cout << "Embeddings size: " << size << endl;
    return embeddings;
}

int main(int argc, char **argv) {
    bool cos_mul = false, cased = false, error = false;
    int voc_size = 0;
    
    for (int i = 3; i < argc; i++) {
        if (string(argv[i]) == string("--cos-mul")) {
            cos_mul = true;
        } else if (string(argv[i]) == string("--cased")) {
            cased = true;
        } else if (string(argv[i]) == string("--voc-size") and argc > i + 1) {
            voc_size = atoi(argv[i + 1]);
            i++;
        } else {
            error = true;
        }
    }
    
    if (argc < 3 or error) {
        cout << "Usage: " << argv[0] << " MODEL_FILE QUESTIONS [--voc-size VOC_SIZE] [--cos-mul] [--cased]" << endl;
        return 0;
    }
    
    ifstream model_file(argv[1]);
    auto embeddings = read_embeddings(model_file, voc_size);
    model_file.close();
    
    ifstream question_file(argv[2]);
    computeAccuracy(question_file, embeddings, voc_size, cos_mul, cased);
    question_file.close();
}
