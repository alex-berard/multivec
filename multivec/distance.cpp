#include "monolingual.hpp"
#include "bilingual.hpp"


/**
 * @brief Compute cosine similarity between word1 and word2.
 * For the score to be in [0,1], the weights need to be normalized beforehand.
 * Return 0 if word1 or word2 is unknown.
 */
float MonolingualModel::similarity(const string& word1, const string& word2, int policy) const {
    auto it1 = vocabulary.find(word1);
    auto it2 = vocabulary.find(word2);

    if (it1 == vocabulary.end() || it2 == vocabulary.end()) {
        return 0.0;
    } else if (it1->second.index == it2->second.index) {
        return 1.0;
    } else {
        vec v1 = wordVec(it1->second.index, policy);
        vec v2 = wordVec(it2->second.index, policy);
        return cosineSimilarity(v1, v2);
    }
}

float MonolingualModel::distance(const string& word1, const string& word2, int policy) const {
    return 1 - similarity(word1, word2, policy);
}


static bool comp(const pair<string, float>& p1, const pair<string, float>& p2) {
    return p1.second > p2.second;
}

/**
 * @brief Return an ordered list of the `n` closest words to `word` according to cosine similarity.
 */
vector<pair<string, float>> MonolingualModel::closest(const string& word, int n, int policy) const {
    vector<pair<string, float>> res;
    auto it = vocabulary.find(word);

    if (it == vocabulary.end()) {
        throw runtime_error("OOV word");
    }

    int index = it->second.index;
    vec v1 = wordVec(index, policy);

    for (auto it = vocabulary.begin(); it != vocabulary.end(); ++it) {
        if (it->second.index != index) {
            vec v2 = wordVec(it->second.index, policy);
            res.push_back({it->second.word, cosineSimilarity(v1, v2)});
        }
    }

    std::partial_sort(res.begin(), res.begin() + n, res.end(), comp);
    if (res.size() > n) res.resize(n);
    return res;
}

vector<pair<string, float>> MonolingualModel::closest(const vec& v, int n, int policy) const {
    vector<pair<string, float>> res;

    for (auto it = vocabulary.begin(); it != vocabulary.end(); ++it) {
        vec v2 = wordVec(it->second.index, policy);
        res.push_back({it->second.word, cosineSimilarity(v, v2)});
    }

    std::partial_sort(res.begin(), res.begin() + n, res.end(), comp);
    if (res.size() > n) res.resize(n);
    return res;
}

/**
 * @brief Return sorted list of `words` according to their similarity to `word`.
 */
vector<pair<string, float>> MonolingualModel::closest(const string& word, const vector<string>& words, int policy) const {
    vector<pair<string, float>> res;
    auto it = vocabulary.find(word);

    if (it == vocabulary.end()) {
        throw runtime_error("OOV word");
    }

    int index = it->second.index;
    vec v1 = wordVec(index, policy);

    for (auto it = words.begin(); it != words.end(); ++it) {
        auto node_it = vocabulary.find(*it);
        if (node_it != vocabulary.end()) {
            vec v2 = wordVec(node_it->second.index, policy);
            res.push_back({node_it->second.word, cosineSimilarity(v1, v2)});
        }
    }

    std::sort(res.begin(), res.end(), comp);
    return res;
}

float MonolingualModel::similarityNgrams(const string& seq1, const string& seq2, int policy) const {
    auto words1 = split(seq1);
    auto words2 = split(seq2);

    if (words2.size() != words2.size()) {
        throw runtime_error("input sequences don't have the same size");
    }

    float res = 0;
    int n = 0;
    for (size_t i = 0; i < words1.size(); ++i) {
        try {
            res += similarity(words1[i], words2[i], policy);
            n += 1;
        }
        catch (runtime_error) {}
    }

    if (n == 0) {
        throw runtime_error("all word pairs are unknown (OOV)");
    } else {
        return res / n;
    }
}

void normalizeWeights(mat& weights) {
    if (weights.empty()) {
        return;
    }

    int dim = weights[0].size();

    vec min_values = weights[0];
    vec max_values = weights[0];
    for (size_t i = 1; i < weights.size(); ++i) {
        for (size_t j = 0; j < dim; ++j) {
            min_values[j] = min(min_values[j], weights[i][j]);
            max_values[j] = max(max_values[j], weights[i][j]);
        }
    }

    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < dim; ++j) {
            if (max_values[j] != min_values[j]) {
                weights[i][j] = (weights[i][j] - min_values[j]) / (max_values[j] - min_values[j]);
            }
        }
    }
}

void MonolingualModel::normalizeWeights() {
    ::normalizeWeights(input_weights);
    ::normalizeWeights(output_weights);
    ::normalizeWeights(output_weights_hs);
    ::normalizeWeights(sent_weights);
}

float MonolingualModel::similaritySentence(const string& seq1, const string& seq2, int policy) const {
    auto words1 = split(seq1);
    auto words2 = split(seq2);
    
    vec vec1(config->dimension);
    vec vec2(config->dimension);
    
    for (auto it = words1.begin(); it != words1.end(); ++it) {
        try {
            vec1 += wordVec(*it, policy);
        }
        catch (runtime_error) {}
    }
    
    for (auto it = words2.begin(); it != words2.end(); ++it) {
        try {
            vec2 += wordVec(*it, policy);
        }
        catch (runtime_error) {}
    }
    
    float length = vec1.norm() * vec2.norm();
    
    if (length == 0) {
        return 0.0;
    } else {
        return vec1.dot(vec2) / length;
    }
}

float MonolingualModel::softWER(const string& hyp, const string& ref, int policy) const {
    auto s1 = split(hyp);
    auto s2 = split(ref);
    const size_t len1 = s1.size(), len2 = s2.size();
    vector<vector<float>> d(len1 + 1, vector<float>(len2 + 1));

    d[0][0] = 0;
    for (size_t i = 1; i <= len1; ++i) d[i][0] = i;
    for (size_t i = 1; i <= len2; ++i) d[0][i] = i;

    for (size_t i = 1; i <= len1; ++i) {
        for (size_t j = 1; j <= len2; ++j) {
            // uses distance between word embeddings as a substitution cost
            // FIXME: distances tend to be well below 1, even for very different words.
            // This is rather unbalanced with deletion and insertion costs, which remain at 1.
            // Also, distance can (but will rarely) be greater than 1.
            float sub_cost = distance(s1[i - 1], s2[j - 1], policy);
            
            d[i][j] = min({ d[i - 1][j] + 1,  // deletion
                            d[i][j - 1] + 1,  // insertion
                            d[i - 1][j - 1] + sub_cost });  // substitution
        }
    }
    
    return d[len1][len2] / len2;
}


/**
 *
 * Bilingual Methods
 *
 */


/**
 * @brief Compute cosine similarity between word1 in the source model and word2 in the target model.
 * For the score to be in [0,1], the weights need to be normalized beforehand.
 * Return 0 if word1 or word2 is unknown.
 */
float BilingualModel::similarity(const string& src_word, const string& trg_word, int policy) const {
    auto it1 = src_model.vocabulary.find(src_word);
    auto it2 = trg_model.vocabulary.find(trg_word);

    if (it1 == src_model.vocabulary.end() || it2 == trg_model.vocabulary.end()) {
        return 0.0;
    } else {
        vec v1 = src_model.wordVec(it1->second.index, policy);
        vec v2 = trg_model.wordVec(it2->second.index, policy);
        return cosineSimilarity(v1, v2);
    }
}


float BilingualModel::distance(const string& src_word, const string& trg_word, int policy) const {
    return 1 - similarity(src_word, trg_word, policy);
}


vector<pair<string, float>> BilingualModel::trg_closest(const string& src_word, int n, int policy) const {
    vector<pair<string, float>> res;
    auto it = src_model.vocabulary.find(src_word);

    if (it == src_model.vocabulary.end()) {
        throw runtime_error("OOV word");
    }

    vec v = src_model.wordVec(it->second.index, policy);
    return trg_model.closest(v, n, policy);
}


vector<pair<string, float>> BilingualModel::src_closest(const string& trg_word, int n, int policy) const {
    vector<pair<string, float>> res;
    auto it = trg_model.vocabulary.find(trg_word);

    if (it == trg_model.vocabulary.end()) {
        throw runtime_error("OOV word");
    }

    vec v = trg_model.wordVec(it->second.index, policy);
    return src_model.closest(v, n, policy);
}


float BilingualModel::similarityNgrams(const string& src_seq, const string& trg_seq, int policy) const {
    auto src_words = split(src_seq);
    auto trg_words = split(trg_seq);

    if (trg_words.size() != trg_words.size()) {
        throw runtime_error("input sequences don't have the same size");
    }

    float res = 0;
    int n = 0;
    for (size_t i = 0; i < src_words.size(); ++i) {
        try {
            res += similarity(src_words[i], trg_words[i], policy);
            n += 1;
        }
        catch (runtime_error) {}
    }

    if (n == 0) {
        throw runtime_error("all word pairs are unknown (OOV)");
    } else {
        return res / n;
    }
}

float BilingualModel::similaritySentence(const string& src_seq, const string& trg_seq, int policy) const {
    auto src_words = split(src_seq);
    auto trg_words = split(trg_seq);
    
    vec src_vec(config->dimension);
    vec trg_vec(config->dimension);
    
    for (auto it = src_words.begin(); it != src_words.end(); ++it) {
        try {
            src_vec += src_model.wordVec(*it, policy);
        }
        catch (runtime_error) {}
    }
    
    for (auto it = trg_words.begin(); it != trg_words.end(); ++it) {
        try {
            trg_vec += trg_model.wordVec(*it, policy);
        }
        catch (runtime_error) {}
    }
    
    float length = src_vec.norm() * trg_vec.norm();
    
    if (length == 0) {
        return 0.0;
    } else {
        return src_vec.dot(trg_vec) / length;
    }
}

/**
 * POS weights according to Universal Tagset from http://github.com/slavpetrov/universal-pos-tags
 */
const static std::map<std::string, float> syntax_weights = {
    { "VERB", 0.75 },
    { "NOUN", 1.00 },
    { "PRON", 0.10 },
    { "ADJ",  0.75 },
    { "ADV",  0.50 },
    { "ADP",  0.10 },
    { "CONJ", 0.10 },
    { "DET",  0.10 },
    { "NUM",  0.50 },
    { "PRT",  0.10 },
    { "X",    0.50 },
    { ".",    0.05 }
};

float MonolingualModel::similaritySentenceSyntax(const string& seq1, const string& seq2, const string& tags1, const string& tags2, int policy) const {
    auto words1 = split(seq1);
    auto words2 = split(seq2);

      auto pos_tags1 = split(tags1);
    auto pos_tags2 = split(tags2);
    
    vec vec1(config->dimension);
    vec vec2(config->dimension);
    
    for (size_t i = 0; i < words1.size() && i < pos_tags1.size(); ++i) {
        try {
            vec1 += wordVec(words1[i], policy) * syntax_weights.at(pos_tags1[i]);
        }
        catch (runtime_error) {}
    }
    
    for (size_t i = 0; i < words2.size() && i < pos_tags2.size(); ++i) {
        try {
            vec2 += wordVec(words2[i], policy) * syntax_weights.at(pos_tags2[i]);
        }
        catch (runtime_error) {}
    }
    
    float length = vec1.norm() * vec2.norm();
    
    if (length == 0) {
        return 0.0;
    } else {
        return vec1.dot(vec2) / length;
    }
}

float BilingualModel::similaritySentenceSyntax(const string& src_seq, const string& trg_seq, const string& src_tags, const string& trg_tags, int policy) const {    
    auto src_words = split(src_seq);
    auto trg_words = split(trg_seq);
    
    auto src_pos_tags = split(src_tags);
    auto trg_pos_tags = split(trg_tags);
    
    vec src_vec(config->dimension);
    vec trg_vec(config->dimension);
    
    for (size_t i = 0; i < src_words.size() && i < src_pos_tags.size(); ++i) {
        try {
            src_vec += src_model.wordVec(src_words[i], policy) * syntax_weights.at(src_pos_tags[i]);
        }
        catch (runtime_error) {}
    }
    for (size_t i = 0; i < trg_words.size() && i < trg_pos_tags.size(); ++i) {
        try {
            trg_vec += trg_model.wordVec(trg_words[i], policy) * syntax_weights.at(trg_pos_tags[i]);
        }
        catch (runtime_error) {}
    }
    
    float length = src_vec.norm() * trg_vec.norm();
    
    if (length == 0) {
        return 0.0;
    } else {
        return src_vec.dot(trg_vec) / length;
    }
}

vector<pair<string, vector<pair<string, float>>>> BilingualModel::list_trg_closest(int n, int policy) const {
    // TODO : this should be multithreaded !!!!
    vector<pair<string, float>> res;
    vector<pair<string, vector<pair<string, float>>>> to_return;
    auto it = src_model.vocabulary.begin();
    int lc=0;
    while  (it != src_model.vocabulary.end()) {
                // cerr << it->first << endl;
        vec v = src_model.wordVec(it->second.index, policy);
        res=trg_model.closest(v, n, policy);        
        to_return.push_back(pair<string, vector<pair<string, float>>>(it->first,res));
        it++;
        lc++;
        if (lc % 10 == 0) { cerr << '.'; }
        if (lc % 100 == 0) { cerr << " [" << lc << "]\n" << flush; }
    }
    return to_return;
}


vector<vector<pair<string,vec>>> chunk_vectors(unordered_map<string, HuffmanNode> l_vocabulary, int nbr_chunks)
{
    int l_size_chunk=(int)l_vocabulary.size() / nbr_chunks;
    int l_size_rest=(int)l_vocabulary.size() % nbr_chunks;
    if (l_size_rest > 0) l_size_chunk++;
    int l_inc;
    vector<vector<pair<string,vec>>> to_return;
    vector<pair<string,vec>> to_process;
    auto it = l_vocabulary.begin();
    while  (it != l_vocabulary.end())
    {
        if (l_inc < l_size_chunk)
        {
            to_process.push_back(pair<string,vec>(it->first,it->second.index));
            l_inc++;
            it++;
        }
        else
        {
            l_inc=0;
            to_return.push_back(to_process);
            to_process.clear();
        }
    }
    if (not to_process.empty())
    {
        to_return.push_back(to_process);
    }
    return to_return;
}

void MonolingualModel::closest_chunk(const vector<pair<string,vec>>& v, int n, int policy, vector<pair<string,vector<pair<string, float>>>>& ret) {
    auto it_chunk = v.begin();
    int lc=0;
    while (it_chunk != v.end())
    {
        vec l_v=it_chunk->second;
        vector<pair<string, float>> res = closest(l_v,n,policy);
        pair<string,vector<pair<string, float>>> p_ret(it_chunk->first,res);
        ret.push_back(p_ret);
        it_chunk++;
        lc++;
        if (lc % 10 == 0) { cerr << '.';  }
        if (lc % 100 == 0) { cerr << " [" << lc << "]\n" << flush; }
    }
}



// vector<pair<string, vector<pair<string, float>>>> BilingualModel::list_src_closest(int n, int policy) const {
//     // TODO : this should be multithreaded !!!!
//     vector<pair<string, float>> res;
//     vector<pair<string, vector<pair<string, float>>>> to_return;
//     int l_threads = config->threads;
//     vector<pair<string, vector<pair<string, float>>>> tab_res(l_threads);
//     
//     auto it = trg_model.vocabulary.begin();
//     int lc=0;
//     int l_curr_thread=0;
//     while  (it != trg_model.vocabulary.end()) {
// //                cerr << it->first << endl;
//         vec v = trg_model.wordVec(it->second.index, policy);
//         res=src_model.closest(v, n, policy);        
//         to_return.push_back(pair<string, vector<pair<string, float>>>(it->first,res));
//         it++;
//         lc++;
//         if (lc % 10 == 0) { cerr << '.';  }
//         if (lc % 100 == 0) { cerr << " [" << lc << "]\n" << flush; }
//     }
//     return to_return;
// }


vector<pair<string, vector<pair<string, float>>>> BilingualModel::list_src_closest(int n, int policy) const {
    // TODO : this should be multithreaded !!!!
    int l_threads = config->threads;
    vector<vector<pair<string,vec>>> to_process = chunk_vectors(trg_model.vocabulary,l_threads);
    vector<pair<string, float>> res;
    vector<pair<string, vector<pair<string, float>>>> to_return;
    vector<vector<pair<string, vector<pair<string, float>>>>> tab_res(l_threads);
    if (l_threads == 1)
    {
        auto it = trg_model.vocabulary.begin();
        int lc=0;
        int l_curr_thread=0;
        while  (it != trg_model.vocabulary.end()) {
    //                cerr << it->first << endl;
            vec v = trg_model.wordVec(it->second.index, policy);
            res=src_model.closest(v, n, policy);        
            to_return.push_back(pair<string, vector<pair<string, float>>>(it->first,res));
            it++;
            lc++;
            if (lc % 10 == 0) { cerr << '.';  }
            if (lc % 100 == 0) { cerr << " [" << lc << "]\n" << flush; }
        }
    }
    else
    {
        vector<thread> threads;
        for (int i = 0; i < config->threads; ++i) {
            threads.push_back(thread(&MonolingualModel::closest_chunk, this->trg_model, 
                                      std::ref(to_process.at(i)), n, policy,  std::ref(tab_res.at(i))));
        }

        for (auto it = threads.begin(); it != threads.end(); ++it) {
            it->join();
        }   
        
        auto res_it=tab_res.begin();
        while (res_it != tab_res.end())
        {
            to_return.insert(to_return.end(),(*res_it).begin(),(*res_it).end());
        }
    }
    return to_return;
}

void BilingualModel::save_srcpt(int n, string file) const {
    stringstream ssoutput;
    ofstream outputfile;
    std::cerr << "Saving Prob table src->trg" <<endl;
    auto list_src = list_trg_closest(n,0);
    outputfile.open(file);
    auto it_src=list_src.begin();
    while (it_src!=list_src.end()) {
        auto list_trg=it_src->second;
        auto it_trg=list_trg.begin();
        while (it_trg!=list_trg.end()) {
            outputfile << it_src->first << "\t" << it_trg->first << "\t" << log(it_trg->second) << endl;
                        it_trg++;
        }
                it_src++;
//     ofstream outputfile;
//     outputfile.open(file);
//     outputfile << ssoutput.str();
    }
    outputfile.close();
}

void BilingualModel::save_trgpt(int n, string file) const  {
    stringstream ssoutput;
    ofstream outputfile;
    std::cerr << "Saving Prob table trg->src" <<endl;
    auto list_trg = list_src_closest(n,0);
    outputfile.open(file);
    auto it_trg=list_trg.begin();
    while (it_trg!=list_trg.end()) {
        auto list_src=it_trg->second;
        auto it_src=list_src.begin();
        while (it_src!=list_src.end()) {
            outputfile << it_trg->first << "\t" << it_src->first << "\t" << log(it_src->second) << endl;
                        it_src++;
        }
                it_trg++;
//     outputfile << ssoutput.str();
    }
    outputfile.close();
}


