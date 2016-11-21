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

/**
 * POS weights according to "A Universal Part-of-Speech Tagset"
 * by Slav Petrov, Dipanjan Das and Ryan McDonald
 * for more details, see:
 * paper: http://arxiv.org/abs/1104.2086
 * project url: https://github.com/slavpetrov/universal-pos-tags
 * 
 * VERB - verbs (all tenses and modes)
 * NOUN - nouns (common and proper)
 * PRON - pronouns 
 * ADJ - adjectives
 * ADV - adverbs
 * ADP - adpositions (prepositions and postpositions)
 * CONJ - conjunctions
 * DET - determiners
 * NUM - cardinal numbers
 * PRT - particles or other function words
 * X - other: foreign words, typos, abbreviations
 * . - punctuation
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

/**
* @brief Compute a cosine similarity between two variable-size sequences according of part-of-speech and inverse document frequencies of terms in the sequences.
* @param seq1 First sequence of terms.
* @param seq2 Second sequence of terms.
* @param tags1 Part-of-speech tags of terms of the first sequence. The i^th tag in tags1 corresponds of the i^th term of seq1. The POS tags need to be normalized under the Universal Tagset.
* @param tags2 Part-of-speech tags of terms of the second sequence. The i^th tag in tags2 corresponds of the i^th term of seq2. The POS tags need to be normalized under the Universal Tagset.
* @param idf1 Inverse document frequencies (IDF) of terms of the first sequence. The i^th value in idf1 corresponds of the i^th term of seq1.
* @param idf2 Inverse document frequencies (IDF) of terms of the second sequence. The i^th value in idf2 corresponds of the i^th term of seq2.
* @param alpha Weighting coefficient for the use of IDF weights against POS weights. 0 to only use POS weights and 1 to only use IDF weights.
* @param policy Indice for determining the weights to use in the word embeddings.
* @return Return a float between 0 and 1 representing the similarity between the two sequences.
*/
float MonolingualModel::similaritySentenceSyntax(const string& seq1, const string& seq2, const string& tags1, const string& tags2,
                                                 const vector<float>& idf1, const vector<float>& idf2, float alpha, int policy) const {
    auto words1 = split(seq1);
    auto words2 = split(seq2);
    auto pos_tags1 = split(tags1);
    auto pos_tags2 = split(tags2);
    
    vec vec1(config->dimension);
    vec vec2(config->dimension);
    
    for (size_t i = 0; i < words1.size() && i < pos_tags1.size() && i < idf1.size(); ++i) {
        try {
            vec1 += wordVec(words1[i], policy) * pow(syntax_weights.at(pos_tags1[i]), 1 - alpha) * pow(idf1[i], alpha);
        }
        catch (runtime_error) {}
    }
    
    for (size_t i = 0; i < words2.size() && i < pos_tags2.size() && i < idf2.size(); ++i) {
        try {
            vec2 += wordVec(words2[i], policy) * pow(syntax_weights.at(pos_tags2[i]), 1 - alpha) * pow(idf2[i], alpha);
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
* @brief Compute a cosine similarity between two variable-size sequences in two different languages according of part-of-speech and inverse document frequencies of terms in the sequences.
* @param src_seq First sequence of terms.
* @param trg_seq Second sequence of terms.
* @param src_tags Part-of-speech tags of terms of the first sequence. The i^th tag in src_tags corresponds of the i^th term of src_seq. The POS tags need to be normalized under the Universal Tagset.
* @param trg_tags Part-of-speech tags of terms of the second sequence. The i^th tag in trg_tags corresponds of the i^th term of trg_seq. The POS tags need to be normalized under the Universal Tagset.
* @param src_idf Inverse document frequencies (IDF) of terms of the first sequence. The i^th value in src_idf corresponds of the i^th term of src_seq.
* @param trg_idf Inverse document frequencies (IDF) of terms of the second sequence. The i^th value in trg_idf corresponds of the i^th term of trg_seq.
* @param alpha Weighting coefficient for the use of IDF weights against POS weights. 0 to only use POS weights and 1 to only use IDF weights.
* @param policy Indice for determining the weights to use in the word embeddings.
* @return Return a float between 0 and 1 representing the similarity between the two sequences.
*/
float BilingualModel::similaritySentenceSyntax(const string& src_seq, const string& trg_seq, const string& src_tags, const string& trg_tags,
                                               const vector<float>& src_idf, const vector<float>& trg_idf, float alpha, int policy) const {    
    auto src_words = split(src_seq);
    auto trg_words = split(trg_seq);
    auto src_pos_tags = split(src_tags);
    auto trg_pos_tags = split(trg_tags);
    
    vec src_vec(config->dimension);
    vec trg_vec(config->dimension);
    
    for (size_t i = 0; i < src_words.size() && i < src_pos_tags.size() && i < src_idf.size(); ++i) {
        try {
            src_vec += src_model.wordVec(src_words[i], policy) * pow(syntax_weights.at(src_pos_tags[i]), 1 - alpha) * pow(src_idf[i], alpha);
        }
        catch (runtime_error) {}
    }
    for (size_t i = 0; i < trg_words.size() && i < trg_pos_tags.size() && i < trg_idf.size(); ++i) {
        try {
            trg_vec += trg_model.wordVec(trg_words[i], policy) * pow(syntax_weights.at(trg_pos_tags[i]), 1 - alpha) * pow(trg_idf[i], alpha);
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
