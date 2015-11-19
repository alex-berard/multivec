#include "monolingual.hpp"

/**
 * @brief Compute cosine similarity between word1 and word2. Throw runtime_error if 
 * those words are unknown. For the score to be in [0,1], the weights need to be normalized beforehand.
 */
/*
float MonolingualModel::similarity(const string& word1, const string& word2, int policy) const {
    if (word1 == word2) {
        return 1.0;
    } else {
        vec v1 = wordVec(word1, policy);
        vec v2 = wordVec(word2, policy);
        return cosineSimilarity(v1,v2);
    }
}
*/
float MonolingualModel::similarity(const string& word1, const string& word2, int policy) const {
    // Try to find word1, if not return 0.0
    auto it1 = vocabulary.find(word1);
    if (it1 == vocabulary.end()) {
	return 0.0;
    }
    // Try to find word2, if not return 0.0
    auto it2 = vocabulary.find(word2);
    if (it2 == vocabulary.end()) {
	return 0.0;
    }
    // if word1 == word2 return 1.0 , else compute the cosine distance
    if (it1->second.index == it2->second.index) {
        return 1.0;
    } else {
        vec v1 = wordVec(it1->second.index, policy);
        vec v2 = wordVec(it2->second.index, policy);
        return cosineSimilarity(v1,v2);
    }
}


float MonolingualModel::distance(const string& word1, const string& word2, int policy) const {
    return 1 - similarity(word1, word2, policy);
}

/**
 * @brief Return an ordered list of "maxNbest" closest words with their cosine similarity score
 */
vector< pair< string, float > > MonolingualModel::closest(const string& word, int policy) const
{
    auto it = vocabulary.find(word);
    vector< pair< string, float > > toReturn;

    if (it == vocabulary.end()) {
        cerr <<"out of vocabulary!" << endl;
        return toReturn;
    } 
    toReturn.push_back(pair("Nothing",-1000.0)); // add a dummy word for comparison
    int index1 = it->second.index;
    vec v1 = wordVec(index1, policy);
    vec v2;
    for (it = vocabulary.begin(); it != vocabulary.end(); it) {
	if (index1 != it->second.index) {
	    v2 = wordVec(it->second.index, policy);
	    float sim = cosineSimilarity(v1,v2);
	    for (size_t i = 0; ((i < toReturn.size()) && (i < maxNbest + 1)); i++) {
		if (sim > toReturn.at(i).second)
		{
		    toReturn.insert(i,pair(it->second.word,sim));
		    break;
		}
	    }
	}
    }
    toReturn.pop_back(); // remove the dummy word
    while ((int)toReturn.size() > maxNbest)
    {
	toReturn.pop_back();
    }
    return toReturn;
}

/**
 * @brief Return an ordered list of closest words given in the vector with their cosine similarity score
 */
vector< pair< string, float > > MonolingualModel::closest(const string& word1, const vector< string >& vecword, int policy) const
{
    auto it = vocabulary.find(word1);
    vector< pair< string, float > > toReturn;

    if (it == vocabulary.end()) {
        cerr <<"out of vocabulary!" << endl;
        return toReturn;
    } 
    toReturn.push_back(pair("Nothing",-1000.0)); // add a dummy word for comparison
    vec v1 = wordVec(it->second.index, policy);
    vec v2;
    for (size_t i = 0; i < vecword.size(); i++) {
        it = vocabulary.find(vecword.at(i));
	float sim = 0.0;
	if (it != vocabulary.end()) {
	    v2 = wordVec(it->second.index, policy);
	    sim = cosineSimilarity(v1,v2);
	} 
	for (size_t j = 0; ((j < toReturn.size()) && (j < maxNbest + 1)); j++) {
	    if (sim > toReturn.at(j).second)
	    {
		toReturn.insert(j,pair(vecword.at(i),sim));
		break;
	    }
	}
    }
    toReturn.pop_back(); // remove the dummy word
    return toReturn;
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