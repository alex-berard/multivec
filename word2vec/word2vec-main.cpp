#include "word2vec.hpp"
#include <getopt.h>
#include <vector>
#include <iomanip>

struct option_plus {
    const char *name;
    int         has_arg;
    int        *flag;
    int         val;
    const char *desc;
};

static std::vector<option_plus> options_plus = {
    {"help",              no_argument,       0, 'h', "print this help message"},
    {"verbose",           no_argument,       0, 'v', "verbose mode"},
    {"dimension",         required_argument, 0, 'a', "dimension of the word embeddings"},
    {"min-count",         required_argument, 0, 'b', "minimum count of vocabulary words"},
    {"window-size",       required_argument, 0, 'c', "size of the window"},
    {"threads",           required_argument, 0, 'd', "number of threads"},
    {"iter",              required_argument, 0, 'e', "number of training epochs"},
    {"negative",          required_argument, 0, 'f', "number of negative samples (0 for no negative sampling)"},
    {"alpha",             required_argument, 0, 'g', "initial learning rate"},
    {"subsampling",       required_argument, 0, 'i', "subsampling (usually between 1e-03 and 1e-05)"},
    {"sg",                no_argument,       0, 'j', "skip-gram model (default: CBOW)"},
    {"hs",                no_argument,       0, 'k', "hierarchical softmax (default off)"},
    {"sent-vector",       no_argument,       0, 'l', "train sentence vectors"},
    {"train",             required_argument, 0, 'm', "train with given training file"},
    {"save-vectors",      required_argument, 0, 'n', "save word vectors"},
    {"save-vectors-bin",  required_argument, 0, 'o', "save word vectors in binary format"},
    {0, 0, 0, 0, 0}
};

void print_usage() {
    std::cout << "Options:" << std::endl;
    for (auto it = options_plus.begin(); it != options_plus.end(); ++it) {
        if (it->name == 0) continue;
        std::string name(it->name);
        if (it->has_arg == required_argument) name += " arg";
        std::cout << std::setw(26) << std::left << "  --" + name << " " << it->desc << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char **argv) {
    std::vector<option> options;
    for (auto it = options_plus.begin(); it != options_plus.end(); ++it) {
        option op = {it->name, it->has_arg, it->flag, it->val};
        options.push_back(op);
    }

    Config config;
    std::string train_file;
    std::string save_vectors;
    std::string save_vectors_bin;

    while (1) {
        int option_index = 0;
        int opt = getopt_long(argc, argv, "hv", options.data(), &option_index);
        if (opt == -1) break;

        switch (opt) {
            case 0:                                           break;
            case 'h': print_usage();                          return 0;
            case 'v': config.verbose = true;                  break;
            case 'a': config.dimension = atoi(optarg);        break;
            case 'b': config.min_count = atoi(optarg);        break;
            case 'c': config.window_size = atoi(optarg);      break;
            case 'd': config.threads = atoi(optarg);        break;
            case 'e': config.iterations = atoi(optarg);   break;
            case 'f': config.negative = atoi(optarg);         break;
            case 'g': config.learning_rate = atof(optarg);   break;
            case 'i': config.subsampling = atof(optarg);      break;
            case 'j': config.skip_gram = true;                break;
            case 'k': config.hierarchical_softmax = true;     break;
            case 'l': config.sent_vector = true;              break;
            case 'm': train_file = std::string(optarg);       break;
            case 'n': save_vectors = std::string(optarg);     break;
            case 'o': save_vectors_bin = std::string(optarg); break;
            default:                                          abort();
        }
    }

    if (!train_file.empty()) {
        if (!save_vectors.empty()) {
            Main(train_file, save_vectors, config);
            return 0;
        }
        else if (!save_vectors_bin.empty()) {
            config.binary = true;
            Main(train_file, save_vectors_bin, config);
            return 0;
        }
    }

    print_usage();
    return 0;
}
