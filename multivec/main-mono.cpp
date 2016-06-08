#include "monolingual.hpp"
#include <getopt.h>

struct option_plus { // same as option with an additional description field
    const char *name;
    int         has_arg;
    int        *flag;
    int         val;
    const char *desc;
};

static vector<option_plus> options_plus = {
    {"help",              no_argument,       0, 'h', "print this help message"},
    {"verbose",           no_argument,       0, 'v', "verbose mode"},
    {"dimension",         required_argument, 0, 'a', "dimension of the word embeddings"},
    {"min-count",         required_argument, 0, 'b', "minimum count of vocabulary words"},
    {"window-size",       required_argument, 0, 'c', "size of the window"},
    {"threads",           required_argument, 0, 'd', "number of threads"},
    {"iter",              required_argument, 0, 'e', "number of training epochs"},
    {"negative",          required_argument, 0, 'f', "number of negative samples (0 for no negative sampling)"},
    {"saving-policy",     required_argument, 0, 'g', "saving policy (0: only input weights, 1: concat, 2: sum, 3: only output weights)"},
    {"alpha",             required_argument, 0, 'i', "initial learning rate"},
    {"subsampling",       required_argument, 0, 'j', "subsampling (usually between 1e-03 and 1e-05)"},
    {"sg",                no_argument,       0, 'k', "skip-gram model (default: CBOW)"},
    {"hs",                no_argument,       0, 'l', "hierarchical softmax (default off)"},
    {"sent-vector",       no_argument,       0, 'm', "train sentence vectors"},
    {"train",             required_argument, 0, 'n', "train with given training file"},
    {"load",              required_argument, 0, 'o', "load model"},
    {"save",              required_argument, 0, 'p', "save model"},
    {"save-vectors",      required_argument, 0, 'q', "save word vectors"},
    {"save-sent-vectors", required_argument, 0, 'r', "save sentence vectors"},
    {"save-vectors-bin",  required_argument, 0, 's', "save word vectors in binary format"},
    {"train-online",      required_argument, 0, 't', "use existing model to train online sentence vectors"},
    {0, 0, 0, 0, 0}
};

void print_usage() {
    std::cout << "Options:" << std::endl;
    for (auto it = options_plus.begin(); it != options_plus.end(); ++it) {
        if (it->name == 0) continue;
        string name(it->name);
        if (it->has_arg == required_argument) name += " arg";
        std::cout << std::setw(26) << std::left << "  --" + name << " " << it->desc << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char **argv) {
    vector<option> options;
    for (auto it = options_plus.begin(); it != options_plus.end(); ++it) {
        option op = {it->name, it->has_arg, it->flag, it->val};
        options.push_back(op);
    }

    string load_file;

    // first pass on parameters to find out if a model file is provided
    while (1) {
        int option_index = 0;
        int opt = getopt_long(argc, argv, "hv", options.data(), &option_index);
        if (opt == -1) break;

        switch (opt) {
            case 'o': load_file = string(optarg);           break;
            default:                                        break;
        }
    }

    Config config;
    MonolingualModel model(&config);

    // model file needs to be loaded before anything else (otherwise it overwrites the parameters)
    if (!load_file.empty()) {
        model.load(load_file);
    }

    int saving_policy = 0;
    string train_file;
    string save_file;
    string save_vectors;
    string save_sent_vectors;
    string save_vectors_bin;
    string online_train_file;

    optind = 0;  // necessary to parse arguments twice
    while (1) {
        int option_index = 0;
        int opt = getopt_long(argc, argv, "hv", options.data(), &option_index);
        if (opt == -1) break;

        switch (opt) {
            case 0:                                         break;
            case 'h': print_usage();                        return 0;
            case 'v': config.verbose = true;                break;
            case 'a': config.dimension = atoi(optarg);      break;
            case 'b': config.min_count = atoi(optarg);      break;
            case 'c': config.window_size = atoi(optarg);    break;
            case 'd': config.threads = atoi(optarg);      break;
            case 'e': config.iterations = atoi(optarg); break;
            case 'f': config.negative = atoi(optarg);       break;
            case 'g': saving_policy = atoi(optarg);         break;
            case 'i': config.learning_rate = atof(optarg); break;
            case 'j': config.subsampling = atof(optarg);    break;
            case 'k': config.skip_gram = true;              break;
            case 'l': config.hierarchical_softmax = true;   break;
            case 'm': config.sent_vector = true;            break;
            case 'n': train_file = string(optarg);          break;
            case 'o':                                       break;
            case 'p': save_file = string(optarg);           break;
            case 'q': save_vectors = string(optarg);        break;
            case 'r': save_sent_vectors = string(optarg);   break;
            case 's': save_vectors_bin = string(optarg);    break;
            case 't': online_train_file = string(optarg);   break;
            default:                                        abort();
        }
    }
    // TODO: possibility to provide vocabulary file

    if (load_file.empty() && train_file.empty()) {  // one of those actions is required
        print_usage();
        return 0;
    }

    std::cout << "MultiVec-mono" << std::endl;
    config.print();

    if (!train_file.empty()) {
        model.train(train_file, load_file.empty());
    }

    if (!online_train_file.empty()) {
        throw runtime_error("not implemented");  // TODO
    }
    
    // saving methods (TODO: save model periodically/when training is interrupted)
    if(!save_file.empty()) {
        model.save(save_file);
    }
    if (!save_vectors.empty()) {
        model.saveVectors(save_vectors, saving_policy);
    }
    if (!save_vectors_bin.empty()) {
        model.saveVectorsBin(save_vectors_bin, saving_policy);
    }
    if (!save_sent_vectors.empty() && config.sent_vector) {
        model.saveSentVectors(save_sent_vectors);
    }

    return 0;
}
