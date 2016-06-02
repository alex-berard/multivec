#include "bilingual.hpp"
#include <getopt.h>

struct option_plus {
    const char *name;
    int         has_arg;
    int        *flag;
    int         val;
    const char *desc;
};

static vector<option_plus> options_plus = {
    {"help",          no_argument,       0, 'h', "print this help message"},
    {"verbose",       no_argument,       0, 'v', "verbose mode"},
    {"dimension",     required_argument, 0, 'a', "dimension of the word embeddings"},
    {"min-count",     required_argument, 0, 'b', "minimum count of vocabulary words"},
    {"window-size",   required_argument, 0, 'c', "size of the window"},
    {"threads",       required_argument, 0, 'd', "number of threads"},
    {"iter",          required_argument, 0, 'e', "number of training epochs"},
    {"negative",      required_argument, 0, 'f', "number of negative samples (0 for no negative sampling)"},
    {"alpha",         required_argument, 0, 'g', "initial learning rate"},
    {"beta",          required_argument, 0, 'i', "bilingual training weight"},
    {"subsampling",   required_argument, 0, 'j', "subsampling (usually between 1e-03 and 1e-05)"},
    {"sg",            no_argument,       0, 'k', "skip-gram model (default: CBOW)"},
    {"hs",            no_argument,       0, 'l', "hierarchical softmax (default off)"},
    {"train-src",     required_argument, 0, 'm', "specify source file for training"},
    {"train-trg",     required_argument, 0, 'n', "specify target file for training"},
    {"load",          required_argument, 0, 'o', "load model"},
    {"save",          required_argument, 0, 'p', "save model"},
    {"save-src",      required_argument, 0, 'q', "save source model"},
    {"save-trg",      required_argument, 0, 'r', "save target model"},
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

    BilingualConfig config;
    BilingualModel model(&config);

    // model file needs to be loaded before anything else (otherwise it overwrites the parameters)
    if (!load_file.empty()) {
        model.load(load_file);
    }

    string train_src_file;
    string train_trg_file;
    string save_file;
    string save_src_file;
    string save_trg_file;

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
            case 'g': config.learning_rate = atof(optarg); break;
            case 'i': config.beta = atof(optarg);           break;
            case 'j': config.subsampling = atof(optarg);    break;
            case 'k': config.skip_gram = true;              break;
            case 'l': config.hierarchical_softmax = true;   break;
            case 'm': train_src_file = string(optarg);      break;
            case 'n': train_trg_file = string(optarg);      break;
            case 'o':                                       break;
            case 'p': save_file = string(optarg);           break;
            case 'q': save_src_file = string(optarg);       break;
            case 'r': save_trg_file = string(optarg);       break;
            default:                                        abort();
        }
    }

    if (load_file.empty() && (train_src_file.empty() || train_trg_file.empty())) {
        print_usage();
        return 0;
    }

    std::cout << "MultiVec-bi" << std::endl;
    config.print();

    if (!train_src_file.empty() && !train_trg_file.empty()) {
        model.train(train_src_file, train_trg_file, load_file.empty());
    }

    if(!save_file.empty()) {
        model.save(save_file);
    }
    if(!save_src_file.empty()) {
        model.src_model.save(save_src_file);
    }
    if(!save_trg_file.empty()) {
        model.trg_model.save(save_trg_file);
    }

    return 0;
}
