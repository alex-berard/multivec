#include "bivec.h"
#include "boost/program_options.hpp"

namespace po = boost::program_options;

vector<string> check_arg_count(const po::variables_map& vm, const string& name, int count) {
    auto args = vm[name].as<vector<string>>();
    if (args.size() != count) {
        throw po::error("option --" + name + " wrong number of arguments");
    }
    return args;
}

int main(int argc, char **argv) {
    BilingualConfig config;
    po::options_description desc("Options");

    desc.add_options()
        ("help", "Print help message")
        ("alpha",       po::value<float>(&config.starting_alpha),   "Learning rate")
        ("dimension",   po::value<int>(&config.dimension),          "Dimension of the embeddings")
        ("min-count",   po::value<int>(&config.min_count),          "Minimum count of each word in the vocabulary")
        ("window-size", po::value<int>(&config.window_size),        "Window size")
        ("threads",     po::value<int>(&config.n_threads),          "Number of threads")
        ("iter",        po::value<int>(&config.max_iterations),     "Number of training iterations")
        ("subsampling", po::value<float>(&config.subsampling),      "Subsampling parameter (0 for no subsampling)")
        ("sg",          po::bool_switch(&config.skip_gram),         "Skip-gram-model (cbow model by default)")
        ("hs",          po::bool_switch(&config.hierarchical_softmax), "Hierarchical softmax (negative sampling by default)")
        ("verbose,v",   po::bool_switch(&config.verbose),           "Verbose mode")
        ("negative",    po::value<int>(&config.negative),           "Number of negative samples")
        ("bi-weight",   po::value<float>(&config.bi_weight),        "Bilingual training weight")
        ("load",        po::value<std::string>(),                   "Load existing model")
        ("save",        po::value<std::string>(),                   "Save entire model")
        ("save-src",    po::value<std::string>(),                   "Save source model")
        ("save-trg",    po::value<std::string>(),                   "Save target model")
        ("train",       po::value<vector<string>>()->multitoken(),  "Training files");

    po::variables_map vm;

    vector<string> training_files;

    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            cout << desc << endl;
            return 0;
        }

        po::notify(vm); // checks required arguments

        if (vm.count("train")) {
            training_files = check_arg_count(vm, "train", 2);
        }

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    BilingualModel model(config);

    if (!training_files.empty()) {
        model.train(training_files[0], training_files[1]);
    } else if (vm.count("load")) {
        model.load(vm["load"].as<std::string>());
    } else {
        return 0;
    }

    if (vm.count("save")) {
        model.save(vm["save"].as<std::string>());
    }
    if (vm.count("save-src")) {
        model.src_model.save(vm["save-src"].as<std::string>());
    }
    if (vm.count("save-trg")) {
        model.trg_model.save(vm["save-trg"].as<std::string>());
    }

    return 0;
}
