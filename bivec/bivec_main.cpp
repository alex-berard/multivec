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
        ("alpha",       po::value<float>(&config.starting_alpha),  "Learning rate")
        ("dimension",   po::value<int>(&config.dimension),         "Dimension of the embeddings")
        ("min-count",   po::value<int>(&config.min_count),         "Minimum count of each word in the vocabulary")
        ("window-size", po::value<int>(&config.window_size),       "Window size")
        ("threads",     po::value<int>(&config.n_threads),         "Number of threads")
        ("iter",        po::value<int>(&config.max_iterations),    "Number of training iterations")
        ("sampling",    po::value<float>(&config.sampling),        "Subsampling parameter (0 for no subsampling)")
        ("sg",          po::value<bool>(&config.skip_gram),        "Skip-gram-model (cbow model by default)")
        ("ns",          po::value<bool>(&config.negative_sampling),"Negative sampling (hierarchical softmax by default)")
        ("negative",    po::value<int>(&config.negative),          "Number of negative samples")
        ("bi-weight",   po::value<float>(&config.bi_weight),       "Bilingual training weight")
        ("save",        po::value<vector<string>>()->multitoken(), "Save embeddings")
        ("full-save",   po::value<vector<string>>()->multitoken(), "Save entire model")
        ("train",       po::value<vector<string>>()->multitoken(), "Training files");

    po::variables_map vm;

    vector<string> training_files;
    vector<string> save_files;
    vector<string> fullsave_files;

    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            cout << desc << endl;
            return 1;
        }

        po::notify(vm); // checks required arguments

        if (vm.count("train")) {
            training_files = check_arg_count(vm, "train", 2);
        }

        if (vm.count("save")) {
            save_files = check_arg_count(vm, "save", 2);
        }
        if (vm.count("full-save")) {
            fullsave_files = check_arg_count(vm, "full-save", 2);
        }

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    if (!vm.count("train")) return 1;

    BilingualModel model(config);
    model.train(training_files[0], training_files[1]);

    if (!save_files.empty()) {
        model.saveEmbeddings(save_files[0], save_files[1]);
    }
    if (!fullsave_files.empty()) {
        model.save(fullsave_files[0], fullsave_files[1]);
    }

    return 1;
}
