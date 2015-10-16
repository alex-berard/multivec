#include "word2vec.hpp"
#include "boost/program_options.hpp"

int main(int argc, char **argv) {
    Config config;

    namespace po = boost::program_options;
    po::options_description desc("Options");

    desc.add_options()
        ("help,h", "Print help message")
        ("alpha",       po::value<float>(&config.starting_alpha),  "Learning rate")
        ("dimension",   po::value<int>(&config.dimension),         "Dimension of the embeddings")
        ("min-count",   po::value<int>(&config.min_count),         "Minimum count of each word in the vocabulary")
        ("window-size", po::value<int>(&config.window_size),       "Window size")
        ("threads",     po::value<int>(&config.n_threads),         "Number of threads")
        ("iter",        po::value<int>(&config.max_iterations),    "Number of training iterations")
        ("subsampling", po::value<float>(&config.subsampling),     "Subsampling parameter (0 for no subsampling)")
        ("sg",          po::bool_switch(&config.skip_gram),        "Skip-gram-model (cbow model by default)")
        ("hs",          po::bool_switch(&config.hierarchical_softmax), "Hierarchical softmax (negative sampling by default)")
        ("verbose,v",   po::bool_switch(&config.verbose),          "Verbose mode")
        ("negative",    po::value<int>(&config.negative),          "Number of negative samples")
        ("load",        po::value<std::string>(),                  "Load existing model")
        ("save",        po::value<std::string>(),                  "Save entire model")
        ("train",       po::value<std::string>(),                  "Training file")
        ("evaluate",    po::value<int>(),                          "Compute accuracy of the model with max vocabulary size")
        ("save-embeddings", po::value<std::string>(),              "Save embeddings")
        ("save-embeddings-txt", po::value<std::string>(),          "Save embeddings in the txt format")
        ("sent-ids",    po::bool_switch(&config.sent_ids),         "Training file includes sentence ids")
        ;

    po::variables_map vm;

    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 0;
        }

        // if model is skip-gram, default learning rate is 0.025
        if (!vm.count("alpha") && config.skip_gram) {
            config.starting_alpha = 0.025;
        }

        po::notify(vm); // checks required arguments
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    MonolingualModel model(config);

    if (vm.count("train")) {
        model.train(vm["train"].as<std::string>());
    } else if (vm.count("load")) {
        model.load(vm["load"].as<std::string>());
    } else {
        return 0; // either train a new model or load an existing model
    }

    if (vm.count("save-embeddings")) {
        model.saveEmbeddings(vm["save-embeddings"].as<std::string>());
    }
    if (vm.count("save-embeddings-txt")) {
        model.saveEmbeddingsTxt(vm["save-embeddings-txt"].as<std::string>());
    }
    if (vm.count("save")) {
        model.save(vm["save"].as<std::string>());
    }
    if (vm.count("evaluate")) {
        model.computeAccuracy(std::cin, vm["evaluate"].as<int>());
    }

    return 0;
}
