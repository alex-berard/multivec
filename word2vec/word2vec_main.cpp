#include "word2vec.h"
#include "boost/program_options.hpp"

int main(int argc, char **argv) {
    namespace po = boost::program_options;
    po::options_description desc("Options");
    Config config;

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
        ("train",       po::value<std::string>(),  "Training file")
        ("save-embeddings", po::value<std::string>(),"Output file")
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

    if (vm.count("train") && vm.count("save-embeddings")) {
        string training_file = vm["train"].as<std::string>();
        string output_file = vm["save-embeddings"].as<std::string>();

        Main(training_file, output_file, config);
    }

    return 0;
}
