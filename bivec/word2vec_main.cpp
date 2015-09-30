#include "word2vec.h"
#include "boost/program_options.hpp"

int main(int argc, char **argv) {
    Config config;

    namespace po = boost::program_options;
    po::options_description desc("Options");

    desc.add_options()
        ("help", "Print help message")
        //("alpha",       po::value<float>(&config.starting_alpha),  "Learning rate")
        ("alpha",       po::value<float>(),                        "Learning rate")
        ("dimension",   po::value<int>(&config.dimension),         "Dimension of the embeddings")
        ("min-count",   po::value<int>(&config.min_count),         "Minimum count of each word in the vocabulary")
        ("window-size", po::value<int>(&config.window_size),       "Window size")
        ("threads",     po::value<int>(&config.n_threads),         "Number of threads")
        ("iter",        po::value<int>(&config.max_iterations),    "Number of training iterations")
        ("sampling",    po::value<float>(&config.sampling),        "Subsampling parameter (0 for no subsampling)")
        ("sg",          po::value<bool>(&config.skip_gram),        "Skip-gram-model (cbow model by default)")
        ("ns",          po::value<bool>(&config.negative_sampling),"Negative sampling (hierarchical softmax by default)")
        ("negative",    po::value<int>(&config.negative),          "Number of negative samples")
        ("train",       po::value<std::string>(),                  "Training file")
        ("save",        po::value<std::string>(),                  "Save embeddings")
        ("full-save",   po::value<std::string>(),                  "Save entire model")
        ;

    po::variables_map vm;

    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 0;
        }

        // if model is skip-gram, default learning rate is 0.025
        if (vm.count("alpha")) {
            config.starting_alpha = vm["alpha"].as<float>();
        } else if (config.skip_gram) {
            config.starting_alpha = 0.025;
        }

        po::notify(vm); // checks required arguments
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    if (!vm.count("train")) return 1;

    MonolingualModel model(config);

    model.train(vm["train"].as<std::string>());

    if (vm.count("save")) {
        model.saveEmbeddings(vm["save"].as<std::string>());
    }
    if (vm.count("full-save")) {
        model.save(vm["full-save"].as<std::string>());
    }

    return 0;
}
