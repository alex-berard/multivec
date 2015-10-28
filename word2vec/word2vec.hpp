#pragma once
#include <iostream>
#include <chrono>

using namespace std::chrono;
//using namespace std;

struct Config {
    float starting_alpha;
    int dimension;
    int min_count;
    int max_iterations;
    int window_size;
    int n_threads;
    float subsampling;
    bool verbose;
    bool hierarchical_softmax;
    bool skip_gram;
    int negative;
    bool sent_vector;
    bool binary;

    Config() :
        starting_alpha(0.05),
        dimension(100),
        min_count(5),
        max_iterations(5),
        window_size(5),
        n_threads(4),
        subsampling(1e-03),
        verbose(false),
        hierarchical_softmax(false),
        skip_gram(false),
        negative(5),
        sent_vector(false),
        binary(false)
        {}

    void print() const {
        std::cout << std::boolalpha;
        std::cout << "dimension:   " << dimension << std::endl;
        std::cout << "window size: " << window_size << std::endl;
        std::cout << "min count:   " << min_count << std::endl;
        std::cout << "alpha:       " << starting_alpha << std::endl;
        std::cout << "iterations:  " << max_iterations << std::endl;
        std::cout << "threads:     " << n_threads << std::endl;
        std::cout << "subsampling: " << subsampling << std::endl;
        std::cout << "skip-gram:   " << skip_gram << std::endl;
        std::cout << "HS:          " << hierarchical_softmax << std::endl;
        std::cout << "negative:    " << negative << std::endl;
        std::cout << "sent vector: " << sent_vector << std::endl;
    }
};

void Main(std::string train_file, std::string output_file, Config config);
