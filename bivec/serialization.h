#pragma once

#include "bivec.h"
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

namespace boost {
namespace serialization {
    template<class Archive>
    void serialize(Archive& ar, Config& cfg, const unsigned int version) {
        ar & cfg.starting_alpha
           & cfg.dimension
           & cfg.min_count
           & cfg.max_iterations
           & cfg.window_size
           & cfg.n_threads
           & cfg.subsampling
           //& cfg.debug
           //& cfg.verbose
           & cfg.hierarchical_softmax
           & cfg.skip_gram
           & cfg.negative;
    }

    template<class Archive>
    void serialize(Archive& ar, BilingualConfig& cfg, const unsigned int version) {
        ar & dynamic_cast<Config&>(cfg);
        ar & cfg.bi_weight;
    }

    template<class Archive>
    void serialize(Archive& ar, HuffmanNode& node, const unsigned int version) {
        ar & node.word & node.code & node.parents & node.index & node.count;
    }
}
}
