#include <scratch_ml/supervised/RandomForest.hpp>

RandomForest::RandomForest(size_t numTrees, size_t numFeatures) {
    this->maxFeatures = numFeatures;
    this->numTrees = numTrees;
    this->forest.reserve(this->numTrees);
}
