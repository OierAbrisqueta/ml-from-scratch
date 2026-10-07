#ifndef ML_FROM_SCRATCH_RANDOMFOREST_HPP
#define ML_FROM_SCRATCH_RANDOMFOREST_HPP
#include <memory>

#include "CART.hpp"
#include "Classifier.hpp"

class RandomForest: public Classifier {
public:
    RandomForest(size_t numTrees, size_t numFeatures);

    void fit(const std::vector<std::vector<std::string>>& xTrain, const std::vector<std::string>& yTrain) override;
    [[nodiscard]] std::vector<std::string> predict(const std::vector<std::vector<std::string>>& xTrain) const override;

private:
    size_t numTrees;
    size_t maxFeatures;
    std::vector<std::unique_ptr<CART>> forest;
};

#endif //ML_FROM_SCRATCH_RANDOMFOREST_HPP