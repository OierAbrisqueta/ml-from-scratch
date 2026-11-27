#ifndef ML_FROM_SCRATCH_RANDOMFOREST_HPP
#define ML_FROM_SCRATCH_RANDOMFOREST_HPP
#include <memory>
#include <optional>
#include <random>

#include "CART.hpp"
#include "Classifier.hpp"

class RandomForest: public Classifier {
public:
    // Sample up to numFeatures distinct columns per split. Zero uses all columns.
    // A supplied seed is reused on each fit. The tree count must be positive.
    RandomForest(std::size_t numTrees, std::size_t numFeatures,
                 std::optional<std::mt19937::result_type> seed = std::nullopt);

    void fit(const std::vector<std::vector<std::string>>& xTrain, const std::vector<std::string>& yTrain) override;
    [[nodiscard]] std::vector<std::string> predict(const std::vector<std::vector<std::string>>& xTest) const override;
    [[nodiscard]] std::vector<std::unordered_map<std::string, double>> predictionProbabilities(const std::vector<std::vector<std::string>> &xSample) const override;

private:
    std::size_t numTrees;
    std::size_t maxFeatures;
    std::optional<std::mt19937::result_type> seed;
    std::size_t featureCount = 0;
    std::vector<std::unique_ptr<CART>> forest;
};

#endif //ML_FROM_SCRATCH_RANDOMFOREST_HPP
