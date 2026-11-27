#include <scratch_ml/supervised/RandomForest.hpp>
#include <random>

RandomForest::RandomForest(std::size_t numTrees, std::size_t numFeatures,
                           std::optional<std::mt19937::result_type> seed)
    : numTrees(numTrees), maxFeatures(numFeatures), seed(seed) {
    if (numTrees == 0) {
        throw std::invalid_argument("RandomForest requires at least one tree");
    }
}

void RandomForest::fit(const std::vector<std::vector<std::string>>& xTrain, const std::vector<std::string>& yTrain) {
    validateTrainingInput(xTrain, yTrain);
    std::vector<std::unique_ptr<CART>> newForest;
    newForest.reserve(numTrees);
    std::mt19937 gen(seed ? *seed : std::random_device{}());
    std::uniform_int_distribution<std::size_t> dist(0, xTrain.size() - 1);

    for (auto i{0uz}; i < this->numTrees; i++) {
        std::vector<std::size_t> randomIndex(xTrain.size());
        for (auto j{0uz}; j < randomIndex.size(); j++) {
            randomIndex[j] = dist(gen);
        }

        std::vector<std::vector<std::string>> tempX;
        std::vector<std::string> tempY;
        tempX.reserve(xTrain.size());
        tempY.reserve(yTrain.size());

        for (auto j{0uz}; j < randomIndex.size(); j++) {
            std::size_t iRandom = randomIndex[j];
            tempX.push_back(xTrain[iRandom]);
            tempY.push_back(yTrain[iRandom]);
        }

        auto tree = std::make_unique<CART>(maxFeatures, gen());
        tree->fit(tempX, tempY);
        newForest.push_back(std::move(tree));
    }
    forest = std::move(newForest);
    featureCount = xTrain.front().size();
}

std::vector<std::string> RandomForest::predict(const std::vector<std::vector<std::string>>& xTest) const {
    const auto probabilities = predictionProbabilities(xTest);
    std::vector<std::string> predictions;
    predictions.reserve(probabilities.size());
    for (const auto& row : probabilities) {
        predictions.push_back(mostProbableClass(row));
    }
    return predictions;
}

std::vector<std::unordered_map<std::string, double>> RandomForest::predictionProbabilities(const std::vector<std::vector<std::string> > &xSample) const {
    validatePredictionInput(xSample, !forest.empty(), featureCount);
    std::vector<std::unordered_map<std::string, double>> result(xSample.size());

    for (const auto& tree : forest) {
        std::vector<std::unordered_map<std::string, double>> treeProb = tree->predictionProbabilities(xSample);
        for (auto j{0uz}; j < treeProb.size(); j++) {
            for (const auto& [key, value] : treeProb[j]) {
                result[j][key] += value;
            }
        }
    }

    for (auto j{0uz}; j < result.size(); j++) {
        for (auto& [key, value] : result[j]) {
            value /= static_cast<double>(this->forest.size());
        }
    }

    return result;
}
