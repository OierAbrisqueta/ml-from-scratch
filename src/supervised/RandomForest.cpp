#include <scratch_ml/supervised/RandomForest.hpp>
#include <random>

RandomForest::RandomForest(std::size_t numTrees, std::size_t numFeatures) {
    this->maxFeatures = numFeatures;
    this->numTrees = numTrees;
    this->forest.reserve(this->numTrees);
}

void RandomForest::fit(const std::vector<std::vector<std::string>>& xTrain, const std::vector<std::string>& yTrain) {
    std::random_device rd;
    std::mt19937 gen(rd());
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

        auto tree = std::make_unique<CART>();
        tree->fit(tempX, tempY);
        this->forest.push_back(std::move(tree));
    }
}

std::vector<std::string> RandomForest::predict(const std::vector<std::vector<std::string>>& xTest) const {
    std::vector<std::vector<std::string>> predictions;
    predictions.reserve(this->numTrees);

    for (auto i{0uz}; i < this->numTrees; i++) {
        const std::unique_ptr<CART>& tree = this->forest[i];
        std::vector<std::string> prediction = tree->predict(xTest);
        predictions.push_back(std::move(prediction));
    }

    std::vector<std::string> finalPredictions;
    finalPredictions.reserve(predictions.front().size());
    for (auto j{0uz}; j < xTest.size(); j++) {
        std::unordered_map<std::string, int> frequencies;
        for (auto i{0uz}; i < this->numTrees; i++) {
            frequencies[predictions[i][j]]++;
        }

        int max = 0;
        std::string maxString;
        for (const auto& frequency : frequencies) {
            if (frequency.second > max) {
                max = frequency.second;
                maxString = frequency.first;
            }
        }
        finalPredictions.push_back(maxString);
    }

    return finalPredictions;
}