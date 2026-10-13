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