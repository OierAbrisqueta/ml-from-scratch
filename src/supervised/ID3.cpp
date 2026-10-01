#include <bits/ranges_algo.h>
#include <scratch_ml/supervised/ID3.hpp>
#include <scratch_ml/core/matrix.hpp>

void ID3::fit(const std::vector<std::vector<std::string>>& xTrain, const std::vector<std::string>& yTrain) {
    size_t nRows= xTrain.size();
    size_t nCols = xTrain.front().size();

    this->featureValue.resize(nCols);

    std::vector<std::vector<int>> preMatrix(nRows, std::vector<int>(nCols, 0));
    for (auto i{0uz}; i < nCols; i++) {
        int lastValue = 0;
        for (auto j{0uz}; j < nRows; j++) {
            std::string category = xTrain.at(j).at(i);
            if (!featureValue.at(i).contains(category)) {
                featureValue[i][category] = lastValue;
                lastValue++;
            }
            preMatrix.at(j).at(i) = featureValue.at(i).at(category);
        }
    }
    Matrix m(preMatrix);

    this->data = {};
    data.X = m;
    data.y = yTrain;

    std::vector<int> activeRows(nRows);
    std::ranges::iota(activeRows.begin(), activeRows.end(), 0);

    std::vector<int> activeFeatures(nCols);
    std::ranges::iota(activeFeatures.begin(), activeFeatures.end(), 0);

    root = id3Recursive(activeRows, activeFeatures);
}

std::unique_ptr<ID3::Node> ID3::id3Recursive(const std::vector<int>& activeRows, const std::vector<int>& availableFeatures) {
    //Base Case 1
    std::string firstLabel = data.y[activeRows.front()];
    bool isNode = true;

    for (auto i{0uz}; i < activeRows.size(); i++) {
        if (firstLabel != data.y[activeRows[i]]) {
            isNode = false;
            break;
        }
    }

    if (isNode) {
        auto leafNode = std::make_unique<Node>();
        leafNode->isLeaf = true;
        leafNode->predictedClass = firstLabel;
        return leafNode;
    }

    //Base Case 2
    if (availableFeatures.empty()) {
        std::unordered_map<std::string, int> frequencies;
        for (auto i{0uz}; i < activeRows.size(); i++) {
            std::string current = data.y[activeRows[i]];
            frequencies[current]++;
        }

        int maxFrequency = -1;
        std::string mode;
        for (const auto& entry : frequencies) {
            if (entry.second > maxFrequency) {
                maxFrequency = entry.second;
                mode = entry.first;
            }
        }

        auto leafNode = std::make_unique<Node>();
        leafNode->isLeaf = true;
        leafNode->predictedClass = mode;
        return leafNode;
    }
}