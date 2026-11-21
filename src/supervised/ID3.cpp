#include <algorithm>
#include <scratch_ml/supervised/ID3.hpp>
#include <scratch_ml/core/matrix.hpp>
#include <cmath>
#include <numeric>

void ID3::fit(const std::vector<std::vector<std::string>>& xTrain, const std::vector<std::string>& yTrain) {
    allClasses.clear();
    for (const std::string& s : yTrain) {
        this->allClasses.insert(s);
    }

    size_t nRows= xTrain.size();
    size_t nCols = xTrain.front().size();

    this->featureValue.clear();
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
    std::iota(activeRows.begin(), activeRows.end(), 0);

    std::vector<int> activeFeatures(nCols);
    std::iota(activeFeatures.begin(), activeFeatures.end(), 0);

    root = id3Recursive(activeRows, activeFeatures);
}

std::unique_ptr<ID3::Node> ID3::id3Recursive(const std::vector<int>& activeRows, const std::vector<int>& availableFeatures) {
    //Get the frequence of each class in active rows
    std::unordered_map<std::string, int> freq;
    for (const std::string& s : this->allClasses) {
        freq[s] = 0;
    }

    for (auto i{0uz}; i < activeRows.size(); i++) {
        freq[data.y[activeRows.at(i)]]++;
    }

    //Compute the proportions
    std::unordered_map<std::string, double> proportions;
    for (const auto& [key, value] : freq) {
        proportions[key] = static_cast<double>(value) / static_cast<double>(activeRows.size());
    }

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
        leafNode->classProbabilities = proportions;
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
        leafNode->classProbabilities = proportions;
        return leafNode;
    }

    int bestSplit = getFeatureMaxInformationGain(activeRows, availableFeatures);
    auto node = std::make_unique<Node>();
    node->isLeaf = false;
    node->value = bestSplit;
    node->classProbabilities = proportions;

    std::vector<int> remainingFeatures = availableFeatures;
    std::erase(remainingFeatures, bestSplit);

    std::unordered_map<int, std::vector<int>> branches;
    for (size_t i{0uz}; i < activeRows.size(); i++) {
        int rowIndex = activeRows[i];
        int featureVal = data.X(rowIndex, bestSplit);
        branches[featureVal].push_back(rowIndex);
    }

    for (const auto& branch : branches) {
        int uniqueFeatureValue = branch.first;
        const std::vector<int>& childRows = branch.second;
        node->childrenNodes[uniqueFeatureValue] = id3Recursive(childRows, remainingFeatures);
    }

    return node;
}

int ID3::getFeatureMaxInformationGain(const std::vector<int>& activeRows,
                const std::vector<int>& availableFeatures) {
    double maxInformationGain = -1.0;
    int maxFeature = -1;
    for (int feature: availableFeatures) {
        double currentInformationGain = getInformationGain(feature, activeRows);
        if (currentInformationGain > maxInformationGain) {
            maxFeature = feature;
            maxInformationGain = currentInformationGain;
        }
    }

    return maxFeature;
}

double ID3::getInformationGain(int feature, const std::vector<int>& activeRows) {
    double entropy = getEntropy(activeRows);

    std::unordered_map<int, std::vector<int>> subsets;
    for (auto i{0uz}; i < activeRows.size(); i++) {
        int rowIndex = activeRows[i];
        int currentFeatureValue = data.X(rowIndex, feature);
        subsets[currentFeatureValue].push_back((rowIndex));
    }

    double conditionalEntropy = 0.0;
    for (const auto& entry : subsets) {
        double weight = (double) entry.second.size() / (double) activeRows.size();
        double entropySubset = getEntropy(entry.second);
        conditionalEntropy += weight * entropySubset;
    }

    return entropy - conditionalEntropy;
}

double ID3::getEntropy(const std::vector<int>& activeRows) {
    std::unordered_map<std::string, int> frequencies;
    for (auto i{0uz}; i < activeRows.size(); i++) {
        std::string current = data.y[activeRows[i]];
        frequencies[current]++;
    }

    double entropy = 0.0;
    for (const auto& entry : frequencies) {
        double pi = (double)entry.second / (double)activeRows.size();
        entropy -= (pi * std::log2(pi));
    }

    return entropy;
}

std::vector<std::string> ID3::predict(const std::vector<std::vector<std::string>>& xTrain) const {
    std::vector<std::string> predictions;

    for (auto i{0uz}; i < xTrain.size(); i++) {
        Node* currentNode = root.get();
        bool predictionFound = true;

        while (!currentNode->isLeaf) {
            int value = currentNode->value;
            const std::string& prediction = xTrain[i][value];

            if (!featureValue[value].contains(prediction)) {
                predictionFound = false;
                break;
            }

            int predictionValue = featureValue[value].at(prediction);
            if (!currentNode->childrenNodes.contains(predictionValue)) {
                predictionFound = false;
                break;
            }

            currentNode = currentNode->childrenNodes.at(predictionValue).get();
        }
        if (predictionFound) {
            predictions.push_back(currentNode->predictedClass);
        } else {
            predictions.push_back("Unknown");
        }

    }

    return predictions;
}

//Calculates the probability of each class
std::vector<std::unordered_map<std::string, double>> ID3::predictionProbabilities(const std::vector<std::vector<std::string> > &xSample) const {
    std::vector<std::unordered_map<std::string, double>> classProbabilities;

    for (auto i{0uz}; i < xSample.size(); i++) {
        Node* currentNode = root.get();
        const std::vector<std::string>& sample = xSample[i];

        while (!currentNode->isLeaf) {
            int feature = currentNode->value;
            const std::string& category = sample[feature];

            auto codeIt = featureValue[feature].find(category);
            if (codeIt == featureValue[feature].end()) {
                break;
            }

            auto childIt = currentNode->childrenNodes.find(codeIt->second);
            if (childIt == currentNode->childrenNodes.end()) {
                break;
            }

            currentNode = childIt->second.get();
        }

        classProbabilities.push_back(currentNode->classProbabilities);
    }

    return classProbabilities;
}
