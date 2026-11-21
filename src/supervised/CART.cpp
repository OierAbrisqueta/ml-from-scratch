#include <scratch_ml/supervised/CART.hpp>
#include <unordered_set>
#include <numeric>

void CART::fit(const std::vector<std::vector<std::string>>& xTrain, const std::vector<std::string>& yTrain) {
    this->allClasses.clear();
    for (const std::string& s: yTrain) {
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

    this->root = buildTreeRecursive(activeRows);
}

std::vector<std::string> CART::predict(const std::vector<std::vector<std::string>>& xTrain) const {
    std::vector<std::string> predictions;

    for (auto i{0uz}; i < xTrain.size(); i++) {
        Node* currentNode = root.get();
        bool predictionFound = true;

        while (!currentNode->isLeaf) {
            int currentFeature = currentNode->splitFeature;
            const std::string& categoryStr = xTrain[i][currentFeature];

            if (!featureValue[currentFeature].contains(categoryStr)) {
                predictionFound = false;
                break;
            }

            int categoryInt = featureValue[currentFeature].at(categoryStr);
            if (currentNode->splitValue == categoryInt) {
                currentNode = currentNode->left.get();
            } else {
                currentNode = currentNode->right.get();
            }
        }
        if (predictionFound) {
            predictions.push_back(currentNode->predictedClass);
        } else {
            predictions.push_back("Unknown");
        }

    }

    return predictions;
}

std::unique_ptr<CART::Node> CART::buildTreeRecursive(const std::vector<int>& activeRows) {
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
    std::string value = data.y[activeRows[0]];
    bool areSame = true;
    for (auto i{0uz}; i < activeRows.size(); i++) {
        if (data.y[activeRows[i]] != value) {
            areSame = false;
            break;
        }
    }
    if (areSame) {
        auto leafNode = std::make_unique<Node>();
        leafNode->isLeaf = true;
        leafNode->predictedClass = value;
        leafNode->classProbabilities = proportions;
        return leafNode;
    }

    int bestFeature = -1;
    int bestValue = -1;
    getBestSplit(activeRows, bestFeature, bestValue);

    //Base Case 2
    if (bestFeature == -1) {
        std::unordered_map<std::string, int> frequencies;
        for (auto i{0uz}; i < activeRows.size(); i++) {
            frequencies[data.y[activeRows[i]]]++;
        }

        std::string mostCommon;
        int maxFreq = -1;
        for (const auto freq : frequencies) {
            if (freq.second > maxFreq) {
                mostCommon = freq.first;
                maxFreq = freq.second;
            }
        }

        auto leafNode = std::make_unique<Node>();
        leafNode->isLeaf = true;
        leafNode->predictedClass = mostCommon;
        leafNode->classProbabilities = proportions;
        return leafNode;
    }

    std::vector<int> leftNodes;
    std::vector<int> rightNodes;
    splitData(activeRows, bestFeature, bestValue, leftNodes, rightNodes);

    //If split concluded in no meaningful split
    if (rightNodes.empty() || leftNodes.empty()) {
        std::unordered_map<std::string, int> frequencies;
        for (auto i{0uz}; i < activeRows.size(); i++) {
            frequencies[data.y[activeRows[i]]]++;
        }

        std::string mostCommon;
        int maxFreq = -1;
        for (const auto freq : frequencies) {
            if (freq.second > maxFreq) {
                mostCommon = freq.first;
                maxFreq = freq.second;
            }
        }

        auto leafNode = std::make_unique<Node>();
        leafNode->isLeaf = true;
        leafNode->predictedClass = mostCommon;
        leafNode->classProbabilities = proportions;
        return leafNode;
    }

    auto node = std::make_unique<Node>();
    node->classProbabilities = proportions;
    node->isLeaf = false;
    node->left = buildTreeRecursive(leftNodes);
    node ->right = buildTreeRecursive(rightNodes);
    node->splitFeature = bestFeature;
    node->splitValue = bestValue;
    return node;
}

double CART::getGiniImpurity(const std::vector<int>& activeRows) const {
    std::unordered_map<std::string, int> frequencies;
    for (size_t i{0uz}; i < activeRows.size(); i++) {
        std::string current = data.y[activeRows[i]];
        frequencies[current]++;
    }

    double gini = 0.0;
    for (const auto& freq : frequencies) {
        double pi = static_cast<double>(freq.second) / static_cast<double>(activeRows.size());
        gini += pi * pi;
    }

    return 1 - gini;
}

void CART::splitData(const std::vector<int>& activeRows, int feature, int value,
                std::vector<int>& leftRows, std::vector<int>& rightRows) {
    for (int i = 0; i < activeRows.size(); i++) {
        int realRow = activeRows[i];
        int valueFeature = data.X(realRow, feature);
        if (value == valueFeature) {
            leftRows.push_back(realRow);
        } else {
            rightRows.push_back(realRow);
        }
    }
}

void CART::getBestSplit(const std::vector<int>& activeRows, int& bestFeature, int& bestValue) {
    double lowestImpurity = 1.1;
    for (int i = 0; i < data.X.getNumberColumns(); i++) {
        std::unordered_set<int> uniqueValues;
        for (size_t j{0uz}; j < activeRows.size(); j++) {
            int row = activeRows[j];
            int value = data.X(row, i);
            uniqueValues.insert(value);
        }

        for (int candidate : uniqueValues) {
            std::vector<int> right;
            std::vector<int> left;
            splitData(activeRows, i, candidate, left, right);
            double giniLeft = getGiniImpurity(left);
            double giniRight = getGiniImpurity(right);
            double weightedGini = (static_cast<double>(left.size())/static_cast<double>(activeRows.size())) * giniLeft + (static_cast<double>(right.size())/static_cast<double>(activeRows.size())) * giniRight;
            if (lowestImpurity > weightedGini) {
                lowestImpurity = weightedGini;
                bestFeature = i;
                bestValue = candidate;
            }
        }
    }
}

std::vector<std::unordered_map<std::string, double> > CART::predictionProbabilities(const std::vector<std::vector<std::string> > &xSample) const {
    std::vector<std::unordered_map<std::string, double>> classProbabilities;

    for (auto i{0uz}; i < xSample.size(); i++) {
        Node* currentNode = root.get();
        const std::vector<std::string>& sample = xSample[i];

        while (!currentNode->isLeaf) {
            int feature = currentNode->splitFeature;
            const std::string& category = sample[feature];

            auto codeIt = featureValue[feature].find(category);
            if (codeIt == featureValue[feature].end()) {
                break;
            }

            if (codeIt->second == currentNode->splitValue) {
                currentNode = currentNode->left.get();
            } else {
                currentNode = currentNode->right.get();
            }
        }

        classProbabilities.push_back(currentNode->classProbabilities);
    }

    return classProbabilities;
}
