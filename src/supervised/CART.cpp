#include <scratch_ml/supervised/CART.hpp>
#include <algorithm>
#include <unordered_set>
#include <numeric>

void CART::fit(const std::vector<std::vector<std::string>>& xTrain, const std::vector<std::string>& yTrain) {
    validateTrainingInput(xTrain, yTrain);
    root.reset();

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

std::vector<std::string> CART::predict(const std::vector<std::vector<std::string>>& rows) const {
    validatePredictionInput(rows, root != nullptr, featureValue.size());
    std::vector<std::string> predictions;
    predictions.reserve(rows.size());
    for (const auto& row : rows) {
        predictions.push_back(mostProbableClass(predictionNode(row).classProbabilities));
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
        leafNode->classProbabilities = proportions;
        return leafNode;
    }

    int bestFeature = -1;
    int bestValue = -1;
    getBestSplit(activeRows, bestFeature, bestValue);

    //Base Case 2
    if (bestFeature == -1) {
        auto leafNode = std::make_unique<Node>();
        leafNode->isLeaf = true;
        leafNode->classProbabilities = proportions;
        return leafNode;
    }

    std::vector<int> leftNodes;
    std::vector<int> rightNodes;
    splitData(activeRows, bestFeature, bestValue, leftNodes, rightNodes);

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
    if (activeRows.empty()) {
        return 0.0;
    }

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
    std::vector<int> candidates(data.X.getNumberColumns());
    std::iota(candidates.begin(), candidates.end(), 0);
    if (maxFeatures > 0 && maxFeatures < candidates.size()) {
        std::shuffle(candidates.begin(), candidates.end(), featureGenerator);
        candidates.resize(maxFeatures);
    }

    double lowestImpurity = 1.1;
    for (int feature : candidates) {
        std::unordered_set<int> uniqueValues;
        for (size_t j{0uz}; j < activeRows.size(); j++) {
            int row = activeRows[j];
            int value = data.X(row, feature);
            uniqueValues.insert(value);
        }

        for (int candidate : uniqueValues) {
            std::vector<int> right;
            std::vector<int> left;
            splitData(activeRows, feature, candidate, left, right);
            if (left.empty() || right.empty()) {
                continue;
            }
            double giniLeft = getGiniImpurity(left);
            double giniRight = getGiniImpurity(right);
            double weightedGini = (static_cast<double>(left.size())/static_cast<double>(activeRows.size())) * giniLeft + (static_cast<double>(right.size())/static_cast<double>(activeRows.size())) * giniRight;
            if (lowestImpurity > weightedGini) {
                lowestImpurity = weightedGini;
                bestFeature = feature;
                bestValue = candidate;
            }
        }
    }
}

const CART::Node& CART::predictionNode(const std::vector<std::string>& sample) const {
    const Node* currentNode = root.get();
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
    return *currentNode;
}

std::vector<std::unordered_map<std::string, double>> CART::predictionProbabilities(
    const std::vector<std::vector<std::string>>& rows) const {
    validatePredictionInput(rows, root != nullptr, featureValue.size());
    std::vector<std::unordered_map<std::string, double>> probabilities;
    probabilities.reserve(rows.size());
    for (const auto& row : rows) {
        probabilities.push_back(predictionNode(row).classProbabilities);
    }
    return probabilities;
}
