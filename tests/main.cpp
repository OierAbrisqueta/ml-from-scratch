#include <iostream>
#include <scratch_ml/supervised/ID3.hpp>
#include <scratch_ml/supervised/CART.hpp>
#include <scratch_ml/supervised/RandomForest.hpp>

static bool testFeatureSampling() {
    const std::vector<std::vector<std::string>> rows = {
        {"0", "0"}, {"0", "1"}, {"1", "0"}, {"1", "1"}
    };
    const std::vector<std::string> labels = {"no", "yes", "yes", "no"};

    for (std::size_t limit : {0uz, 2uz, 3uz}) {
        CART tree(limit, 42);
        tree.fit(rows, labels);
        if (tree.predict(rows) != labels) {
            return false;
        }
    }

    bool fittedBothColumns = false;
    bool missedSplit = false;
    for (unsigned seed = 1; seed <= 128; ++seed) {
        CART tree(1, seed);
        tree.fit(rows, labels);
        if (tree.predict(rows) == labels) {
            fittedBothColumns = true;
        } else {
            missedSplit = true;
        }
    }
    return fittedBothColumns && missedSplit;
}

static bool testConstantFeatureSplits() {
    CART tree;
    const std::vector<std::vector<std::string>> rows = {
        {"fixed", "0", "0"}, {"fixed", "0", "1"},
        {"fixed", "1", "0"}, {"fixed", "1", "1"}
    };
    const std::vector<std::string> labels = {"no", "yes", "yes", "no"};
    tree.fit(rows, labels);
    if (tree.predict(rows) != labels) {
        return false;
    }

    tree.fit({{"fixed"}, {"fixed"}, {"fixed"}}, {"yes", "no", "yes"});
    return tree.predict({{"fixed"}}).at(0) == "yes"
        && tree.predictionProbabilities({{"fixed"}}).at(0).at("yes") == 2.0 / 3.0;
}

int main(void) {
    if (!testConstantFeatureSplits()) {
        std::cerr << "Constant-feature split regression\n";
        return 1;
    }
    if (!testFeatureSampling()) {
        std::cerr << "Per-split feature sampling regression\n";
        return 1;
    }
    std::vector<std::vector<std::string>> xTrain = {
        {"Sunny", "Hot", "High", "Weak"},
        {"Sunny", "Hot", "High", "Strong"},
        {"Cloudy", "Hot", "High", "Weak"},
        {"Rain", "Mild", "High", "Weak"},
        {"Rain", "Cool", "Normal", "Weak"},
        {"Rain", "Cool", "Normal", "Strong"},
        {"Cloudy", "Cool", "Normal", "Strong"},
        {"Sunny", "Mild", "High", "Weak"}
    };

    std::vector<std::string> yTrain = {
        "No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No"
    };

    std::vector<std::vector<std::string>>   xTest = {
        // Was trained with this combination. Should predict NO.
        {"Sunny", "Hot", "High", "Weak"},

        // Completely new combination (Wasn't trained with this combination)
        {"Sunny", "Cool", "Normal", "Strong"},

        // Edge Case: Contains unseen category. Should return unknown.
        {"Rain", "Mild", "High", "Hurricane"},

        {"Sunny", "Hot", "High", "Weak"},
    };

    //ID3 Algorithm
    ID3 id3DecisionTree;
    id3DecisionTree.fit(xTrain, yTrain);

    std::vector<std::string> predictionsID3 = id3DecisionTree.predict(xTest);
    std::vector<std::unordered_map<std::string, double>> predictionProbabilities = id3DecisionTree.predictionProbabilities(xTest);

    std::cout << std::endl;
    std::cout << "----- ID3 CLASSIFICATION ALGORITHM -----" << std::endl;
    for (auto i{0uz}; i < predictionsID3.size(); i++) {
        std::cout << "Prediction made for combination in row " << i << ": " << predictionsID3[i] << std::endl;
        std::cout << "Probabilities for this row" << std::endl;
        for (const auto& [key, value] : predictionProbabilities[i]) {
            std::cout << "Probabilities for " << key << "is: " << value << std::endl;
        }
    }

    //CART Algorithm
    CART cartDecisionTree;
    cartDecisionTree.fit(xTrain, yTrain);

    std::vector<std::string> predictionsCART = cartDecisionTree.predict(xTest);

    std::cout << std::endl;
    std::cout << "----- CART CLASSIFICATION ALGORITHM -----" << std::endl;
    for (auto i{0uz}; i < predictionsCART.size(); i++) {
        std::cout << "Prediction made for combination in row " << i << ": " << predictionsCART[i] << std::endl;
    }

    //Random Forest Algorithm
    RandomForest rf(100, 2);
    rf.fit(xTrain, yTrain);

    std::vector<std::string> predictionsRF = rf.predict(xTest);

    std::cout << std::endl;
    std::cout << "----- RANDOM FOREST CLASSIFICATION ALGORITHM -----" << std::endl;
    for (auto i{0uz}; i < predictionsRF.size(); i++) {
        std::cout << "Prediction made for combination in row " << i << ": " << predictionsRF[i] << std::endl;
    }

    return 0;
}
