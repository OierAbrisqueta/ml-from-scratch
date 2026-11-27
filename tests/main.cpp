#include <iostream>
#include <stdexcept>
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

template <typename Exception>
static bool throwsException(auto operation) {
    try {
        operation();
    } catch (const Exception&) {
        return true;
    }
    return false;
}

static bool testClassifierContract(Classifier& classifier) {
    if (!throwsException<std::logic_error>([&] { (void)classifier.predict({}); }) ||
        !throwsException<std::logic_error>([&] { (void)classifier.predictionProbabilities({{"a"}}); }) ||
        !throwsException<std::invalid_argument>([&] { classifier.fit({}, {}); }) ||
        !throwsException<std::invalid_argument>([&] { classifier.fit({{"a"}}, {}); }) ||
        !throwsException<std::invalid_argument>([&] { classifier.fit({{"a"}, {"a", "b"}}, {"yes", "no"}); })) {
        return false;
    }

    classifier.fit({{"a"}, {"a"}, {"b"}}, {"yes", "yes", "no"});
    if (!throwsException<std::invalid_argument>([&] { (void)classifier.predict({{"a"}, {}}); }) ||
        !throwsException<std::invalid_argument>([&] { (void)classifier.predictionProbabilities({{"a", "b"}}); }) ||
        !throwsException<std::invalid_argument>([&] { classifier.fit({}, {}); })) {
        return false;
    }

    const auto probabilities = classifier.predictionProbabilities({{"unseen"}}).at(0);
    const auto prediction = classifier.predict({{"unseen"}}).at(0);
    const auto no = probabilities.find("no");
    const double noProbability = no == probabilities.end() ? 0.0 : no->second;
    const auto expected = probabilities.at("yes") > noProbability ? "yes" : "no";
    return prediction == expected && classifier.predict({{"a"}}).at(0) == "yes"
        && classifier.predict({}).empty() && classifier.predictionProbabilities({}).empty();
}

static bool testSeededForest() {
    const std::vector<std::vector<std::string>> rows = {{"a", "0"}, {"a", "1"}, {"b", "0"}, {"b", "1"}};
    const std::vector<std::string> labels = {"no", "yes", "yes", "no"};
    RandomForest forest(16, 1, 0);
    RandomForest secondForest(16, 1, 0);
    forest.fit(rows, labels);
    secondForest.fit(rows, labels);
    const auto probabilities = forest.predictionProbabilities(rows);
    if (probabilities != secondForest.predictionProbabilities(rows)) {
        return false;
    }

    forest.fit(rows, labels);
    return probabilities == forest.predictionProbabilities(rows)
        && throwsException<std::invalid_argument>([] { RandomForest invalid(0, 1); });
}

int main(void) {
    ID3 id3;
    CART cart;
    RandomForest forest(32, 1, 42);
    if (!testClassifierContract(id3) || !testClassifierContract(cart) ||
        !testClassifierContract(forest) || !testSeededForest()) {
        std::cerr << "Classifier contract regression\n";
        return 1;
    }
    for (Classifier* tree : {static_cast<Classifier*>(&id3), static_cast<Classifier*>(&cart)}) {
        tree->fit({{"a"}, {"b"}}, {"yes", "no"});
        if (tree->predict({{"unseen"}}).at(0) != "no" ||
            tree->predictionProbabilities({{"unseen"}}).at(0).at("yes") != 0.5) {
            std::cerr << "Unseen category fallback regression\n";
            return 1;
        }
    }

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

        // Edge Case: An unseen category uses the last reachable node.
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
