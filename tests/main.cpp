#include <iostream>
#include <scratch_ml/supervised/ID3.hpp>
#include <scratch_ml/supervised/CART.hpp>

int main(void) {
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

    ID3 id3DecisionTree;
    id3DecisionTree.fit(xTrain, yTrain);

    std::vector<std::vector<std::string>> xTest = {
        // Was trained with this combination. Should predict NO.
        {"Sunny", "Hot", "High", "Weak"},

        // Completely new combination (Wasn't trained with this combination)
        {"Sunny", "Cool", "Normal", "Strong"},

        // Edge Case: Contains unseen category. Should return unknown.
        {"Rain", "Mild", "High", "Hurricane"},

        {"Sunny", "Hot", "High", "Weak"},
    };

    std::vector<std::string> predictionsID3 = id3DecisionTree.predict(xTest);

    std::cout << "----- ID3 CLASSIFICATION ALGORITHM -----" << std::endl;
    for (auto i{0uz}; i < predictionsID3.size(); i++) {
        std::cout << "Prediction made for combination in row " << i << ": " << predictionsID3[i] << std::endl;
    }

    CART cartDecisionTree;
    cartDecisionTree.fit(xTrain, yTrain);

    std::vector<std::string> predictionsCART = cartDecisionTree.predict(xTest);

    std::cout << "----- CART CLASSIFICATION ALGORITHM -----" << std::endl;
    for (auto i{0uz}; i < predictionsCART.size(); i++) {
        std::cout << "Prediction made for combination in row " << i << ": " << predictionsCART[i] << std::endl;
    }

    return 0;
}