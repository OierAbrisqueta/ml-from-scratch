#ifndef ML_FROM_SCRATCH_ID3_HPP
#define ML_FROM_SCRATCH_ID3_HPP
#include "Classifier.hpp"
#include <unordered_map>
#include <memory>
#include <unordered_set>

class ID3: public Classifier {
public:
    ID3() = default;

    void fit(const std::vector<std::vector<std::string>>& xTrain, const std::vector<std::string>& yTrain) override;
    [[nodiscard]] std::vector<std::string> predict(const std::vector<std::vector<std::string>>& xTrain) const override;
    std::vector<std::unordered_map<std::string, double>> predictionProbabilities(const std::vector<std::vector<std::string>> &xSample) const override;

private:
    struct Node {
        bool isLeaf = false;
        int value = -1;
        std::unordered_map<int, std::unique_ptr<Node>> childrenNodes;
        std::unordered_map<std::string, double> classProbabilities;
    };

    // Stop at the last reachable node when a category cannot be followed.
    [[nodiscard]] const Node& predictionNode(const std::vector<std::string>& sample) const;

    DataSet data;
    std::unique_ptr<Node> root;
    std::unordered_set<std::string> allClasses;

    std::vector<std::unordered_map<std::string, int>> featureValue;

    std::unique_ptr<Node> id3Recursive(const std::vector<int>& activeRows,
                const std::vector<int>& availableFeatures);
    double getEntropy(const std::vector<int>& activeRows);
    double getInformationGain(int feature, const std::vector<int>& activeRows);
    int getFeatureMaxInformationGain(const std::vector<int>& activeRows,
                const std::vector<int>& availableFeatures);
};

#endif //ML_FROM_SCRATCH_ID3_HPP