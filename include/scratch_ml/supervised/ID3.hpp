#ifndef ML_FROM_SCRATCH_ID3_HPP
#define ML_FROM_SCRATCH_ID3_HPP
#include "DecisionTree.hpp"

class ID3: public DecisionTree {
public:
    ID3() = default;

    void fit(const std::vector<std::vector<std::string>>& xTrain, const std::vector<std::string>& yTrain) override;
    [[nodiscard]] std::vector<std::string> predict(const std::vector<std::vector<std::string>>& xTrain) const override;

private:
    struct Node {
        bool isLeaf = false;
        int value = -1;
        std::unordered_map<int, std::unique_ptr<Node>> childrenNodes;
        std::string predictedClass;
    };

    DataSet data;
    std::unique_ptr<Node> root;

    std::vector<std::unordered_map<std::string, int>> featureValue;

    std::unique_ptr<Node> id3Recursive(const std::vector<int>& activeRows,
                const std::vector<int>& availableFeatures);
    double getEntropy(const std::vector<int>& activeRows);
    double getInformationGain(int feature, const std::vector<int>& activeRows);
    int getFeatureMaxInformationGain(const std::vector<int>& activeRows,
                const std::vector<int>& availableFeatures);
};

#endif //ML_FROM_SCRATCH_ID3_HPP