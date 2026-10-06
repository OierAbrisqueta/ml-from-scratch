#ifndef ML_FROM_SCRATCH_CART_HPP
#define ML_FROM_SCRATCH_CART_HPP
#include <vector>
#include <scratch_ml/supervised/DecisionTree.hpp>

class CART: public DecisionTree {
public:
    CART() = default;

    void fit(const std::vector<std::vector<std::string>>& xTrain, const std::vector<std::string>& yTrain) override;
    [[nodiscard]] std::vector<std::string> predict(const std::vector<std::vector<std::string>>& xTrain) const override;

private:
    struct Node {
        bool isLeaf = false;
        int splitFeature = -1;
        int splitValue = -1;
        std::string predictedClass;

        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
    };

    DataSet data;
    std::unique_ptr<Node> root;

    std::vector<std::unordered_map<std::string, int>> featureValue;

    std::unique_ptr<Node> buildTreeRecursive(const std::vector<int>& activeRows);
    [[nodiscard]] double getGiniImpurity(const std::vector<int>& activeRows) const;
    void getBestSplit(const std::vector<int>& activeRows, int& bestFeature, int& bestValue);
    void splitData(const std::vector<int>& activeRows, int feature, int value,
                    std::vector<int>& leftRows, std::vector<int>& rightRows);
};

#endif //ML_FROM_SCRATCH_CART_HPP