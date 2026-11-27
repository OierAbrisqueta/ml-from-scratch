#ifndef ML_FROM_SCRATCH_CART_HPP
#define ML_FROM_SCRATCH_CART_HPP
#include <vector>
#include <scratch_ml/supervised/Classifier.hpp>
#include <unordered_map>
#include <memory>
#include <random>
#include <unordered_set>

class CART: public Classifier {
public:
    CART() = default;
    CART(std::size_t maxFeatures, std::minstd_rand::result_type seed)
        : maxFeatures(maxFeatures), featureGenerator(seed) {}

    void fit(const std::vector<std::vector<std::string>>& xTrain, const std::vector<std::string>& yTrain) override;
    [[nodiscard]] std::vector<std::string> predict(const std::vector<std::vector<std::string>>& xTrain) const override;
    [[nodiscard]] std::vector<std::unordered_map<std::string, double>> predictionProbabilities(const std::vector<std::vector<std::string>> &xSample) const override;

private:
    struct Node {
        bool isLeaf = false;
        int splitFeature = -1;
        int splitValue = -1;
        std::unordered_map<std::string, double> classProbabilities;

        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
    };

    // Stop at the last reachable node when a category cannot be followed.
    [[nodiscard]] const Node& predictionNode(const std::vector<std::string>& sample) const;

    DataSet data;
    std::size_t maxFeatures = 0;
    std::minstd_rand featureGenerator;
    std::unique_ptr<Node> root;

    std::vector<std::unordered_map<std::string, int>> featureValue;
    std::unordered_set<std::string> allClasses;

    std::unique_ptr<Node> buildTreeRecursive(const std::vector<int>& activeRows);
    [[nodiscard]] double getGiniImpurity(const std::vector<int>& activeRows) const;
    void getBestSplit(const std::vector<int>& activeRows, int& bestFeature, int& bestValue);
    void splitData(const std::vector<int>& activeRows, int feature, int value,
                    std::vector<int>& leftRows, std::vector<int>& rightRows);
};

#endif //ML_FROM_SCRATCH_CART_HPP
