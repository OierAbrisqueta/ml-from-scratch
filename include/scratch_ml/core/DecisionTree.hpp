#ifndef ML_FROM_SCRATCH_DECISIONTREE_HPP
#define ML_FROM_SCRATCH_DECISIONTREE_HPP

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class DecisionTree {
public:
    virtual ~DecisionTree() = default;

    virtual void fit(const std::vector<std::vector<std::string>>& xTrain, const std::vector<std::string>& yTrain) = 0;
    [[nodiscard]] virtual std::vector<std::string> predict(const std::vector<std::vector<std::string>>& xTrain) const = 0;
};
#endif //ML_FROM_SCRATCH_DECISIONTREE_HPP