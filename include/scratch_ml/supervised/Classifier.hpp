#ifndef ML_FROM_SCRATCH_CLASSIFIER_HPP
#define ML_FROM_SCRATCH_CLASSIFIER_HPP

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <scratch_ml/core/matrix.hpp>

struct DataSet {
    Matrix X;
    std::vector<std::string> y;
    std::vector<std::string> features;
};

class Classifier {
public:
    virtual ~Classifier() = default;

    // Training requires at least one row, uniform row widths, and one label per row
    virtual void fit(const std::vector<std::vector<std::string>>& xTrain, const std::vector<std::string>& yTrain) = 0;
    // Both prediction methods require a fitted model
    // and rows matching the training set
    // A fitted model accepts an empty prediction set
    // Labels maximize the returned probabilities
    [[nodiscard]] virtual std::vector<std::string> predict(const std::vector<std::vector<std::string>>& xTrain) const = 0;
    [[nodiscard]] virtual std::vector<std::unordered_map<std::string, double>> predictionProbabilities(const std::vector<std::vector<std::string>>& xSample) const = 0;
protected:
    static void validateTrainingInput(const std::vector<std::vector<std::string>>& rows,
                                      const std::vector<std::string>& labels) {
        if (rows.empty() || rows.size() != labels.size()) {
            throw std::invalid_argument("Training requires at least one row and one label per row");
        }
        validateRowWidths(rows, rows.front().size());
    }

    static void validatePredictionInput(const std::vector<std::vector<std::string>>& rows,
                                        bool fitted, std::size_t width) {
        if (!fitted) {
            throw std::logic_error("Call fit before predicting");
        }
        validateRowWidths(rows, width);
    }

    static std::string mostProbableClass(const std::unordered_map<std::string, double>& probabilities) {
        return std::max_element(probabilities.begin(), probabilities.end(),
            [](const auto& a, const auto& b) {
                if (a.second == b.second) {
                    return a.first > b.first;
                }
                return a.second < b.second;
            })->first;
    }

private:
    static void validateRowWidths(const std::vector<std::vector<std::string>>& rows, std::size_t width) {
        for (const auto& row : rows) {
            if (row.size() != width) {
                throw std::invalid_argument("All rows must match the training feature count");
            }
        }
    }
};
#endif //ML_FROM_SCRATCH_CLASSIFIER_HPP