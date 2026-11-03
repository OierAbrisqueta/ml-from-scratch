#include <algorithm>
#include <stdexcept>
#include <scratch_ml/evaluation/ROCCurve.hpp>

/*ROCCurve::ROCCurve(const std::vector<std::unordered_map<std::string, double> > &yProbabilities, const std::vector<std::string> &yReal) {
    if (yProbabilities.empty() || yProbabilities.size() != yReal.size()) {
        throw std::invalid_argument("yReal and yProbabilities should have the same size and can't be empty");
    }

}*/

std::vector<ROCPoint> ROCCurve::computeBinaryCurve(const std::vector<std::pair<double, bool>> &scoredLabels,
    std::size_t totalPositives, std::size_t totalNegatives) {
    //Copy and sort the scored labels
    std::vector<std::pair<double, bool>> sorted = scoredLabels;
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    std::vector<ROCPoint> curve;
    // Start at (0, 0) with a threshold slightly above 1.0 (nothing predicted positive)
    curve.push_back({0.0, 0.0, 1.0});

    if (totalPositives == 0 || totalNegatives == 0) {
        curve.push_back({1.0, 1.0, 0.0});
        return curve;
    }

    std::size_t tp = 0;
    std::size_t fp = 0;

    for (auto i{0uz}; i < sorted.size(); i++) {
        if (sorted[i].second) {
            tp++;
        } else {
            fp++;
        }

        // Only emit a point if this is the last element or if the next probability is different
        bool isLastElement = (i + 1 == sorted.size());
        if (isLastElement || sorted[i].first != sorted[i + 1].first) {
            double fpr = static_cast<double>(fp) / static_cast<double>(totalNegatives);
            double tpr = static_cast<double>(tp) / static_cast<double>(totalPositives);
            curve.push_back({fpr, tpr, sorted[i].first});
        }
    }

    return curve;
}
