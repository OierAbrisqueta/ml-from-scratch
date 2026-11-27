#ifndef ML_FROM_SCRATCH_ROCCURVE_HPP
#define ML_FROM_SCRATCH_ROCCURVE_HPP
#include <string>
#include <unordered_map>
#include <vector>

struct ROCPoint {
    double fpr; // 1 - Specificity
    double tpr; //Sensitivity or Recall
    double threshold; // Probability cutoff
};

class ROCCurve {
public:
    ROCCurve(const std::vector<std::unordered_map<std::string, double>>& yProbabilities, const std::vector<std::string>& yReal);

    [[nodiscard]] const std::unordered_map<std::string, std::vector<ROCPoint>>& getClassCurves() const;

    [[nodiscard]] const std::vector<ROCPoint>& getMicroCurve() const;

    // Throws std::out_of_range if targetClass has no ROC curve.
    [[nodiscard]] double getClassAUC(const std::string& targetClass) const;

    [[nodiscard]] double getMicroAUC() const;

private:
    std::unordered_map<std::string, std::vector<ROCPoint>> classCurves;
    std::vector<ROCPoint> microCurve;
    std::unordered_map<std::string, int> classSupport;

    [[nodiscard]] static std::vector<ROCPoint> computeBinaryCurve(const std::vector<std::pair<double, bool>>& scoredLabels,
        std::size_t totalPositives, std::size_t totalNegatives);
    [[nodiscard]] static double computeAUC(const std::vector<ROCPoint>& curve);
};

#endif //ML_FROM_SCRATCH_ROCCURVE_HPP
