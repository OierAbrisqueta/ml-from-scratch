#ifndef ML_FROM_SCRATCH_CONFUSIONMATRIX_HPP
#define ML_FROM_SCRATCH_CONFUSIONMATRIX_HPP

#include <vector>
#include <unordered_map>
#include "scratch_ml/core/matrix.hpp"

enum class Average {
    MACRO,
    MICRO,
    WEIGHTED
};

class ConfusionMatrix {
public:
    ConfusionMatrix(const std::vector<std::string>& yPred, const std::vector<std::string>& yReal);

    [[nodiscard]] double accuracy() const;
    [[nodiscard]] double errorRate() const;

    //Binary/Specific class Evaluation Methods
    [[nodiscard]] double sensitivity(const std::string& targetClass) const;
    [[nodiscard]] double recall(const std::string& targetClass) const;
    [[nodiscard]] double specificity(const std::string& targetClass) const;
    [[nodiscard]] double precision(const std::string& targetClass) const;

    //Multiclass Evaluation Methods
    [[nodiscard]] double sensitivity(Average averageType) const;
    [[nodiscard]] double recall(Average averageType) const;
    [[nodiscard]] double specificity(Average averageType) const;
    [[nodiscard]] double precision(Average averageType) const;
    [[nodiscard]] double fMeasure(Average averageType, double beta = 1.0) const;
    [[nodiscard]] double mcc() const;

private:
    int totalPredictions;
    Matrix m;
    std::unordered_map<std::string, std::size_t> catToInt;
    std::unordered_map<std::size_t, std::string> intToCat;

    [[nodiscard]] int getTP(size_t classIndex) const;
    [[nodiscard]] int getTN(size_t classIndex) const;
    [[nodiscard]] int getFP(size_t classIndex) const;
    [[nodiscard]] int getFN(size_t classIndex) const;

    [[nodiscard]] std::vector<double> perClassSensitivity() const;
    [[nodiscard]] std::vector<double> perClassSpecificity() const;
    [[nodiscard]] std::vector<double> perClassPrecision() const;
};

#endif //ML_FROM_SCRATCH_CONFUSIONMATRIX_HPP