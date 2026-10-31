#ifndef ML_FROM_SCRATCH_CONFUSIONMATRIX_HPP
#define ML_FROM_SCRATCH_CONFUSIONMATRIX_HPP

#include <vector>
#include <unordered_map>
#include "scratch_ml/core/matrix.hpp"

class ConfusionMatrix {
public:
    ConfusionMatrix(const std::vector<std::string>& yPred, const std::vector<std::string>& yReal);
private:
    int totalPredictions;
    Matrix m;
    std::unordered_map<std::string, std::size_t> catToInt;
    std::unordered_map<std::size_t, std::string> intToCat;

    [[nodiscard]] int getTP(size_t classIndex) const;
    [[nodiscard]] int getTN(size_t classIndex) const;
    [[nodiscard]] int getFP(size_t classIndex) const;
    [[nodiscard]] int getFN(size_t classIndex) const;
};

#endif //ML_FROM_SCRATCH_CONFUSIONMATRIX_HPP