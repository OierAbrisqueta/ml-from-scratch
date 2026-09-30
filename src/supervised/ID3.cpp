#include <bits/ranges_algo.h>
#include <scratch_ml/supervised/ID3.hpp>
#include <scratch_ml/core/matrix.hpp>

void ID3::fit(const std::vector<std::vector<std::string>>& xTrain, const std::vector<std::string>& yTrain) {
    size_t nRows= xTrain.size();
    size_t nCols = xTrain.front().size();

    this->featureValue.resize(nCols);

    std::vector<std::vector<int>> preMatrix(nRows, std::vector<int>(nCols, 0));
    for (auto i{0uz}; i < nCols; i++) {
        int lastValue = 0;
        for (auto j{0uz}; j < nRows; j++) {
            std::string category = xTrain.at(j).at(i);
            if (!featureValue.at(i).contains(category)) {
                featureValue[i][category] = lastValue;
                lastValue++;
            }
            preMatrix.at(j).at(i) = featureValue.at(i).at(category);
        }
    }
    Matrix m(preMatrix);

    this->data = {};
    data.X = m;
    data.y = yTrain;

    std::vector<int> activeRows(nRows);
    std::ranges::iota(activeRows.begin(), activeRows.end(), 0);

    std::vector<int> activeFeatures(nCols);
    std::ranges::iota(activeFeatures.begin(), activeFeatures.end(), 0);

    //root = id3_recursive(activeRows, availableFeatures);
}