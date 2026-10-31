#include "scratch_ml/evaluation/ConfusionMatrix.hpp"
#include <unordered_map>
#include <unordered_set>

ConfusionMatrix::ConfusionMatrix(const std::vector<std::string> &yPred, const std::vector<std::string> &yReal) {
    if (yPred.size() != yReal.size()) {
        throw std::invalid_argument("Prediction and real vectors must be the same size");
    }

    //Find every unique category
    std::unordered_set<std::string> everyCategory;
    for (const std::string& s : yPred) {
        everyCategory.insert(s);
    }
    for (const std::string& s : yReal) {
        everyCategory.insert(s);
    }

    //Coordinate map and the Reverse coordinate map
    std::size_t counter{0};
    for (const std::string& s : everyCategory) {
        catToInt[s] = counter;
        intToCat[counter] = s;
        counter++;
    }

    //Build and fill the confusion matrix
    std::size_t n = everyCategory.size();
    std::vector<std::vector<int>> zeroMatrix(n, std::vector<int>(n, 0));
    m = Matrix(zeroMatrix);
    for (auto i{0uz}; i < yReal.size(); i++) {
        const std::string& rowString = yReal.at(i);
        std::size_t row = catToInt[rowString];

        const std::string& colString = yPred.at(i);
        std::size_t col = catToInt[colString];

        m(row, col) = m(row, col)++;
    }
}
