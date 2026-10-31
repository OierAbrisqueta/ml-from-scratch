#include "scratch_ml/evaluation/ConfusionMatrix.hpp"
#include <unordered_map>
#include <unordered_set>

ConfusionMatrix::ConfusionMatrix(const std::vector<std::string> &yPred, const std::vector<std::string> &yReal) {
    if (yPred.size() != yReal.size()) {
        throw std::invalid_argument("Prediction and real vectors must be the same size");
    }

    totalPredictions = yPred.size();

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

        this->m(row, col)++;
    }
}

int ConfusionMatrix::getTP(size_t classIndex) const {
    return m(classIndex, classIndex);
}

int ConfusionMatrix::getTN(size_t classIndex) const {
    return this->totalPredictions - (getFN(classIndex) + getFP(classIndex) + getTP(classIndex));
}

int ConfusionMatrix::getFP(size_t classIndex) const {
    int fp{0};
    for (auto i{0uz}; i < m.getNumberRows(); i++) {
        if (classIndex != i) fp += m(i, classIndex);
    }
    return fp;
}

int ConfusionMatrix::getFN(size_t classIndex) const {
    int fn{0};
    for (auto i{0uz}; i < m.getNumberColumns(); i++) {
        if (classIndex != i) fn += m(classIndex, i);
    }
    return fn;
}

double ConfusionMatrix::accuracy() const {
    int totalCorrect = 0;
    for (auto i{0uz}; i < this->m.getNumberRows(); i++) {
        totalCorrect += this->getTP(i);
    }

    return static_cast<double>(totalCorrect) / this->totalPredictions;
}

double ConfusionMatrix::errorRate() const {
    return 1 - accuracy();
}