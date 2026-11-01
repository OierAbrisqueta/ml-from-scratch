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

//Returns the accuracy of the model
double ConfusionMatrix::accuracy() const {
    int totalCorrect = 0;
    for (auto i{0uz}; i < this->m.getNumberRows(); i++) {
        totalCorrect += this->getTP(i);
    }

    return static_cast<double>(totalCorrect) / this->totalPredictions;
}

//Returns the error rate of the model
double ConfusionMatrix::errorRate() const {
    return 1 - accuracy();
}

//Returns the sensitivity of the model: it is calculated differently depending on the average type selected
double ConfusionMatrix::sensitivity(Average averageType) const {
    switch (averageType) {
        case Average::MICRO: {
            int tp{0};
            int fn{0};
            for (auto i{0uz}; i < this->m.getNumberColumns(); i++) {
                tp += this->getTP(i);
                fn += this->getFN(i);
            }

            if (tp + fn == 0) return 0.0;
            return static_cast<double>(tp) / (tp + fn);
        }

        case Average::MACRO: {
            std::vector<double> sensitivitiesMacro = perClassSensitivity();

            double sumMacro{0.0};
            for (auto i{0uz}; i < sensitivitiesMacro.size(); i++) {
                sumMacro += sensitivitiesMacro[i];
            }
            return sumMacro / static_cast<double>(sensitivitiesMacro.size());
        }

        case Average::WEIGHTED: {
            std::vector<double> sensitivitiesWeighted = perClassSensitivity();

            std::vector<double> weights;
            for (auto i{0uz}; i < this->m.getNumberColumns(); i++) {
                int total = getTP(i) + getFN(i);
                weights.push_back(static_cast<double>(total) / this->totalPredictions);
            }

            double sumWeighted{0.0};
            for (auto i{0uz}; i < sensitivitiesWeighted.size(); i++) {
                sumWeighted += sensitivitiesWeighted[i] * weights[i];
            }
            return sumWeighted;
        }

        default:
            throw std::invalid_argument("Not a proper Average Type");
    }
}

std::vector<double> ConfusionMatrix::perClassSensitivity() const {
    std::vector<double> sensitivities;
    for (auto i{0uz}; i < this->m.getNumberColumns(); i++) {
        int tp{0};
        int fn{0};
        tp += this->getTP(i);
        fn += this->getFN(i);
        if (tp + fn == 0) sensitivities.push_back(0.0);
        else {
            sensitivities.push_back(static_cast<double>(tp) / (tp + fn));
        }
    }

    return sensitivities;
}

double ConfusionMatrix::recall(Average averageType) const {
    return sensitivity(averageType);
}

//Returns the specificity of the model: it is calculated differently depending on the average type selected
double ConfusionMatrix::specificity(Average averageType) const {
    switch (averageType) {
        case Average::MICRO: {
            int tn{0};
            int fp{0};
            for (auto i{0uz}; i < this->m.getNumberColumns(); i++) {
                tn += this->getTN(i);
                fp += this->getFP(i);
            }

            if (tn + fp == 0) return 0.0;
            return static_cast<double>(tn) / (tn + fp);
        }

        case Average::MACRO: {
            std::vector<double> specificityMacro = perClassSpecificity();

            double sumMacro{0.0};
            for (auto i{0uz}; i < specificityMacro.size(); i++) {
                sumMacro += specificityMacro[i];
            }
            return sumMacro / static_cast<double>(specificityMacro.size());
        }

        case Average::WEIGHTED: {
            std::vector<double> specificitiesWeighted = perClassSpecificity();

            std::vector<double> weights;
            for (auto i{0uz}; i < this->m.getNumberColumns(); i++) {
                int total = getTP(i) + getFN(i);
                weights.push_back(static_cast<double>(total) / this->totalPredictions);
            }

            double sumWeighted{0.0};
            for (auto i{0uz}; i < specificitiesWeighted.size(); i++) {
                sumWeighted += specificitiesWeighted[i] * weights[i];
            }
            return sumWeighted;
        }

        default:
            throw std::invalid_argument("Not a proper Average Type");
    }
}

std::vector<double> ConfusionMatrix::perClassSpecificity() const {
    std::vector<double> sensitivities;
    for (auto i{0uz}; i < this->m.getNumberColumns(); i++) {
        int tn{0};
        int fp{0};
        tn += this->getTN(i);
        fp += this->getFP(i);
        if (tn + fp == 0) sensitivities.push_back(0.0);
        else {
            sensitivities.push_back(static_cast<double>(tn) / (tn + fp));
        }
    }

    return sensitivities;
}

double ConfusionMatrix::precision(Average averageType) const {
    switch (averageType) {
        case Average::MICRO: {
            int tp{0};
            int fp{0};
            for (auto i{0uz}; i < this->m.getNumberColumns(); i++) {
                tp += this->getTP(i);
                fp += this->getFP(i);
            }

            if (tp + fp == 0) return 0.0;
            return static_cast<double>(tp) / (tp + fp);
        }

        case Average::MACRO: {
            std::vector<double> precisionMacro = perClassPrecision();

            double sumMacro{0.0};
            for (auto i{0uz}; i < precisionMacro.size(); i++) {
                sumMacro += precisionMacro[i];
            }
            return sumMacro / static_cast<double>(precisionMacro.size());
        }

        case Average::WEIGHTED: {
            std::vector<double> precisionWeighted = perClassPrecision();

            std::vector<double> weights;
            for (auto i{0uz}; i < this->m.getNumberColumns(); i++) {
                int total = getTP(i) + getFN(i);
                weights.push_back(static_cast<double>(total) / this->totalPredictions);
            }

            double sumWeighted{0.0};
            for (auto i{0uz}; i < precisionWeighted.size(); i++) {
                sumWeighted += precisionWeighted[i] * weights[i];
            }
            return sumWeighted;
        }

        default:
            throw std::invalid_argument("Not a proper Average Type");
    }
}

std::vector<double> ConfusionMatrix::perClassPrecision() const {
    std::vector<double> precisions;
    for (auto i{0uz}; i < this->m.getNumberColumns(); i++) {
        int tp{0};
        int fp{0};
        tp += this->getTP(i);
        fp += this->getFP(i);
        if (tp + fp == 0) precisions.push_back(0.0);
        else {
            precisions.push_back(static_cast<double>(tp) / (tp + fp));
        }
    }

    return precisions;
}