#include "scratch_ml/evaluation/ConfusionMatrix.hpp"
#include <unordered_map>
#include <unordered_set>
#include <cmath>

#include <stdexcept>

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

//Global Metrics

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

//Binary / Specific class Evaluation Methods

//Returns the sensitivity of the model
double ConfusionMatrix::sensitivity(const std::string& targetClass) const {
    auto it = this->catToInt.find(targetClass);
    if (it == this->catToInt.end()) throw std::invalid_argument("The target class is not in the confusion matrix");

    int index = it->second;

    int tp = getTP(index);
    int fn = getFN(index);

    if (tp + fn == 0) return 0.0;

    return static_cast<double>(tp) / (tp + fn);
}

//Returns the recall of the model
double ConfusionMatrix::recall(const std::string& targetClass) const {
    return sensitivity(targetClass);
}

//Returns the specificity of the model
double ConfusionMatrix::specificity(const std::string& targetClass) const {
    auto it = this->catToInt.find(targetClass);
    if (it == this->catToInt.end()) throw std::invalid_argument("The target class is not in the confusion matrix");

    int index = it->second;

    int tn = getTN(index);
    int fp = getFP(index);

    if (tn + fp == 0) return 0.0;

    return static_cast<double>(tn) / (tn + fp);
}

//Returns the precision of the model
double ConfusionMatrix::precision(const std::string& targetClass) const {
    auto it = this->catToInt.find(targetClass);
    if (it == this->catToInt.end()) throw std::invalid_argument("The target class is not in the confusion matrix");

    int index = it->second;

    int tp = getTN(index);
    int fp = getFP(index);

    if (tp + fp == 0) return 0.0;

    return static_cast<double>(tp) / (tp + fp);
}

//Returns the f measure of the model
double ConfusionMatrix::fMeasure(const std::string& targetClass, double beta) const {
    auto it = this->catToInt.find(targetClass);
    if (it == this->catToInt.end()) throw std::invalid_argument("The target class is not in the confusion matrix");

    double prec = precision(targetClass);
    double rec = recall(targetClass);
    return (1 + beta * beta) * prec * rec / (beta * beta * prec + rec);
}

double ConfusionMatrix::mcc(const std::string& targetClass) const {
    auto it = this->catToInt.find(targetClass);
    if (it == this->catToInt.end()) {
        throw std::invalid_argument("Class label not found in the confusion matrix.");
    }

    std::size_t classIndex = it->second;

    double tp = static_cast<double>(this->getTP(classIndex));
    double tn = static_cast<double>(this->getTN(classIndex));
    double fp = static_cast<double>(this->getFP(classIndex));
    double fn = static_cast<double>(this->getFN(classIndex));

    double numerator = (tp * tn) - (fp * fn);

    double denominator = std::sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));

    if (denominator == 0.0) return 0.0;
    return numerator / denominator;
}

//Multiclass Evaluation Methods

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

//Returns the precision of the model: it is calculated differently depending on the average type selected
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

double ConfusionMatrix::fMeasure(Average averageType, double beta) const {
    double prec = precision(averageType);
    double rec = recall(averageType);
    return (1 + beta * beta) * prec * rec / (beta * beta * prec + rec);
}

double ConfusionMatrix::mcc() const {
    double s = static_cast<double>(this->totalPredictions);
    double c = 0.0;
    double sum_pk_tk = 0.0;
    double sum_pk_squared = 0.0;
    double sum_tk_squared = 0.0;

    for (auto k{0uz}; k < this->m.getNumberColumns(); k++) {
        double tp = static_cast<double>(this->getTP(k));
        double tk = static_cast<double>(this->getTP(k) + this->getFN(k));
        double pk = static_cast<double>(this->getTP(k) + this->getFP(k));

        c += tp;
        sum_pk_tk += (pk * tk);
        sum_pk_squared += (pk * pk);
        sum_tk_squared += (tk * tk);
    }

    double numerator = (c * s) - sum_pk_tk; // c * s - sum(pk * tk)
    double denominator = std::sqrt((s * s - sum_pk_squared) * (s * s - sum_tk_squared)); // sqrt((s^2 - sum(pk^2)) * (s^2 - sum(tk^2)))

    if (denominator == 0.0) return 0.0;
    return numerator / denominator;
}
