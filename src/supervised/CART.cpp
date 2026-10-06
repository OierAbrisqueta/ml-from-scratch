#include <scratch_ml/supervised/CART.hpp>
#include <unordered_set>

double CART::getGiniImpurity(const std::vector<int>& activeRows) const {
    std::unordered_map<std::string, int> frequencies;
    for (size_t i{0uz}; i < activeRows.size(); i++) {
        std::string current = data.y[activeRows[i]];
        frequencies[current]++;
    }

    double gini = 0.0;
    for (const auto& freq : frequencies) {
        double pi = static_cast<double>(freq.second) / static_cast<double>(activeRows.size());
        gini += pi * pi;
    }

    return 1 - gini;
}

void CART::splitData(const std::vector<int>& activeRows, int feature, int value,
                std::vector<int>& leftRows, std::vector<int>& rightRows) {
    for (int i = 0; i < activeRows.size(); i++) {
        int valueFeature = data.X(i, feature);
        if (value == valueFeature) {
            leftRows.push_back(i);
        } else {
            rightRows.push_back(i);
        }
    }
}

void CART::getBestSplit(const std::vector<int>& activeRows, int& bestFeature, int& bestValue) {
    double lowestImpurity = 1.1;
    for (int i = 0; i < data.features.size(); i++) {
        std::unordered_set<int> uniqueValues;
        for (size_t j{0uz}; j < activeRows.size(); i++) {
            int row = activeRows[j];
            int value = data.X(row, i);
            uniqueValues.insert(value);
        }

        for (int candidate : uniqueValues) {
            std::vector<int> right;
            std::vector<int> left;
            splitData(activeRows, i, candidate, left, right);
            double giniLeft = getGiniImpurity(left);
            double giniRight = getGiniImpurity(right);
            double weightedGini = (static_cast<double>(left.size())/static_cast<double>(activeRows.size())) * giniLeft + (static_cast<double>(right.size())/static_cast<double>(activeRows.size())) * giniRight;
            if (lowestImpurity < weightedGini) {
                lowestImpurity = weightedGini;
                bestFeature = i;
                bestValue = candidate;
            }
        }
    }
}