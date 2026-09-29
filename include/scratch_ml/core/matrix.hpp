#ifndef ML_FROM_SCRATCH_MATRIX_HPP
#define ML_FROM_SCRATCH_MATRIX_HPP

#include <vector>
#include <string>

class Matrix {
public:
    Matrix() = default;
    Matrix(std::vector<std::vector<int>>& rows);
    Matrix(size_t nRows, size_t nCols, std::vector<int>& data);

    int& operator()(size_t i, size_t j);
    int operator()(size_t i, size_t j) const;

    [[nodiscard]] size_t getNumberRows() const;
    [[nodiscard]] size_t getNumberColumns() const;

private:
    size_t nRows = 0, nCols = 0;
    std::vector<int> values;
};

#endif //ML_FROM_SCRATCH_MATRIX_HPP