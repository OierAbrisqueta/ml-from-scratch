#include <stdexcept>
#include <scratch_ml/core/matrix.hpp>

Matrix::Matrix(std::vector<std::vector<int>>& rows): nRows(rows.size()), nCols(rows.empty() ? 0: rows.at(0).size()) {
        this->values.reserve(nCols * nRows);
        for (const auto& row : rows) {
                if (row.size() != nCols) throw std::invalid_argument("jagged rows");
                values.insert(values.end(), row.begin(), row.end());
        }
}

Matrix::Matrix(std::size_t nRows, std::size_t nCols, std::vector<int>& data):
        nRows(nRows), nCols(nCols), values(data){}

int& Matrix::operator()(std::size_t i, std::size_t j) {
        return values[i * nCols + j];
}

int Matrix::operator()(std::size_t i, std::size_t j) const {
        return values[i * nCols + j];
}

std::size_t Matrix::getNumberRows() const {
        return nRows;
}

size_t Matrix::getNumberColumns() const {
        return nCols;
}