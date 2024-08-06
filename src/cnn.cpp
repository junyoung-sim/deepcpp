#include <vector>
#include <cstdlib>
#include "../lib/cnn.hpp"

void ConvPool2D::use_pooling(unsigned int pool_rows, unsigned int pool_cols) {
    _pool_rows = pool_rows;
    _pool_cols = pool_cols;
}

std::vector<std::vector<float>> ConvPool2D::forward(std::vector<std::vector<float>> &x) {
    std::vector<std::vector<float>> out;
    for(unsigned int i0 = 0; i0 <= x.size() - _kernel_rows; i0++) {
        std::vector<float> row;
        for(unsigned int j0 = 0; j0 <= x[i0].size() - _kernel_cols; j0++) {
            float sum = 0.0f;
            for(unsigned int i = i0; i < i0 + _kernel_rows; i++) {
                for(unsigned int j = j0; j < j0 + _kernel_cols; j++)
                    sum += x[i][j] * _weight[i-i0][j-j0];
            }
            row.push_back(relu(sum));
        }
        out.push_back(row);
    }
    return out;
}