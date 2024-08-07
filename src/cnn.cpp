#include <vector>
#include <cstdlib>
#include "../lib/cnn.hpp"

#include <iostream>

std::vector<std::vector<float>> ConvPool2D::forward(std::vector<std::vector<float>> &x) {
    std::vector<std::vector<float>> conv;
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
        conv.push_back(row);
    }

    for(unsigned int i = 0; i < conv.size(); i++) {
        for(unsigned int j = 0; j < conv[0].size(); j++)
            std::cout << conv[i][j] << " ";
        std::cout << "\n";
    }
    std::cout << "\n";

    if(_pool_rows == 0 && _pool_cols == 0) return conv;

    std::vector<std::vector<float>> pool;
    for(unsigned int i0 = 0; i0 <= conv.size() - _pool_rows; i0 += _pool_rows) {
        std::vector<float> row;
        for(unsigned int j0 = 0; j0 <= conv[0].size() - _pool_cols; j0 += _pool_cols) {
            float max_value = conv[i0][j0];
            for(unsigned int i = i0; i < i0 + _pool_rows; i++) {
                for(unsigned int j = j0; j < j0 + _pool_cols; j++)
                    max_value = std::max(conv[i][j], max_value);
            }
            row.push_back(max_value);
        }
        pool.push_back(row);
    }

    for(unsigned int i = 0; i < pool.size(); i++) {
        for(unsigned int j = 0; j < pool[0].size(); j++)
            std::cout << pool[i][j] << " ";
        std::cout << "\n";
    }
    std::cout << "\n";

    return pool;
}