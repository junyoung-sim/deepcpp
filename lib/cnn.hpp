#ifndef __CNN_HPP_
#define __CNN_HPP_

#include <vector>
#include <random>

#include "math.hpp"

#include <iostream>

class ConvPool2D
{
private:
    unsigned int _pool_rows;
    unsigned int _pool_cols;
    unsigned int _kernel_rows;
    unsigned int _kernel_cols;
    std::vector<std::vector<float>> _weight;
public:
    ConvPool2D() {
        _pool_rows = _pool_cols = 0;
        _kernel_rows = _kernel_cols = 0;
    }
    ConvPool2D(std::vector<unsigned int> shape, std::default_random_engine &seed) {
        _kernel_rows = shape[0];
        _kernel_cols = shape[1];
        _weight.resize(_kernel_rows, std::vector<float>(_kernel_cols));
        
        std::normal_distribution<float> gaussian(0.0f, 1.0f);
        for(unsigned int i = 0; i < _kernel_rows; i++) {
            for(unsigned int j = 0; j < _kernel_cols; j++) {
                _weight[i][j] = gaussian(seed) / (_kernel_rows * _kernel_cols);
                std::cout << _weight[i][j] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";

        _pool_rows = shape[2];
        _pool_cols = shape[3];
    }
    ~ConvPool2D() {}

    std::vector<std::vector<float>> forward(std::vector<std::vector<float>> &x);
};

#endif