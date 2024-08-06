#ifndef __CNN_HPP_
#define __CNN_HPP_

#include <vector>
#include <random>

#include "math.hpp"

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
    ConvPool2D(unsigned int kernel_rows, unsigned int kernel_cols, std::default_random_engine &seed) {
        _kernel_rows = kernel_rows;
        _kernel_cols = kernel_cols;
        _weight.resize(kernel_rows, std::vector<float>(kernel_cols));
        
        std::normal_distribution<float> gaussian(0.0f, 1.0f);
        for(unsigned int i = 0; i < kernel_rows; i++) {
            for(unsigned int j = 0; j < kernel_cols; j++)
                _weight[i][j] = gaussian(seed);
        }
    }
    ~ConvPool2D() {}

    void use_pooling(unsigned int pool_rows, unsigned int pool_cols);

    std::vector<std::vector<float>> forward(std::vector<std::vector<float>> &x);
};

#endif