#ifndef __CNN_HPP_
#define __CNN_HPP_

#include <vector>
#include "mlp.hpp"

class CNN
{
private:
    unsigned int _input_rows;
    unsigned int _input_cols;
    unsigned int _num_of_layers;
    std::vector<std::vector<float>> _kernel;
    std::vector<std::vector<unsigned int>> _kernel_shape;
    std::vector<std::vector<float>> _pool;
    std::vector<std::vector<unsigned int>> _pool_shape;
    MLP _mlp;
public:
    CNN() {
        _input_rows = _input_cols = _num_of_layers = 0;
    }
    ~CNN() {}

    unsigned int input_rows();
    unsigned int input_cols();
    unsigned int num_of_layers();
    std::vector<std::vector<float>> *kernel();
    std::vector<std::vector<unsigned int>> *kernel_shape();
    std::vector<std::vector<float>> *pool();
    std::vector<std::vector<unsigned int>> *pool_shape();
    MLP *mlp();

    
};

#endif