#include <vector>
#include <cstdlib>
#include "../lib/cnn.hpp"

unsigned int CNN::input_rows() { return _input_rows; }
unsigned int CNN::input_cols() { return _input_cols; }
unsigned int CNN::num_of_layers() { return _num_of_layers; }
std::vector<std::vector<float>> *CNN::kernel() { return &_kernel; }
std::vector<std::vector<unsigned int>> *CNN::kernel_shape() { return &_kernel_shape; }
std::vector<std::vector<float>> *CNN::pool() { return &_pool; }
std::vector<std::vector<unsigned int>> *CNN::pool_shape() { return &_pool_shape; }
MLP *CNN::mlp() { return &_mlp; }

