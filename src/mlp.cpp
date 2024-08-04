#include <random>
#include <chrono>
#include <cstdlib>

#include <iostream>

#include "../lib/mlp.hpp"

float relu(float x) { return std::max(0.0f, x); }
float drelu(float x) { return (float)(x > 0.0f); }

void MLP::set_input_size(unsigned int size) {
    _input_size = size;
}

void MLP::add_layer(unsigned int size) {
    if(_num_of_layers < MAX_LAYERS) {
        _shape[_num_of_layers] = size;
        _num_of_layers++;
    }
}

void MLP::initialize(std::default_random_engine &seed) {
    std::normal_distribution<float> gaussian(0.0f, 1.0f);
    for(unsigned int l = 0; l < _num_of_layers; l++) {
        for(unsigned int n = 0; n < _shape[l]; n++) {
            unsigned int in_features = (l == 0 ? _input_size : _shape[l-1]);
            for(unsigned int i = 0; i < in_features; i++)
                _weight[l][n][i] = gaussian(seed) / in_features;
            _bias[l][n] = _sum[l][n] = _act[l][n] = _err[l][n] = 0.0f;
        }
    }
}