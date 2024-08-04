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
    _shape.push_back(size);
    _bias.push_back(std::vector<float>(size, 0.0f));
    _sum.push_back(std::vector<float>(size, 0.0f));
    _act.push_back(std::vector<float>(size, 0.0f));
    _err.push_back(std::vector<float>(size, 0.0f));
    _weight.push_back(std::vector<std::vector<float>>(size, std::vector<float>()));
}

void MLP::initialize(std::default_random_engine &seed) {
    std::normal_distribution<float> gaussian(0.0f, 1.0f);
    for(unsigned int l = 0; l < _shape.size(); l++) {
        for(unsigned int n = 0; n < _shape[l]; n++) {
            unsigned int in_features = (l == 0 ? _input_size : _shape[l-1]);
            for(unsigned int i = 0; i < in_features; i++) {
                _weight[l][n].push_back(gaussian(seed) / in_features);
                std::cout << _weight[l][n][i] << " ";
            }
            std::cout << _bias[l][n] << " ";
            std::cout << _sum[l][n] << " ";
            std::cout << _act[l][n] << " ";
            std::cout << _err[l][n] << "\n";
        }
    }
}