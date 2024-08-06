#include <cmath>
#include <random>
#include <chrono>
#include <cstdlib>

#include "../lib/mlp.hpp"

float relu(float x) { return std::max(0.0f, x); }
float drelu(float x) { return (float)(x > 0.0f); }

float dot_product(std::vector<float> &a, std::vector<float> &b) {
    try {
        float dot = 0.0f;
        for(unsigned int i = 0; i < a.size(); i++)
            dot += a[i] * b[i];
        return dot;
    }
    catch(...) { return (float)RAND_MAX; }
}

unsigned int MLP::input_size() { return _input_size; }
std::vector<unsigned int> *MLP::shape() { return &_shape; }
std::vector<std::vector<float>> *MLP::bias() { return &_bias; }
std::vector<std::vector<float>> *MLP::sum() { return &_sum; }
std::vector<std::vector<float>> *MLP::act() { return &_act; }
std::vector<std::vector<float>> *MLP::err() { return &_err; }
std::vector<std::vector<std::vector<float>>> *MLP::weight() { return &_weight; }
std::string MLP::output_type() { return _output_type; }

void MLP::set_input_size(unsigned int size) { _input_size = size; }

void MLP::set_output_type(std::string output_type) { _output_type = output_type; }

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
        unsigned int in_features = (l == 0 ? _input_size : _shape[l-1]);
        for(unsigned int n = 0; n < _shape[l]; n++) {
            for(unsigned int i = 0; i < in_features; i++)
                _weight[l][n].push_back(gaussian(seed) / in_features);
        }
    }
}

std::vector<float> MLP::forward(std::vector<float> &x) {
    float exp_sum = 0.0f;
    for(unsigned int l = 0; l < _shape.size(); l++) {
        for(unsigned int n = 0; n < _shape[l]; n++) {
            if(l == 0) _sum[l][n] = dot_product(x, _weight[l][n]);
            else _sum[l][n] = dot_product(_act[l-1], _weight[l][n]);
            _sum[l][n] += _bias[l][n];

            if(l != _shape.size() - 1) {
                _act[l][n] = relu(_sum[l][n]);
                continue;
            }

            if(_output_type == "linear") _act[l][n] = _sum[l][n];
            if(_output_type == "softmax") exp_sum += exp(_sum[l][n]);
        }
    }

    if(_output_type == "softmax") {
        for(unsigned int n = 0; n < _shape.back(); n++)
            _act.back()[n] = exp(_sum.back()[n]) / exp_sum;
    }
    
    return _act.back();
}

std::vector<float> MLP::update(std::vector<float> &x, std::vector<float> &y, float alpha, float lambda) {
    std::vector<float> out = forward(x);
    std::vector<float> ierr(_input_size, 0.0f);
    for(int l = _shape.size() - 1; l >= 0; l--) {
        float partial_gradient = 0.0f, full_gradient = 0.0f;
        unsigned int in_features = (l == 0 ? _input_size : _shape[l-1]);
        for(unsigned int n = 0; n < _shape[l]; n++) {
            if(l == _shape.size() - 1) {
                if(_output_type == "linear") partial_gradient = -2.0f * (y[n] - out[n]);
                if(_output_type == "softmax") partial_gradient = out[n] - y[n];
            }
            else partial_gradient = _err[l][n] * drelu(_sum[l][n]);

            _bias[l][n] -= alpha * partial_gradient;
            for(unsigned int i = 0; i < in_features; i++) {
                if(l == 0) {
                    full_gradient = partial_gradient * x[i];
                    ierr[i] += partial_gradient * _weight[l][n][i];
                }
                else {
                    full_gradient = partial_gradient * _act[l-1][i];
                    _err[l-1][i] = 0.0f;
                    _err[l-1][i] += partial_gradient * _weight[l][n][i];
                }
                full_gradient += lambda * _weight[l][n][i];
                _weight[l][n][i] -= alpha * full_gradient;
            }
        }
    }
    return ierr;
}