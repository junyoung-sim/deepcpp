#include <cmath>
#include <random>
#include <chrono>
#include <cstdlib>
#include <cassert>

#include "../lib/mlp.hpp"

unsigned int MLP::input_size() { return _input_size; }
std::vector<unsigned int> *MLP::shape() { return &_shape; }
std::vector<std::vector<float>> *MLP::sum() { return &_sum; }
std::vector<std::vector<float>> *MLP::act() { return &_act; }
std::vector<std::vector<float>> *MLP::err() { return &_err; }
std::vector<std::vector<float>> *MLP::bias() { return &_bias; }
std::vector<std::vector<float>> *MLP::bias_grad() { return &_bias_grad; }
std::vector<std::vector<std::vector<float>>> *MLP::weight() { return &_weight; }
std::vector<std::vector<std::vector<float>>> *MLP::weight_grad() { return &_weight_grad; }
std::string MLP::output_type() { return _output_type; }

void MLP::set_input_size(unsigned int size) { _input_size = size; }

void MLP::set_output_type(std::string output_type) { _output_type = output_type; }

void MLP::add_layer(unsigned int size) {
    _shape.push_back(size);
    _sum.push_back(std::vector<float>(size, 0.0f));
    _act.push_back(std::vector<float>(size, 0.0f));
    _err.push_back(std::vector<float>(size, 0.0f));
    _bias.push_back(std::vector<float>(size, 0.0f));
    _bias_grad.push_back(std::vector<float>(size, 0.0f));
    _weight.push_back(std::vector<std::vector<float>>(size, std::vector<float>()));
    _weight_grad.push_back(std::vector<std::vector<float>>(size, std::vector<float>()));
}

void MLP::initialize(std::default_random_engine &seed) {
    std::normal_distribution<float> gaussian(0.0f, 1.0f);
    for(unsigned int l = 0; l < _shape.size(); l++) {
        unsigned int in_features = (l == 0 ? _input_size : _shape[l-1]);
        for(unsigned int n = 0; n < _shape[l]; n++) {
            for(unsigned int i = 0; i < in_features; i++) {
                _weight[l][n].push_back(gaussian(seed) / in_features);
                _weight_grad[l][n].push_back(0.0f);
            }
        }
    }
    initialized = true;
}

std::vector<float> MLP::forward(std::vector<float> &x) {
    float exp_sum = 0.0f;
    for(unsigned int l = 0; l < _shape.size(); l++) {
        for(unsigned int n = 0; n < _shape[l]; n++) {
            _sum[l][n] = _act[l][n] = _err[l][n] = 0.0f;
            _sum[l][n] = dot_product((l == 0 ? x : _act[l-1]), _weight[l][n]) + _bias[l][n];
            if(l != _shape.size() - 1) {
                _act[l][n] = relu(_sum[l][n]);
                continue;
            }
            if(_output_type == "linear") _act[l][n] = _sum[l][n];
            if(_output_type == "sigmoid") _act[l][n] = sigmoid(_sum[l][n]);
            if(_output_type == "softmax") exp_sum += exp(_sum[l][n]);
        }
    }
    if(_output_type == "softmax") {
        for(unsigned int n = 0; n < _shape.back(); n++)
            _act.back()[n] = exp(_sum.back()[n]) / exp_sum;
    }
    return _act.back();
}

void MLP::backward(std::vector<float> &x, std::vector<float> &y, float alpha, float lambda) {
    std::vector<float> out = forward(x);
    //std::vector<float> ierr(_input_size, 0.0f);
    for(int l = _shape.size() - 1; l >= 0; l--) {
        float partial_gradient = 0.0f, full_gradient = 0.0f;
        unsigned int in_features = (l == 0 ? _input_size : _shape[l-1]);
        for(unsigned int n = 0; n < _shape[l]; n++) {
            if(l == _shape.size() - 1) {
                if(_output_type == "linear") partial_gradient = -2.0f * (y[n] - out[n]);
                if(_output_type == "sigmoid" || _output_type == "softmax") partial_gradient = out[n] - y[n];
            }
            else partial_gradient = _err[l][n] * drelu(_sum[l][n]);

            _bias_grad[l][n] += alpha * partial_gradient;
            for(unsigned int i = 0; i < in_features; i++) {
                if(l == 0) {
                    full_gradient = partial_gradient * x[i];
                    //ierr[i] += partial_gradient * _weight[l][n][i];
                }
                else {
                    full_gradient = partial_gradient * _act[l-1][i];
                    _err[l-1][i] += partial_gradient * _weight[l][n][i];
                }
                full_gradient += lambda * _weight[l][n][i];
                _weight_grad[l][n][i] += alpha * full_gradient;
            }
        }
    }
    _backward_count++;
    //return ierr;
}

void MLP::step() {
    assert(_backward_count != 0);
    for(unsigned int l = 0; l < _shape.size(); l++) {
        unsigned int in_features = (l == 0 ? _input_size : _shape[l-1]);
        for(unsigned int n = 0; n < _shape[l]; n++) {
            _bias[l][n] -= _bias_grad[l][n] / _backward_count;
            for(unsigned int i = 0; i < in_features; i++)
                _weight[l][n][i] -= _weight_grad[l][n][i] / _backward_count;
        }
    }
}

void MLP::zero_grad() {
    for(unsigned int l = 0; l < _shape.size(); l++) {
        unsigned int in_features = (l == 0 ? _input_size : _shape[l-1]);
        for(unsigned int n = 0; n < _shape[l]; n++) {
            _bias_grad[l][n] = 0.0f;
            for(unsigned int i = 0; i < in_features; i++)
                _weight_grad[l][n][i] = 0.0f;
        }
    }
    _backward_count = 0;
}