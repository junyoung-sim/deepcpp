#ifndef __MLP_HPP_
#define __MLP_HPP_

#include <vector>
#include <random>
#include <string>

#include "math.hpp"

class MLP
{
private:
    unsigned int _input_size;
    unsigned int _backward_count;
    std::vector<unsigned int> _shape;
    std::vector<std::vector<float>> _sum;
    std::vector<std::vector<float>> _act;
    std::vector<std::vector<float>> _err;
    std::vector<std::vector<float>> _bias;
    std::vector<std::vector<float>> _bias_grad;
    std::vector<std::vector<std::vector<float>>> _weight;
    std::vector<std::vector<std::vector<float>>> _weight_grad;
    std::string _output_type;

    bool initialized;

public:
    MLP() {
        _input_size = 0;
        _backward_count = 0;
        initialized = false;
    }
    ~MLP() {}

    unsigned int input_size();
    std::vector<unsigned int> *shape();
    std::vector<std::vector<float>> *sum();
    std::vector<std::vector<float>> *act();
    std::vector<std::vector<float>> *err();
    std::vector<std::vector<float>> *bias();
    std::vector<std::vector<float>> *bias_grad();
    std::vector<std::vector<std::vector<float>>> *weight();
    std::vector<std::vector<std::vector<float>>> *weight_grad();
    std::string output_type();

    void set_input_size(unsigned int size);
    void set_output_type(std::string output_type);
    void add_layer(unsigned int size);
    void initialize(std::default_random_engine &seed);

    std::vector<float> forward(std::vector<float> &x);

    void backward(std::vector<float> &x, std::vector<float> &y, float alpha, float lambda);
    void step();
    void zero_grad();
};

#endif