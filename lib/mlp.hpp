#ifndef __MLP_HPP_
#define __MLP_HPP_

#include <vector>
#include <random>

float relu(float x);
float drelu(float x);

class MLP
{
private:
    unsigned int _input_size;
    std::vector<unsigned int> _shape;
    std::vector<std::vector<float>> _bias;
    std::vector<std::vector<float>> _sum;
    std::vector<std::vector<float>> _act;
    std::vector<std::vector<float>> _err;
    std::vector<std::vector<std::vector<float>>> _weight;
public:
    MLP() {
        _input_size = 0;
    }
    ~MLP() {}

    void set_input_size(unsigned int size);
    void add_layer(unsigned int size);
    void initialize(std::default_random_engine &seed);
};

#endif