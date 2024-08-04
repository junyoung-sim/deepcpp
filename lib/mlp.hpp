#ifndef __MLP_HPP_
#define __MLP_HPP_

#include <vector>
#include <random>

typedef struct std::vector<float> vec1d;
typedef struct std::vector<vec1d> vec2d;
typedef struct std::vector<vec2d> vec3d;

#define MAX_LAYERS 100
#define MAX_NODES 1000

float relu(float x);
float drelu(float x);

class MLP
{
private:
    vec2d _bias;
    vec2d _sum;
    vec2d _act;
    vec2d _err;
    vec3d _weight;
    unsigned int _input_size;
    unsigned int _num_of_layers;
    unsigned int _shape[MAX_LAYERS];
public:
    MLP() {
        _bias.resize(MAX_LAYERS, vec1d(MAX_NODES));
        _sum.resize(MAX_LAYERS, vec1d(MAX_NODES));
        _act.resize(MAX_LAYERS, vec1d(MAX_NODES));
        _err.resize(MAX_LAYERS, vec1d(MAX_NODES));
        _weight.resize(MAX_LAYERS, vec2d(MAX_NODES, vec1d(MAX_NODES)));
        
        _input_size = _num_of_layers = 0;
    }
    ~MLP() {}

    void set_input_size(unsigned int size);
    void add_layer(unsigned int size);
    void initialize(std::default_random_engine &seed);
};

#endif