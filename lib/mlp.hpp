#ifndef __MLP_HPP_
#define __MLP_HPP_

#include <cstdlib>

#define MAX_LAYERS 100
#define MAX_NODES 1000

class MLP
{
private:
    float _bias[MAX_LAYERS][MAX_NODES];
    float _sum[MAX_LAYERS][MAX_NODES];
    float _act[MAX_LAYERS][MAX_NODES];
    float _err[MAX_LAYERS][MAX_NODES];
    float _weight[MAX_LAYERS][MAX_NODES][MAX_NODES];
    unsigned int _shape[MAX_LAYERS];
    unsigned int _num_of_layers;
public:
    MLP(): _num_of_layers(0) {}
    ~MLP() {}

    void add_layer(unsigned int size);
};

#endif