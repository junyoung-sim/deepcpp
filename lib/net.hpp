#ifndef __NET_HPP_
#define __NET_HPP_

#include <vector>
#include <string>
#include <random>

typedef std::vector<float> vec1d;
typedef std::vector<std::vector<float>> vec2d;

float relu(float x);
float drelu(float x);

class Node
{
private:
    float _bias;
    float _sum;
    float _act;
    float _err;
    vec1d _weight;
public:
    Node() {}
    Node(unsigned int in_features) {
        _bias = _sum = _act = _err = 0.0f;
        _weight.resize(in_features, 0.0f);
    }
    ~Node() { vec1d().swap(_weight); }

    float bias() { return _bias; }
    float sum() { return _sum; };
    float act() { return _act; };
    float err() { return _err; };
    float weight(unsigned int index) { return _weight[index]; }

    void zero() { _sum = _act = _err = 0.0f; }
    void set_bias(float x) { _bias = x; }
    void set_sum(float x) { _sum = x; }
    void set_act(float x) { _act = x; }
    void add_err(float x) { _err = x; }
    void set_weight(unsigned int index, float x) { _weight[index] = x; }
};

class Layer
{
private:
    unsigned int _in_features;
    unsigned int _out_features;
    std::vector<Node> _node;
public:
    Layer() {}
    ~Layer() { std::vector<Node>().swap(_node); }

    unsigned int in_features() { return _in_features; }
    unsigned int out_features() { return _out_features; }

    Node *node(unsigned int index) { return &_node[index]; }
};



#endif