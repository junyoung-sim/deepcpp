#include <vector>
#include <cstdlib>
#include "../lib/math.hpp"

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