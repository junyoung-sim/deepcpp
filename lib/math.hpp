#ifndef __MATH_HPP
#define __MATH_HPP

#include <vector>

float relu(float x);
float drelu(float x);

float dot_product(std::vector<float> &a, std::vector<float> &b);

#endif