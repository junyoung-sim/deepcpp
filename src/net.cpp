#include <cmath>
#include <vector>
#include <string>
#include <cstdlib>

#include "../lib/net.hpp"

float relu(float x) { return std::max(0.0f, x); }
float drelu(float x) { return (float)(x > 0.0f); }