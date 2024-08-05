#include <cmath>
#include <random>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <iostream>

#include "../lib/mlp.hpp"

std::default_random_engine seed(std::chrono::system_clock::now().time_since_epoch().count());

int main(int argc, char *argv[])
{
    std::cout << std::fixed;
    std::cout.precision(12);

    /*
    std::vector<float> x = {-1.0, 0.0, 1.0};
    std::vector<float> y = {0.5};

    MLP net;
    net.set_input_size(3);
    net.add_layer(3);
    net.add_layer(1);
    net.set_output_type("linear");
    net.initialize(seed);

    std::cout << "\n";
    for(unsigned int t = 0; t < 10; t++) {
        net.update(x, y, 0.000001, 0.01);
        std::vector<float> out = net.forward(x);
        std::cout << pow(out[0] - y[0], 2) << "\n";
    }
    */

    std::vector<float> x = {-1.0, 0.0, 1.0};
    std::vector<float> y = {1.0, 0.0};

    MLP net;
    net.set_input_size(3);
    net.add_layer(3);
    net.add_layer(2);
    net.set_output_type("softmax");
    net.initialize(seed);

    std::cout << "\n";
    for(unsigned int t = 0; t < 10; t++) {
        net.update(x, y, 0.000001, 0.01);
        std::vector<float> out = net.forward(x);
        std::cout << -1.0f * (y[0]*log(out[0]) + y[1]*log(out[1])) << "\n";
    }

    return 0;
}