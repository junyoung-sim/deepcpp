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

    MLP net;
    net.set_input_size(3);
    net.set_output_type("softmax");
    net.add_layer(3);
    net.add_layer(2);
    net.initialize(seed);

    std::vector<float> x = {-1.0, 0.0, 1.0};
    std::vector<float> y = net.forward(x);

    std::cout << y[0] << " " << y[1] << "\n";

    return 0;
}