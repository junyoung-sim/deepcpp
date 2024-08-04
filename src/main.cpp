#include <random>
#include <chrono>
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
    net.add_layer(3);
    net.add_layer(1);
    net.initialize(seed);

    return 0;
}