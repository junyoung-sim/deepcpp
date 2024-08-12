#include <cmath>
#include <random>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <iostream>

#include "../lib/mlp.hpp"

std::default_random_engine seed(std::chrono::system_clock::now().time_since_epoch().count());

void test_regression() {
    std::normal_distribution<float> gaussian(0.0f, 1.0f);
    std::vector<float> x;
    for(unsigned int i = 0; i < 5; i++)
        x.push_back(gaussian(seed));
    std::vector<float> y = {0.25, 0.50, 0.75};

    MLP reg;
    reg.set_input_size(5);
    reg.add_layer(5);
    reg.add_layer(5);
    reg.add_layer(3);
    reg.set_output_type("linear");
    reg.initialize(seed);

    std::cout << "REGRESSION\n";
    for(unsigned int t = 1; t <= 10000; t++) {
        reg.update(x, y, 0.001, 0.01);

        if(t % 1000) continue;

        std::vector<float> out = reg.forward(x);

        float loss = 0.0f;
        for(unsigned int i = 0; i < out.size(); i++)
            loss += pow(y[i] - out[i], 2);
        loss /= out.size();
        
        std::cout << "L=" << loss << " [";
        std::cout << out[0] << " ";
        std::cout << out[1] << " ";
        std::cout << out[2] << "]\n";
    }
    std::cout << "\n";
}

void test_logistic_regression() {
    std::normal_distribution<float> gaussian(0.0f, 1.0f);
    std::vector<float> x;
    for(unsigned int i = 0; i < 5; i++)
        x.push_back(gaussian(seed));
    std::vector<float> y = {1.0, 1.0, 0.0};

    MLP logreg;
    logreg.set_input_size(5);
    logreg.add_layer(5);
    logreg.add_layer(5);
    logreg.add_layer(3);
    logreg.set_output_type("sigmoid");
    logreg.initialize(seed);

    std::cout << "LOGISTIC REGRESSION\n";
    for(unsigned int t = 1; t <= 10000; t++) {
        logreg.update(x, y, 0.001, 0.01);

        if(t % 1000) continue;

        std::vector<float> out = logreg.forward(x);

        float loss = 0.0f;
        for(unsigned int i = 0; i < out.size(); i++)
            loss += -1.0f * y[i] * log(out[i]) - (1.0f - y[i]) * log(1.0f - out[i]);
        
        std::cout << "L=" << loss << " [";
        std::cout << out[0] << " ";
        std::cout << out[1] << " ";
        std::cout << out[2] << "]\n";
    }
    std::cout << "\n";
}

void test_classification() {
    std::normal_distribution<float> gaussian(0.0f, 1.0f);
    std::vector<float> x;
    for(unsigned int i = 0; i < 5; i++)
        x.push_back(gaussian(seed));
    std::vector<float> y = {1.0, 0.0, 0.0};

    MLP classifier;
    classifier.set_input_size(5);
    classifier.add_layer(5);
    classifier.add_layer(5);
    classifier.add_layer(3);
    classifier.set_output_type("softmax");
    classifier.initialize(seed);

    std::cout << "CLASSIFICATION\n";
    for(unsigned int t = 1; t <= 10000; t++) {
        classifier.update(x, y, 0.001, 0.01);

        if(t % 1000) continue;

        std::vector<float> out = classifier.forward(x);

        float loss = 0.0f;
        for(unsigned int i = 0; i < out.size(); i++)
            loss += -1.0f * y[i] * log(out[i]);
        
        std::cout << "L=" << loss << " [";
        std::cout << out[0] << " ";
        std::cout << out[1] << " ";
        std::cout << out[2] << "]\n";
    }
    std::cout << "\n";
}


int main(int argc, char *argv[])
{
    std::cout << std::fixed;
    std::cout.precision(12);

    test_regression();
    test_logistic_regression();
    test_classification();

    return 0;
}