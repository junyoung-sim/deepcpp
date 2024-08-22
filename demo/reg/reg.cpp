#include <cmath>
#include <random>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <iostream>

#include "../../lib/mlp.hpp"

unsigned int batch = 10;
unsigned int input_size = 100;
unsigned int output_size = 3;
unsigned int epochs = 1000;

float learning_rate = 0.001;
float l2_regularization = 0.01;

std::normal_distribution<float> gaussian(0.0f, 1.0f);
std::default_random_engine seed(std::chrono::system_clock::now().time_since_epoch().count());

int main(int argc, char *argv[])
{
    std::cout << std::fixed;
    std::cout.precision(12);

    std::vector<std::vector<float>> x(batch, std::vector<float>(input_size, 0.0f));
    for(unsigned int i = 0; i < batch; i++) {
        for(unsigned int j = 0; j < input_size; j++)
            x[i][j] = gaussian(seed);
    }

    std::vector<std::vector<float>> y(batch, std::vector<float>(output_size, 0.0f));
    for(unsigned int i = 0; i < batch; i++) {
        for(unsigned int j = 0; j < output_size; j++)
            y[i][j] = gaussian(seed);
    }
    
    MLP classifier;
    classifier.set_input_size(input_size);
    classifier.add_layer(input_size);
    classifier.add_layer(input_size);
    classifier.add_layer(input_size);
    classifier.add_layer(output_size);
    classifier.set_output_type("linear");
    classifier.initialize(seed);

    for(unsigned int epoch = 1; epoch <= epochs; epoch++) {
        classifier.zero_grad();
        for(unsigned int i = 0; i < batch; i++)
            classifier.backward(x[i], y[i], learning_rate, l2_regularization);
        classifier.step();

        float batch_loss = 0.0f;
        for(unsigned int i = 0; i < batch; i++) {
            float rss = 0.0f;
            std::vector<float> out = classifier.forward(x[i]);
            for(unsigned int j = 0; j < output_size; j++)
                rss += pow(y[i][j] - out[j], 2);
            batch_loss += rss / output_size;
        }
        batch_loss /= batch;

        if(epoch % (epochs / 10)) continue;

        std::cout << "[" << epoch << "/" << epochs << "] ";
        std::cout << "L=" << batch_loss << "\n";
    }

    return 0;
}