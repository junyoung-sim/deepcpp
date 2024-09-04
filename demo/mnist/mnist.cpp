#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <algorithm>

#include "../../lib/mlp.hpp"

std::ifstream images;
std::ifstream labels;

std::vector<std::vector<float>> x;
std::vector<std::vector<float>> y;

int magic_number, num_of_images, rows, cols;
int label_magic_number, num_of_labels;
int num_of_classes = 10;

unsigned int batch = 10;
unsigned int epochs = 100;
float learning_rate = 0.001;
float l2_regularization = 0.01;

std::uniform_int_distribution<int> randint(0, num_of_classes-1);
std::default_random_engine seed(std::chrono::system_clock::now().time_since_epoch().count());

int reverse(int i) {
    unsigned char b1, b2, b3, b4;
    b1 = i & 255;
    b2 = (i >> 8) & 255;
    b3 = (i >> 16) & 255;
    b4 = (i >> 24) & 255;
    return (int)(b1 << 24) + (int)(b2 << 16) + (int)(b3 << 8) + b4;
}

int parse_mnist_dataset() {
    images.open("train-images-idx3-ubyte", std::ios::binary);
    labels.open("train-labels-idx1-ubyte", std::ios::binary);

    if(!images.is_open() || !labels.is_open()) {
        std::cout << "Failed to open MNIST dataset...\n";
        return 0;
    }

    images.read((char*)&magic_number, sizeof(magic_number));
    images.read((char*)&num_of_images, sizeof(num_of_images));
    images.read((char*)&rows, sizeof(rows));
    images.read((char*)&cols, sizeof(cols));

    labels.read((char*)&label_magic_number, sizeof(label_magic_number));
    labels.read((char*)&num_of_labels, sizeof(num_of_labels));

    magic_number = reverse(magic_number);
    num_of_images = reverse(num_of_images);
    rows = reverse(rows);
    cols = reverse(cols);
    label_magic_number = reverse(label_magic_number);
    num_of_labels = reverse(num_of_labels);

    x.resize(num_of_images, std::vector<float>());
    y.resize(num_of_labels, std::vector<float>(num_of_classes, 0.0));

    for(unsigned int i = 0; i < num_of_images; i++) {
        for(unsigned int r = 0; r < rows; r++) {
            for(unsigned int c = 0; c < cols; c++) {
                unsigned char pixel;
                images.read((char*)&pixel, sizeof(pixel));
                x[i].push_back((int)pixel);
            }
        }
    }

    for(unsigned int i = 0; i < num_of_labels; i++) {
        unsigned char label;
        labels.read((char*)&label, sizeof(label));
        y[i][label] = 1.0f;
    }

    images.close();
    labels.close();

    return 1;
}

int main(int argc, char *argv[])
{
    std::cout << std::fixed;
    std::cout.precision(12);

    if(!parse_mnist_dataset()) return 0;

    MLP mnist;
    mnist.set_input_size(rows * cols);
    mnist.add_layer(rows * cols);
    mnist.add_layer(rows * cols);
    mnist.add_layer(rows * cols);
    mnist.add_layer(num_of_classes);
    mnist.set_output_type("softmax");
    mnist.initialize(seed);

    for(unsigned int epoch = 1; epoch <= epochs; epoch++) {
        std::vector<unsigned int> index(num_of_images);
        std::iota(index.begin(), index.end(), 0);
        std::shuffle(index.begin(), index.end(), seed);

        for(unsigned int i = 0; i < num_of_images; i++) {
            if(i > 0 && i % batch == 0) {
                mnist.step();
                mnist.zero_grad();
            }
            mnist.backward(x[i], y[i], learning_rate, l2_regularization);
        }

        float loss = 0.0f;
        for(unsigned int i = 0; i < num_of_images; i++) {
            std::vector<float> out = mnist.forward(x[i]);
            for(unsigned int k = 0; k < num_of_classes; k++)
                loss += -1.0f * y[i][k] * log(out[k]);
        }
        loss /= num_of_images;

        std::cout << "[" << epoch << "/" << epochs << "] L=" << loss << "\n";
    }

    return 0;
}