#include <vector>
#include <cstdlib>
#include <fstream>
#include <iostream>

std::ifstream images;
std::ifstream labels;

int magic_number, num_of_images, rows, cols;
int label_magic_number, num_of_labels;

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

    return 1;
}

int main(int argc, char *argv[])
{
    if(!parse_mnist_dataset()) return 0;

    std::cout << magic_number << "\n";
    std::cout << num_of_images << "\n";
    std::cout << rows << "x" << cols << "\n";
    std::cout << label_magic_number << "\n";
    std::cout << num_of_labels << "\n";

    return 0;
}