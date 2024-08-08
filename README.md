# DeepCPP

DeepCPP is a simple deep learning framework in C++ built from scratch.

## Multi-Layer Perceptron (Deep Neural Network)

The following is a general guideline for implementing multi-layer perceptrons, also known as deep neural networks.

```cpp
#include "../lib/mlp.hpp"

// declare a global random engine initialized with the system clock
std::default_random_engine seed(std::chrono::system_clock::now().time_since_epoch().count());

int main(int argc, char *argv[])
{
    MLP net;
    net.set_input_size(...); // specify input size
    net.add_layer(...); // specify number of neurons in the layer being added
    /*
        add as many layers as needed (ReLU is used)
    */
    net.set_output_type(...); // specify one of the following: {"linear", "softmax", "relu"}
    net.initialize(seed);

    // training (application-specific)
    for(...) {
        net.update(x, y, learning_rate, l2_regularization);
        /*
            stochastic gradient descent
            x, y: std::vector<float>
            learning_rate, l2_regularization: float
        */
    }

    // inference
    std::vector<float> out = net.forward(x); // x: std::vector<float>

    return 0;
}
```

## Convolutional Neural Network

CONVPOOL2D CLASS IS COMPLETE (2024-08-07). CNN CLASS WILL BE IMPLEMENTED SOON.
