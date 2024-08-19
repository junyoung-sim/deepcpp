# DeepCPP

DeepCPP is a deep learning framework in C++ built from scratch for regression or classification tasks that may require efficiently configurable and deployable deep neural networks.

The following is a general guideline:

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
    net.set_output_type(...); // specify one of the following: {"linear", "softmax", "sigmoid"}
    net.initialize(seed);

    // training (application-specific)
    for(...) {
        net.zero_grad();
        for(...) {
            net.backward(x, y, learning_rate, l2_regularization);
            // x: std::vector<float>
            // y: std::vector<float>
            // learning_rate: float
            // l2_regularization: float
        }
        net.step();
    }

    // inference
    std::vector<float> out = net.forward(x); // x: std::vector<float>

    return 0;
}
```

Demo source code and executable files can be found in [./demo](./demo).