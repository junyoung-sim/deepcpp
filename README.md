## Multi-Layer Perceptron (Deep Neural Network)

This framework allows easily configurable and deployable multi-layer perceptrons, also known as deep neural networks. The following is a general implementation guideline:

```cpp
// declare a global random engine initialized with the system clock
std::default_random_engine seed(std::chrono::system_clock::now().time_since_epoch().count());

int main(int argc, char *argv[])
{
    MLP net;
    net.set_input_size(...); // specify input size
    net.add_layer(...); // specify number of neurons in the layer being added
    /*
        add as many layers as needed
    */
    net.set_output_type(...); // one of the following: {"linear", "softmax", "relu"}
    net.initialize(seed);

    // training (application-specific)
    for(...) {
        net.update(x, y, learning_rate, l2_regularization); // x, y: std::vector<float>
                                                            // learning_rate, l2_regularization: float
    }

    // inference
    std::vector<float> out = net.forward(x); // x: std::vector<float>

    return 0;
}
```

ATTENTION: MODEL PARAMETER SAVE/LOAD (TO BE IMPLEMENTED)

## Convolutional Neural Network

COMING SOON!

CURRENT PROGRESS (2024-08-07): CONVPOOL2D CLASS IS COMPLETE AND VERIFIED.
