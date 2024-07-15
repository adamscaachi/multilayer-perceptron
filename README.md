# multilayer-perceptron

The multilayer perceptron contains 3 layers: an input layer, a hidden layer, and an output layer. The weights are initialised using the He initialisation method and the biases are initialised using a uniform distribution. The hidden layer uses the ReLU activation function and the output layer uses the softmax activation function. The gradients of the cross-entropy loss function with respect to the weights and biases are calculated and the parameters are updated via batch gradient descent. A demonstration of the model being trained on data from the MNIST database is provided (learning_rate = 0.1, hidden_layer_size = 64).

<img src="https://github.com/user-attachments/assets/a6be99b3-a6b0-4fcd-bec8-bb2f3a865f2e" alt="demonstration" width="600" />
