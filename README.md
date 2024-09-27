# Neural Network from Scratch

This project implements a simple feed-forward neural network with backpropagation from scratch using Python and NumPy.

## Structure

### 1. **Base Class - `Layer`**
   This is a base class that all other layers inherit from. It defines two essential methods for neural networks:
   - **`forward_propagation(self, input)`**: This method computes the output of the layer for a given input.
   - **`backward_propagation(self, output_error, learning_rate)`**: This method computes the error derivative with respect to the input, which is used during backpropagation.

### 2. **Fully Connected Layer (`FCLayer`)**
   - **Purpose**: This is a fully connected (or dense) layer where each neuron in the layer is connected to every neuron in the previous layer.
   - **Parameters**:
     - `input_size`: The number of neurons in the previous layer.
     - `output_size`: The number of neurons in this layer.
   - **Methods**:
     - **`forward_propagation(self, input_data)`**: This method calculates the layer's output using the formula `output = input * weights + bias`.
     - **`backward_propagation(self, output_error, learning_rate)`**: This method computes the gradients of the weights and bias with respect to the error and updates the parameters using the learning rate.

### 3. **Activation Layer (`ActivationLayer`)**
   - **Purpose**: This layer applies an activation function (like `tanh`, `ReLU`, etc.) element-wise to the output of a previous layer.
   - **Methods**:
     - **`forward_propagation(self, input_data)`**: Applies the activation function to the input.
     - **`backward_propagation(self, output_error, learning_rate)`**: Computes the derivative of the activation function and passes it backward in the network for error correction.

### 4. **Activation Functions (`tanh`, `tanh_prime`)**
   - **`tanh(x)`**: The hyperbolic tangent function, a common activation function.
   - **`tanh_prime(x)`**: The derivative of the `tanh` function, used during backpropagation to compute how the error propagates through the activation.

### 5. **Loss Functions (`mse`, `mse_prime`)**
   - **`mse(y_true, y_pred)`**: The Mean Squared Error (MSE) loss function, which measures the difference between the predicted output and the true output.
   - **`mse_prime(y_true, y_pred)`**: The derivative of the MSE function, which is used during backpropagation to adjust weights.

### 6. **Network Class (`Network`)**
   This class ties everything together and manages the entire neural network.
   - **Methods**:
     - **`add(self, layer)`**: Adds a new layer (like FCLayer or ActivationLayer) to the network.
     - **`use(self, loss, loss_prime)`**: Defines the loss function to be used during training.
     - **`predict(self, input_data)`**: Feeds input through the network using forward propagation to predict the output.
     - **`fit(self, x_train, y_train, epochs, learning_rate)`**: This is the training loop, where the network:
       1. Runs forward propagation to compute the output.
       2. Computes the error using the loss function.
       3. Runs backward propagation to update the weights and biases using the computed error.

### 7. **Training Script**
   - **Loading and preprocessing MNIST dataset**: This part loads the dataset, reshapes it, and normalizes it so that it can be used by the neural network. The labels (digits 0-9) are one-hot encoded.
   - **Building the network**: The code defines the structure of the neural network with fully connected layers and activation layers.
   - **Training the network**: The `fit` method is called to train the network on the MNIST dataset using backpropagation and gradient descent.

### 8. **XOR Example**
   In the second part of the code, we built a smaller neural network to solve the XOR problem, which demonstrates how a neural network can learn a non-linear problem.
   - It has two fully connected layers and uses the `tanh` activation function.
   - The network is trained on the XOR truth table, and after training, it can predict the correct output for inputs like [0, 0], [0, 1], [1, 0], and [1, 1].

## Summary
- **Layer**: Base class for different types of layers.
- **FCLayer**: A fully connected layer that performs matrix multiplication and adds bias.
- **ActivationLayer**: Applies a non-linear activation function to the output of the fully connected layer.
- **Network**: Manages the forward pass, backpropagation, and training process.
- **Activation and Loss functions**: These handle the non-linearity and error calculation necessary for learning.
