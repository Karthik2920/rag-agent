# Artificial Neural Networks (ANN)

## Structure of ANN

Artificial Neural Networks (ANNs) are composed of layers of interconnected neurons, including input, hidden, and output layers. Each neuron receives inputs, applies weights, and passes the result through an activation function. This layered structure allows ANNs to learn complex patterns in data. The connections between neurons determine how information flows through the network, making it possible to model nonlinear relationships.

## Activation Functions

Activation functions introduce non-linearity into neural networks, enabling them to learn complex mappings. Common activation functions include ReLU, sigmoid, and tanh. ReLU is widely used because it helps mitigate the vanishing gradient problem and speeds up training. Without activation functions, neural networks would behave like linear models regardless of their depth.

## Backpropagation

Backpropagation is the learning algorithm used to train neural networks. It calculates the gradient of the loss function with respect to each weight by applying the chain rule. These gradients are then used to update weights using optimization techniques like gradient descent. Backpropagation allows the network to minimize errors and improve performance over time.