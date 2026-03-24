# Convolutional Neural Networks (CNN)

## Convolution Layer

A convolution layer is a fundamental building block of Convolutional Neural Networks (CNNs). It is designed to automatically and adaptively learn spatial hierarchies of features from input images. Instead of connecting every neuron to all inputs, convolution layers use small filters (kernels) that slide across the input image. These filters detect patterns such as edges, textures, and shapes. Each filter produces a feature map that highlights where specific patterns occur in the image. This parameter sharing reduces computational cost and allows CNNs to generalize well to new data. Convolution layers are especially effective for image processing tasks because they preserve spatial relationships between pixels.

## Pooling Layer

Pooling layers are used in CNNs to reduce the spatial dimensions of feature maps while retaining important information. This helps decrease computational complexity and prevents overfitting. The most common type is max pooling, which selects the maximum value from a region of the feature map. This operation ensures that the most prominent features are preserved while reducing noise. Pooling also provides translation invariance, meaning the model can recognize features even if they shift slightly in position. By progressively reducing dimensions, pooling layers allow deeper networks to focus on higher-level features.

## Fully Connected Layer

Fully connected layers are typically placed at the end of a CNN architecture. They take the high-level features extracted by convolution and pooling layers and use them to perform classification or regression tasks. In this layer, every neuron is connected to all neurons from the previous layer, allowing the model to combine learned features into final predictions. These layers act as a decision-making component of the network. While convolution layers extract features, fully connected layers interpret those features to produce outputs such as class probabilities.