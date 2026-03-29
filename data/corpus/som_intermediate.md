# Self-Organizing Maps (SOM)

## What is a SOM

A Self-Organizing Map (SOM) is an unsupervised neural network that learns to produce a low-dimensional, typically two-dimensional, representation of high-dimensional input data. Unlike most neural networks that use backpropagation, SOMs are trained through competitive learning — neurons compete to respond to each input, and only the winner is updated. The result is a topology-preserving map where similar inputs are mapped to nearby locations on the grid. SOMs are used for clustering, visualization, and dimensionality reduction without requiring labeled data.

## Competitive Learning

Competitive learning is the mechanism that drives SOM training. For each input, every neuron in the map computes its distance to the input vector. The neuron with the smallest distance — called the Best Matching Unit (BMU) — wins the competition. Only the BMU and its neighbors are updated to move closer to the input. This winner-takes-most approach ensures that neurons specialize in representing particular regions of the input space, leading to a structured map where similar neurons respond to similar inputs.

## Neighbourhood Function

The neighbourhood function determines how strongly neurons near the BMU are updated relative to the BMU itself. Neurons close to the BMU are updated significantly, while those farther away are updated less. The most common neighbourhood function is the Gaussian, which produces a smooth gradient of influence. The radius of the neighbourhood shrinks over time during training — early in training, large updates spread across the map to establish global structure; later, smaller updates refine local structure. This two-phase behavior allows SOMs to first organize broadly, then tune fine-grained representations.

## Training Process

SOM training proceeds over many iterations. At each step: an input vector is selected, the BMU is identified by finding the neuron with the minimum Euclidean distance to the input, and all neurons within the neighbourhood radius are updated by moving their weight vectors toward the input. The learning rate and neighbourhood radius both decay over time according to a schedule. After training, each neuron's weight vector represents a prototype of the inputs it responds to. The final map can be visualized as a U-matrix showing distances between neighboring neurons, revealing cluster boundaries.

## Applications

SOMs are widely used for visualizing high-dimensional data, clustering, and anomaly detection. In practice, they have been applied to customer segmentation, gene expression analysis, and network intrusion detection. Because the output is a 2D grid, SOMs provide an interpretable visual summary of complex datasets that would be difficult to understand otherwise. Unlike PCA, SOMs can capture non-linear structure in data. Their unsupervised nature makes them particularly useful when labeled data is scarce or unavailable.
