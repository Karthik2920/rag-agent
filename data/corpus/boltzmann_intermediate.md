# Boltzmann Machines

## What is a Boltzmann Machine

A Boltzmann Machine is a type of stochastic recurrent neural network that learns a probability distribution over its inputs. It consists of visible units, which represent observed data, and hidden units, which capture latent structure. All units are connected to each other — including connections between hidden units — making the network fully connected. The network is trained to assign high probability to configurations that match the training data and low probability to others. Boltzmann Machines are generative models, meaning they can produce new samples by sampling from the learned distribution.

## Energy Function

Boltzmann Machines are defined by an energy function that assigns a scalar value to every configuration of visible and hidden units. Lower energy states correspond to more probable configurations. The probability of a configuration is proportional to the negative exponential of its energy — this is the Boltzmann distribution from statistical physics, which gives the model its name. Training drives the model to reduce the energy of training data configurations and increase the energy of other configurations. The challenge is that computing the exact probability requires summing over all possible configurations, which is exponentially expensive.

## Restricted Boltzmann Machine

A Restricted Boltzmann Machine (RBM) is a simplified version of the Boltzmann Machine where connections between units of the same layer are removed. Visible units connect only to hidden units, and there are no visible-to-visible or hidden-to-hidden connections. This restriction makes inference and training tractable. In an RBM, given the visible units, all hidden units are conditionally independent of each other, allowing efficient block Gibbs sampling. RBMs became foundational in deep learning as building blocks for Deep Belief Networks, where multiple RBMs are stacked to learn hierarchical representations.

## Training with Contrastive Divergence

Training an RBM exactly requires computing an intractable partition function. Contrastive Divergence (CD) is an approximation algorithm that makes training feasible. In CD-k, starting from a training example, the model performs k steps of Gibbs sampling to produce a reconstructed data point. The weight update is the difference between the outer products of the original data and the reconstruction. CD-1 (a single Gibbs step) works surprisingly well in practice. The algorithm increases the probability of the training data while decreasing the probability of the model's own reconstructions, pushing the model to represent the data distribution accurately.

## Applications

Boltzmann Machines and RBMs have been applied to collaborative filtering (notably the Netflix Prize), dimensionality reduction, and feature learning. Stacked RBMs trained greedily layer by layer form Deep Belief Networks, which were among the first deep architectures trained successfully before modern techniques like batch normalization and ReLU activations became standard. While largely superseded by VAEs and GANs for generative modeling, Boltzmann Machines remain important historically as they introduced the idea of learning latent probabilistic representations of data.
