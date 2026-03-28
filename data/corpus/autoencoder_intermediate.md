# Autoencoders

## What is an Autoencoder

An autoencoder is a type of neural network trained to reconstruct its input at the output layer. It learns to compress the input into a lower-dimensional representation called the latent space or bottleneck, and then reconstruct the original input from this compressed representation. The network is trained in an unsupervised manner — no labels are needed — because the target output is simply the input itself. The goal is not perfect reconstruction but to force the network to learn the most important features of the data.

## Encoder, Bottleneck, and Decoder

An autoencoder has three components: the encoder, the bottleneck, and the decoder. The encoder maps the high-dimensional input to a low-dimensional latent representation by passing it through one or more hidden layers with decreasing sizes. The bottleneck is the narrowest layer, where the compressed representation lives — this is the learned feature space. The decoder mirrors the encoder and maps the latent representation back to the original input dimensions. The reconstruction error — typically mean squared error — is minimized during training, forcing the bottleneck to retain only the most informative features.

## Dimensionality Reduction and Feature Learning

The primary use of autoencoders is dimensionality reduction and unsupervised feature learning. Unlike PCA, which is limited to linear transformations, autoencoders with non-linear activation functions can learn complex non-linear mappings to a lower-dimensional space. The latent representation learned by the encoder captures the underlying structure of the data in a compact form. These learned features can then be used for downstream tasks such as classification, clustering, or visualization. Autoencoders are particularly useful when labeled data is scarce, because they learn from unlabeled examples.

## Denoising and Variational Autoencoders

Two important variants extend the basic autoencoder. A denoising autoencoder is trained to reconstruct clean inputs from corrupted versions — noise is added to the input before encoding, and the target is the original clean input. This forces the model to learn robust features that capture the true structure of the data rather than memorizing exact values. A variational autoencoder (VAE) replaces the deterministic bottleneck with a probabilistic one — the encoder outputs the parameters of a probability distribution, and the decoder samples from this distribution. VAEs can generate new data samples by sampling from the learned latent space, making them a type of generative model.
