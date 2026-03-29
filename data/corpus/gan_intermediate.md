# Generative Adversarial Networks (GAN)

## What is a GAN

A Generative Adversarial Network (GAN) is a deep learning framework for training generative models. A GAN consists of two neural networks — a generator and a discriminator — that are trained simultaneously in opposition to each other. The generator learns to produce synthetic data that resembles the training distribution, while the discriminator learns to distinguish real data from generated data. Through this adversarial process, the generator is pushed to produce increasingly realistic outputs. GANs are capable of generating high-quality images, audio, and text, and have become one of the most influential generative modeling approaches in deep learning.

## Generator and Discriminator

The generator takes a random noise vector sampled from a simple prior distribution (typically Gaussian) as input and maps it to a data sample in the target domain. It never sees real data directly — it only receives feedback through the discriminator. The discriminator takes a data sample (either real or generated) and outputs a probability indicating whether the sample is real. It is trained as a binary classifier. The two networks have opposing objectives: the generator tries to fool the discriminator, and the discriminator tries not to be fooled. This creates a dynamic competitive training process.

## Adversarial Training

GAN training alternates between updating the discriminator and updating the generator. In each iteration, the discriminator is updated to maximize its ability to classify real and fake samples correctly. Then the generator is updated to maximize the probability that the discriminator misclassifies its outputs as real. The theoretical optimal outcome is a Nash equilibrium where the generator produces samples indistinguishable from real data and the discriminator outputs 0.5 for every sample. In practice, reaching this equilibrium is difficult and training is notoriously unstable, requiring careful tuning of learning rates and architectures.

## Mode Collapse

Mode collapse is a common failure mode in GAN training where the generator produces only a narrow variety of outputs, ignoring large portions of the real data distribution. For example, a GAN trained on digits might generate only 1s and 3s perfectly while ignoring all other digits. This happens because the generator finds a small set of outputs that reliably fool the discriminator and exploits them repeatedly. The discriminator then adapts, but the generator shifts to another mode rather than covering the full distribution. Techniques to address mode collapse include Wasserstein GAN (which replaces the discriminator with a critic trained with a different loss), minibatch discrimination, and feature matching.

## Applications

GANs have enabled remarkable applications including photorealistic image synthesis, image-to-image translation (e.g., converting sketches to photographs), super-resolution, data augmentation, and deepfake generation. StyleGAN produces high-resolution human faces indistinguishable from real photographs. Pix2Pix and CycleGAN enable paired and unpaired image translation. In medicine, GANs are used to generate synthetic training data when real data is scarce. Despite their power, GANs remain challenging to train stably and evaluate rigorously — there is no single reliable metric equivalent to perplexity in language models.
