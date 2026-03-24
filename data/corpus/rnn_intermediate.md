# Recurrent Neural Networks (RNN)

## Sequential Data Processing

Recurrent Neural Networks (RNNs) are designed to process sequential data such as time series, text, or speech. Unlike traditional neural networks, RNNs maintain a hidden state that captures information from previous inputs. This allows them to model dependencies over time. Each step in the sequence updates the hidden state, enabling the network to retain context. This makes RNNs particularly useful for tasks like language modeling and speech recognition.

## Vanishing Gradient Problem

One major challenge with RNNs is the vanishing gradient problem. During training, gradients used to update weights can become extremely small as they propagate backward through time. This prevents the network from learning long-term dependencies effectively. As a result, RNNs struggle with tasks that require remembering information from earlier in the sequence. This limitation led to the development of more advanced architectures like LSTM and GRU.

## Hidden State Mechanism

The hidden state in an RNN acts as memory that carries information across time steps. At each step, the hidden state is updated based on the current input and the previous hidden state. This allows the network to accumulate knowledge over a sequence. However, because the same weights are reused at each step, the model can suffer from instability and difficulty learning long-range dependencies. Proper initialization and architecture improvements are often required to make RNNs effective.