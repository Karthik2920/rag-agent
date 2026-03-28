# Long Short-Term Memory (LSTM)

## The Problem LSTM Solves

Standard recurrent neural networks suffer from the vanishing gradient problem, where gradients shrink exponentially as they are backpropagated through many time steps. This makes it nearly impossible for RNNs to learn long-term dependencies — connections between events that are far apart in a sequence. LSTM networks were designed specifically to overcome this limitation by introducing a gated memory cell that can store information over long periods without degradation.

## LSTM Cell Structure and Gates

An LSTM cell contains three gates that control the flow of information: the forget gate, the input gate, and the output gate. The forget gate decides what information to discard from the cell state by outputting values between 0 and 1 — a value of 0 means completely forget, and 1 means completely keep. The input gate determines what new information to store in the cell state. The output gate controls what part of the cell state is passed to the next hidden state. This gating mechanism allows LSTMs to selectively remember or forget information at each time step.

## Cell State and Hidden State

The LSTM has two forms of memory: the cell state and the hidden state. The cell state acts as a long-term memory that runs through the entire sequence, with only minor linear modifications at each step — this is what allows gradients to flow without vanishing. The hidden state is a short-term memory that carries information relevant to the current output. At each time step, the gates update the cell state and produce a new hidden state, which is passed to the next time step and used for predictions.

## Why LSTMs Outperform Vanilla RNNs

LSTMs significantly outperform vanilla RNNs on tasks that require understanding long-range dependencies, such as language modeling, machine translation, and speech recognition. The key advantage is the cell state highway — information can pass through many time steps with minimal modification, preserving gradients during backpropagation through time. In practice, LSTMs learn much faster and achieve better accuracy than standard RNNs on sequences longer than 10 to 20 steps. This makes them the preferred choice when temporal context over long sequences is required.
