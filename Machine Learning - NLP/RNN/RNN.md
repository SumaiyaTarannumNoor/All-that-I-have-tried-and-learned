# RNN
**Recurrent Neural Networks (RNNs)** differ from regular neural networks in how they process information. While standard neural networks pass information in one direction 
i.e from input to output, RNNs feed information back into the network at each step.

Imagine reading a sentence and you try to predict the next word, you don’t rely only on the current word but also remember the words that came before. RNNs work similarly by
“remembering” past information and passing the output from one step as input to the next i.e it considers all the earlier words to choose the most likely next word. This memory
of previous steps helps the network understand context and make better predictions.

[Reference: GeeksforGeeks – RNN](https://www.geeksforgeeks.org/introduction-to-recurrent-neural-network/) <br>

## Key Components of RNNs

There are mainly two components of RNNs that we will discuss.

---

### 1. Recurrent Neurons

The fundamental processing unit in RNN is a **Recurrent Unit**. They hold a hidden state that maintains information about previous inputs in a sequence. Recurrent units can "remember" information from prior steps by feeding back their hidden state, allowing them to capture dependencies across time.

![Recurrent Neuron](./images/recurrent-neuron.png)
*Recurrent Neuron*

---

### 2. RNN Unfolding

**RNN unfolding** or **unrolling** is the process of expanding the recurrent structure over time steps. During unfolding, each step of the sequence is represented as a separate layer in a series, illustrating how information flows across each time step.

This unrolling enables **Backpropagation Through Time (BPTT)** — a learning process where errors are propagated across time steps to adjust the network’s weights, enhancing the RNN’s ability to learn dependencies within sequential data.

![RNN Unfolding](./images/Unfolding-660.png)
*RNN Unfolding*

---

### Recurrent Neural Network Architecture

RNNs share similarities in input and output structures with other deep learning architectures but differ significantly in how information flows from input to output. Unlike traditional deep neural networks, where each dense layer has distinct weight matrices, RNNs use **shared weights across time steps**, allowing them to remember information over sequences.
