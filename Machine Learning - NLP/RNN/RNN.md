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

![Recurrent Neuron](https://media.geeksforgeeks.org/wp-content/uploads/20241030134529497963/recurrent-neuron.png)
*Recurrent Neuron*

---

### 2. RNN Unfolding

**RNN unfolding** or **unrolling** is the process of expanding the recurrent structure over time steps. During unfolding, each step of the sequence is represented as a separate layer in a series, illustrating how information flows across each time step.

This unrolling enables **Backpropagation Through Time (BPTT)** — a learning process where errors are propagated across time steps to adjust the network’s weights, enhancing the RNN’s ability to learn dependencies within sequential data.

![RNN Unfolding](https://media.geeksforgeeks.org/wp-content/uploads/20231204131012/Unfolding-660.png)
*RNN Unfolding*

---

### Recurrent Neural Network Architecture

RNNs share similarities in input and output structures with other deep learning architectures but differ significantly in how information flows from input to output. Unlike traditional deep neural networks, where each dense layer has distinct weight matrices, RNNs use **shared weights across time steps**, allowing them to remember information over sequences.

## Recurrent Neural Network (RNN) Architecture and Formula Breakdown

---

## Recurrent Neural Network Architecture

- RNNs share similarities in input and output structures with other deep learning architectures.
  - However, they differ in how information flows from input to output.
    - Traditional deep neural networks have distinct weight matrices per layer.
    - RNNs use **shared weights across time steps** to remember sequence information.

---

## Hidden State Calculation

- In RNNs, a hidden state \( H_i \) is calculated for each input \( X_i \) to retain sequential dependencies.
  - The core formula for the hidden state update is:

    \[
    h = σ(U . X + W . h_{t-1} + B)
    \]

---

## Explanation of the Formula

- This formula updates the hidden state \( h \) at time step \( t \).
  - It combines the current input and previous hidden state.
    - This creates a memory-aware representation of the input sequence.

---

## Term-by-Term Table

- Symbols and their meanings:

  | Symbol             | Meaning                                                                 |
  |--------------------|-------------------------------------------------------------------------|
  | \( h \)            | The current hidden state — memory of the network                        |
  | \( X \)            | The input at time \( t \)                                               |
  | \( h_{t-1} \)      | The hidden state from the previous time step                            |
  | \( U \)            | Weight matrix for the input                                             |
  | \( W \)            | Weight matrix for the previous hidden state                             |
  | \( B \)            | Bias term                                                               |
  | \( σ \) (Sigma)           | Activation function (e.g., tanh, sigmoid, ReLU)                         |

---

## Step-by-Step Breakdown

- Computation process:

  - Multiply input \( X \) by matrix \( U \).
  - Multiply previous hidden state \( h_{t-1} \) by matrix \( W \).
  - Add bias vector \( B \).
  - Apply activation function \( σ \).
  - Output the new hidden state \( h \).

---

## Why It Matters

- The RNN formulation allows:
  - Retention of past information.
    - Important for temporal/sequential patterns.
  - Adaptation to variable input lengths.
  - Learning long-term dependencies in data like:
    - Language
    - Time series
    - Speech

---

## Example Using tanh Activation

- Using the tanh function as the activation function \( σ \):

  \[
  h_t = \tanh(U . X_t + W . h_{t-1} + B)
  \]

  - This non-linearity put the outputs into the range [-1, 1].
    - Helps in stabilizing gradients and learning non-linear patterns.

---

### Summary

- The core RNN update formula:

  \[
  h = σ(U . X + W . h_{t-1} + B)
  \]

  - Merges current input with previous memory.
    - Allows the model to make context-aware predictions.
  - Fundamental for sequence modeling tasks like:
    - Text generation
    - Time series forecasting
    - Speech recognition
   
### Output Calculation

The output \( Y \) is computed by applying an activation function \( O \) to the weighted hidden state. The formula is:

\[
Y = O(V \cdot h + C)
\]

- **\( Y \)**: The output at time step \( t \).
- **\( O \)**: An activation function (e.g., sigmoid, softmax, tanh) applied to the weighted hidden state.
- **\( V \)**: Weight matrix for the hidden state.
- **\( C \)**: Bias term for the output calculation.

This formula transforms the hidden state into the output for the network, which can be used for predictions or passed to the next layer in the network. The activation function \( O \) is applied to scale the result based on the network's needs, such as classification or regression tasks.

---

### Overall Function

The overall function for the RNN is:

\[
Y = f(X, h, W, U, V, B, C)
\]

- **\( X \)**: The input sequence fed into the RNN.
- **\( h \)**: The hidden state sequence which holds the network's memory over time.
- **\( W \), \( U \), \( V \)**: Weight matrices for the input, hidden states, and output respectively.
- **\( B \), \( C \)**: Bias terms for the hidden state and output.

This function defines the entire RNN operation, where the input sequence \( X \) is processed over time, and at each time step, a hidden state \( h \) is updated. The RNN remembers past inputs and computes a corresponding output \( Y \) for each time step. The state matrix \( S \) holds the state at each time step \( i \), where each element \( s_i \) represents the network's state at that particular time.

---

## Term-by-Term Table

| Symbol        | Meaning                                                           |
|---------------|--------------------------------------------------------------------|
| \( h_t \)     | Current hidden state at time step \( t \)                         |
| \( X_t \)     | Input vector at time step \( t \)                                 |
| \( h_{t-1} \) | Hidden state from previous time step                              |
| \( U \)       | Weight matrix applied to the input                                |
| \( W \)       | Weight matrix applied to the previous hidden state                |
| \( B \)       | Bias term                                                         |
| \( \sigma \)  | Activation function (e.g., tanh, sigmoid, ReLU)                   |
| \( Y \)       | Output of the RNN at time step \( t \)                            |
| \( O \)       | Output activation function                                        |
| \( V \)       | Weight matrix applied to the hidden state for output calculation  |
| \( C \)       | Bias term applied to the output                                   |
| \( f \)       | The overall function for RNN operation                            |
| \( S \)       | State matrix representing the network's state at each time step  |
| \( s_i \)     | The network's state at time step \( i \)                          |

---

## Summary of Key Formulas

1. **Hidden State Update**:  
    \[
    h_t = \sigma(U \cdot X_t + W \cdot h_{t-1} + B)
    \]

2. **Output Calculation**:  
    \[
    Y = O(V \cdot h + C)
    \]

3. **Overall Function**:  
    \[
    Y = f(X, h, W, U, V, B, C)
    \]

These formulas show how RNNs handle sequential data by retaining information in the hidden state, transforming it through weights and bias, and generating outputs based on that memory.

---

### Step-by-Step Breakdown

- **Hidden State Calculation**: Compute  
    \[
    h_t = \sigma(U \cdot X_t + W \cdot h_{t-1} + B)
    \]
    - Multiply the input \( X_t \) with weight matrix \( U \)
    - Multiply the previous hidden state \( h_{t-1} \) with weight matrix \( W \)
    - Add the bias term \( B \)
    - Apply the activation function \( \sigma \) (e.g., tanh)
    - Output the new hidden state \( h_t \)

- **Output Calculation**: Compute  
    \[
    Y = O(V \cdot h + C)
    \]
    - Multiply the hidden state \( h \) with weight matrix \( V \)
    - Add the bias term \( C \)
    - Apply the output activation function \( O \) (e.g., softmax, sigmoid)
    - Output the result \( Y \)

- **Overall Function**:  
    \[
    Y = f(X, h, W, U, V, B, C)
    \]
    - Defines the complete RNN operation, processing the input sequence \( X \) over time and producing output \( Y \).

---

### Final Remarks

In summary, the RNN architecture is defined by key formulas that handle the sequential dependencies of data. The hidden state is updated at each time step, and the output is calculated by transforming the hidden state. The overall function defines how these updates and transformations occur over time, allowing RNNs to process sequential data effectively.
 


## How does an RNN work?

A Recurrent Neural Network (RNN) processes sequences by maintaining a memory of previous inputs in its hidden state. At each time step, the RNN updates its hidden state based on the current input and its previous hidden state, enabling it to capture patterns over time.

### 1. Hidden State Update

At each time step \( t \), the hidden state \( h_t \) is updated as follows:

\[
h_t = f(h_{t-1}, x_t)
\]

Where:

- \( h_t \) is the current hidden state (memory of the network)
- \( h_{t-1} \) is the previous hidden state
- \( x_t \) is the input at time step \( t \)
- \( f \) is a nonlinear activation function (commonly tanh or ReLU)

A more specific common formulation is:

\[
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t)
\]

Where:

- \( W_{hh} \) is the weight matrix for the hidden state (recurrent weights)
- \( W_{xh} \) is the weight matrix for the input

### 2. Output Calculation

The output at time step \( t \) is typically computed as:

\[
y_t = W_{hy} h_t
\]

Where:

- \( y_t \) is the output at time step \( t \)
- \( W_{hy} \) is the weight matrix for mapping the hidden state to the output

### 3. Learning (Backpropagation Through Time)

RNNs are trained using a variation of backpropagation, called **Backpropagation Through Time (BPTT)**, which unfolds the network through the sequence and updates the weights based on the errors at each time step.

---

**Summary:**  
RNNs maintain a hidden state that acts as memory, updating it at each time step using the current input and the previous state. This mechanism makes them suitable for processing sequential data such as text, audio, or time series.
