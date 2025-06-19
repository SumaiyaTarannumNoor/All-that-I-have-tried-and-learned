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

---

# Backpropagation Through Time (BPTT) in RNNs

Since RNNs process sequential data, **Backpropagation Through Time (BPTT)** is used to update the network's parameters. The loss function \( L(\theta) \) depends on the final hidden state \( h_3 \), and each hidden state relies on preceding ones, forming a sequential dependency chain:

\[
h_3 \text{ depends on } h_2, \quad h_2 \text{ depends on } h_1, \quad \ldots, \quad h_1 \text{ depends on } h_0
\]

---
![Backpropagation Through Time (BPTT) In RNN](https://media.geeksforgeeks.org/wp-content/uploads/20231204132128/Backpropagation-Through-Time-(BPTT).webp)  

## Backpropagation Through Time (BPTT) in RNN

In BPTT, gradients are backpropagated through each time step. This is essential for updating network parameters based on temporal dependencies.

### Simplified Gradient Calculation

\[
\frac{\partial L(\theta)}{\partial W} = \frac{\partial L(\theta)}{\partial h_3} \cdot \frac{\partial h_3}{\partial W}
\]

### Handling Dependencies in Layers

Each hidden state is updated based on its dependencies:

\[
h_3 = \sigma(W \cdot h_2 + b)
\]

### Gradient Calculation with Explicit and Implicit Parts

The gradient is broken down into explicit and implicit parts, summing up the indirect paths from each hidden state to the weights:

\[
\frac{\partial h_3}{\partial W} = \frac{\partial h_3}{\partial W} + \frac{\partial h_3}{\partial h_2} \cdot \frac{\partial h_2}{\partial W}
\]

### Final Gradient Expression

The final derivative of the loss function with respect to the weight matrix \( W \) is computed as:

\[
\frac{\partial L(\theta)}{\partial W} = \frac{\partial L(\theta)}{\partial h_3} \cdot \sum_{k=1}^{3} \frac{\partial h_3}{\partial h_k} \cdot \frac{\partial h_k}{\partial W}
\]

This iterative process is the essence of backpropagation through time.

---

## Types Of Recurrent Neural Networks

There are four types of RNNs based on the number of inputs and outputs in the network:

### 1. One-to-One RNN

This is the simplest type of neural network architecture where there is a single input and a single output. It is used for straightforward classification tasks such as binary classification where no sequential data is involved.

![One to One RNN](https://media.geeksforgeeks.org/wp-content/uploads/20231204131135/One-to-One-300.webp)  
[Reference: GeeksforGeeks – RNN](https://www.geeksforgeeks.org/introduction-to-recurrent-neural-network/) <br>

---

### 2. One-to-Many RNN

In a One-to-Many RNN, the network processes a single input to produce multiple outputs over time. This is useful in tasks where one input triggers a sequence of predictions (outputs). For example, in image captioning, a single image can be used as input to generate a sequence of words as a caption.

![One to Many RNN](https://media.geeksforgeeks.org/wp-content/uploads/20231204131355/Many-to-One-300.webp)  
[Reference: GeeksforGeeks – RNN](https://www.geeksforgeeks.org/introduction-to-recurrent-neural-network/) <br>

---

### 3. Many-to-One RNN

The Many-to-One RNN receives a sequence of inputs and generates a single output. This type is useful when the overall context of the input sequence is needed to make one prediction. In sentiment analysis, the model receives a sequence of words (like a sentence) and produces a single output, like positive, negative, or neutral.

![Many to One RNN](https://media.geeksforgeeks.org/wp-content/uploads/many-to-one-rnn.png)  
[Reference: GeeksforGeeks – RNN](https://www.geeksforgeeks.org/introduction-to-recurrent-neural-network/) <br>

---

### 4. Many-to-Many RNN

The Many-to-Many RNN type processes a sequence of inputs and generates a sequence of outputs. In language translation tasks, a sequence of words in one language is given as input and a corresponding sequence in another language is generated as output.

![Many to Many RNN](https://media.geeksforgeeks.org/wp-content/uploads/20231204131436/Many-to-Many-300.webp)  
[Reference: GeeksforGeeks – RNN](https://www.geeksforgeeks.org/introduction-to-recurrent-neural-network/) <br>

---

## Variants of Recurrent Neural Networks (RNNs)

There are several variations of RNNs, each designed to address specific challenges or optimize for certain tasks:

### 1. Vanilla RNN

This simplest form of RNN consists of a single hidden layer where weights are shared across time steps. Vanilla RNNs are suitable for learning short-term dependencies but are limited by the vanishing gradient problem, which hampers long-sequence learning.

---

### 2. Bidirectional RNNs

Bidirectional RNNs process inputs in both forward and backward directions, capturing both past and future context for each time step. This architecture is ideal for tasks where the entire sequence is available, such as named entity recognition and question answering.

---

### 3. Long Short-Term Memory Networks (LSTMs)

Long Short-Term Memory Networks (LSTMs) introduce a memory mechanism to overcome the vanishing gradient problem. Each LSTM cell has three gates:

- **Input Gate**: Controls how much new information should be added to the cell state.
- **Forget Gate**: Decides what past information should be discarded.
- **Output Gate**: Regulates what information should be output at the current step.

This selective memory enables LSTMs to handle long-term dependencies, making them ideal for tasks where earlier context is critical.

---

### 4. Gated Recurrent Units (GRUs)

Gated Recurrent Units (GRUs) simplify LSTMs by combining the input and forget gates into a single update gate and streamlining the output mechanism. This design is computationally efficient, often performing similarly to LSTMs, and is useful in tasks where simplicity and faster training are beneficial.

---

## How RNN Differs from Feedforward Neural Networks?

**Feedforward Neural Networks (FNNs)** process data in one direction from input to output without retaining information from previous inputs. This makes them suitable for tasks with independent inputs like image classification. However, FNNs struggle with sequential data since they lack memory.

**Recurrent Neural Networks (RNNs)** solve this by incorporating loops that allow information from previous steps to be fed back into the network. This feedback enables RNNs to remember prior inputs, making them ideal for tasks where context is important.

![Recurrent Vs Feedforward networks](https://media.geeksforgeeks.org/wp-content/uploads/20231204130132/RNN-vs-FNN-660.png)  
[Reference: GeeksforGeeks – RNN](https://www.geeksforgeeks.org/introduction-to-recurrent-neural-network/) <br>

---

## Implementing a Text Generator Using Recurrent Neural Networks (RNNs)

In this section, we create a character-based text generator using Recurrent Neural Network (RNN) in TensorFlow and Keras. We'll implement an RNN that learns patterns from a text sequence to generate new text character-by-character.

### 1. Importing Necessary Libraries

We start by importing essential libraries for data handling and building the neural network.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
```

---

### 2. Defining the Input Text and Prepare Character Set

We define the input text and identify unique characters in the text which we’ll encode for our model.

```python
text = "This is GeeksforGeeks a software training institute"
chars = sorted(list(set(text)))
char_to_index = {char: i for i, char in enumerate(chars)}
index_to_char = {i: char for i, char in enumerate(chars)}
```

---

### 3. Creating Sequences and Labels

To train the RNN, we need sequences of fixed length (`seq_length`) and the character following each sequence as the label.

```python
seq_length = 3
sequences = []
labels = []

for i in range(len(text) - seq_length):
    seq = text[i:i + seq_length]
    label = text[i + seq_length]
    sequences.append([char_to_index[char] for char in seq])
    labels.append(char_to_index[label])

X = np.array(sequences)
y = np.array(labels)
```

---

### 4. Converting Sequences and Labels to One-Hot Encoding

For training, we convert `X` and `y` into one-hot encoded tensors.

```python
X_one_hot = tf.one_hot(X, len(chars))
y_one_hot = tf.one_hot(y, len(chars))
```

---

### 5. Building the RNN Model

We create a simple RNN model with a hidden layer of 50 units and a Dense output layer with softmax activation.

```python
model = Sequential()
model.add(SimpleRNN(50, input_shape=(seq_length, len(chars)), activation='relu'))
model.add(Dense(len(chars), activation='softmax'))
```

---

### 6. Compiling and Training the Model

We compile the model using the `categorical_crossentropy` loss and train it for 100 epochs.

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_one_hot, y_one_hot, epochs=100)
```

*Output: Training the RNN model* <br>
![Reference: Training](https://media.geeksforgeeks.org/wp-content/uploads/20250519153759669306/training.png) <br>

### 7. Generating New Text Using the Trained Model

After training, we use a starting sequence to generate new text character by character.

```python
start_seq = "This is G"
generated_text = start_seq

for i in range(50):
    x = np.array([[char_to_index[char] for char in generated_text[-seq_length:]]])
    x_one_hot = tf.one_hot(x, len(chars))
    prediction = model.predict(x_one_hot)
    next_index = np.argmax(prediction)
    next_char = index_to_char[next_index]
    generated_text += next_char

print("Generated Text:")
print(generated_text)
```

*Output: Predicting the next word*

![Reference: Predicting](https://media.geeksforgeeks.org/wp-content/uploads/20250519153923303648/prediction.png) <br>

## Advantages of Recurrent Neural Networks

- **Sequential Memory:** RNNs retain information from previous inputs, making them ideal for time-series predictions where past data is crucial.
- **Enhanced Pixel Neighborhoods:** RNNs can be combined with convolutional layers to capture extended pixel neighborhoods, improving performance in image and video data processing.

---

## Limitations of Recurrent Neural Networks (RNNs)

While RNNs excel at handling sequential data, they face two main training challenges: **vanishing gradient** and **exploding gradient** problems:

- **Vanishing Gradient:** During backpropagation, gradients diminish as they pass through each time step, leading to minimal weight updates. This limits the RNN’s ability to learn long-term dependencies, which is crucial for tasks like language translation.
- **Exploding Gradient:** Sometimes gradients grow uncontrollably, causing excessively large weight updates that de-stabilize training.

These challenges can hinder the performance of standard RNNs on complex, long-sequence tasks.

---

## Applications of Recurrent Neural Networks

RNNs are used in various applications where data is sequential or time-based:

- **Time-Series Prediction:** RNNs excel in forecasting tasks, such as stock market predictions and weather forecasting.
- **Natural Language Processing (NLP):** RNNs are fundamental in NLP tasks like language modeling, sentiment analysis and machine translation.
- **Speech Recognition:** RNNs capture temporal patterns in speech data, aiding in speech-to-text and other audio-related applications.
- **Image and Video Processing:** When combined with convolutional layers, RNNs help analyze video sequences, facial expressions and gesture recognition.

[Reference: GeeksforGeeks – RNN](https://www.geeksforgeeks.org/introduction-to-recurrent-neural-network/) <br>
