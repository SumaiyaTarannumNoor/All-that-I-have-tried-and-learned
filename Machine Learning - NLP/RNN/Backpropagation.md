## Backpropagation
Backpropagation is a supervised learning algorithm used to train artificial neural networks. It works by calculating the gradient of the loss function with respect
to the network's weights, allowing the network to adjust its weights to minimize the error between predicted and actual outputs. In essence, it's the process of 
propagating error information backward through the network to refine its internal parameters. 

[Reference:](https://www.geeksforgeeks.org/machine-learning/ml-back-propagation-through-time/)

- **Recurrent Neural Networks (RNNs)** are designed to process sequential data.
- Unlike traditional neural networks, RNN outputs depend not only on the current input but also on previous inputs through a memory element.
- This memory allows RNNs to capture temporal dependencies in data, such as time series or language.
- **Training RNNs** involves a specialized version of backpropagation:
  - Instead of updating weights based only on the current timestep, RNNs consider all previous timesteps (\( t, t-1, t-2, t-3, \ldots \)).
- This training method is called **Backpropagation Through Time (BPTT)**:
  - BPTT extends traditional backpropagation to sequential data by "unfolding" the network over time.
  - Gradients are summed across all relevant time steps.
  - This enables RNNs to learn complex temporal patterns in sequential data.
