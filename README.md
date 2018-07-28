# Symbolic-Auto-Differentiation-Executer

## Intuition

For the original Back Propagation, there are some disadvantages:

• You always need to keep intermediate data in the memory during the forward pass in case it will be used in the backpropagation.

• Lack of flexibility, e.g., compute the gradient of gradient.

As an example, in my another project [q-net](https://github.com/Huqicheng/Q-Net), when I implemented the LSTM, I felt it's really complicated to translate the back propagation process to numpy statements.

Objectives:

To make it more convenient for developers to create awesome deep learning algorithms.

To be used to simplify the development of neural networks.
