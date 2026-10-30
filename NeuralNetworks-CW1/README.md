# Neural Networks — Coursework 1

### Incremental Backpropagation in a Feedforward Neural Network

## Overview

This project implements incremental (online) backpropagation from scratch in MATLAB for a feedforward neural network. The implementation demonstrates the mathematical principles underlying neural network learning, including forward propagation, error backpropagation, and gradient-based weight updates.

The network consists of two input units, three sigmoid hidden units, and one linear output unit. It uses a sparse connectivity structure with additional direct input-to-output connections and no bias terms.

The project was developed as part of the MSc Computational Cognitive Neuroscience programme at Goldsmiths, University of London.

### Network Architecture

The network contains:

* Input layer: Two input units (x1 and x2).
* Hidden layer: Three units with logistic sigmoid activation functions.
* Output layer: One unit with a linear activation function.
* Connectivity: Sparse input-to-hidden connections, alongside direct input-to-output connections.
* Bias terms: None.

The network weights are represented using matrices and vectors, with absent connections constrained to zero.

### Computational Methods

#### 1. Forward Propagation

The net input to each hidden unit is calculated as the weighted sum of its inputs. A sigmoid activation function transforms this net input into a hidden-unit activation.

The network output is calculated as the sum of weighted hidden-unit activations and direct input-to-output contributions.

#### 2. Backpropagation

The output error signal is calculated from the difference between the target and predicted output.

Hidden-layer error signals are then calculated using the derivative of the sigmoid activation function and the weights connecting the hidden units to the output.

These calculations implement the chain rule underlying backpropagation.

#### 3. Incremental Weight Updates

Weights are updated immediately after each training example using a learning rate of 0.2.

The implementation explicitly calculates weight updates for three connection types:

* Hidden-to-output connections.
* Input-to-output connections.
* Input-to-hidden connections.

The sparse architecture is preserved by constraining absent input-to-hidden connections to zero.

#### 4. Numerical Verification

The MATLAB implementation reproduces the forward-pass outputs, error signals, and updated weights obtained through manual backpropagation calculations.

### Key Results

The network processes two training examples sequentially, performing one incremental weight update after each example.

Training example	Input (x1, x2)	Target	Predicted output	Output error signal
1	(0, 1)	1	-0.1025	1.1025
2	(1, 0)	1	0.2654	0.7346

The second example is processed using the weights updated after the first example, demonstrating the sequential nature of incremental learning.

The numerical results agree with the manually calculated values to four decimal places.

### Limitations

This implementation is designed to demonstrate the mathematical operations underlying backpropagation rather than to train a neural network for predictive performance.

The network processes only two training examples in a single pass. Consequently, the implementation does not assess convergence, generalisation, or performance across multiple training epochs.

The network architecture and initial weights are predefined to reproduce the coursework calculations.

### Skills Demonstrated

* MATLAB programming
* Feedforward neural network implementation
* Forward propagation
* Backpropagation and the chain rule
* Gradient-based optimisation
* Incremental (online) learning
* Matrix and vector operations
* Numerical verification of mathematical calculations

### Running the Code

Open MATLAB and navigate to the directory containing
[IncrementalBackpropagation.m](IncrementalBackpropagation.m).

Run the function from the MATLAB Command Window:

```matlab
IncrementalBackpropagation

The program displays the initial weights, hidden-unit activations, network outputs, error signals, weight changes, and updated weights for both training examples.

The implementation uses standard MATLAB functionality and does not require a specialised neural network toolbox.
