# Neural Networks — Coursework 2

### Newton's Method for Training Multilayer Perceptron Neural Networks

## Overview

This project implements and evaluates Newton-based optimisation for training multilayer perceptron (MLP) neural networks in MATLAB. The implementation demonstrates the mathematical principles underlying second-order neural network optimisation, including forward propagation, analytical Jacobian computation, and damped Gauss–Newton weight updates.

The project comprises two experiments. The first validates the Newton-based training algorithm using a small neural network trained on the XOR problem. The second compares conventional backpropagation and damped Gauss–Newton optimisation for predicting annual sunspot activity using historical time-series data.

The project was developed as part of the MSc Computational Cognitive Neuroscience programme at Goldsmiths, University of London.

### Network Architecture

Two feedforward neural network architectures are implemented.

**Experiment 1: XOR Validation**

The network contains:

* Input layer: Two input units (x1 and x2).
* Hidden layer: Three units with logistic sigmoid activation functions.
* Output layer: One unit with a linear activation function.
* Connectivity: Sparse connections between the input and hidden layers, alongside direct input-to-output connections.
* Bias terms: None.
* Trainable parameters: Eight connection weights.

The network is trained to reproduce the XOR function using four input–target pairs.

**Experiment 2: Sunspots Time-Series Prediction**

The network contains:

* Input layer: Ten input units representing the preceding ten annual sunspot observations.
* Hidden layer: Five units with logistic sigmoid activation functions.
* Output layer: One unit with a linear activation function.
* Connectivity: Fully connected input-to-hidden and hidden-to-output layers.
* Bias terms: Included in the hidden and output layers.
* Trainable parameters: 61.

The same network architecture is used to compare conventional backpropagation and damped Gauss–Newton optimisation.

### Computational Methods

#### 1. Forward Propagation

The net input to each hidden unit is calculated as the weighted sum of its inputs, including bias terms where applicable.

A logistic sigmoid activation function transforms each hidden-unit net input into an activation.

The network output is calculated as a linear combination of hidden-unit activations and the corresponding output weights.

For the XOR network, direct input-to-output connections also contribute to the final output.

#### 2. Analytical Jacobian Computation

The Jacobian matrix describes how the network outputs change with respect to each trainable parameter.

The implementation calculates the partial derivatives of network outputs with respect to the connection weights using the chain rule.

These derivatives are used to construct the Gauss–Newton approximation to the Hessian matrix, providing second-order information for optimisation.

#### 3. Damped Gauss–Newton Optimisation

The Newton-based training algorithm updates network parameters using an approximation to the second-order curvature of the error function.

The implementation explicitly calculates:

* The difference between target and predicted outputs.
* The Jacobian matrix of network outputs with respect to the weights.
* The Gauss–Newton approximation to the Hessian matrix.
* A damping term to improve numerical stability.
* The resulting parameter updates.

The algorithm iteratively updates the network weights to minimise the sum of squared errors (SSE).

For the Sunspots experiment, the Newton-based model is trained for a maximum of 20 iterations using a damping parameter of 0.01.

#### 4. Conventional Backpropagation

A conventional backpropagation algorithm is implemented to provide a comparison with Newton-based optimisation.

The algorithm calculates output errors, propagates error signals through the hidden layer, and updates the network weights using gradient-based learning.

Weights are updated incrementally during training.

The backpropagation model is trained for 200 epochs using a learning rate of 0.01.

#### 5. Time-Series Prediction

Historical annual sunspot observations are loaded from MATLAB's `sunspot.dat` dataset.

The data are normalised and transformed into input–target pairs, with the preceding ten annual observations used to predict the subsequent observation.

Both training algorithms are evaluated using the same network architecture.

Training performance is assessed using convergence curves and normalised mean squared error (NMSE).

### Key Results

**Experiment 1: XOR Validation**

The damped Gauss–Newton algorithm successfully trained the small neural network to reproduce the XOR function.

The final sum of squared errors was:

**SSE = 2.1618 × 10⁻¹⁹**

The training error approached zero, demonstrating successful optimisation of the network parameters.

![XOR validation using Newton optimisation](Figures/XOR_Newton_Convergence.png)

*Figure 1. Sum of squared errors across training iterations during Newton-based optimisation of the XOR network.*

**Experiment 2: Sunspots Time-Series Prediction**

The two training algorithms produced the following normalised mean squared errors:

| Training algorithm | NMSE |
|--------------------|------|
| Conventional backpropagation | 0.2322 |
| Damped Gauss–Newton optimisation | 0.0791 |

Damped Gauss–Newton optimisation achieved a lower NMSE than conventional backpropagation in this experiment.

The Newton-based method also demonstrated faster reductions in training error, although its initial optimisation steps exhibited fluctuations.

![Sunspots training convergence comparison](Figures/Sunspots_Training_Convergence.png)

*Figure 2. Comparison of training convergence between conventional backpropagation and damped Gauss–Newton optimisation for the Sunspots prediction task. SSE is displayed on a logarithmic scale.*

These results demonstrate the potential advantages of second-order optimisation for neural network training under the experimental conditions examined.

### Limitations

The implementation demonstrates the mathematical and computational principles underlying Newton-based neural network training rather than establishing the general superiority of one optimisation algorithm.

The experiments use relatively small neural networks and a single time-series dataset. Consequently, the findings may not generalise to larger architectures or more complex prediction tasks.

Damped Gauss–Newton optimisation requires the calculation and manipulation of Jacobian-related matrices, which can introduce substantial computational costs as the number of trainable parameters increases.

Additionally, convergence depends on factors including weight initialisation, learning rates, damping parameters, and network architecture.

Lower training error does not necessarily indicate better generalisation to unseen data.

### Skills Demonstrated

* MATLAB programming
* Multilayer perceptron implementation
* Feedforward neural network architectures
* Forward propagation
* Backpropagation and gradient-based optimisation
* Newton-based and damped Gauss–Newton optimisation
* Analytical Jacobian computation
* Hessian approximation
* Matrix and vector operations
* Time-series forecasting
* Numerical optimisation and convergence analysis
* Model performance evaluation
* Scientific visualisation

### Running the Code

Open MATLAB and navigate to the directory containing
[CW2_main.m](CW2_main.m) and [sunspot.dat](sunspot.dat).

Run the script from the MATLAB Command Window:

```matlab
CW2_main
```

The program executes the XOR validation and Sunspots prediction experiments.

It displays the final XOR sum of squared errors, the normalised mean squared errors for both Sunspots training algorithms, and the corresponding training convergence figures.

The implementation uses standard MATLAB functionality and requires the `sunspot.dat` dataset to be available in the working directory.
