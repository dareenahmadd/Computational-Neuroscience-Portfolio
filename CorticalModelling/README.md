## Cortical Modelling: Neural Decision-Making Network

### Overview

This project implements a stochastic, two-population neural rate model to investigate the computational dynamics of decision-making.

The model simulates competition between two recurrently connected neural populations, incorporating self-excitation, mutual inhibition, and Gaussian noise in synaptic inputs.

Using MATLAB, I investigated how external input differences influence decision probabilities, decision times, and neural population activity. I also examined the model’s dynamical behaviour through phase-plane trajectories, numerical nullclines, and animated visualisations.

This project was originally developed as part of my MSc in Computational Cognitive Neuroscience at Goldsmiths, University of London, and received a coursework mark of 98%.

Following assessment feedback, I extended the original analysis by implementing nonlinear model fitting to explore the relationship between input bias and decision time.

## Computational Methods

**Programming language:** MATLAB

The model consists of two competing neural populations with recurrent self-excitation and mutual inhibition. Population firing rates are determined by a nonlinear input-output function, while synaptic activity evolves according to coupled differential equations.

The equations were numerically integrated using the forward Euler method with a time step of 1 ms. Independent Gaussian noise was introduced into the synaptic inputs to generate trial-to-trial variability.

### Decision-Making Simulations

The network was simulated across 100 trials per input condition.

External input currents were systematically adjusted to investigate conditions targeting approximately 50%, 60%, 80%, and 100% Population 1 decisions.

Decision outcomes were classified according to which population first crossed an activity threshold of 0.5.

### Decision-Time Analysis

Mean decision times were examined as a function of the difference between the external input currents.

An additional post-assessment analysis fitted an exponential decay function to the simulation-derived decision-time summaries using MATLAB's `fminsearch` optimisation function.

### Dynamical Systems Analysis

The model's behaviour was investigated using:

- Two-dimensional phase-plane trajectories.
- Three-dimensional plots of population activity derivatives.
- Numerical approximations of nullclines.
- Identification of candidate fixed-point regions.
- Animated visualisations of neural population activity.
