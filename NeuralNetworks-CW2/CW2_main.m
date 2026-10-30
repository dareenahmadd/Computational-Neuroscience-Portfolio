%% SECTION 2 — Simple Network (Newton Validation)
clear; close all; clc

% 1. Define dataset (small and nonlinear)
X = [0 0; % input matrix where each row is one training pattern 
     0 1;
     1 0;
     1 1]; % 4 patterns, 2 inputs 

T = [0; 1; 1; 0];   % XOR target vector 

% 2. Initial weights (8 parameters for 2-3-1 network)
w0 = [-0.25; 0.33; 0.14; -0.17; 0.16; 0.43; 0.21; -0.25];

% 3. Train using Newton function 
[w_final, E_hist] = newton_simple_net(X, T, w0, 50); % X = training inputs, T = target, w0 = initial weights, 50 = maximum iterations 
% w_final = trained weights
% E_hist = error history per iteration 

% 4. Plot error convergence - shows error vs iteration 
figure(1);
plot(E_hist,'LineWidth',2);
xlabel('Iteration');
ylabel('Sum of Squared Error');
title('Newton Training – Simple Network');
grid on;

% few iterations, rapid convergence, and perhaps non-monotonic early steps 

% Validate trained simple network

Y = zeros(size(T));
for i = 1:length(T)
    Y(i) = forward_simple(X(i,:)', w_final);
end

SSE_final = 0.5 * sum((T - Y).^2);

fprintf('\nSection 2 Validation\n');
fprintf('Final SSE: %.4e\n', SSE_final);

% for each iteration, compute network outputs, compute residuals, build
% jacobian, approximate Hessian, and apply damped newton update. 

%% SECTION 3 — Sunspots 10–5–1 Network

clearvars -except w_final E_hist   % keep Section 2 results 
close all; % closes all figures from section 2 

rng(1); % for reproducibility

% loading dataset 
load sunspot.dat
spots = sunspot(:,2); % extracts column 2, the sunspot numbers and ignores years 
% spots becomes raw time series 

% lagged dataset (10 inputs to target pairs; 10 prior years for next year
% prediction)
L = 10; % 10 previous time steps to predict next value 
N = length(spots) - L; % number of usable samples 

% preallocating input matrix and target vector 
X = zeros(N,L);
T = zeros(N,1);

% constructing lagged inputs 
for i = 1:N 
    X(i,:) = spots(i:i+L-1)'; % inputs 10 consecutive past values 
    T(i)   = spots(i+L); % target is next value 
end

% normalising as sunspots values are large; normalisation to keep activations
% in useful range also improves conditioning
scale = max(spots);
X = X / scale;
T = T / scale;

% initialising parameters
% 50 input-hidden weights + 5 hidden biases + 5 hidden-output weights + 1
% output bias = 61 trainable parameters
w0 = randn(61,1) * 0.05; % using randn function 

% backprop training
eta = 0.01; % small learning rate 
maxEpochs = 200; % training rate; requires many epochs to converge because it has small step size

% training network using gradient descent; returns trained weights and SSE history
[w_bp, E_bp] = backprop_sunspots(X, T, w0, eta, maxEpochs);

% backprop plots 
figure(2);
plot(E_bp,'LineWidth',2);
xlabel('Epoch');
ylabel('SSE');
title('Backpropagation – Sunspots');
grid on;

% plotting log scales to show convergence rate; linear behaviour = linear convergence 
figure(3);
semilogy(E_bp,'LineWidth',2);
xlabel('Epoch');
ylabel('SSE (log scale)');
title('Backpropagation – Log Scale');
grid on;

% newton training
maxIter = 20; % second-order method so converges much faster, and needs less iterations

% uses Gauss-Newton 
[w_newton, E_newton] = newton_sunspots(X, T, w0, maxIter);

% newton plots 
figure(5);
plot(E_newton,'-o','LineWidth',2);
xlabel('Iteration');
ylabel('SSE');
title('Newton – Sunspots');
grid on;

% plots log scale for convergence rate 
figure(6);
semilogy(E_newton,'-o','LineWidth',2);
xlabel('Iteration');
ylabel('SSE (log scale)');
title('Newton – Log Scale');
grid on;

% predictions 
Y_bp = zeros(N,1);
Y_newton = zeros(N,1);

% computing final outputs using trained weights 
for i = 1:N
    Y_bp(i)     = forward_sunspots(X(i,:), w_bp);
    Y_newton(i) = forward_sunspots(X(i,:), w_newton);
end

% NMSE to normalise error relative to data variance
mse_bp = mean((T - Y_bp).^2);
mse_newton = mean((T - Y_newton).^2);

var_T = var(T);

NMSE_bp = mse_bp / var_T;
NMSE_newton = mse_newton / var_T;

fprintf('\nNMSE (Backprop): %.4f\n', NMSE_bp);
fprintf('NMSE (Newton):   %.4f\n', NMSE_newton);

figure(7);
plot(T,'k','LineWidth',1.5); hold on;
plot(Y_bp,'b');
plot(Y_newton,'r');
legend('Target','Backprop','Newton');
xlabel('Sample Index');
ylabel('Normalized Sunspot Value');
title('Sunspots Approximation Comparison');
grid on;

% overlaying backprop and newton on the same log plot for comparison
figure(8);
semilogy(E_bp,'LineWidth',2)
hold on
semilogy(1:length(E_newton),E_newton,'o-','LineWidth',2)

xlabel('Training step')
ylabel('SSE (log scale)')
legend('Backpropagation','Newton')
title('Training Convergence Comparison')
grid on

%% LOCAL FUNCTIONS

function [w, E_hist] = newton_simple_net(X, T, w0, maxIter)

% initialise weight vector 
w = w0(:); % ensuring weights are a column vector 

% N = number of training patterns
N = size(X,1);

% M = number of parameters (weights) 
M = length(w);

% store sum-of-squared error per iteration
E_hist = zeros(maxIter,1);

for iter = 1:maxIter

    % Forward Pass --here 
    % computing the network outputs for all patterns 
    y = zeros(N,1);
    for p = 1:N
        y(p) = forward_simple(X(p,:)', w);
    end

    % residuals; 
    r = T - y; %r = t - y where t is the target and y is the network

    % prediction; sum of squared error
    E_hist(iter) = 0.5 * sum(r.^2); % E(w) = 1/2 * sum_p (t_p - y_p)^2

    % jacobian matrix where J contains derivatives of network output w.r.t
    % weights 
    % so, J(p,i) = ∂y_p / ∂w_i
    J = zeros(N,M); % each row corresponds to one training pattern
    for p = 1:N
        J(p,:) = jacobian_simple(X(p,:)', w);
    end

   % Damped Gauss-Newton Hessian approximation for least squares problem

   % true Hessian is H = H = JᵀJ + Σ r_p H_p but Gauss-Newton
   % approximation  = H ≈ JᵀJ

    lambda = 1e-3; % damping term
    H = J'*J + lambda*eye(M); % λI is used as a damping term to stabilise the inversion when JᵀJ is ill-conditioned
    
    % Gradient of SSE
    % E(w) = 1/2 Σ (t - y)^2 and ∇E = -Jᵀ r
    grad = -J'*r; % where r = (t-y) are the residual errors

    % newton parameter update; Δw = −H⁻¹ ∇E
    delta_w = -H \ grad;

    % then,
    w = w + delta_w;

    % Stopping criterion if update step becomes very small
    if norm(delta_w) < 1e-6 % maxIter (50) for safety limit
        E_hist = E_hist(1:iter);
        break
    end
end
end

% -------------
function y = forward_simple(x, w)

% function to compute forward pass of small 2-3-1 neural network used to
% validate Newton's algorithm

% two inputs
x1 = x(1);
x2 = x(2);

% Hidden neuron 1
% net input: net1 = w1 * x1
net1 = w(1) * x1;

% sigmoid activation:
% equation used for all:  y1 = σ(net1) = 1 / (1 + e^(-net1))
y1   = 1 / (1 + exp(-net1));

% Hidden neuron 2
% net input: net2 = w4 * x2
net2 = w(4) * x2;

% sigmoid activation 
y2   = 1 / (1 + exp(-net2));

% Hidden neuron 3
% net input combines both inputs: net 3 = w3*x1 + w5*x2
net3 = w(3) * x1 + w(5) * x2;

% sigmoid activation
y3   = 1 / (1 + exp(-net3));

% output neuron is linear (no sigmoid); combining direct input connection
% and hidden activations

% y = w2*x1 + w6*y1 + w7*y2 + w8*y3
y = w(2) * x1 + w(6) * y1 + w(7) * y2 + w(8) * y3;

end

% -------------
function Jrow = jacobian_simple(x, w)

% computing one row of the Jacobian matrix: J(p,i) = ∂y / ∂w_i; so
% derivative output w.r.t each weight

% 2 inputs in the network
x1 = x(1);
x2 = x(2);

% hidden unit 1
net1 = w(1) * x1; % net input to H1
y1   = 1 / (1 + exp(-net1)); % sigmoid activation σ(net1)
d1   = y1 * (1 - y1); % derivative σ'(net1)

% hidden unit 2
net2 = w(4) * x2;
y2   = 1 / (1 + exp(-net2));
d2   = y2 * (1 - y2);

% hidden unit 3
net3 = w(3) * x1 + w(5) * x2;
y3   = 1 / (1 + exp(-net3));
d3   = y3 * (1 - y3);

% allocating Jacobian row, one derivative per weight
Jrow = zeros(1,8);

% ∂y/∂w1 : weight x1 to H1
% y = w6*y1 + ...
% ∂y/∂w1 = w6 * σ'(net1) * x1
Jrow(1) = w(6) * d1 * x1;
% ∂y/∂w2 : direct input x1 to output
% y = w2*x1 + ...
Jrow(2) = x1;
% ∂y/∂w3 : weight x1 to H3
Jrow(3) = w(8) * d3 * x1;
% ∂y/∂w4 : weight x2 to H2
Jrow(4) = w(7) * d2 * x2;
% ∂y/∂w5 : weight x2 to H3
Jrow(5) = w(8) * d3 * x2;
% ∂y/∂w6 : hidden H1 to output
Jrow(6) = y1;
% ∂y/∂w7 : hidden H2 to output
Jrow(7) = y2;
% ∂y/∂w8 : hidden H3 to output
Jrow(8) = y3;

end

% -------------
function y = forward_sunspots(x, w) % forward pass for the sunspots network
% 10 inputs, 5 sigmoid hidden units with biases, 1 linear output with bias

% w contains 61 trainable parameters where 50 are from input to hidden
% connections (10x5), 5 are hidden biases, 5 are hidden to output
% connections, and 1 is the output bias

% reshaping the first 50 parameters into the input to hidden weight matrix
W1 = reshape(w(1:50), [10,5]); % 10 inputs x 5 hidden units

% extracting hidden biases
b1 = w(51:55)'; % 1x5 hidden bias vector

% extracting hidden to output weights
W2 = w(56:60); % 5 hidden units and 1 output

% extracting output bias
b2 = w(61); % scalar output bias

% a_j = sum_i x_i * W1(i,j) + b1(j)
a = x * W1 + b1;                 %(1×10)*(10×5) = 1×5

% h_j = σ(a_j) = 1/(1 + e^{-a_j})
h = 1 ./ (1 + exp(-a));     % sigmoid activation for hidden

% y = sum_j h_j * W2(j) + b2
y = h * W2 + b2;                 % linear output; % (1×5)*(5×1) = scalar

end

% -------------
function J = jacobian_sunspots(x, w) % computes Jacobian row for one training example

% J(i) = ∂y / ∂w_i
% 10 inputs: 5 sigmoid units with biases and 1 linear output with bias

% extract weight matrices and bias vectors from parameter vector
W1 = reshape(w(1:50), [10,5]); % input to hidden weights
b1 = w(51:55)'; % hidden biases
W2 = w(56:60); % hidden to output weights

% here is the forward pass for derivative computation

% net input to hidden units
a  = x * W1 + b1; % a_j = Σ_i x_i W1(i,j) + b1(j)
% hidden activations
h  = 1 ./ (1 + exp(-a)); % sigmoid activation σ(a)

% derivative of the sigmoid
dh = h .* (1 - h); % σ'(a) = σ(a)(1 − σ(a))

% allocating jacobian row, one derivative per trainable parameter
J = zeros(1,61);

% so derivatives w.r.t. for input to hidden weights
% Using chain rule:
% y = Σ_j h_j W2_j + b2
% h_j = σ(a_j)
% a_j = Σ_i x_i W1(i,j) + b1(j)
% therefore, ∂y/∂W1(i,j) = W2(j) * σ'(a_j) * x(i)

idx = 1;
for j = 1:5 % looping over hidden units
    for i = 1:10 % loop over inputs 
        J(idx) = W2(j) * dh(j) * x(i);
        idx = idx + 1;
    end
end

% derivatives w.r.t. hidden biases
% ∂y/∂b1(j) = W2(j) * σ'(a_j)
for j = 1:5
    J(idx) = W2(j) * dh(j);
    idx = idx + 1;
end

% now, derivatives w.r.t. hidden to output weights
% output layer is linear so y = Σ_j h_j W2_j + b2
% therefore, ∂y/∂W2(j) = h_j
J(56:60) = h;

% derivative w.r.t. output bias
% ∂y/∂b2 = 1
J(61) = 1;

end

% -------------
function [w, E_hist] = newton_sunspots(X, T, w0, maxIter)

% the Gauss-newton training for sunspots; 10 inputs, 5 sigmoid hidden units
% with biases and 1 linear output with bias

% initialising weight vector
w = w0(:); % ensure weights are column vector 

N = size(X,1); % number of training samples
M = length(w); % 61 trainable parameters

E_hist = zeros(maxIter,1); % records SSE per iteration
lambda = 0.01;    % this is damping term for numerical stability (Levenberg–Marquardt style regularisation)

for k = 1:maxIter

    % intialising hessian approximation and gradient 
    H = zeros(M,M); % gauss-newton hessian approximation
    g = zeros(M,1); % gradient vector
    E = 0; % total SSE for current iteration

    % loops through all training samples
    for p = 1:N

        x = X(p,:); % input vector (1x10)
        t = T(p); % target value

        % forward pass
        y = forward_sunspots(x, w);
        e = t - y; % residual error
        E = E + 0.5 * e^2; % accumulating sum of squared errors

        % computes jacobian row; J(p,i) = ∂y_p / ∂w_i
        J = jacobian_sunspots(x, w);

        % gauss-newton hessian approximation
        H = H + J' * J; % H ≈ JᵀJ

        % the gradient of SSE objective
        g = g + J' * e; % ∇E = -Jᵀ r
    end

    % the newton parameter update; Δw = (H + λI)⁻¹ g
    delta_w = (H + lambda*eye(M)) \ g; % λI stabilises inversion if H is ill-conditioned
   % updates weights
    w = w + delta_w;

    % stores training error for current iteration
    E_hist(k) = E;

    % this stopping criterion, stops early if the update becomes very small
    % and converges
    if norm(delta_w) < 1e-6
        E_hist = E_hist(1:k);
        break
    end
end

end

% -------------
function [w, E_hist] = backprop_sunspots(X, T, w0, eta, maxEpochs)
% the backpropagation training for sunspots; learning through stochastic
% gradient descent

% initialises weight vector 
w = w0(:); % ensuring its a column vector
N = size(X,1); % number of training samples

E_hist = zeros(maxEpochs,1); % stores SSE per epoch

for epoch = 1:maxEpochs
    
    E = 0; % resets SSE for current epoch
    
    % looping through training samples
    for p = 1:N 
        
        x = X(p,:); % input vector is 1x10
        t = T(p); % target value 
        
        % Forward pass
        W1 = reshape(w(1:50), [10,5]); % reshapes weight vector into matrices so input to hidden weights
        b1 = w(51:55)'; % hidden biases
        W2 = w(56:60); % hidden to output weights 
        b2 = w(61); % output bias
        
        % for hidden layer computation
        a = x * W1 + b1; % net input to hidden units
        h = 1 ./ (1 + exp(-a)); % sigmoid activation
        % linear output unit
        y = h * W2 + b2;
        % this is prediction error 
        e = t - y;

        E = E + 0.5 * e^2; % then accumulate sum of squared errors 
        
        % Backprop gradients
        % for linear output unit: δ_out = (t − y)
        delta_out = e;

        % hidden layer deltas: % δ_hidden = σ'(a) ⊙ (W2 * δ_out)
        % where σ'(a) = h(1 − h)
        delta_hidden = (delta_out * W2') .* (h .* (1 - h));
        
        % Gradient for hidden to output weights
        grad_W2 = h' * delta_out;  % ∂E/∂W2 = -δ_out * h

        % gradient for output bias
        grad_b2 = delta_out;

        % gradient for input to hidden weights
        grad_W1 = x' * delta_hidden; % ∂E/∂W1 = -δ_hidden * x

        % gradient for hidden biases
        grad_b1 = delta_hidden;
        
        % weight update for stochastic gradient descent 
        w(56:60) = w(56:60) + eta * grad_W2; % w_new = w + η * gradient
        w(61)    = w(61)    + eta * grad_b2;
        w(1:50)  = w(1:50)  + eta * grad_W1(:);
        w(51:55) = w(51:55) + eta * grad_b1(:);
        
    end
    
    E_hist(epoch) = E; % stores training error for current epoch
end

end