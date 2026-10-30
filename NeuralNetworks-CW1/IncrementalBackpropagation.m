function IncrementalBackpropagation()

% ============================================================
% NEURAL NETWORKS — COURSEWORK 1
% Incremental Backpropagation in a Feedforward Neural Network
% ============================================================
%
% Purpose:
% Implement incremental (online) backpropagation from scratch
% in MATLAB, reproducing the forward and backward passes and
% weight updates for two training examples.
%
% Network architecture:
% - 2 input units
% - 3 sigmoid hidden units
% - 1 linear output unit
% - No bias terms
% - Sparse input-to-hidden connectivity
% - Direct input-to-output connections
%
% Training:
% - Learning rate: 0.2
% - Two training examples
% - Incremental weight updates after each example
%
% Implementation:
% - Matrix and vector representations of network weights
% - Explicit loops for forward and backward propagation
% - Gradient-based weight updates
% - Comparison with manually calculated results
%
% Adapted from the original MSc Neural Networks coursework.
% The function name and documentation have been revised.
% The original computational implementation is preserved.
% ============================================================


% ------------------------------------------------------------
% Learning rate and network dimensions
% ------------------------------------------------------------

eta = 0.2; % Learning rate

Ni = 2;    % Number of input units (x1, x2)
Nh = 3;    % Number of hidden units

% The network contains one linear output unit.


% ------------------------------------------------------------
% Weight initialisation
% ------------------------------------------------------------

% Input-to-hidden weight matrix: Wih(i,j)
% Rows correspond to input units; columns to hidden units.
%
% Original scalar weight mapping:
% w1: x1 -> H1
% w3: x1 -> H2
% w4: x2 -> H2
% w6: x2 -> H3
%
% Sparse connectivity:
% H1 receives input from x1 only.
% H2 receives input from both x1 and x2.
% H3 receives input from x2 only.
%
% Absent connections: Wih(2,1) and Wih(1,3).

% Initialise the 2x3 input-to-hidden weight matrix with zeros.
Wih = zeros(Ni,Nh);

% Initialise the existing connections.
Wih(1,1) =  0.1;   % w1: x1 -> H1
Wih(1,2) = -0.2;   % w3: x1 -> H2
Wih(2,2) =  0.1;   % w4: x2 -> H2
Wih(2,3) =  0.1;   % w6: x2 -> H3

% Wih(1,3) and Wih(2,1) remain zero (absent connections).

% Hidden-to-output weights: Who_h(j)
% Original mapping: w9 (H1), w8 (H2), w7 (H3).
Who_h = [0.1; 0.1; -0.2];

% Input-to-output weights: Who_x(i)
% Original mapping: w2 (x1), w5 (x2).
Who_x = [0.1; -0.1];


% ------------------------------------------------------------
% Training dataset
% ------------------------------------------------------------

% Two training examples, each containing two inputs and
% one target output.
%
% D(n,:) = [x1, x2, target]
% Rows represent individual training examples.

D = [0 1 1;     % Example 1: x1=0, x2=1, t=1
     1 0 1];    % Example 2: x1=1, x2=0, t=1


% ------------------------------------------------------------
% Display initial weights
% ------------------------------------------------------------

% Display the initial weight matrices and vectors before training.

fprintf('PART 2 — Question 1 (Incremental backprop, two training examples)\n');
fprintf('Initial weights (matrix form):\n');

disp('Wih (input->hidden):'); disp(Wih);
disp('Who_h (hidden->out):'); disp(Who_h');
disp('Who_x (input->out): '); disp(Who_x');


% ------------------------------------------------------------
% Incremental backpropagation: main training loop
% ------------------------------------------------------------

% Process each training example sequentially.
%
% For each example:
% 1. Perform forward propagation to calculate the output.
% 2. Compute output and hidden-layer error signals.
% 3. Update the weights immediately using backpropagation.
%
% A single pass is performed over the two training examples.

for n = 1:size(D,1) % Index of the current training example

    x = D(n,1:2)';    % Column vector [x1; x2]
    t = D(n,3);       % Scalar target (t=1)

    % Display the corresponding coursework question labels.
    if n == 1
        fprintf('\nQuestion 1.a  (First training example)\n');
        fprintf('1.a.i) Forward pass outputs\n');
    else
        fprintf('\nQuestion 1.b  (Second training example)\n');
        fprintf('1.b.i) Forward pass outputs\n');
    end


    % --------------------------------------------------------
    % Forward propagation
    % --------------------------------------------------------

    % Calculate the net input and activation of each hidden unit.
    %
    % Hidden-unit net input:
    % netH(j) = sum_i [x(i) * Wih(i,j)]
    %
    % Hidden-unit activation:
    % yH(j) = sigmoid(netH(j))
    %
    % The sigmoid function transforms the net input into an
    % activation between 0 and 1.

    netH = zeros(Nh,1); % Preallocate hidden-unit net inputs
    yH   = zeros(Nh,1); % Preallocate hidden-unit activations

    % Compute the net input and sigmoid activation of each hidden unit.
    for j = 1:Nh
        for i = 1:Ni
            netH(j) = netH(j) + x(i)*Wih(i,j);
        end
        yH(j) = sigmoid(netH(j));
    end


    % --------------------------------------------------------
    % Linear output unit
    % --------------------------------------------------------

    % Calculate the output as the weighted sum of:
    % 1. Hidden-unit activations.
    % 2. Direct input-to-output connections.
    %
    % netOut = sum_j [yH(j) * Who_h(j)]
    %        + sum_i [x(i) * Who_x(i)]
    %
    % The output unit uses a linear activation function.

    netOut = 0; % Initialise the output net input

    % Add contributions from the hidden units.
    for j = 1:Nh
        netOut = netOut + yH(j)*Who_h(j);
    end

    % Add direct contributions from the input units.
    for i = 1:Ni
        netOut = netOut + x(i)*Who_x(i);
    end

    % Linear activation: output equals net input.
    yOut = netOut;


    % --------------------------------------------------------
    % Display forward-pass results
    % --------------------------------------------------------

    fprintf('Example %d: x1=%d, x2=%d, t=%.4f\n', n, x(1), x(2), t);

    fprintf('Hidden nets: ');
    fprintf('%.4f ', netH);

    fprintf('\nHidden activations: ');
    fprintf('%.4f ', yH);

    fprintf('\nOutput yOut = %.4f\n', yOut);


    % --------------------------------------------------------
    % Backward propagation: error signals (deltas)
    % --------------------------------------------------------

    % Calculate the error signals required for weight updates.
    %
    % The output error signal depends on the difference between
    % the target and predicted output.
    %
    % Hidden-layer error signals are calculated by propagating
    % the output error backwards through the network.

    % Display the corresponding coursework question labels.
    if n == 1
        fprintf('1.a.ii) Error signals (deltas)\n');
    else
        fprintf('1.b.ii) Error signals (deltas)\n');
    end


    % --------------------------------------------------------
    % Output-layer error signal
    % --------------------------------------------------------

    % dOut = (t - yOut) * f'(netOut)
    %
    % Because the output unit is linear, f'(netOut) = 1.
    % Therefore, the output delta equals the prediction error.

    dOut = (t - yOut); % Target minus predicted output


    % --------------------------------------------------------
    % Hidden-layer error signals
    % --------------------------------------------------------

    % dH(j) = yH(j) * (1 - yH(j)) * Who_h(j) * dOut
    %
    % The derivative of the sigmoid activation function is:
    % sigmoid'(netH(j)) = yH(j) * (1 - yH(j))
    %
    % Each hidden-unit delta combines the sigmoid derivative,
    % its connection weight to the output, and the output delta.

    dH = zeros(Nh,1); % Preallocate hidden-layer error signals

    for j = 1:Nh
        dH(j) = yH(j) * (1 - yH(j)) * (Who_h(j) * dOut);
    end


    % --------------------------------------------------------
    % Display error signals
    % --------------------------------------------------------

    fprintf('dOut = %.4f\n', dOut);

    fprintf('Hidden deltas: ');
    fprintf('%.4f ', dH);
    fprintf('\n');


    % --------------------------------------------------------
    % Incremental weight updates
    % --------------------------------------------------------

    % Update the network weights immediately after processing
    % each training example.
    %
    % General weight-update rule:
    % delta_w = eta * delta_target * activation_source
    %
    % where:
    % eta               = learning rate
    % delta_target      = error signal of the receiving unit
    % activation_source = activation of the sending unit

    % Display the corresponding coursework question labels.
    if n == 1
        fprintf('1.a.iii) Weight updates (Δw) and new weights\n');
    else
        fprintf('1.b.iii) Weight updates (Δw) and new weights\n');
    end


    % --------------------------------------------------------
    % Hidden-to-output weight updates
    % --------------------------------------------------------

    % Calculate the change in each hidden-to-output connection.
    %
    % dWho_h(j) = eta * dOut * yH(j)
    %
    % Each weight is updated by adding its calculated change.

    dWho_h = zeros(Nh,1);

    for j = 1:Nh
        dWho_h(j) = eta * dOut * yH(j);
        Who_h(j)  = Who_h(j) + dWho_h(j);
    end


    % --------------------------------------------------------
    % Input-to-output weight updates
    % --------------------------------------------------------

    % Update the direct connections from input units to the
    % linear output unit.
    %
    % dWho_x(i) = eta * dOut * x(i)

    dWho_x = zeros(Ni,1);

    for i = 1:Ni
        dWho_x(i) = eta * dOut * x(i);
        Who_x(i)  = Who_x(i) + dWho_x(i);
    end


    % --------------------------------------------------------
    % Input-to-hidden weight updates
    % --------------------------------------------------------

    % Update the connections between the input and hidden layers.
    %
    % dWih(i,j) = eta * dH(j) * x(i)
    %
    % The hidden-layer error signals were calculated during
    % backward propagation using the original connection weights.

    dWih = zeros(Ni,Nh);

    for j = 1:Nh
        for i = 1:Ni
            dWih(i,j) = eta * dH(j) * x(i);
            Wih(i,j)  = Wih(i,j) + dWih(i,j);
        end
    end


    % --------------------------------------------------------
    % Preserve sparse network connectivity
    % --------------------------------------------------------

    % The original network architecture excludes two connections:
    % x2 -> H1 and x1 -> H3.
    %
    % Explicitly reset their weight changes and connection weights
    % to zero to ensure these connections remain absent.

    dWih(2,1) = 0;   % x2 -> H1 does not exist
    dWih(1,3) = 0;   % x1 -> H3 does not exist

    Wih(2,1)  = 0;   % Keep this weight at zero
    Wih(1,3)  = 0;   % Keep this weight at zero


    % --------------------------------------------------------
    % Display updated weights
    % --------------------------------------------------------

    % Display the calculated weight changes and updated weights.
    % Numerical results are rounded to four decimal places.

    fprintf('ΔWho_h (H->Out): ');
    fprintf('%.4f ', dWho_h);

    fprintf('\nΔWho_x (X->Out): ');
    fprintf('%.4f ', dWho_x);

    fprintf('\nΔWih (X->H):\n');
    disp(round(dWih,4));

    fprintf('Weights after example %d:\n', n);

    disp('Wih:');   disp(round(Wih,4));
    disp('Who_h:'); disp(round(Who_h',4));
    disp('Who_x:'); disp(round(Who_x',4));

end % End of training loop

end % End of IncrementalBackpropagation function


% ------------------------------------------------------------
% Sigmoid activation function
% ------------------------------------------------------------

% Applies the logistic sigmoid transformation:
%
% sigmoid(x) = 1 / (1 + exp(-x))
%
% Maps the net input of each hidden unit to an activation
% between 0 and 1.
%
% The derivative is calculated directly from the hidden-unit
% activations during backpropagation.

function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end % End of sigmoid function