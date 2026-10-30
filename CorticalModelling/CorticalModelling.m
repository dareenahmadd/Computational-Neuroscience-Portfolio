%% Simulating the rate network
close all;clear all;clc

% defining time constant 
tau = 0.1; % 100 ms

% defining scaling function (gamma)
y = 0.641; 

% defining response function for non-linear gain; converts synaptic input
% into firing rate (like sigmoid function)
a = 270 * 10^9; 
b = 108; 
d = 0.15;

% defining synaptic weights
 w_11 = 0.2609 * 10^-9; % recurrent excitation from population 1-1
 w_22 = 0.2609 * 10^-9; % recurrent excitation from population 2-2
 w_12 = 0.0497 * 10^-9; % inhibition from population 1-2
 w_21 = 0.0497 * 10^-9; % inhibition from population 2-1

 % defining external current applied
 I_1 = 0.3310 * 10^-9; % external current applied to population 1 
 I_2 = 0.3304 * 10^-9; % external current applied to population 2

 % simulation time 
 dt = 0.001; % time step of 1ms 
 time_span = 7; % 7 seconds max per trial (the longer = better population stablisation)
 t = 0:dt:time_span; % time vector 

 % defining threshold; when S1 or S2 cross this, a decision is made 
 thresh = 0.63; % decision threshold at 0.63, can change later but I ran this script 
 % and changed this value until I saw changes in the model. First started
 % off at 0.8 but that was too high for the model to make a decision. 

total_trial = 100; % a total of 100 simulation trials

 % S population activity initialisation
S1_all = zeros(total_trial, length(t));
S2_all = zeros(total_trial, length(t));

% for calculating firing rate percentage later 
r1_all = zeros(total_trial, length(t));
r2_all = zeros(total_trial, length(t));

 % preallocation for decision outcomes where 1 = S1 wins, 2 = S2 wins
model_decision = zeros(1, total_trial);
duration = zeros(1, total_trial);

 % simulation loop over 100 trials 
 for trial = 1:total_trial
     S1 = 0.1; % initial state is 0.1
     S2 = 0.1; % initial state is 0.1
     output_decision = false; 

     for i = 1:length(t)
         current_time = t(i); % tracks the current time 

         % noise inputs on pop 1 and 2
         I_noise1 = 0.03 * 10^-9 * randn; % normal noise on inputs to population 1
         I_noise2 = 0.03 * 10^-9 * randn; % normal noise on inputs to population 2

         % computing synpatic for each pop 
         % excitation + inhbition + input + noise
         I_syn1 = w_11*S1 - w_12*S2 + I_1 + I_noise1; % equation 5
         I_syn2 = w_22*S2 - w_21*S1 + I_2 + I_noise2; % equation 6

         % computing firing rates using non-linear transfer function from
         % synaptic input
         r1 = (a*I_syn1 - b) / (1 - exp(-d*(a*I_syn1 - b))); % equation 3
         r2 = (a*I_syn2 - b) / (1 - exp(-d*(a*I_syn2 - b))); % equation 4

       % catching NaN values 
       if isnan(r1) || isinf(r1)
           r1 = 0;
       end
       if isnan(r2) || isinf(r2)
           r2 = 0;
       end

       % this stores firing rates 
       r1_all(trial, i) = r1;
       r2_all(trial, i) = r2;

         % updating S1 and S2 using Euler's method 
         dS1 = (-S1 / tau + (1-S1) * y * r1) * dt; % equation 1 
         dS2 = (-S2 / tau + (1-S2) * y * r2) * dt; % equation 2 

         % updates the pop activity values
         S1 = S1 + dS1;
         S2 = S2 + dS2;

         % catching NaNs again, for unstable activity
         if isnan(S1) || isinf(S1)
             S1 = 0;
         end
         if isnan(S2) || isinf(S2)
             S2 = 0;
         end

         % this stores updated activity 
         S1_all(trial, i) = S1;
         S2_all(trial, i) = S2;

     % checking for threshold crossing and decision outcome
if ~output_decision && (S1 >= thresh || S2 >= thresh)
    output_decision = true;
    duration(trial) = current_time; % time to reach threshold or make decision
    decision_index(trial) = i; % indexing this decision time

    % decision determined as soon as firing rate of model reaches threshold
    % 
    if S1 >= thresh
        model_decision(trial) = 1; % pop 1 wins
    else
        model_decision(trial) = 2; % pop 2 wins
    end

   % decided to hold activity constant after decision 
  S1_all(trial, i:end) = S1;
  S2_all(trial, i:end) = S2;
    break; % so the trial ends after the decision is made
end
     end
 end
 
% displaying the summary for initial input parameters 
no_decision_trials = sum(model_decision == 0); % essentially counts how many trials 
% failed to reach a decision - this fluctuates each time script is run
% because of randn

% this is what I used to tell me which threshold to use, I lowered the
% threshold until I saw outputs here
fprintf('Trials with no decision: %d out of %d\n', no_decision_trials, total_trial); 
fprintf('Proportion of trials choosing Option 1: %.2f\n', mean(model_decision == 1));
fprintf('Proportion of trials choosing Option 2: %.2f\n', mean(model_decision == 2));
fprintf('Mean decision time: %.3f seconds\n', mean(duration(duration > 0)));
%% Question 1: Plot the trajectory of 100 trials over time. 

figure(1); % for plotting trajectory of 100 trials over time 
hold on;

for trial = 1:total_trial % total_trial = 100
    if decision_index(trial) > 0 % trial is only plotted if decision was made and threshold crossed
        idx = decision_index(trial); % time index where decision occured
   plot(t(1:idx) * 1000, S1_all(trial, 1:idx), 'b', 'LineWidth', 0.3);  % Pop 1 activity over time up to decision
   plot(t(1:idx) * 1000, S2_all(trial, 1:idx), 'r', 'LineWidth', 0.3);  % Pop 2 activity over time up to decision
    end
end

xlabel('Time (ms)'); % up to 7000ms (so 7 seconds)
ylabel('Population activity S_1 and S_2'); % from 0.1 to 0.63
title('Time course of firing rates over 100 trials - Dareen');
yline(thresh, '--k', 'Threshold'); % represents when the decision was made (set as 0.63)
legend('S_1', 'S_2'); % S1 in blue, S2 in red 

hold off;

% plot reflects trial-to-trial variability because of noise
%% Question 2: Calculate the percentage of trials in which Population 1 reaches the decision threshold first.

% setting the decision threshold
decision_thresh = 0.5;

% initialising population that crossed the threshold first, where 1 is
% pop 1, 2 is pop 2, and 0 is neither
model_choice = zeros(total_trial, 1); 

% looping over all 100 trials 
for trial = 1:total_trial
    % find() to determine the first index where the threshold is crossed for each t
    S1_crossed = S1_all(trial, :) >= decision_thresh; % true if S1 crosses decision threshold
    S2_crossed = S2_all(trial, :) >= decision_thresh; % true if S2 crosses decision threshold

    % if either population crossed 0.5, then if sum > 0
    reached_thresh1 = sum(S1_crossed) > 0; % checking if S1 reached threshold
    reached_thresh2 = sum(S2_crossed) > 0; % checking if S2 reached threshold

    if reached_thresh1 && ~reached_thresh2
        model_choice(trial) = 1; % for population 1 if reached threshold first
    elseif ~reached_thresh1 && reached_thresh2
        model_choice(trial) = 2; % for population 2 if reached threshold first
    elseif reached_thresh1 && reached_thresh2 % if both reached threshold... then....

        % find which population reached first
        first1 = find(S1_crossed, 1); % indexing of S1 first crossing
        first2 = find(S2_crossed, 1); % indexing of S2 first crossing

        if first1 < first2
            model_choice(trial) = 1; % if S1 reached first it wins decision
        elseif first2 < first1
            model_choice(trial) = 2; % if S2 reached first it wins decision
        end
    end
end

% the proportion of trials choosing choice 1 
prop_choice1 = mean(model_choice == 1) * 100; % percentage is calculated

% displaying this percentage to include in document
fprintf('The model chose choice 1 in %.2f%% of trials.\n', prop_choice1);
%% Question 3: Calculate the average decision time when choosing choice 1

% setting the decision threshold
decision_thresh = 0.5; % decision when pop activity crosses this

%1: identifying the trials where pop 1 reaches 0.5 first
% initialising vector to store decision times (in seconds)
decision_times = NaN(total_trial, 1); % NaN to exclude trials that didn't reach the 0.5 first

% looping over the 100 simulation trials 
for trial = 1:total_trial
    % find() to get first time point when the threshold is crossed 
    S1_crossed = S1_all(trial, :) >= decision_thresh; % >= if pop1 reaches 0.5
    S2_crossed = S2_all(trial, :) >= decision_thresh; % >= if pop2 reaches 0.5

    % if sum > 0, then either population did cross 0.5
    reached_thresh1 = sum(S1_crossed) > 0;
    reached_thresh2 = sum(S2_crossed) > 0;

    % determining if pop 1 reached 0.5 first
    if reached_thresh1 && ~reached_thresh2
        % 2: time point when firing rate of pop 1 surpasses 0.5
        decision_times(trial) = t(find(S1_crossed, 1)); % decision time of pop 1 reaching 0.5
        % if both pop crossed 0.5
    elseif reached_thresh1 && reached_thresh2
        % find() for time where each pop reached 0.5
        first1 = find(S1_crossed, 1);
        first2 = find(S2_crossed, 1);
        
        if first1 < first2
            % storing decision time only if population 1 reached 0.5
            decision_times(trial) = t(first1); % time pop1 reached 0.5
        end
    end
end

% 3: computing mean of decision times (converted to ms)
% filtering non-NaN for trials where pop 1 didn't reach 0.5 (isnan)
avg_decision_time_choice1 = mean(decision_times(~isnan(decision_times))) * 1000;

% displaying the average decision time when model chooses choice 1
fprintf('Average decision time when choosing choice 1: %.0f ms\n', avg_decision_time_choice1);
%% Question 4: Adjust I_1 and I_2 so that population 1 reaches the threshold of 0.5 first in approximately 50% of trials.

% I just used the same rate network code as above but changed the I_1 and
% I_2 values (i did this until Q7) to see how often pop 1 wins

close all;clear all;clc

% defining time constant 
tau = 0.1; % 100 ms

% defining scaling factor for synaptic input
y = 0.641; 

% defining response function 
a = 270 * 10^9; % gain function: steepness
b = 108; % gain function: threshold shift
d = 0.15; % gain function: non-linearity 

% defining synaptic weights
 w_11 = 0.2609 * 10^-9; % recurrent exictation from population 1-1
 w_22 = 0.2609 * 10^-9; % recurrent excitation from population 2-2
 w_12 = 0.0497 * 10^-9; % inhibition from population 1-2
 w_21 = 0.0497 * 10^-9; % inhibition from population 2-1

 % defining external current applied for 50%; 
 % balancing inputs, but there is slight variability, so needs to be run multiple times
 I_1 = 0.3305 * 10^-9; % external current applied to population 1 
 I_2 = 0.3304 * 10^-9; % external current applied to population 2

 % simulation time 
 dt = 0.001; % time step of 1ms 
 time_span = 7; % 6 seconds max per trial
 t = 0:dt:time_span; % time vector 

 % defining threshold 
 thresh = 0.63; % decision threshold at 0.63, same as above, I still kept it at 0.63, not 0.5

 total_trial = 100; % a total of 100 simulation trials

 % S population activity initialisation
S1_all = zeros(total_trial, length(t));
S2_all = zeros(total_trial, length(t));

% for calculating firing rate percentage later 
r1_all = zeros(total_trial, length(t));
r2_all = zeros(total_trial, length(t));

 % preallocation for decision making
model_decision = zeros(1, total_trial); % stores decision (1 for S1, 2 for S2)
duration = zeros(1, total_trial); % stores decision time

 % simulation loop over trials 
 for trial = 1:total_trial
     S1 = 0.1; % initial state 
     S2 = 0.1; % initial state 
     output_decision = false; % tracking decision

     for i = 1:length(t)
         current_time = t(i); % current simulation time

         % noise inputs on pop 1 and 2
         I_noise1 = 0.03 * 10^-9 * randn; % normal noise on inputs to population 1
         I_noise2 = 0.03 * 10^-9 * randn; % normal noise on inputs to population 2

         % computing synpatic input currents for each pop
         I_syn1 = w_11*S1 - w_12*S2 + I_1 + I_noise1; % equation 5
         I_syn2 = w_22*S2 - w_21*S1 + I_2 + I_noise2; % equation 6

         % computing firing rates using non-linear function 
         r1 = (a*I_syn1 - b) / (1 - exp(-d*(a*I_syn1 - b))); % equation 3
         r2 = (a*I_syn2 - b) / (1 - exp(-d*(a*I_syn2 - b))); % equation 4

       % catching NaN values 
       if isnan(r1) || isinf(r1)
           r1 = 0;
       end
       if isnan(r2) || isinf(r2)
           r2 = 0;
       end

       % stores firing rates 
       r1_all(trial, i) = r1;
       r2_all(trial, i) = r2;

         % updating S1 and S2 using Euler's method 
         dS1 = (-S1 / tau + (1-S1) * y * r1) * dt; % equation 1 
         dS2 = (-S2 / tau + (1-S2) * y * r2) * dt; % equation 2 

         % updates population activity
         S1 = S1 + dS1;
         S2 = S2 + dS2;

         % catching NaNs
         if isnan(S1) || isinf(S1)
             S1 = 0;
         end
         if isnan(S2) || isinf(S2)
             S2 = 0;
         end

         % stores the pop activity 
         S1_all(trial, i) = S1;
         S2_all(trial, i) = S2;

     % checking for threshold crossing 
if ~output_decision && (S1 >= thresh || S2 >= thresh)
    output_decision = true;
    duration(trial) = current_time;
    decision_index(trial) = i;

    if S1 >= thresh
        model_decision(trial) = 1;
    else
        model_decision(trial) = 2;
    end

    % decided to hold activity constant after decision 
  S1_all(trial, i:end) = S1;
  S2_all(trial, i:end) = S2;
    break; % ends the trial early, just to avoid any more random fluctations
end
     end
 end
 
% summary stats display
no_decision_trials = sum(model_decision == 0);
fprintf('Trials with no decision: %d out of %d\n', no_decision_trials, total_trial);
fprintf('Proportion of trials choosing Option 1: %.2f\n', mean(model_decision == 1));
fprintf('Proportion of trials choosing Option 2: %.2f\n', mean(model_decision == 2));
fprintf('Mean decision time: %.3f seconds\n', mean(duration(duration > 0)));

% setting the decision threshold
decision_thresh = 0.5;

% Initialising population that crossed the threshold first, where 1 is
% population 1
model_choice = zeros(total_trial, 1); 

% looping over the 100 trials 
for trial = 1:total_trial
    % find() to determine whether threshold is reached for each t
    S1_crossed = S1_all(trial, :) >= decision_thresh;
    S2_crossed = S2_all(trial, :) >= decision_thresh;

    % if either population crossed 0.5, then if sum > 0
    reached_thresh1 = sum(S1_crossed) > 0;
    reached_thresh2 = sum(S2_crossed) > 0;

    if reached_thresh1 && ~reached_thresh2
        model_choice(trial) = 1; % for population 1 if reached threshold first
    elseif ~reached_thresh1 && reached_thresh2
        model_choice(trial) = 2; % for population 2 if reached threshold first
    elseif reached_thresh1 && reached_thresh2 % if both reached threshold...
        % find which population reached first
        first1 = find(S1_crossed, 1);
        first2 = find(S2_crossed, 1);

        if first1 < first2
            model_choice(trial) = 1;
        elseif first2 < first1
            model_choice(trial) = 2;
        end
    end
end

% the proportion of trials choosing choice 1 * 100 for %
prop_choice1 = mean(model_choice == 1) * 100;

fprintf('The model chose choice 1 in %.2f%% of trials.\n', prop_choice1);

% the same code as above is used until Q7, only difference is changes in
% tuning I_1 and I_2

% later use for Q9
% decision times  for trials where pop1 won
pop1_decision_times = duration(model_decision == 1);  % in seconds

% mean decision time
avg_decision_time_pop1 = mean(pop1_decision_times);  % in seconds

% displaying
fprintf('Average decision time for Population 1: %.3f seconds (%.0f ms)\n', ...
    avg_decision_time_pop1, avg_decision_time_pop1 * 1000);
%% Question 5: Adjust I_1 and I_2 so that population 1 reaches the threshold of 0.5 first in approximately 60% of trials. 

close all;clear all;clc

% defining time constant 
tau = 0.1; % 100 ms

% defining scaling function
y = 0.641; 

% defining response function 
a = 270 * 10^9; 
b = 108; 
d = 0.15;

% defining synaptic weights
 w_11 = 0.2609 * 10^-9; % from population 1-1
 w_22 = 0.2609 * 10^-9; % from population 2-2
 w_12 = 0.0497 * 10^-9; % from population 1-2
 w_21 = 0.0497 * 10^-9; % from population 2-1

 % defining external current applied; so pop 1 wins 60% of the time
 I_1 = 0.3310 * 10^-9; % external current applied to population 1 
 I_2 = 0.3306 * 10^-9; % external current applied to population 2

 % simulation time 
 dt = 0.001; % time step of 1ms 
 time_span = 7; % 6 seconds max per trial
 t = 0:dt:time_span; % time vector 

 % defining threshold 
 thresh = 0.63; % decision threshold at 0.63, can change later 

 total_trial = 100; % a total of 100 simulation trials

 % S population activity initialisation
S1_all = zeros(total_trial, length(t));
S2_all = zeros(total_trial, length(t));

% for calculating firing rate percentage later 
r1_all = zeros(total_trial, length(t));
r2_all = zeros(total_trial, length(t));

 % preallocation for decision making
model_decision = zeros(1, total_trial);
duration = zeros(1, total_trial);

 % simulation loop over trials 
 for trial = 1:total_trial
     S1 = 0.1;
     S2 = 0.1;
     output_decision = false; 

     for i = 1:length(t)
         current_time = t(i);

         % noise inputs on population 1 and 2
         I_noise1 = 0.03 * 10^-9 * randn; % normal noise on inputs to population 1
         I_noise2 = 0.03 * 10^-9 * randn; % normal noise on inputs to population 2

         % computing synpatic input currents
         I_syn1 = w_11*S1 - w_12*S2 + I_1 + I_noise1;
         I_syn2 = w_22*S2 - w_21*S1 + I_2 + I_noise2;

         % computing firing rates using non-linear function 
         r1 = (a*I_syn1 - b) / (1 - exp(-d*(a*I_syn1 - b)));
         r2 = (a*I_syn2 - b) / (1 - exp(-d*(a*I_syn2 - b)));

       % Catch NaN values from numerical overflow
       if isnan(r1) || isinf(r1)
           r1 = 0;
       end
       if isnan(r2) || isinf(r2)
           r2 = 0;
       end

       r1_all(trial, i) = r1;
       r2_all(trial, i) = r2;

         % updating S1 and S2 using Euler's method 
         dS1 = (-S1 / tau + (1-S1) * y * r1) * dt; % equation 1 
         dS2 = (-S2 / tau + (1-S2) * y * r2) * dt; % equation 2 

         S1 = S1 + dS1;
         S2 = S2 + dS2;

         % Catch NaNs or Infs in S values
         if isnan(S1) || isinf(S1)
             S1 = 0;
         end
         if isnan(S2) || isinf(S2)
             S2 = 0;
         end

         S1_all(trial, i) = S1;
         S2_all(trial, i) = S2;

     % checking for threshold crossing 
if ~output_decision && (S1 >= thresh || S2 >= thresh)
    output_decision = true;
    duration(trial) = current_time;
    decision_index(trial) = i;

    if S1 >= thresh
        model_decision(trial) = 1;
    else
        model_decision(trial) = 2;
    end

  S1_all(trial, i:end) = S1;
  S2_all(trial, i:end) = S2;
    break;
end
     end
 end
 
% Summary stats (moved outside of loop)
no_decision_trials = sum(model_decision == 0);
fprintf('Trials with no decision: %d out of %d\n', no_decision_trials, total_trial);
fprintf('Proportion of trials choosing Option 1: %.2f\n', mean(model_decision == 1));
fprintf('Proportion of trials choosing Option 2: %.2f\n', mean(model_decision == 2));
fprintf('Mean decision time: %.3f seconds\n', mean(duration(duration > 0))); % this is for all trials. 
       
% setting the decision threshold
decision_thresh = 0.5;

% Initialising population that crossed the threshold first, where 1 is
% population 1
model_choice = zeros(total_trial, 1); 

% looping over the 100 trials 
for trial = 1:total_trial
    % find() to determine whether threshold is reached for each t
    S1_crossed = S1_all(trial, :) >= decision_thresh;
    S2_crossed = S2_all(trial, :) >= decision_thresh;

    % if either population crossed 0.5, then if sum > 0
    reached_thresh1 = sum(S1_crossed) > 0;
    reached_thresh2 = sum(S2_crossed) > 0;

    if reached_thresh1 && ~reached_thresh2
        model_choice(trial) = 1; % for population 1 if reached threshold first
    elseif ~reached_thresh1 && reached_thresh2
        model_choice(trial) = 2; % for population 2 if reached threshold first
    elseif reached_thresh1 && reached_thresh2 % if both reached threshold...
        % find which population reached first
        first1 = find(S1_crossed, 1);
        first2 = find(S2_crossed, 1);

        if first1 < first2
            model_choice(trial) = 1;
        elseif first2 < first1
            model_choice(trial) = 2;
        end
    end
end

% the proportion of trials choosing choice 1 * 100 for %
prop_choice1 = mean(model_choice == 1) * 100;

% this displays the percentage of trials choosing choice 1 - it will
% fluctuate
fprintf('The model chose choice 1 in %.2f%% of trials.\n', prop_choice1); 


% irrelevent to this question, but I use later use for Q9 
% decision times  for trials where pop1 won
pop1_decision_times = duration(model_decision == 1);  % in seconds

% mean decision time
avg_decision_time_pop1 = mean(pop1_decision_times);  % in seconds

% displaying
fprintf('Average decision time for Population 1: %.3f seconds (%.0f ms)\n', ...
    avg_decision_time_pop1, avg_decision_time_pop1 * 1000);
%% Question 6: Adjust I_1 and I_2 so that population 1 reaches the threshold of 0.5 first in approximately 80% of trials. Determine the appropriate values for I_1 and I_2. 

close all;clear all;clc

% defining time constant 
tau = 0.1; % 100 ms

% defining scaling function
y = 0.641; 

% defining response function 
a = 270 * 10^9; 
b = 108; 
d = 0.15;

% defining synaptic weights
 w_11 = 0.2609 * 10^-9; % from population 1-1
 w_22 = 0.2609 * 10^-9; % from population 2-2
 w_12 = 0.0497 * 10^-9; % from population 1-2
 w_21 = 0.0497 * 10^-9; % from population 2-1

 % defining external current applied; so pop 1 wins 80% of the time
 I_1 = 0.3315 * 10^-9; % external current applied to population 1 
 I_2 = 0.3301 * 10^-9; % external current applied to population 2

 % simulation time 
 dt = 0.001; % time step of 1ms 
 time_span = 7; % 6 seconds max per trial
 t = 0:dt:time_span; % time vector 

 % defining threshold 
 thresh = 0.63; % decision threshold at 0.63, can change later 

 total_trial = 100; % a total of 100 simulation trials

 % S population activity initialisation
S1_all = zeros(total_trial, length(t));
S2_all = zeros(total_trial, length(t));

% for calculating firing rate percentage later 
r1_all = zeros(total_trial, length(t));
r2_all = zeros(total_trial, length(t));

 % preallocation for decision making
model_decision = zeros(1, total_trial);
duration = zeros(1, total_trial);

 % simulation loop over trials 
 for trial = 1:total_trial
     S1 = 0.1;
     S2 = 0.1;
     output_decision = false; 

     for i = 1:length(t)
         current_time = t(i);

         % noise inputs on population 1 and 2
         I_noise1 = 0.03 * 10^-9 * randn; % normal noise on inputs to population 1
         I_noise2 = 0.03 * 10^-9 * randn; % normal noise on inputs to population 2

         % computing synpatic input currents
         I_syn1 = w_11*S1 - w_12*S2 + I_1 + I_noise1;
         I_syn2 = w_22*S2 - w_21*S1 + I_2 + I_noise2;

         % computing firing rates using non-linear function 
         r1 = (a*I_syn1 - b) / (1 - exp(-d*(a*I_syn1 - b)));
         r2 = (a*I_syn2 - b) / (1 - exp(-d*(a*I_syn2 - b)));

       % Catch NaN values from numerical overflow
       if isnan(r1) || isinf(r1)
           r1 = 0;
       end
       if isnan(r2) || isinf(r2)
           r2 = 0;
       end

       r1_all(trial, i) = r1;
       r2_all(trial, i) = r2;

         % updating S1 and S2 using Euler's method 
         dS1 = (-S1 / tau + (1-S1) * y * r1) * dt; % equation 1 
         dS2 = (-S2 / tau + (1-S2) * y * r2) * dt; % equation 2 

         S1 = S1 + dS1;
         S2 = S2 + dS2;

         % removing NaNs 
         if isnan(S1) || isinf(S1)
             S1 = 0;
         end
         if isnan(S2) || isinf(S2)
             S2 = 0;
         end

         S1_all(trial, i) = S1;
         S2_all(trial, i) = S2;

     % checking for threshold crossing 
if ~output_decision && (S1 >= thresh || S2 >= thresh)
    output_decision = true;
    duration(trial) = current_time;
    decision_index(trial) = i;

    if S1 >= thresh
        model_decision(trial) = 1;
    else
        model_decision(trial) = 2;
    end

  S1_all(trial, i:end) = S1;
  S2_all(trial, i:end) = S2;
    break;
end
     end
 end
 
% displaying summary 
no_decision_trials = sum(model_decision == 0);
fprintf('Trials with no decision: %d out of %d\n', no_decision_trials, total_trial);
fprintf('Proportion of trials choosing Option 1: %.2f\n', mean(model_decision == 1));
fprintf('Proportion of trials choosing Option 2: %.2f\n', mean(model_decision == 2));
fprintf('Mean decision time: %.3f seconds\n', mean(duration(duration > 0)));

% setting the decision threshold
decision_thresh = 0.5;

% Initialising population that crossed the threshold first, where 1 is
% population 1
model_choice = zeros(total_trial, 1); 

% looping over the 100 trials 
for trial = 1:total_trial
    % find() to determine whether threshold is reached for each t
    S1_crossed = S1_all(trial, :) >= decision_thresh;
    S2_crossed = S2_all(trial, :) >= decision_thresh;

    % if either population crossed 0.5, then if sum > 0
    reached_thresh1 = sum(S1_crossed) > 0;
    reached_thresh2 = sum(S2_crossed) > 0;

    if reached_thresh1 && ~reached_thresh2
        model_choice(trial) = 1; % for population 1 if reached threshold first
    elseif ~reached_thresh1 && reached_thresh2
        model_choice(trial) = 2; % for population 2 if reached threshold first
    elseif reached_thresh1 && reached_thresh2 % if both reached threshold...
        % find which population reached first
        first1 = find(S1_crossed, 1);
        first2 = find(S2_crossed, 1);

        if first1 < first2
            model_choice(trial) = 1;
        elseif first2 < first1
            model_choice(trial) = 2;
        end
    end
end

% the proportion of trials choosing choice 1 * 100 for %
prop_choice1 = mean(model_choice == 1) * 100;

fprintf('The model chose choice 1 in %.2f%% of trials.\n', prop_choice1);

% irrelevent to this question, but I use later use for Q9 
% decision times  for trials where pop1 won
pop1_decision_times = duration(model_decision == 1);  % in seconds

% mean decision time
avg_decision_time_pop1 = mean(pop1_decision_times);  % in seconds

% displaying
fprintf('Average decision time for Population 1: %.3f seconds (%.0f ms)\n', ...
    avg_decision_time_pop1, avg_decision_time_pop1 * 1000);
%% Question 7: Adjust I_1 and I_2 so that population 1 reaches the threshold of 0.5 first in approximately 100% of trials. 

close all;clear all;clc

% defining time constant 
tau = 0.1; % 100 ms

% defining scaling function
y = 0.641; 

% defining response function 
a = 270 * 10^9; 
b = 108; 
d = 0.15;

% defining synaptic weights
 w_11 = 0.2609 * 10^-9; % from population 1-1
 w_22 = 0.2609 * 10^-9; % from population 2-2
 w_12 = 0.0497 * 10^-9; % from population 1-2
 w_21 = 0.0497 * 10^-9; % from population 2-1

 % defining external current applied, so pop 1 wins 100% of the time
 I_1 = 0.3336 * 10^-9; % external current applied to population 1 is much stronger
 I_2 = 0.3300 * 10^-9; % external current applied to population 2

 % simulation time 
 dt = 0.001; % time step of 1ms 
 time_span = 7; % 6 seconds max per trial
 t = 0:dt:time_span; % time vector 

 % defining threshold 
 thresh = 0.63; % decision threshold at 0.63, can change later 

 total_trial = 100; % a total of 100 simulation trials

 % S population activity initialisation
S1_all = zeros(total_trial, length(t));
S2_all = zeros(total_trial, length(t));

% for calculating firing rate percentage later 
r1_all = zeros(total_trial, length(t));
r2_all = zeros(total_trial, length(t));

 % preallocation for decision making
model_decision = zeros(1, total_trial);
duration = zeros(1, total_trial);

 % simulation loop over trials 
 for trial = 1:total_trial
     S1 = 0.1;
     S2 = 0.1;
     output_decision = false; 

     for i = 1:length(t)
         current_time = t(i);

         % noise inputs on population 1 and 2
         I_noise1 = 0.03 * 10^-9 * randn; % normal noise on inputs to population 1
         I_noise2 = 0.03 * 10^-9 * randn; % normal noise on inputs to population 2

         % computing synpatic input currents
         I_syn1 = w_11*S1 - w_12*S2 + I_1 + I_noise1;
         I_syn2 = w_22*S2 - w_21*S1 + I_2 + I_noise2;

         % computing firing rates using non-linear function 
         r1 = (a*I_syn1 - b) / (1 - exp(-d*(a*I_syn1 - b)));
         r2 = (a*I_syn2 - b) / (1 - exp(-d*(a*I_syn2 - b)));

       % Catch NaN values from numerical overflow
       if isnan(r1) || isinf(r1)
           r1 = 0;
       end
       if isnan(r2) || isinf(r2)
           r2 = 0;
       end

       r1_all(trial, i) = r1;
       r2_all(trial, i) = r2;

         % updating S1 and S2 using Euler's method 
         dS1 = (-S1 / tau + (1-S1) * y * r1) * dt; % equation 1 
         dS2 = (-S2 / tau + (1-S2) * y * r2) * dt; % equation 2 

         S1 = S1 + dS1;
         S2 = S2 + dS2;

         % Catch NaNs or Infs in S values
         if isnan(S1) || isinf(S1)
             S1 = 0;
         end
         if isnan(S2) || isinf(S2)
             S2 = 0;
         end

         S1_all(trial, i) = S1;
         S2_all(trial, i) = S2;

     % checking for threshold crossing 
if ~output_decision && (S1 >= thresh || S2 >= thresh)
    output_decision = true;
    duration(trial) = current_time;
    decision_index(trial) = i;

    if S1 >= thresh
        model_decision(trial) = 1;
    else
        model_decision(trial) = 2;
    end

  S1_all(trial, i:end) = S1;
  S2_all(trial, i:end) = S2;
    break;
end
     end
 end
 
% summary stats display
no_decision_trials = sum(model_decision == 0);
fprintf('Trials with no decision: %d out of %d\n', no_decision_trials, total_trial);
fprintf('Proportion of trials choosing Option 1: %.2f\n', mean(model_decision == 1));
fprintf('Proportion of trials choosing Option 2: %.2f\n', mean(model_decision == 2));
fprintf('Mean decision time: %.3f seconds\n', mean(duration(duration > 0)));
        
% setting the decision threshold
decision_thresh = 0.5; 

% Initialising population that crossed the threshold first, where 1 is
% population 1
model_choice = zeros(total_trial, 1); 

% looping over the 100 trials 
for trial = 1:total_trial
    % find() to determine whether threshold is reached for each t
    S1_crossed = S1_all(trial, :) >= decision_thresh;
    S2_crossed = S2_all(trial, :) >= decision_thresh;

    % if either population crossed 0.5, then if sum > 0
    reached_thresh1 = sum(S1_crossed) > 0;
    reached_thresh2 = sum(S2_crossed) > 0;

    if reached_thresh1 && ~reached_thresh2
        model_choice(trial) = 1; % for population 1 if reached threshold first
    elseif ~reached_thresh1 && reached_thresh2
        model_choice(trial) = 2; % for population 2 if reached threshold first
    elseif reached_thresh1 && reached_thresh2 % if both reached threshold...
        % find which population reached first
        first1 = find(S1_crossed, 1);
        first2 = find(S2_crossed, 1);

        if first1 < first2
            model_choice(trial) = 1;
        elseif first2 < first1
            model_choice(trial) = 2;
        end
    end
end

% the proportion of trials choosing choice 1 * 100 for %
prop_choice1 = mean(model_choice == 1) * 100;

fprintf('The model chose choice 1 in %.2f%% of trials.\n', prop_choice1);

% later use for Q9
% decision times  for trials where pop1 won
pop1_decision_times = duration(model_decision == 1);  % in seconds

% mean decision time
avg_decision_time_pop1 = mean(pop1_decision_times);  % in seconds

% displaying results 
fprintf('Average decision time for Population 1: %.3f seconds (%.0f ms)\n', ...
    avg_decision_time_pop1, avg_decision_time_pop1 * 1000);
%% Question 8: Plot the percentage of trials in which population 1 reaches the threshold of 0.5 first as a function of the difference between I_1 and I_2. 

% Input differences; values used in Q4-Q7
I_1_values = [0.3305, 0.3310, 0.3315, 0.3336]; % for pop1 in nA
I_2_values = [0.3304, 0.3306, 0.3301, 0.3300]; % for pop2 in nA

% calculating the input differences defined for x-axis
input_diff = I_1_values - I_2_values; % how much more input pop1 got

% percentage of trails where pop1 reached 0.5 first
pop1_condition = [50, 60, 80, 100]; % for the y-axis

% plotting difference between input currents over percentage of trials
% where pop1 reaches 0.5 first. 
figure(2);
plot(input_diff, pop1_condition, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('I_1 - I_2 (nA)'); % difference between two inputs 
ylabel('Percentage of trials where Population 1 wins (%)'); % percentage of trials where pop1 
% reaches the threshold first
title('Population 1 Win Rate vs Input Current Difference - Dareen', 'FontSize', 14);
grid on; % for visuals

% figure shows that increasing the difference between input currents (from 50% to 100%) affects
% likelihood of pop1 winning
%% Question 9: Plot the mean decision time for trials where Pop1 wins, as a function of input difference

% defining I_1 and I_2 values in nA (im just re-doing it again)
I_1_values = [0.3305, 0.3310, 0.3315, 0.3336];
I_2_values = [0.3304, 0.3306, 0.3301, 0.3300];

% how much more input pop 1 received; value for x-axis
input_diff = I_1_values - I_2_values; 

% Define average decision times for Pop 1 (manually computed from simulation where % were exact for each percentage, because of randn, values may vary slightly)
%  decision times where Pop 1 reached threshold first
avg_decision_times = [2.638, 2.329, 1.836, 1.240];  % in seconds

% Convert to milliseconds for clarity (optional)
avg_decision_times_ms = avg_decision_times * 1000;

% plotting average decision time for trials where pop1 wins 
figure(3);
plot(input_diff, avg_decision_times_ms, 'o-', 'LineWidth', 2, 'MarkerSize', 8); % so input difference over mean decision times
xlabel('I_1 - I_2 (nA)', 'FontSize', 12);
ylabel('Average Decision Time for Choice 1 (ms)', 'FontSize', 12);
title('Population 1 Decision Time vs Input Current Difference - Dareen', 'FontSize', 14);
grid on; % for better visuals 

% the larger the input difference, the faster pop1 reached 0.5 threshold. 
% the non-linear curve maybe means rapid changes with small bias because
% more input received by pop 1
%% Question 9 Extension: Nonlinear Fit to Decision Time Data

% Fit an exponential decay model to the mean decision times
% as a function of the external input difference.

% Define the exponential decay function
exp_model = @(p, x) p(1) * exp(-p(2) * x) + p(3);

% Initial parameter estimates
initial_params = [2000, 500, 500];

% Estimate parameters by minimising squared residuals
fitted_params = fminsearch(@(p) ...
    sum((avg_decision_times_ms - exp_model(p, input_diff)).^2), ...
    initial_params);

% Generate smooth x-values for plotting
x_fit = linspace(min(input_diff), max(input_diff), 200);

% Calculate fitted decision times
y_fit = exp_model(fitted_params, x_fit);

% Plot observed data and nonlinear fit
figure(4);
plot(input_diff, avg_decision_times_ms, 'o', ...
    'LineWidth', 2, 'MarkerSize', 8);
hold on;

plot(x_fit, y_fit, '-', 'LineWidth', 2);

xlabel('I_1 - I_2 (nA)');
ylabel('Average Decision Time for Choice 1 (ms)');
title('Nonlinear Fit: Decision Time vs Input Current Difference');

legend('Observed Data', 'Exponential Fit', 'Location', 'best');
grid on;
hold off;

% Calculate goodness of fit
predicted_times = exp_model(fitted_params, input_diff);

SS_res = sum((avg_decision_times_ms - predicted_times).^2);
SS_tot = sum((avg_decision_times_ms - mean(avg_decision_times_ms)).^2);

R_squared = 1 - SS_res / SS_tot;

fprintf('Exponential model R-squared: %.3f\n', R_squared);
%% Question 10: Analyse the model in the phase plane for one trial

% 1:Plot in the phase plane

% plotting S1 on the x-axis and S2 on the y-axis for one trial to show how
% the system evolves over time in the space - so the trajectory

% visualising how population activity evolve over time, and where the
% system is stable or unstable 

trial = 1; % choosing trial 1 to plot

% plotting in the phase plane the trajectory of S_1 and S_2 for trial 1
figure(4);
plot(S1_all(trial, :), S2_all(trial, :), 'b', 'LineWidth', 1.5);
hold on;
xlabel('S_1'); % S1 on x-axis
ylabel('S_2'); % S2 on y-axis 
% shows how population activity evolves during the trial 
title('Phase Plane Trajectory for Trial One - Dareen', 'FontSize', 14);
grid on;
hold off; 
% shows how the system moves through state space 

% phase plane shows low activity for both pop, S2 quickly rises but then
% declines as S1 rises, settles into a final state around 0.6 in favour of
% pop1

% The trajectory spirals or flows toward a stable state (an attractor), showing decision formation.


% 2: Try to numerically calculate and plot the nullclines

% S_1 nullcline = all points where dS1/dt = 0
%S_1 nullcline = all points where dS2/dt = 0
% equllibrium when points intersect; so find where its rate of change = 0


% so, grid of S_1 and S_2 

% compute dS_1 and d_S2 at every grid point

% find points that are close to zero and plot those points

% numerical calculation
% defining S1 and S2 ranges for nullclines 
S1_vals = 0:0.01:1;
S2_vals = 0:0.01:1;

% preallocating matrices for dS1 and dS2; setting up the grid 
dS1 = zeros(length(S1_vals), length(S2_vals)); % pop 1
dS2 = zeros(length(S1_vals), length(S2_vals)); % pop 2

% looping over S1 and S2 grid to calculate dS1 and dS2
for i = 1:length(S1_vals)
    for j = 1:length(S2_vals)
        S1 = S1_vals(i);
        S2 = S2_vals(j);

        % Synaptic inputs calculated for each pop
        I_syn1 = w_11 * S1 - w_12 * S2 + I_1;
        I_syn2 = w_22 * S2 - w_21 * S1 + I_2;

        % Firing rates using nonlinear response function (gain) calculated for each pop
        r1 = (a * I_syn1 - b) / (1 - exp(-d * (a * I_syn1 - b)));
        r2 = (a * I_syn2 - b) / (1 - exp(-d * (a * I_syn2 - b)));

        % catching NaN values 
        if isnan(r1) || isinf(r1)
            r1 = 0;
        end
        if isnan(r2) || isinf(r2)
            r2 = 0;
        end

        % computing the derivatives; where rate of change is 0 (the
        % nullclines)
        dS1(i, j) = (-S1 / tau) + (1 - S1) * y * r1;
        dS2(i, j) = (-S2 / tau) + (1 - S2) * y * r2;
    end
end

% using surf(); surface plots to plot across a grid for S1 rate of change
figure(5); 
surf(S1_vals, S2_vals, dS1');
xlabel('S_1'); 
ylabel('S_2'); 
zlabel('dS_1/dt');
title('Surface Plot of dS_1/dt - Dareen');
% shows how dS1/dt varies across state space


% same thing but for S2 rate of change 
figure(6);
surf(S1_vals, S2_vals, dS2');
xlabel('S_1'); 
ylabel('S_2'); 
zlabel('dS_2/dt');
title('Surface Plot of dS_2/dt - Dareen');
% shows how dS2/dt varies across state space

% flatter regions are closer to steady-state

% help locate nullclines visually; flat plane is where derivative = 0

% also plotted nullcline on a 2D graph to locate fixed points

% S1 nullcline
[ind1, ind2, ~] = find(dS1 > -0.01 & dS1 < 0.01); % finding all points where dS1/dt = 0 (

figure(7);
plot(S2_vals(ind2), S1_vals(ind1), 'r.', 'MarkerSize', 10) % plot in red to show where S1 derivative = 0

% S2 nullcline 
[ind1, ind2, ~] = find(dS2 > -0.01 & dS2 < 0.01); %  finding all points where dS2/dt = 0

hold on; 

plot(S2_vals(ind2), S1_vals(ind1), 'b*', 'MarkerSize', 8) % plot in blue to show where S2 derivative = 0

xlabel('S_2')
ylabel('S_1')
legend({'dS_1/dt ≈ 0 (S_1 nullcline)', 'dS_2/dt ≈ 0 (S_2 nullcline)'})
title('Numerical Nullclines of the Model - Dareen')
grid on; 
% intersection points are the  fixed points
% stable attractors if trajectory flows toward these fixed points
% (attractors)

%% 
% 2.5: optionally, overlaying phase trajectory for one trial
hold on;
plot(S2_all(trial, :), S1_all(trial, :), 'k', 'LineWidth', 1.2); % black trajectory line onto nullcline plot

% green mark for final state of the system 
plot(S2_all(trial,end), S1_all(trial,end), 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
%% additional visualistion of decision dynamics - animated, like in lab8

figure(8);

% retrieving the population activity trajectories for trial 1
S1 = S1_all(trial, :);
S2 = S2_all(trial, :);

% for consistency
s_min = min([S1, S2]) - 0.02; % min activity across populations
s_max = max([S1, S2]) + 0.02; % max activity across populations

% looping in steps of 5
for i = 1:5:length(t) % t is 7 seconds
    subplot(2,1,1); % phase plane plot
    cla; % this clears the previous frame 
    hold on;
    plot(S1, S2, 'k'); % x-axis is pop1; y-axis is pop2 (full trajectory)
    plot(S1(i), S2(i), 'o', 'MarkerFaceColor', 'b'); % plots current point in the trajectory in blue
    xlabel('S_1'); ylabel('S_2'); 
    title(' Phase Plane for Trial One - Dareen');

    % again, consistency for axis limits
    xlim([s_min s_max]); 
    ylim([s_min s_max]);

% phase plane shows the trajectory starting from 0.1 for both S1 and S2
% system fluctuates upwards (maybe due to noise or system uncertainty;randn), then increasing S1
% and decreasing S2.
% trajectory settles around 0.63 for S1, and 0.1 for S2. The attractor state is in favour of S1 

    subplot(2,1,2); % plots the time series of S1 and S2 activity 
    cla; 
    hold on;
    plot(t(1:i), S1(1:i), 'b'); % plots S1 in blue up to time i
    plot(t(1:i), S2(1:i), 'r'); % plots S2 in red up to time i
    plot(t(i), S1(i), 'o', 'MarkerFaceColor', 'b'); % current point in blue for S1
    plot(t(i), S2(i), 'o', 'MarkerFaceColor', 'r'); % current point in red for S2
    xlabel('Time'); 
    ylabel('Activity'); 
    title('Population Activity Over Time for Trial One - Dareen');
    legend('S_1','S_2'); 

% fixed axes
    xlim([0 max(t)]); 
    ylim([s_min s_max]);  
    pause(0.05); % short pause for speed
end

% population activity over time shows S1 passing the threshold of 0.63,
% then plateaus 
% S2, however, rises earlier than S1 but becomes suppressed 
% S1 wins 
%%
