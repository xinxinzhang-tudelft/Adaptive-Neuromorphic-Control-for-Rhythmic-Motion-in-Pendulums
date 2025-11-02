%% ================================================================================
% DUAL HCO Neuromorphic Pendulum Controller - Frequency Sweep Gain Adaptation
% This script tunes neuromorphic torque gains for pendulum control across
% a wide frequency range (1–100 Hz), generating a lookup table for future use.
%
% Main idea:
%   - Dual Half-Center Oscillator (HCO) neural network produces torque pulses 
%   - Adaptive rule constantly corrects swing amplitude
%   - Final tuned gains are saved for each frequency
%
% Author: Xinxin Zhang
% Date: 2025-07-04
% ================================================================================
clear; clc; close all;
%% ================================================================================
%% SECTION 1: REFERENCE SIGNAL GENERATION - FREQUENCY SWEEP SETUP
%% ================================================================================
A = 1; % Target pendulum amplitude (radians) to maintain across sweep

% Sweep frequency on a log scale from 1 Hz → 100 Hz
frequencies = logspace(0, 2, 50);

% Gain storage arrays: final tuned torque positive/negative gains for each frequency
final_pos_gains = zeros(1, length(frequencies));
final_neg_gains = zeros(1, length(frequencies));

% Simulation length setup (number of oscillation cycles per frequency)
T_periods = 500;
Fs = 1e2; % Normalized sampling resolution w.r.t. frequency
%% ================================================================================
%% SECTION 2: PHYSICAL PENDULUM SYSTEM
%% ================================================================================
rho = 1000; % Material density (e.g., plastics/wood)
L = 0.36;   % Pendulum length (m)
r = 0.02;   % Radius (m)
vol = pi * r^2 * L; 
m = rho * vol; % Mass (kg)
g = 9.81;  % Gravity
% Rotational inertia around pivot (parallel axis theorem included)
Jperp = (1/12)*m*(L^2 + 3*r^2) + m*(L/2)^2;
damp = 1; % Linear damping, compensates friction & air resistance
%% ================================================================================
%% SECTION 3: CONTROLLER PARAMETERS (NEURAL CONSTANTS)
%% ================================================================================
% Time constants ✔ Fixed for stability across frequency sweep
f_base = 1;
tau_f_base  = 0.01005;        % Fast membrane dynamics
tau_s_base  = 20*tau_f_base;  % Slow synaptic recovery
tau_us_base = 100*tau_f_base;  % Ultra-slow activity regulation

% PD feedback disabled for pure neuromorphic testing
Kp_fb = 0;
Kd_fb = 0;
v_threshold = 0; % Neuron activation threshold (soft)

%% ================================================================================
%% SECTION 4: NEUROMORPHIC HCO NETWORK PARAMETERS
%% ================================================================================
% HCO = rhythmic neural oscillator model — antagonistic neurons
g_f      = 2;   % Self excitation
g_s_plus = 1.5; % Mutual inhibition inside each HCO pair (push-pull)
g_s_minus= 1;   % Weak cooperation
g_us     = 1.5; % Ultra-slow adaptation to stabilize oscillations
a_cross  = 1.5; % Cross-HCO inhibition (coordination between torque directions)

% Sigmoidal synaptic transfer
synapse = @(vs,gain) gain./(1 + exp(-2*(vs+1)));

%% ================================================================================
%% MAIN: FREQUENCY SWEEP LOOP
%% ================================================================================
fprintf('\n--- Starting Sweep for Neuromorphic Gain Calibration ---\n\n');
for i = 1:length(frequencies)

    %% Frequency → Reference signal setup
    f_theta = frequencies(i);
    omega   = 2*pi*f_theta;
    Tf = 1/f_theta*T_periods; % Ensure stable convergence per frequency
    dt = 1/(f_theta*Fs);
    t = 0:dt:Tf;

    % Target oscillation (reference)
    theta_ref        = A*sin(omega*t);
    dot_theta_ref_t  = A*omega*cos(omega*t);
    theta_interp     = @(tq) interp1(t, theta_ref,       tq, 'linear','extrap');
    dot_theta_ref    = @(tq) interp1(t, dot_theta_ref_t, tq, 'linear','extrap');

    %% Scale neuron dynamics proportional to oscillation speed
    tau_f  = tau_f_base  * f_base / f_theta;
    tau_s  = tau_s_base  * f_base / f_theta;
    tau_us = tau_us_base * f_base / f_theta;

    %% Begin from equal torque gains → controller will adapt
    torque_gain = 1;
    torque_pos_gain = torque_gain;
    torque_neg_gain = torque_gain;

    I_ext_base = -1.0; % Bias input baseline

    %% Amplitude adaptation parameters (period-based)
    period_len = round(1/f_theta/dt);
    learning_rate_amp = f_theta; % Adapt faster at high frequency

    %% State vector [4 neurons + their slow components + pendulum mech]
    x = zeros(14,length(t));
    x(:,1) = [0.1 0 -1  0 0 0  0 0 -1  0 0 0  0 0]; % Initial conditions

    %% History storage arrays (for visualization/debug, not used further)
    theta_real_hist = zeros(size(t));
    torque_hist = zeros(size(t));
    input_hco1_hist = zeros(size(t));
    input_hco2_hist = zeros(size(t));
    torque_pos_gain_hist = zeros(size(t));
    torque_neg_gain_hist = zeros(size(t));
    cross_inhibition_hco1_hist = zeros(size(t));
    cross_inhibition_hco2_hist = zeros(size(t));

    fprintf('Simulating frequency = %.2f Hz...\n', f_theta);

    %% Core closed-loop simulation
    for k = 2:length(t)

        %%  Feedback modulation → sooner to neurons
        theta_err = theta_interp(t(k-1)) - x(13,k-1);
        dtheta_err= dot_theta_ref(t(k-1)) - x(14,k-1);
        pd_fb = Kp_fb * theta_err + Kd_fb * dtheta_err;

        input_hco1 = I_ext_base + max(pd_fb, 0);
        input_hco2 = I_ext_base + max(-pd_fb,0);

        %% NEURAL DYNAMICS : 4 Neurons (2 per HCO)
        dxdt = zeros(14,1);

        % HCO1
        dxdt(1) = (-x(1)+ g_f*tanh(x(1))- g_s_plus*tanh(x(2))+ ...
                    g_s_minus*tanh(x(2)+0.9)- g_us*tanh(x(3)+0.9) + ...
                    synapse(x(5), -0.2) - a_cross*tanh(x(7)) + input_hco1) / tau_f;
        dxdt(2) = (x(1) - x(2)) / tau_s;
        dxdt(3) = (x(1) - x(3)) / tau_us;
        dxdt(4) = (-x(4)+ g_f*tanh(x(4))- g_s_plus*tanh(x(5))+ ...
                    g_s_minus*tanh(x(5)+0.9) - g_us*tanh(x(6)+0.9) + ...
                    synapse(x(2), -0.2) + a_cross*tanh(x(7)) + input_hco1) / tau_f;
        dxdt(5) = (x(4) - x(5)) / tau_s;
        dxdt(6) = (x(4) - x(6)) / tau_us;

        % HCO2
        dxdt(7) = (-x(7)+ g_f*tanh(x(7))- g_s_plus*tanh(x(8)) + ...
                    g_s_minus*tanh(x(8)+0.9) - g_us*tanh(x(9)+0.9) + ...
                    synapse(x(11), -0.2) - a_cross*tanh(x(1)) + input_hco2) / tau_f;
        dxdt(8) = (x(7) - x(8)) / tau_s;
        dxdt(9) = (x(7) - x(9)) / tau_us;
        dxdt(10)= (-x(10)+g_f*tanh(x(10))- g_s_plus*tanh(x(11)) + ...
                    g_s_minus*tanh(x(11)+0.9) - g_us*tanh(x(12)+0.9) + ...
                    synapse(x(8), -0.2) + a_cross*tanh(x(1)) + input_hco2) / tau_f;
        dxdt(11)= (x(10)-x(11)) / tau_s;
        dxdt(12)= (x(10)-x(12)) / tau_us;

        %% Torque generation: neuron 1 → positive swing, neuron 3 → negative
        torque = torque_pos_gain * (1/(1 + exp(-2*(x(1)-v_threshold)))) - ...
                 torque_neg_gain * (1/(1 + exp(-2*(x(7)-v_threshold))));

        %% Pendulum mechanical dynamics: theta, dtheta
        dxdt(13)= x(14);
        dxdt(14)= (-g*(L/2)*sin(x(13)) - damp*x(14) + torque) / Jperp;

        x(:,k)= x(:,k-1) + dt*dxdt;

        %% AMPLITUDE ADAPTATION per oscillation period
        if mod(k,period_len)==0
            idx = (k-period_len+1):k;
            ctr = (max(x(13,idx))+min(x(13,idx))) / 2;
            pos_mag = max(x(13,idx)) - ctr;
            neg_mag = ctr - min(x(13,idx));

            if pos_mag>1e-4 && neg_mag>1e-4
                torque_pos_gain = torque_pos_gain + learning_rate_amp*tanh(A-pos_mag);
                torque_neg_gain = torque_neg_gain + learning_rate_amp*tanh(A-neg_mag);

                torque_pos_gain = max(min(torque_pos_gain,1e6),1e-4);
                torque_neg_gain = max(min(torque_neg_gain,1e6),1e-4);
            end
        end
    end

    % Save gains after steady-state
    final_pos_gains(i)= torque_pos_gain;
    final_neg_gains(i)= torque_neg_gain;
end
%% ================================================================================
fprintf('\n--- Sweep Completed ✅  Saving tuned gains... ---\n');
save('tuned_gains.mat','frequencies','final_pos_gains','final_neg_gains');
fprintf('✔ Data saved → tuned_gains.mat\n');
%% ================================================================================
