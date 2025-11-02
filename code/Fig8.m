%% ================================================================================
% DUAL HCO NEUROMORPHIC PENDULUM CONTROL WITH FREQUENCY SWEEP
% This script simulates a pendulum driven by a dual Half-Center Oscillator
% (HCO) neural network. The neural network adapts its torque gains to
% maintain a target amplitude across a wide frequency sweep (1–100 Hz).
%
% Author: Xinxin Zhang
% Date: 2025-07-04
%% ================================================================================
clear; clc; close all;

%% ================================================================================
%% SECTION 1: REFERENCE SIGNAL GENERATION
%% ================================================================================
% Purpose: generate target sinusoidal trajectory for pendulum
A = 1; % target amplitude in radians

% Frequency sweep setup (logarithmic from 1 Hz to 100 Hz)
frequencies = logspace(0, 2, 50);

% Preallocate arrays to store final adapted torque gains
final_pos_gains = zeros(1, length(frequencies));
final_neg_gains = zeros(1, length(frequencies));

% Simulation setup
T_periods = 600; % total number of pendulum periods to simulate
Fs = 1e2; % sampling factor for numerical integration

%% ================================================================================
%% SECTION 2: PENDULUM PHYSICAL PARAMETERS
%% ================================================================================
rho = 1000; % density (kg/m³)
L   = 0.36; % length (m)
r   = 0.02; % radius (m)
vol = pi*r^2*L;
m   = rho*vol; % pendulum mass (kg)
g   = 9.81;   % gravitational acceleration (m/s²)
Jperp = (1/12)*m*(L^2 + 3*r^2) + m*(L/2)^2; % rotational inertia (kg·m²)
damp  = 1; % damping coefficient (N·m·s/rad)

%% ================================================================================
%% SECTION 3: CONTROLLER PARAMETER TUNING
%% ================================================================================
% Neuromorphic HCO time constants (fixed across frequencies for stability)
f_base     = 1;           
tau_f_base = 0.01005;    % fast membrane
tau_s_base = 20*tau_f_base;  % slow synaptic
tau_us_base=100*tau_f_base;  % ultra-slow adaptation

% Optional PD feedback gains (here set to zero)
Kp_fb = 0; 
Kd_fb = 0;

v_threshold = 3; % neuron firing threshold (for torque output)

%% ================================================================================
%% SECTION 4: DUAL HCO NETWORK PARAMETERS
%% ================================================================================
g_f      = 2;   % self-excitation
g_s_plus = 1.5; % mutual inhibition within HCO
g_s_minus= 1;   % cross-excitation
g_us     = 1.5; % ultra-slow inhibition
a_cross  = 1.5; % cross-HCO inhibition

% Synaptic transfer function (sigmoidal)
synapse = @(vs,gain) gain./(1 + exp(-2*(vs+1)));

%% ================================================================================
%% MAIN FREQUENCY SWEEP LOOP
%% ================================================================================
fprintf('Starting dual HCO neuromorphic pendulum control frequency sweep...\n');

for i = 1:length(frequencies)
    f_theta = frequencies(i);
    omega   = 2*pi*f_theta; % angular frequency (rad/s)
    
    Tf = T_periods / f_theta;   % total simulation time
    dt = 1/(f_theta*Fs);        % time step
    t  = 0:dt:Tf;               % time vector
    
    % Reference trajectory
    theta_ref       = A*sin(omega*t);
    dot_theta_ref_t = A*omega*cos(omega*t);
    
    % Interpolation functions for continuous-time reference
    theta_interp   = @(tq) interp1(t, theta_ref,       tq, 'linear', 'extrap');
    dot_theta_ref  = @(tq) interp1(t, dot_theta_ref_t, tq, 'linear', 'extrap');
    
    % Scale HCO time constants inversely with frequency
    tau_f  = tau_f_base  * f_base / f_theta;
    tau_s  = tau_s_base  * f_base / f_theta;
    tau_us = tau_us_base * f_base / f_theta;
    
    % Initial torque gains
    torque_gain      = f_theta/2;
    torque_pos_gain  = torque_gain;
    torque_neg_gain  = torque_gain;
    
    % Base external input to neurons
    I_ext_base = -1.0;
    
    % Amplitude adaptation parameters
    period_len       = round(1/f_theta/dt);
    learning_rate_amp= f_theta*1.5; % proportional adaptation gain
    
    % State vector: [v1,s1,u1, v2,s2,u2, v3,s3,u3, v4,s4,u4, theta, omega]
    x = zeros(14, length(t));
    x(:,1) = [0.1, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0];
    
    % History arrays for logging (optional)
    theta_real_hist          = zeros(size(t));
    torque_hist              = zeros(size(t));
    input_hco1_hist          = zeros(size(t));
    input_hco2_hist          = zeros(size(t));
    torque_pos_gain_hist     = zeros(size(t));
    torque_neg_gain_hist     = zeros(size(t));
    cross_inhibition_hco1_hist = zeros(size(t));
    cross_inhibition_hco2_hist = zeros(size(t));
    
    fprintf('Simulating frequency: %.2f Hz...\n', f_theta);
    
    %% ============================================================================
    %% SIMULATION LOOP WITH ADAPTIVE AMPLITUDE CONTROL
    %% ============================================================================
    for k = 2:length(t)
        % PD feedback to neurons
        theta_error     = theta_interp(t(k-1)) - x(13,k-1);
        dot_theta_error = dot_theta_ref(t(k-1)) - x(14,k-1);
        pd_feedback = Kp_fb*theta_error + Kd_fb*dot_theta_error;
        
        % Inputs to HCOs modulated by feedback
        input_hco1 = I_ext_base + max(pd_feedback,0);
        input_hco2 = I_ext_base + max(-pd_feedback,0);
        
        % HCO neural dynamics
        dxdt = zeros(14,1);
        
        % --- HCO1 dynamics ---
        dxdt(1) = (-x(1,k-1) + g_f*tanh(x(1,k-1)) - g_s_plus*tanh(x(2,k-1)) ...
                    + g_s_minus*tanh(x(2,k-1)+0.9) - g_us*tanh(x(3,k-1)+0.9) ...
                    + synapse(x(5,k-1), -0.2) - a_cross*tanh(x(7,k-1)) ...
                    + input_hco1) / tau_f;
        dxdt(2) = (x(1,k-1) - x(2,k-1)) / tau_s;
        dxdt(3) = (x(1,k-1) - x(3,k-1)) / tau_us;
        dxdt(4) = (-x(4,k-1) + g_f*tanh(x(4,k-1)) - g_s_plus*tanh(x(5,k-1)) ...
                    + g_s_minus*tanh(x(5,k-1)+0.9) - g_us*tanh(x(6,k-1)+0.9) ...
                    + synapse(x(2,k-1), -0.2) + a_cross*tanh(x(7,k-1)) ...
                    + input_hco1) / tau_f;
        dxdt(5) = (x(4,k-1) - x(5,k-1)) / tau_s;
        dxdt(6) = (x(4,k-1) - x(6,k-1)) / tau_us;
        
        % --- HCO2 dynamics ---
        dxdt(7) = (-x(7,k-1) + g_f*tanh(x(7,k-1)) - g_s_plus*tanh(x(8,k-1)) ...
                    + g_s_minus*tanh(x(8,k-1)+0.9) - g_us*tanh(x(9,k-1)+0.9) ...
                    + synapse(x(11,k-1), -0.2) - a_cross*tanh(x(1,k-1)) ...
                    + input_hco2) / tau_f;
        dxdt(8) = (x(7,k-1) - x(8,k-1)) / tau_s;
        dxdt(9) = (x(7,k-1) - x(9,k-1)) / tau_us;
        dxdt(10)= (-x(10,k-1)+g_f*tanh(x(10,k-1)) - g_s_plus*tanh(x(11,k-1)) ...
                    + g_s_minus*tanh(x(11,k-1)+0.9) - g_us*tanh(x(12,k-1)+0.9) ...
                    + synapse(x(8,k-1), -0.2) + a_cross*tanh(x(1,k-1)) ...
                    + input_hco2) / tau_f;
        dxdt(11)= (x(10,k-1)-x(11,k-1)) / tau_s;
        dxdt(12)= (x(10,k-1)-x(12,k-1)) / tau_us;
        
        % Torque generation
        torque = torque_pos_gain*(1/(1+exp(-2*(x(1,k-1)-v_threshold)))) ...
               - torque_neg_gain*(1/(1+exp(-2*(x(7,k-1)-v_threshold))));
        
        % Pendulum dynamics
        dxdt(13) = x(14,k-1); % dtheta/dt
        dxdt(14) = (-g*(L/2)*sin(x(13,k-1)) - damp*x(14,k-1) + torque) / Jperp;
        
        % Update state
        x(:,k) = x(:,k-1) + dt*dxdt;
        
        % --- Amplitude adaptation (every period) ---
        if mod(k, period_len)==0
            idx = (k-period_len+1):k;
            center = (max(x(13,idx)) + min(x(13,idx)))/2;
            pos_mag = max(x(13,idx)) - center;
            neg_mag = center - min(x(13,idx));
            
            if pos_mag>1e-4 && neg_mag>1e-4
                torque_pos_gain = torque_pos_gain + learning_rate_amp*tanh(A-pos_mag);
                torque_neg_gain = torque_neg_gain + learning_rate_amp*tanh(A-neg_mag);
                % Bound gains
                torque_pos_gain = max(min(torque_pos_gain, 1e6), 1e-4);
                torque_neg_gain = max(min(torque_neg_gain, 1e6), 1e-4);
            end
        end
    end
    
    % Save final gains for this frequency
    final_pos_gains(i) = torque_pos_gain;
    final_neg_gains(i) = torque_neg_gain;
end

%% ================================================================================
% SAVE RESULTS
%% ================================================================================
fprintf('Frequency sweep complete. Saving data to tuned_gains_vthr3.mat...\n');
save('tuned_gains_vthr3.mat', 'frequencies', 'final_pos_gains', 'final_neg_gains');
fprintf('Data successfully saved.\n');
