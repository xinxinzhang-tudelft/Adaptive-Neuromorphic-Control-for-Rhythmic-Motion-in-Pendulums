%% ================================================================================
%% DUAL HCO NEUROMORPHIC PENDULUM CONTROLLER (V27)
%% ================================================================================
% This script implements a neuromorphic control architecture using a dual
% Half-Center Oscillator (HCO) network to generate periodic torque for a
% pendulum. 
%
% Author: Xinxin Zhang
% Date: 2025-07-04
%% ================================================================================

clear; clc; close all; % Reset MATLAB state for a clean simulation environment

%% ================================================================================
%% SECTION 1: REFERENCE SIGNAL GENERATION
%% ================================================================================
% Generates the sinusoidal desired pendulum motion and supports interpolation
A = 1; % Desired oscillation amplitude (rad)
f_theta = 10; % Target oscillation frequency (Hz)
omega = 2*pi*f_theta; % Convert to angular frequency (rad/s)

% Simulation resolution and duration
Tf = 1/f_theta*50; % Duration = 50 oscillation cycles
Fs = 1e2; % Sample resolution multiplier
dt = 1/f_theta/Fs; % Time step (small to support stiff ODE)
t = 0:dt:Tf; % Time vector

% Reference trajectory and velocity
theta_ref = A*sin(omega*t); % Ideal pendulum angle reference
dot_theta_ref_t = A*omega*cos(omega*t); % Ideal angular velocity

% Build interpolation functions usable inside real-time loop
theta_interp = @(tq) interp1(t, theta_ref, tq, 'linear', 'extrap');
dot_theta_ref = @(tq) interp1(t, dot_theta_ref_t, tq, 'linear', 'extrap');

%% ================================================================================
%% SECTION 2: PENDULUM PHYSICAL PARAMETERS
%% ================================================================================
% Model as a rigid uniform cylinder pendulum
rho = 1000; % Material density (kg/m³)
L = 0.36; % Length (m)
r = 0.02; % Radius (m)
vol = pi * r^2 * L; % Cylinder volume
m = rho * vol; % Mass (kg)
g = 9.81; % Gravity (m/s²)

% Moment of inertia about pivot: composite of body + translation to pivot
Jperp = (1/12)*m*(L^2 + 3*r^2) + m*(L/2)^2;
damp =1; % Joint viscous damping (N·m·s/rad)

%% ================================================================================
%% SECTION 3: CONTROLLER PARAMETER TUNING
%% ================================================================================
% Normalization around baseline frequency ensures stability
f_base = 1;
tau_m_base = 0.01005; % Membrane dynamic speed
tau_s_base = 20*tau_m_base; % Fast adaptation
tau_us_base =100*tau_m_base; % Slow adaptation

% Frequency scaling (↑ freq → ↓ time constants)
kk = 1.0;
tau_m = tau_m_base * f_base / f_theta;
tau_s = tau_s_base * f_base / f_theta;
tau_us = tau_us_base * f_base / f_theta;

% Initial torque gains for positive + negative swing
torque_gain = f_theta;
torque_pos_gain = torque_gain;
torque_neg_gain = torque_gain;

% Tonic input drives oscillation
I_ext_base = -1.0;

% PD feedback interface (default disabled)
Kp_fb = 0;
Kd_fb = 0;

% Neuron firing threshold
v_threshold = 0;

%% ================================================================================
%% SECTION 4: DUAL HCO NETWORK PARAMETERS
%% ================================================================================
% Neural connectivity structure
g_f = 2; % Self excitation
g_s_plus = 1.5; % Local inhibition inside HCO
g_s_minus = 1; % Cross excitation within HCO
g_us = 1.5; % Ultra-slow fatigue feedback
a_cross = 1.5; % Competition between HCOs

% Sigmoid synapse for nonlinear inhibitory control
synapse = @(vs,gain) gain./(1 + exp(-2*(vs+1)));

%% ================================================================================
%% SECTION 5: ADAPTATION PARAMETERS
%% ================================================================================
period_len = round(1/f_theta/dt); % Samples per oscillation period
learning_rate_amp = f_theta*1.5; % Torque adaptation proportionality

%% ================================================================================
%% SECTION 6: SIMULATION STATE INITIALIZATION
%% ================================================================================
% State vector: [6 neural states per HCO ×2] + [theta, angular velocity]
x = zeros(14, length(t));
x(:,1) = [0.1, 0, -1, 0, 0, -0, 0, 0, -1, 0, 0, -0, 0, 0];

% Data logging
theta_real_hist = zeros(size(t));
torque_hist = zeros(size(t));
input_hco1_hist = zeros(size(t));
input_hco2_hist = zeros(size(t));
torque_pos_gain_hist = zeros(size(t));
torque_neg_gain_hist = zeros(size(t));
cross_inhibition_hco1_hist = zeros(size(t));
cross_inhibition_hco2_hist = zeros(size(t));

%% ================================================================================
%% SECTION 7: MAIN SIMULATION LOOP WITH ADAPTATION
%% ================================================================================
fprintf('Starting DUAL HCO neuromorphic pendulum control simulation...\n');
fprintf('Initial torque gain: %.2f\n', torque_gain);

for k = 2:length(t)

    % PD feedback error contribution
    theta_error = theta_interp(t(k-1)) - x(13,k-1);
    dot_theta_error = dot_theta_ref(t(k-1)) - x(14,k-1);
    pd_feedback = Kp_fb * theta_error + Kd_fb * dot_theta_error;

    % Split PD input: positive drives HCO1, negative drives HCO2
    input_hco1 = I_ext_base + max(pd_feedback, 0);
    input_hco2 = I_ext_base + max(-pd_feedback, 0);

    % ===== Neuromorphic population ODE =====
    dxdt = zeros(14,1);

    % HCO1 neuron dynamics (1 & 2)
    dxdt(1) = (-x(1,k-1) + g_f*tanh(x(1,k-1)) - g_s_plus*tanh(x(2,k-1)) + ...
        g_s_minus*tanh(x(2,k-1)+0.9) - a4*tanh(x(3,k-1)+0.9) + ...
        synapse(x(5,k-1), -0.2) - a_cross*tanh(x(7,k-1)) + input_hco1) / tau_m;
    dxdt(2) = (x(1,k-1) - x(2,k-1)) / tau_s;
    dxdt(3) = (x(1,k-1) - x(3,k-1)) / tau_us;

    dxdt(4) = (-x(4,k-1) + g_f*tanh(x(4,k-1)) - g_s_plus*tanh(x(5,k-1)) + ...
        g_s_minus*tanh(x(5,k-1)+0.9) - a4*tanh(x(6,k-1)+0.9) + ...
        synapse(x(2,k-1), -0.2) + a_cross*tanh(x(7,k-1)) + input_hco1) / tau_m;
    dxdt(5) = (x(4,k-1) - x(5,k-1)) / tau_s;
    dxdt(6) = (x(4,k-1) - x(6,k-1)) / tau_us;

    % HCO2 neuron dynamics (3 & 4)
    dxdt(7) = (-x(7,k-1) + g_f*tanh(x(7,k-1)) - g_s_plus*tanh(x(8,k-1)) + ...
        g_s_minus*tanh(x(8,k-1)+0.9) - a4*tanh(x(9,k-1)+0.9) + ...
        synapse(x(11,k-1), -0.2) - a_cross*tanh(x(1,k-1)) + input_hco2) / tau_m;
    dxdt(8) = (x(7,k-1) - x(8,k-1)) / tau_s;
    dxdt(9) = (x(7,k-1) - x(9,k-1)) / tau_us;

    dxdt(10) = (-x(10,k-1) + g_f*tanh(x(10,k-1)) - g_s_plus*tanh(x(11,k-1)) + ...
        g_s_minus*tanh(x(11,k-1)+0.9) - a4*tanh(x(12,k-1)+0.9) + ...
        synapse(x(8,k-1), -0.2) + a_cross*tanh(x(1,k-1)) + input_hco2) / tau_m;
    dxdt(11) = (x(10,k-1) - x(11,k-1)) / tau_s;
    dxdt(12) = (x(10,k-1) - x(12,k-1)) / tau_us;

    % ===== Torque generation based on neuron dominance =====
    torque = torque_pos_gain * (1./(1 + exp(-2*(x(1,k-1) - v_threshold)))) - ...
             torque_neg_gain * (1./(1 + exp(-2*(x(7,k-1) - v_threshold))));

    % ===== Pendulum motion =====
    dxdt(13) = x(14,k-1);
    dxdt(14) = (-g*(L/2)*sin(x(13,k-1)) - damp*x(14,k-1) + torque) / Jperp;

    % ===== Euler integration =====
    x(:,k) = x(:,k-1) + dt * dxdt;

    % ===== Torque adaptation once per oscillatory period =====
    if mod(k, period_len) == 0
        idx = (k - period_len + 1):k;

        current_center = (max(x(13,idx)) + min(x(13,idx))) / 2;
        pos_mag = max(x(13,idx)) - current_center;
        neg_mag = current_center - min(x(13,idx));

        if pos_mag > 1e-4 && neg_mag > 1e-4
            torque_pos_gain = torque_pos_gain + learning_rate_amp * tanh(A - pos_mag);
            torque_neg_gain = torque_neg_gain + learning_rate_amp * tanh(A - neg_mag);
            torque_pos_gain = max(min(torque_pos_gain, 1e6), 1e-4);
            torque_neg_gain = max(min(torque_neg_gain, 1e6), 1e-4);
        end
    end

    % Logging
    theta_real_hist(k) = x(13,k);
    torque_hist(k) = torque;
    input_hco1_hist(k) = input_hco1;
    input_hco2_hist(k) = input_hco2;
    torque_pos_gain_hist(k) = torque_pos_gain;
    torque_neg_gain_hist(k) = torque_neg_gain;
    cross_inhibition_hco1_hist(k) = a_cross * tanh(x(7,k-1));
    cross_inhibition_hco2_hist(k) = a_cross * tanh(x(1,k-1));
end

fprintf('Final positive torque gain: %.2f\n', torque_pos_gain);
fprintf('Final negative torque gain: %.2f\n', torque_neg_gain);

%% ================================================================================
%% SECTION 8: RESULTS VISUALIZATION
%% ================================================================================
fprintf('Simulation completed. Generating plots...\n');

% Plot pendulum motion vs torque
figure(1);
set(gcf, 'Position', [100, 100, 500, 600]);
subplot(3,1,1); plot(t, theta_real_hist, 'r', 'LineWidth', 1.5); ylabel('\theta (rad)');
subplot(3,1,2); plot(t, x(14,:), 'k', 'LineWidth', 1.5); ylabel('\omega (rad/s)');
subplot(3,1,3); plot(t, torque_hist, 'k', 'LineWidth', 2); ylabel('\tau (N·m)'); xlabel('Time (s)');

% Neural voltage monitoring
figure(2);
subplot(3,1,1); plot(t,x(1,:), 'r', t,x(4,:), 'b:'); ylabel('HCO1 Voltage');
subplot(3,1,2); plot(t,x(7,:), 'g', t,x(10,:), 'm:'); ylabel('HCO2 Voltage');
subplot(3,1,3); plot(t, torque_hist,'k'); ylabel('\tau (N·m)'); xlabel('Time (s)');

% Adaptation tracking
figure(3);
plot(t, torque_pos_gain_hist,'r'); hold on;
plot(t, torque_neg_gain_hist,'b');
xlabel('Time (s)'); ylabel('Torque Gain'); title('Gain Adaptation'); grid on;

%% ================================================================================
%% END OF DUAL HCO SIMULATION
%% ================================================================================
