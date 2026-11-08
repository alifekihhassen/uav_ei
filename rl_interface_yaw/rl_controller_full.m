function [U, z_ref, psi_ref, u_T, u_yaw, obs_z, obs_yaw] = rl_controller_full(t, z, vz, az_prev, u_T_prev, ...
                                                                            psi, wz, w_dot_z_prev, u_yaw_prev)
%RL_CONTROLLER_FULL  Altitude policy + yaw policy + mixer, called at each step by EINT_RL_closed_loop_yaw.
%   The same computations as fly() in python/full_mission.py, in the same order:
%     1. altitude policy (policy.mat):     obs_z   -> u_T      (observation of drone_env.Observer)
%     2. yaw policy      (policy_yaw.mat): obs_yaw -> u_yaw    (observation of yaw_env.YawObserver,
%                                                               it sees the u_T chosen at this step)
%     3. mixer: u_yaw clipped to +-(1 - |u_T|) (thrust has priority), then
%               u1 = u3 = u_T + u_yaw,   u2 = u4 = u_T - u_yaw
%   Inputs: measurements at t_k (z, vz, psi, wz), the accelerations of the previous step (az_prev,
%   w_dot_z_prev) and the commands of the previous step (u_T_prev, u_yaw_prev).
%   The two error integrals are persistent variables: "clear rl_controller_full" resets them (start of
%   each flight). Called ONCE per step (Ts = 0.05 s).
persistent PZ PY
if isempty(PZ)
    here = fileparts(mfilename('fullpath'));
    PZ = load_policy(fullfile(here, 'policy.mat'));
    PY = load_policy(fullfile(here, 'policy_yaw.mat'));
end
S = [1; -1; 1; -1];

z_ref = mission_z(t);
psi_ref = mission_psi(t);

obs_z = observation_z(z_ref, z, vz, az_prev, u_T_prev);
u_T = run_policy(PZ, obs_z);

obs_yaw = observation_yaw(psi_ref, psi, wz, w_dot_z_prev, u_yaw_prev, u_T);
u_yaw = run_policy(PY, obs_yaw);

margin = 1 - abs(u_T);
u_yaw = min(max(u_yaw, -margin), margin);
U = u_T + u_yaw * S;
end


%% ------------------------------------------------------------------------------------------------
function P = load_policy(file)
P = load(file);
P.acts = strsplit(P.activations, ',');
end

function u = run_policy(P, obs)
% The exported actor: dense layers + ReLU/tanh, then the output scaling of the algorithm (as rl_controller.m).
x = double(obs(:));
for i = 1:double(P.n_layers)
    x = P.(sprintf('W%d', i)) * x + P.(sprintf('b%d', i));
    switch P.acts{i}
        case 'relu', x = max(x, 0);
        case 'tanh', x = tanh(x);
    end
end
lo = P.action_low(:);
hi = P.action_high(:);
switch P.output
    case 'tanh_scale', x = lo + 0.5 * (tanh(x) + 1) .* (hi - lo);   % SAC
    case 'scale',      x = lo + 0.5 * (x + 1) .* (hi - lo);         % TD3
    otherwise,         x = min(max(x, lo), hi);                     % PPO
end
u = x(1);
end

function obs = observation_z(z_ref, z, vz, az_prev, u_prev)
% drone_env.Observer: obs = [e; z; vz/2; az_prev/10; u_prev; I/I_MAX; z_ref], e = z_ref - z (clipped)
persistent I
if isempty(I), I = 0; end
Ts = 0.05;  I_MAX = 0.5;
e = z_ref - z;
I = min(max(I + Ts * e, -I_MAX), I_MAX);
obs = [min(max(e, -1), 1);
       z;
       min(max(vz / 2, -5), 5);
       min(max(az_prev / 10, -5), 5);
       u_prev;
       I / I_MAX;
       z_ref];
end

function obs = observation_yaw(psi_ref, psi, wz, w_dot_z_prev, u_yaw_prev, u_T)
% yaw_env.YawObserver: obs = [e; wz/5; w_dot_z_prev/20; u_yaw_prev/YAW_MAX; I/I_MAX; u_T],
% e = psi_ref - psi clipped to +-pi (both angles unwrapped: no wrap-around here)
persistent I
if isempty(I), I = 0; end
Ts = 0.05;  I_MAX = 0.5;  YAW_MAX = 0.1;
e = psi_ref - psi;
I = min(max(I + Ts * e, -I_MAX), I_MAX);
obs = [min(max(e, -pi), pi);
       min(max(wz / 5, -5), 5);
       min(max(w_dot_z_prev / 20, -5), 5);
       u_yaw_prev / YAW_MAX;
       I / I_MAX;
       u_T];
end

function z_ref = mission_z(t)
% Full mission: 0 m on the platform until 1 s, 0.5 m until 10 s, then -0.25 m (ground).
% Thresholds shifted by Ts/2, as in Python (full_mission.mission_z).
Ts = 0.05;
if t < 1.0 - Ts / 2
    z_ref = 0;
elseif t < 10.0 - Ts / 2
    z_ref = 0.5;
else
    z_ref = -0.25;
end
end

function psi_ref = mission_psi(t)
% Four quarter turns at t = 2, 4, 6, 8 s: 90, 180, 270, 360 deg (unwrapped), as yaw_env.yaw_mission.
Ts = 0.05;
psi_ref = (pi / 2) * sum(t >= [2.0, 4.0, 6.0, 8.0] - Ts / 2);
end
