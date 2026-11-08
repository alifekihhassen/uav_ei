function [U, z_ref, psi_ref, x_ref, y_ref, u_T, u_yaw, u_x, u_y, obs_z, obs_yaw, obs_x, obs_y] = ...
    rl_controller_follow(t, z, vz, az_prev, u_T_prev, psi, wz, w_dot_z_prev, u_yaw_prev, ...
                         x, vx, wy, w_dot_y_prev, ax_prev, u_x_prev, y, vy, wx, w_dot_x_prev, ay_prev, u_y_prev)
%RL_CONTROLLER_FOLLOW  Advanced tasks 3 and 4: follow the ground vehicle, and land on it. Altitude + yaw +
%   TRACKING x policies and the mixer, called at each step by EINT_RL_closed_loop_follow. The drone takes
%   off from the platform at 1 s and stays 0.5 m above it while the vehicle drives away (2-4 s: 0.5 m/s^2,
%   then 1 m/s along x); with T_LAND_VEH set in the base workspace (task 4), it goes back down onto the
%   platform from that time while still following. The vehicle sends its position, velocity and
%   acceleration (vehicle_mission).
%   Same as rl_controller_full_x.m except: the x policy is policy_x_track.mat, its observation has a 9th
%   component (the reference's acceleration) and uses the velocity relative to the vehicle, and the
%   missions. The same computations as fly(..., tracking=True) in python/full_mission_follow.py:
%     1. altitude policy (policy.mat):     obs_z   -> u_T     (drone_env.Observer)
%     2. yaw policy      (policy_yaw.mat): obs_yaw -> u_yaw   (yaw_env.YawObserver, holds psi = 0)
%     3. x policy        (policy_x_track.mat): obs_x -> u_x   (x_env.XObserver(tracking=True))
%     4. x policy on y   (policy_x_track.mat): obs_y -> u_y     (the axes are symmetric: motors 2/4, the
%                                                             "pitch" of the y axis is -phi, its rate -wx)
%     5. mixer: u_yaw clipped to +-(1 - |u_T|), u_x and u_y each to +-(1 - |u_T| - |u_yaw|), then
%               U = u_T + u_yaw [1 -1 1 -1] + u_x [-1 0 1 0] + u_y [0 -1 0 1]
%   Inputs: measurements at t_k (z, vz, psi, wz, x, vx, wy, y, vy, wx), the accelerations of the previous
%   step (az_prev, w_dot_z_prev, w_dot_y_prev, ax_prev, w_dot_x_prev, ay_prev) and the commands of the
%   previous step. The error integrals and the tilt estimates are persistent variables:
%   "clear rl_controller_follow" resets them (start of each flight). Called ONCE per step (Ts = 0.05 s).
persistent PZ PY PX
if isempty(PZ)
    here = fileparts(mfilename('fullpath'));
    PZ = load_policy(fullfile(here, 'policy.mat'));
    PY = load_policy(fullfile(here, 'policy_yaw.mat'));
    PX = load_policy(fullfile(here, 'policy_x_track.mat'));
end
S_YAW = [1; -1; 1; -1];
S_X = [-1; 0; 1; 0];
S_Y = [0; -1; 0; 1];

z_ref = mission_z(t);
psi_ref = 0;
[x_ref, vx_ref, ax_ref] = vehicle_mission(t);
y_ref = 0;  vy_ref = 0;  ay_ref = 0;

obs_z = observation_z(z_ref, z, vz, az_prev, u_T_prev);
u_T = run_policy(PZ, obs_z);

obs_yaw = observation_yaw(psi_ref, psi, wz, w_dot_z_prev, u_yaw_prev, u_T);
u_yaw = run_policy(PY, obs_yaw);

obs_x = observation_x(1, t, x_ref, x, vx - vx_ref, wy, w_dot_y_prev, ax_prev, az_prev, u_x_prev, u_T, ax_ref);
u_x = run_policy(PX, obs_x);
obs_y = observation_x(2, t, y_ref, y, vy - vy_ref, -wx, -w_dot_x_prev, ay_prev, az_prev, u_y_prev, u_T, ay_ref);
u_y = run_policy(PX, obs_y);

margin = 1 - abs(u_T);
u_yaw = min(max(u_yaw, -margin), margin);
margin = margin - abs(u_yaw);
u_x = min(max(u_x, -margin), margin);
u_y = min(max(u_y, -margin), margin);
U = u_T + u_yaw * S_YAW + u_x * S_X + u_y * S_Y;
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
% yaw_env.YawObserver: obs = [e; wz/5; w_dot_z_prev/20; u_yaw_prev/YAW_MAX; I/I_MAX; u_T]
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

function obs = observation_x(axis, t, p_ref, p, v_rel, q, q_dot_prev, a_prev, az_prev, u_prev, u_T, a_ref)
% x_env.XObserver(tracking=True) for one horizontal axis (1: x, 2: y), each with its own persistent state:
%   obs = [e; v_rel/2; theta/TH_MAX; q; q_dot_prev/5; u_prev/X_MAX; I/I_MAX; u_T; a_ref]
% q is the pitch rate of the axis (wy for x, -wx for y); theta from x_env.TiltEstimator: the gyro
% integral, corrected towards the accelerometer tilt atan2(a_prev, az_prev + g) when there is thrust.
persistent I theta q_last
if isempty(I), I = [0 0]; theta = [0 0]; q_last = [0 0]; end
Ts = 0.05;  I_MAX = 0.5;  TH_MAX = 0.3;  X_MAX = 0.02;  G = 9.81;
ALPHA = exp(-Ts / 1.0);  T_MIN = 2.0;
e = p_ref - p;
I(axis) = min(max(I(axis) + Ts * e, -I_MAX), I_MAX);
if t > 0
    T_prev = az_prev + G;
    if T_prev > T_MIN
        theta(axis) = ALPHA * theta(axis) + (1 - ALPHA) * atan2(a_prev, T_prev);
    end
    theta(axis) = theta(axis) + Ts * (q_last(axis) + q) / 2;
end
q_last(axis) = q;
obs = [min(max(e, -2), 2);
       min(max(v_rel / 2, -5), 5);
       min(max(theta(axis) / TH_MAX, -5), 5);
       min(max(q, -5), 5);
       min(max(q_dot_prev / 5, -5), 5);
       u_prev / X_MAX;
       I(axis) / I_MAX;
       u_T;
       min(max(a_ref, -5), 5)];
end

function z_ref = mission_z(t)
% 0 m on the platform until 1 s, then 0.5 m above it, back to the platform (z = 0) from T_LAND_VEH: task 3
% when T_LAND_VEH = inf, task 4 when it is finite (full_mission_follow.make_mission). T_LAND_VEH is read
% once per flight from the base workspace (default inf), so "clear rl_controller_follow" picks a new value.
persistent T_LAND
if isempty(T_LAND)
    T_LAND = inf;
    if evalin('base', 'exist(''T_LAND_VEH'', ''var'')')
        T_LAND = evalin('base', 'T_LAND_VEH');
    end
end
Ts = 0.05;
z_ref = 0.5 * (t >= 1.0 - Ts / 2 && t < T_LAND - Ts / 2);
end

function [x, v, a] = vehicle_mission(t)
% Platform of the vehicle: at rest until 2 s, 0.5 m/s^2 up to 1 m/s, then 1 m/s, as x_env.vehicle_mission.
Ts = 0.05;  T0 = 2.0;  A = 0.5;  V = 1.0;
t1 = T0 + V / A;
if t < T0 - Ts / 2
    x = 0; v = 0; a = 0;
elseif t < t1 - Ts / 2
    x = 0.5 * A * (t - T0)^2; v = A * (t - T0); a = A;
else
    x = 0.5 * A * (t1 - T0)^2 + V * (t - t1); v = V; a = 0;
end
end
