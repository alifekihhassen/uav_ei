function [u, z_ref, obs] = rl_controller(t, z, vz, az_prev, u_prev)
%RL_CONTROLLER  The trained policy, called at each step by EINT_RL_closed_loop.slx.
%   Write mission() and observation() at the end of this file. policy.mat is written by
%   export_for_matlab (export_policy.py).
persistent P
if isempty(P)
    P = load(fullfile(fileparts(mfilename('fullpath')), 'policy.mat'));
    P.acts = strsplit(P.activations, ',');
end
z_ref = mission(t);
m = struct('t', t, 'z', z, 'vz', vz, 'az_prev', az_prev, 'z_ref', z_ref, 'u_prev', u_prev);
obs = double(observation(m));
x = obs(:);
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
    case 'scale',      x = lo + 0.5 * (x + 1) .* (hi - lo);         % TD3 (tanh in the network)
    otherwise,         x = min(max(x, lo), hi);                     % PPO
end
u = x(1);
end

function obs = observation(m)
% The observation: the same computation as Observer.__call__ in drone_env.py, in the same order.
%   obs = [e; z; vz/2; az_prev/10; u_prev; I/I_MAX; z_ref],  e = z_ref - z (clipped)
%   I = integral of e (anti-windup +/- I_MAX), kept in a persistent variable: it is reset by
%   "clear rl_controller" at the start of each flight. Called ONCE per step (Ts = 0.05 s).
persistent I
if isempty(I), I = 0; end
Ts = 0.05;
I_MAX = 0.5;
e = m.z_ref - m.z;
I = min(max(I + Ts * e, -I_MAX), I_MAX);
obs = [min(max(e, -1), 1);
       m.z;
       min(max(m.vz / 2, -5), 5);
       min(max(m.az_prev / 10, -5), 5);
       m.u_prev;
       I / I_MAX;
       m.z_ref];
end

function z_ref = mission(t)
% The reference altitude at time t [s], the same as mission() in drone_env.py (Figure 1):
%   0 m on the platform until 1 s, 0.5 m until 5 s, then -0.25 m on the ground.
% The thresholds are shifted by Ts/2 so that t = 1 s and t = 5 s fall on the same side as in Python.
Ts = 0.05;
if t < 1.0 - Ts / 2
    z_ref = 0;
elseif t < 5.0 - Ts / 2
    z_ref = 0.5;
else
    z_ref = -0.25;
end
end
