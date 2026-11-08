%% Full mission: altitude policy + yaw policy fly the "real" drone (UAV_model.slx)
%  0-1 s on the platform, 1-10 s at 0.5 m with four quarter turns (90, 180, 270, 360 deg at 2, 4, 6, 8 s),
%  10-13 s landing on the ground (-0.25 m).
%
%  1. checks rl_controller_full.m against Python: replays traj_full.mat (the mission flown on the
%     identified models by python/full_mission.py) step by step; missions, observations and commands
%     must be the same
%  2. builds EINT_RL_closed_loop_yaw.slx in this MATLAB version (build_closed_loop_yaw.m) and flies the
%     mission on the simulator, then plots altitude, heading, commands and the horizontal drift
%  Needs policy.mat, policy_yaw.mat and traj_full.mat, written by: cd python; python full_mission.py
%  Only g_noise is yours to change.

clear; close all; clc;
here = fileparts(mfilename('fullpath'));
cd(here);
addpath(here, fullfile(here, '..', 'Matlab files'));

check_controller(here);

random_init;                   % noise seeds
Ts = 0.05;
T_end = 13;
g_noise = 0.1;                 % 0 to start, 0.1 for tuning, higher to test robustness

mdl = 'EINT_RL_closed_loop_yaw';
build_closed_loop_yaw(mdl);
clear rl_controller_full
out = sim(mdl);
L = out.rl_log;                % t z vz az psi wz w_dot_z phi x y u_T u_yaw z_ref psi_ref u1 u2 u3 u4
t = L(:, 1); z = L(:, 2); psi = L(:, 5); wz = L(:, 6); phi = L(:, 8); x = L(:, 9); y = L(:, 10);
u_T = L(:, 11); u_yaw = L(:, 12); z_ref = L(:, 13); psi_ref = L(:, 14); U = L(:, 15:18);

%% Metrics
land = t >= 10;
e_psi = rad2deg(psi_ref - psi);
settle = nan(1, 4);  over = nan(1, 4);  T_YAW = [2 4 6 8 10];
for j = 1:4
    w = (t >= T_YAW(j) - Ts/2) & (t < T_YAW(j+1) - Ts/2);
    e = abs(e_psi(w));  k = find(e > 5, 1, 'last');
    if isempty(k), settle(j) = 0; elseif k < numel(e), settle(j) = k * Ts; end
    over(j) = max(0, max(rad2deg(psi(w)) - 90 * j));
end
fprintf('\ng_noise = %g\n', g_noise);
fprintf('altitude: RMSE %.3f m, min z %.3f m, below ground while landing %.1f cm\n', ...
        sqrt(mean((z_ref - z).^2)), min(z), 100 * max(0, -0.25 - min(z(land))));
fprintf('yaw     : settling (5 deg) per quarter turn [%s] s, overshoot [%s] deg, final heading %.1f deg\n', ...
        num2str(settle, '%.2f '), num2str(over, '%.1f '), rad2deg(psi(end)));
fprintf('drift   : max |x| %.3f m, max |y| %.3f m, max |phi| %.4f rad\n', max(abs(x)), max(abs(y)), max(abs(phi)));

%% Plots
figure('Name', 'Full mission on the simulator', 'Position', [100 100 900 800]);
subplot(4, 1, 1); hold on; grid on;
stairs(t, z_ref, '--k'); plot(t, z, 'LineWidth', 1.2); yline(-0.25, ':');
ylabel('z [m]'); legend('reference', 'z', 'ground', 'Location', 'best');
title(sprintf('g\\_noise = %g: altitude RMSE %.2f m, min z %.2f m', g_noise, sqrt(mean((z_ref - z).^2)), min(z)));
subplot(4, 1, 2); hold on; grid on;
stairs(t, rad2deg(psi_ref), '--k'); plot(t, rad2deg(psi), 'LineWidth', 1.2);
ylabel('\psi [deg]'); legend('reference', '\psi', 'Location', 'best');
subplot(4, 1, 3); hold on; grid on;
stairs(t, u_T); stairs(t, 10 * u_yaw); ylabel('commands'); legend('u_T', '10 u_{yaw}', 'Location', 'best');
subplot(4, 1, 4); hold on; grid on;
stairs(t, U); ylabel('motors'); xlabel('t [s]'); legend('u_1', 'u_2', 'u_3', 'u_4', 'Location', 'best');


%% Local function

function check_controller(here)
% Replays the measurements of traj_full.mat (the full mission flown on the identified models by
% python/full_mission.py) through rl_controller_full.m, step by step as in Simulink.
if ~isfile(fullfile(here, 'policy.mat')) || ~isfile(fullfile(here, 'policy_yaw.mat')) || ~isfile(fullfile(here, 'traj_full.mat'))
    error('policy.mat, policy_yaw.mat or traj_full.mat is missing: run python/full_mission.py first.');
end
T = load(fullfile(here, 'traj_full.mat'));
clear rl_controller_full
for k = 1:numel(T.t)
    [U, z_ref, psi_ref, u_T, u_yaw, obs_z, obs_yaw] = rl_controller_full(T.t(k), T.z(k), T.vz(k), T.az_prev(k), ...
        T.u_T_prev(k), T.psi(k), T.wz(k), T.w_dot_z_prev(k), T.u_yaw_prev(k));
    where = sprintf('step %d (t = %.2f s)', k - 1, T.t(k));
    if abs(z_ref - T.z_ref(k)) > 1e-9 || abs(psi_ref - T.psi_ref(k)) > 1e-9
        error('%s: z_ref / psi_ref = %g / %g in MATLAB, %g / %g in Python: a mission differs.', ...
              where, z_ref, psi_ref, T.z_ref(k), T.psi_ref(k));
    end
    compare_obs(obs_z, T.obs_z(k, :), 'altitude', where);
    compare_obs(obs_yaw, T.obs_yaw(k, :), 'yaw', where);
    if abs(u_T - T.u_T(k)) > 1e-4 || abs(u_yaw - T.u_yaw(k)) > 1e-5
        error('%s: u_T / u_yaw = %g / %g in MATLAB, %g / %g in Python.', where, u_T, u_yaw, T.u_T(k), T.u_yaw(k));
    end
    if max(abs(U(:)' - T.U(k, :))) > 1e-4
        error('%s: the motor commands differ (mixer).', where);
    end
end
clear rl_controller_full
fprintf('Check OK: rl_controller_full.m flies both policies as in Python (%d steps of traj_full.mat).\n', numel(T.t));
end

function compare_obs(obs, ref, name, where)
if numel(obs) ~= numel(ref)
    error('%s observation: %d values in MATLAB, %d in Python.', name, numel(obs), numel(ref));
end
[e, i] = max(abs(obs(:)' - ref) ./ max(1, abs(ref)));
if e > 1e-5
    error('%s: component %d of the %s observation is %g in MATLAB, %g in Python.', where, i, name, obs(i), ref(i));
end
end
