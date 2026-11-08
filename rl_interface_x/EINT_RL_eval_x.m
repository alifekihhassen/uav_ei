%% Advanced task 2: land at a different position. Altitude + yaw + x policies fly the "real" drone
%  (UAV_model.slx): 0-1 s on the platform, 1-7 s at 0.5 m with a move to x = 1 m from t = 2 s, heading
%  and y held at 0, 7-11 s landing on the ground (-0.25 m) at x = 1 m.
%
%  1. checks rl_controller_full_x.m against Python: replays traj_full_x.mat (the mission flown on the
%     identified models by python/full_mission_x.py) step by step; missions, observations and commands
%     must be the same
%  2. builds EINT_RL_closed_loop_x.slx in this MATLAB version (build_closed_loop_x.m) and flies the
%     mission on the simulator for each g_noise of G_NOISES, then prints the metrics and plots
%  Needs policy.mat, policy_yaw.mat, policy_x.mat and traj_full_x.mat, written by:
%     cd python; python full_mission_x.py

clear; close all; clc;
here = fileparts(mfilename('fullpath'));
cd(here);
addpath(here, fullfile(here, '..', 'Matlab files'));

check_controller(here);

G_NOISES = [0 0.1];            % 0 first, 0.1 = the course's evaluation level, higher to test robustness
Ts = 0.05;
T_end = 11;
T_MOVE = 2;  X_LAND = 1;  T_LAND = 7;

mdl = 'EINT_RL_closed_loop_x';
build_closed_loop_x(mdl);
fig = figure('Name', 'Land at x = 1 m on the simulator', 'Position', [100 100 950 1000]);
ax = gobjects(6, 1);
for j = 1:6, ax(j) = subplot(6, 1, j); hold on; grid on; end
for gn = G_NOISES
    random_init;               % new noise seeds; it also sets Ts and g_noise, so both are set again here
    Ts = 0.05;
    g_noise = gn;
    clear rl_controller_full_x
    out = sim(mdl);
    Lg = out.rl_log;           % t z vz az psi wz x y vx wy ax u_T u_yaw u_x z_ref psi_ref x_ref u_y vy wx y_ref U(4)
    t = Lg(:, 1); z = Lg(:, 2); psi = Lg(:, 5); x = Lg(:, 7); y = Lg(:, 8); wy = Lg(:, 10);
    u_T = Lg(:, 12); u_x = Lg(:, 14); z_ref = Lg(:, 15); x_ref = Lg(:, 17); u_y = Lg(:, 18);
    theta = cumtrapz(t, wy);   % tilt for the plot only (the controller has its own estimate)

    % metrics (same definitions as python/x_env.move_metrics and full_mission_x.summary)
    land = t >= T_LAND;
    w = t >= T_MOVE - Ts / 2;
    e = abs(X_LAND - x(w));  k = find(e > 0.05, 1, 'last');
    if isempty(k), settle = 0; elseif k < numel(e), settle = k * Ts; else, settle = nan; end
    touch = find(land & z < -0.24, 1);        % touchdown: first sample within 1 cm of the ground
    fprintf('\ng_noise = %g\n', g_noise);
    fprintf('altitude: RMSE %.3f m, below ground while landing %.1f cm\n', ...
            sqrt(mean((z_ref - z).^2)), 100 * max(0, -0.25 - min(z(land))));
    fprintf('x       : settling (5 cm) %.2f s, overshoot %.1f cm, at touchdown (t = %.2f s) x = %.3f m, y = %.1f cm\n', ...
            settle, 100 * max(0, max(x(w)) - X_LAND), t(touch), x(touch), 100 * y(touch));
    fprintf('          at the end x = %.3f m, y = %.1f cm, max |y| %.1f cm\n', x(end), 100 * y(end), 100 * max(abs(y)));
    fprintf('attitude: max tilt %.1f deg, max |psi| %.2f deg\n', rad2deg(max(abs(theta))), rad2deg(max(abs(psi))));

    lab = sprintf('g\\_noise = %g', g_noise);
    plot(ax(1), t, z, 'LineWidth', 1.2, 'DisplayName', lab);
    plot(ax(2), t, x, 'LineWidth', 1.2, 'DisplayName', lab);
    plot(ax(3), t, 100 * y, 'LineWidth', 1.2, 'DisplayName', lab);
    plot(ax(4), t, rad2deg(theta), 'DisplayName', lab);
    stairs(ax(5), t, u_T, 'DisplayName', lab);
    stairs(ax(6), t, u_x, 'DisplayName', [lab ', u_x']);
    stairs(ax(6), t, u_y, ':', 'DisplayName', [lab ', u_y']);
end
stairs(ax(1), t, z_ref, '--k', 'DisplayName', 'reference'); yline(ax(1), -0.25, ':', 'HandleVisibility', 'off');
stairs(ax(2), t, x_ref, '--k', 'DisplayName', 'reference');
ylabel(ax(1), 'z [m]'); ylabel(ax(2), 'x [m]'); ylabel(ax(3), 'y [cm]'); ylabel(ax(4), '\theta [deg] (\int wy)');
ylabel(ax(5), 'u_T'); ylabel(ax(6), 'u_x, u_y'); xlabel(ax(6), 't [s]');
legend(ax(1), 'Location', 'best'); legend(ax(2), 'Location', 'best'); legend(ax(6), 'Location', 'best');
title(ax(1), 'UAV\_model.slx: take off, move to x = 1 m, land (altitude + yaw + x policies, y held at 0)');
saveas(fig, fullfile(here, 'eval_x.png'));


%% Local functions

function check_controller(here)
% Replays the measurements of traj_full_x.mat through rl_controller_full_x.m, step by step as in Simulink.
need = {'policy.mat', 'policy_yaw.mat', 'policy_x.mat', 'traj_full_x.mat'};
for i = 1:numel(need)
    if ~isfile(fullfile(here, need{i}))
        error('%s is missing: run python/full_mission_x.py first.', need{i});
    end
end
T = load(fullfile(here, 'traj_full_x.mat'));
clear rl_controller_full_x
for k = 1:numel(T.t)
    [U, z_ref, psi_ref, x_ref, y_ref, u_T, u_yaw, u_x, u_y, obs_z, obs_yaw, obs_x, obs_y] = rl_controller_full_x( ...
        T.t(k), T.z(k), T.vz(k), T.az_prev(k), T.u_T_prev(k), T.psi(k), T.wz(k), T.w_dot_z_prev(k), T.u_yaw_prev(k), ...
        T.x(k), T.vx(k), T.wy(k), T.w_dot_y_prev(k), T.ax_prev(k), T.u_x_prev(k), ...
        T.y(k), T.vy(k), T.wx(k), T.w_dot_x_prev(k), T.ay_prev(k), T.u_y_prev(k));
    where = sprintf('step %d (t = %.2f s)', k - 1, T.t(k));
    refs = [z_ref psi_ref x_ref y_ref] - [T.z_ref(k) T.psi_ref(k) T.x_ref(k) T.y_ref(k)];
    if max(abs(refs)) > 1e-9
        error('%s: a mission differs between MATLAB and Python.', where);
    end
    compare_obs(obs_z, T.obs_z(k, :), 'altitude', where);
    compare_obs(obs_yaw, T.obs_yaw(k, :), 'yaw', where);
    compare_obs(obs_x, T.obs_x(k, :), 'x', where);
    compare_obs(obs_y, T.obs_y(k, :), 'y', where);
    if abs(u_T - T.u_T(k)) > 1e-4 || abs(u_yaw - T.u_yaw(k)) > 1e-5 || abs(u_x - T.u_x(k)) > 2e-6 || abs(u_y - T.u_y(k)) > 2e-6
        error('%s: u_T / u_yaw / u_x / u_y = %g / %g / %g / %g in MATLAB, %g / %g / %g / %g in Python.', where, ...
              u_T, u_yaw, u_x, u_y, T.u_T(k), T.u_yaw(k), T.u_x(k), T.u_y(k));
    end
    if max(abs(U(:)' - T.U(k, :))) > 1e-4
        error('%s: the motor commands differ (mixer).', where);
    end
end
clear rl_controller_full_x
fprintf('Check OK: rl_controller_full_x.m flies the policies as in Python (%d steps of traj_full_x.mat).\n', numel(T.t));
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
