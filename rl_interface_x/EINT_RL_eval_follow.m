%% Advanced tasks 3 and 4: follow the ground vehicle, and land on it. Altitude + yaw + tracking x policies
%  fly the "real" drone (UAV_model.slx): it takes off from the platform at 1 s and stays 0.5 m above it
%  while the vehicle drives away (2-4 s: 0.5 m/s^2 along x, then 1 m/s), y and heading held at 0. 13 s.
%  T_LAND_VEH (base workspace, default inf = task 3): with a finite value (task 4, see
%  EINT_RL_eval_land_vehicle.m) the drone goes back down onto the moving platform from that time.
%  The simulator has no contact: after touchdown the drone keeps "hovering" at z = 0 above the platform.
%
%  1. checks rl_controller_follow.m against Python: replays traj_follow.mat (task 3) or
%     traj_land_vehicle.mat (task 4), the mission flown on the identified models by
%     python/full_mission_follow.py [land], step by step
%  2. builds EINT_RL_closed_loop_follow.slx (build_closed_loop_x.m with the follow controller) and flies
%     the mission on the simulator for each g_noise of G_NOISES, then prints the metrics and plots
%  Needs policy.mat, policy_yaw.mat, policy_x_track.mat and traj_follow.mat / traj_land_vehicle.mat:
%     cd python; python full_mission_follow.py; python full_mission_follow.py land

clearvars -except T_LAND_VEH; close all; clc;
if ~exist('T_LAND_VEH', 'var'), T_LAND_VEH = inf; end
LAND = isfinite(T_LAND_VEH);
here = fileparts(mfilename('fullpath'));
cd(here);
addpath(here, fullfile(here, '..', 'Matlab files'));

if LAND, traj_file = 'traj_land_vehicle.mat'; else, traj_file = 'traj_follow.mat'; end
check_controller(here, traj_file);

G_NOISES = [0 0.1];            % 0 first, 0.1 = the course's evaluation level, higher to test robustness
Ts = 0.05;
T_end = 13;
T_VEH = 2;  T_ACC_END = 4;     % the vehicle accelerates from 2 s to 4 s (0.5 m/s^2 up to 1 m/s)

mdl = 'EINT_RL_closed_loop_follow';
build_closed_loop_x(mdl, 'rl_controller_follow');
fig = figure('Name', 'Follow the vehicle on the simulator', 'Position', [100 100 950 1000]);
ax = gobjects(5, 1);
for j = 1:5, ax(j) = subplot(5, 1, j); hold on; grid on; end
for gn = G_NOISES
    random_init;               % new noise seeds; it also sets Ts and g_noise, so both are set again here
    Ts = 0.05;
    g_noise = gn;
    clear rl_controller_follow
    out = sim(mdl);
    Lg = out.rl_log;           % t z vz az psi wz x y vx wy ax u_T u_yaw u_x z_ref psi_ref x_ref u_y vy wx y_ref U(4)
    t = Lg(:, 1); z = Lg(:, 2); psi = Lg(:, 5); x = Lg(:, 7); y = Lg(:, 8); wy = Lg(:, 10);
    u_x = Lg(:, 14); z_ref = Lg(:, 15); x_ref = Lg(:, 17);
    theta = cumtrapz(t, wy);   % tilt for the plot only (the controller has its own estimate)
    e = x_ref - x;             % how far the drone is behind the platform

    % metrics (same definitions as python/x_env.track_metrics)
    acc = t >= T_VEH & t < T_ACC_END + 1;  cru = t >= T_ACC_END + 2;  up = t >= 2;
    fprintf('\ng_noise = %g\n', g_noise);
    fprintf('x behind the vehicle: max %.1f cm while it accelerates, %+.1f cm mean / %.1f cm max at constant speed\n', ...
            100 * max(abs(e(acc))), 100 * mean(e(cru)), 100 * max(abs(e(cru))));
    fprintf('y: max |y| %.1f cm | altitude above the platform: RMSE %.1f cm | max tilt %.1f deg, max |psi| %.2f deg\n', ...
            100 * max(abs(y)), 100 * sqrt(mean((z_ref(up) - z(up)).^2)), rad2deg(max(abs(theta))), rad2deg(max(abs(psi))));
    if LAND                    % touchdown: first sample within 1 cm of the platform after T_LAND_VEH
        after = t >= T_LAND_VEH - Ts / 2;
        k = find(after & z <= 0.01, 1);
        if isempty(k)
            fprintf('task 4: NO touchdown\n');
        else
            vx = Lg(:, 9);  [~, v_veh] = vehicle_speed(t(k));
            fprintf(['task 4: touchdown at t = %.2f s, %+.1f cm along x / %+.1f cm along y from the centre of the platform,\n' ...
                     '        relative speed %+.2f m/s, descent %+.2f m/s | below the platform %.1f cm, then within %.1f cm of the centre\n'], ...
                    t(k), -100 * e(k), 100 * y(k), vx(k) - v_veh, Lg(k, 3), 100 * max(0, -min(z(after))), ...
                    100 * max(hypot(e(k:end), y(k:end))));
        end
    end

    lab = sprintf('g\\_noise = %g', g_noise);
    plot(ax(1), t, x, 'LineWidth', 1.2, 'DisplayName', ['drone, ' lab]);
    plot(ax(2), t, 100 * e, 'LineWidth', 1.2, 'DisplayName', lab);
    plot(ax(3), t, 100 * y, 'LineWidth', 1.2, 'DisplayName', lab);
    plot(ax(4), t, z, 'LineWidth', 1.2, 'DisplayName', lab);
    stairs(ax(5), t, u_x, 'DisplayName', lab);
end
plot(ax(1), t, x_ref, '--k', 'DisplayName', 'vehicle (platform)');
stairs(ax(4), t, z_ref, '--k', 'DisplayName', 'reference');
xline(ax(2), [T_VEH T_ACC_END], ':', 'HandleVisibility', 'off');
ylabel(ax(1), 'x [m]'); ylabel(ax(2), 'vehicle - drone [cm]'); ylabel(ax(3), 'y [cm]');
ylabel(ax(4), 'z [m] (platform = 0)'); ylabel(ax(5), 'u_x'); xlabel(ax(5), 't [s]');
legend(ax(1), 'Location', 'best'); legend(ax(2), 'Location', 'best');
if LAND
    title(ax(1), sprintf('UAV\\_model.slx: land on the moving vehicle (descent from t = %g s, vehicle at 1 m/s)', T_LAND_VEH));
    saveas(fig, fullfile(here, 'eval_land_vehicle.png'));
else
    title(ax(1), 'UAV\_model.slx: follow the vehicle (0.5 m/s^2 up to 1 m/s from t = 2 s)');
    saveas(fig, fullfile(here, 'eval_follow.png'));
end


%% Local functions

function [x, v] = vehicle_speed(t)
% Position and speed of the platform, as vehicle_mission in rl_controller_follow.m (for the metrics).
T0 = 2.0;  A = 0.5;  V = 1.0;  t1 = T0 + V / A;
if t < T0, x = 0; v = 0;
elseif t < t1, x = 0.5 * A * (t - T0)^2; v = A * (t - T0);
else, x = 0.5 * A * (t1 - T0)^2 + V * (t - t1); v = V;
end
end

function check_controller(here, traj_file)
% Replays the measurements of traj_file through rl_controller_follow.m, step by step as in Simulink.
need = {'policy.mat', 'policy_yaw.mat', 'policy_x_track.mat', traj_file};
for i = 1:numel(need)
    if ~isfile(fullfile(here, need{i}))
        error('%s is missing: run python/full_mission_follow.py [land] first.', need{i});
    end
end
T = load(fullfile(here, traj_file));
clear rl_controller_follow
for k = 1:numel(T.t)
    [U, z_ref, psi_ref, x_ref, y_ref, u_T, u_yaw, u_x, u_y, obs_z, obs_yaw, obs_x, obs_y] = rl_controller_follow( ...
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
clear rl_controller_follow
fprintf('Check OK: rl_controller_follow.m flies the policies as in Python (%d steps of %s).\n', numel(T.t), traj_file);
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
