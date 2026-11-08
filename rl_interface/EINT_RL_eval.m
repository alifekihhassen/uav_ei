%% Track B: your policy (policy.mat) flies the "real" drone (UAV_model.slx)
%  1. checks rl_controller.m against Python: replays traj_model.mat (the Figure 1 mission flown on
%     your model) step by step; the mission, your observation() and the commands must be the same
%  2. flies the mission of Figure 1 on EINT_RL_closed_loop.slx (open it to see the loop) and plots
%  Needs policy.mat and traj_model.mat, written by export_for_matlab in track_b.ipynb.
%  Only g_noise is yours to change.

clear; close all; clc;
here = fileparts(mfilename('fullpath'));
cd(here);
addpath(here, fullfile(here, '..', 'Matlab files'));

check_controller(here);

random_init;                   % noise seeds
Ts = 0.05;
T_end = 10;
g_noise = 0.1;                 % 0 to start, 0.1 for tuning, higher to test robustness

clear rl_controller
out = sim('EINT_RL_closed_loop');
L = out.rl_log;                % t, z, vz, az, u, z_ref
t = L(:, 1); z = L(:, 2); u = L(:, 5); z_ref = L(:, 6);

figure('Name', 'Track B: policy on the simulator');
subplot(2, 1, 1); hold on; grid on;
plot(t, z_ref, '--', t, z, 'LineWidth', 1.2); yline(-0.25, ':');
ylabel('z [m]'); legend('reference', 'z', 'ground');
title(sprintf('g\\_noise = %g: RMSE %.2f m, min z %.2f m', g_noise, sqrt(mean((z_ref - z).^2)), min(z)));
subplot(2, 1, 2); stairs(t, u); grid on; ylabel('u'); xlabel('t [s]');


%% Local function

function check_controller(here)
% Replays the measurements of traj_model.mat (the Figure 1 mission flown on your model, written by
% export_for_matlab in track_b.ipynb) through rl_controller.m, step by step as in Simulink: the
% mission, the observation and the command must be those of Python.
if ~isfile(fullfile(here, 'policy.mat')) || ~isfile(fullfile(here, 'traj_model.mat'))
    error('policy.mat or traj_model.mat is missing: run export_for_matlab in track_b.ipynb first.');
end
T = load(fullfile(here, 'traj_model.mat'));
clear rl_controller
for k = 1:numel(T.t)
    [u, z_ref, obs] = rl_controller(T.t(k), T.z(k), T.vz(k), T.az_prev(k), T.u_prev(k));
    where = sprintf('step %d (t = %.2f s)', k - 1, T.t(k));
    if abs(z_ref - T.z_ref(k)) > 1e-9
        error('%s: z_ref = %g in rl_controller.m, %g in Python: mission() differs.', where, z_ref, T.z_ref(k));
    end
    if numel(obs) ~= size(T.obs, 2)
        error('observation() returns %d values in rl_controller.m, %d in Python.', numel(obs), size(T.obs, 2));
    end
    [e, i] = max(abs(obs(:)' - T.obs(k, :)) ./ max(1, abs(T.obs(k, :))));
    if e > 1e-5
        error('%s: component %d of the observation is %g in rl_controller.m, %g in Python.', ...
              where, i, obs(i), T.obs(k, i));
    end
    if abs(u - T.u(k)) > 1e-4
        error('%s: u = %g in MATLAB, %g in Python, with the same observation: export again.', where, u, T.u(k));
    end
end
clear rl_controller
fprintf('Check OK: rl_controller.m flies the policy as in Python (%d steps of traj_model.mat).\n', numel(T.t));
end
