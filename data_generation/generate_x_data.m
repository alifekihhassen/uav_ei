%% GENERATE_X_DATA  Identification data for the motion along x (advanced task 2: land elsewhere).
%
%  Same bench as generate_yaw_data.m: UAV_model.slx (protected quadcopter_model.p) flown in OPEN LOOP,
%  the 4 motor commands built by a mixer
%
%      u_i = u_T + S_X(i) * u_x        S_X = [-1 0 1 0] or [0 -1 0 1] (found by the probe)
%
%      u_T : collective thrust (what the altitude policy controls)
%      u_x : differential command on one pair of opposite motors; it tilts the drone (theta), and the
%            tilted thrust pushes it along x
%
%  Chain seen in the black-box probe (no noise): u_x -> w_dot_y -> wy -> theta -> ax -> vx -> x.
%  There is NO restoring torque: after a doublet +v/-v the drone keeps its tilt (theta ~ 0.2 rad
%  after a doublet of 0.005 x 0.5 s), so a doublet is not enough to bring it back level.
%  The excitation therefore uses QUADRUPLETS  +v, -v, [hold], -v, +v  (each lasting L samples, the hold
%  H samples): their 0th and 1st moments are zero, so wy AND theta come back to 0 at the end (exactly
%  for a linear or odd-in-v dynamics), while the hold gives stretches at constant tilt (ax vs theta).
%  theta grows like v * L^2, so v is bounded by THETA_MAX / (K_THETA * L^2).
%
%  Note (as for the yaw): the theta OUTPUT is unusable with noise (huge, seed-dependent variance).
%  theta is recovered by integrating wy, which is clean.
%
%  PART 1 (probe) : which pair of motors moves the drone along x without moving it along y, and the sign.
%  PART 2 (data)  : N_EXP experiments of T_EXP seconds, one experiment per row, Ts = 0.05 s.
%
%  Output: data/x/*.csv and data/x/x_dataset.mat (all 18 outputs of the drone).
%  Start with N_EXP = 5 to check that everything runs, then raise it.

clear; close all; clc;
here = fileparts(mfilename('fullpath'));
root = fileparts(here);
addpath(fullfile(root, 'Matlab files'));
out_dir = fullfile(root, 'data', 'x');
if ~isfolder(out_dir), mkdir(out_dir); end

%% Settings
N_EXP     = 400;          % number of experiments (5 to test)
T_EXP     = 30;           % duration of one experiment [s] (short: model errors make theta drift)
G_NOISE   = 0.01;         % sensor noise gain, the value used for the course's dataset
N_REST    = 40;           % first samples at rest (u = 0): rest check az = -g, w_dot_y = 0
UT_RANGE  = [0.2 0.7];    % collective thrust levels (hover is about 0.39)
PAUSE     = [10 30];      % u_x = 0 before / between the two quadruplets of a block [samples]
QUAD_L    = [3 12];       % duration of each pulse of a quadruplet [samples] (motor tau ~0.2 s = 4)
QUAD_H    = [0 30];       % hold at constant tilt in the middle of a quadruplet [samples]
X_MAX     = 0.05;         % max |u_x|
THETA_MAX = 0.3;          % target peak tilt of one quadruplet [rad]
K_THETA   = 0.43;         % theta_peak ~ K_THETA * v * L^2 (probe: 0.216 rad for v = 0.005, L = 10)
P_UT_IN   = 0.15;         % probability that u_T also steps inside a quadruplet (coupling data)
SEED      = 2;            % excitation seed (the sensor-noise seeds are drawn at random)

random_init;                       % defines Ts and the seed_* variables used by UAV_model
Ts = 0.05;
n_t = round(T_EXP / Ts);
NAMES = {'x','y','z','vx','vy','vz','phi','theta','psi','wx','wy','wz', ...
         'ax','ay','az','w_dot_x','w_dot_y','w_dot_z'};
col = @(name) find(strcmp(NAMES, name));

mdl = 'x_data_model';
build_model(mdl);

%% PART 1 - probe: which pair of opposite motors moves the drone along x only?
% u_T = 0.39 (about hover), doublet u_x = +0.005 (1-1.5 s) / -0.005 (1.5-2 s), no noise: the drone
% ends up tilted and accelerating. A pure-x pattern moves x and leaves y (and psi) at 0.
PATTERNS = {[1 0 -1 0], [0 1 0 -1]};
LABELS   = {'1 vs 3', '2 vs 4'};
n_p = round(4 / Ts);
uT_p = 0.39 * ones(n_p, 1);
ux_p = zeros(n_p, 1);
ux_p(round(1/Ts)+1 : round(1.5/Ts)) = 0.005;
ux_p(round(1.5/Ts)+1 : round(2/Ts)) = -0.005;
fprintf('PART 1 - probe (u_T = 0.39, doublet u_x = +-0.005 from 1 s to 2 s, no noise)\n');
fprintf('%-8s %10s %10s %10s %10s %10s\n', 'pattern', 'x(4s)', 'y(4s)', 'theta', 'phi', 'max|psi|');
score = zeros(1, numel(PATTERNS));  x_end = zeros(1, numel(PATTERNS));
for p = 1:numel(PATTERNS)
    Y = run_one(mdl, mixer(uT_p, ux_p, PATTERNS{p}), Ts, 0);
    fprintf('%-8s %10.4f %10.4f %10.4f %10.4f %10.2g\n', LABELS{p}, Y(end, col('x')), ...
            Y(end, col('y')), Y(end, col('theta')), Y(end, col('phi')), max(abs(Y(:, col('psi')))));
    score(p) = abs(Y(end, col('x'))) / (abs(Y(end, col('y'))) + 1e-6);
    x_end(p) = Y(end, col('x'));
end
[~, best] = max(score);
S_X = PATTERNS{best} * sign(x_end(best));     % convention: u_x > 0 pushes the drone towards x > 0
fprintf('-> x pattern: %s, S_X = [%s]  (u_x > 0 moves the drone towards x > 0)\n', ...
        LABELS{best}, num2str(S_X));
Yb = run_one(mdl, mixer(uT_p, ux_p, S_X), Ts, 0);
th_probe = Yb(end, col('theta'));
if abs(Yb(end, col('y'))) > 0.05 * abs(Yb(end, col('x')))
    warning('The best pattern also moves the drone along y: look at the probe before generating data.');
end

%% PART 2 - data generation
rng(SEED);
UT = zeros(N_EXP, n_t);  UX = zeros(N_EXP, n_t);
YALL = zeros(N_EXP, n_t, numel(NAMES), 'single');
fprintf('\nPART 2 - %d experiments of %g s (g_noise = %g)\n', N_EXP, T_EXP, G_NOISE);
t0 = tic;
for e = 1:N_EXP
    [uT, ux] = excitation(n_t, N_REST, UT_RANGE, PAUSE, QUAD_L, QUAD_H, X_MAX, THETA_MAX, K_THETA, P_UT_IN);
    Y = run_one(mdl, mixer(uT, ux, S_X), Ts, G_NOISE);
    UT(e, :) = uT';  UX(e, :) = ux';  YALL(e, :, :) = single(Y);
    fprintf('  experiment %3d / %d  (%.0f s)\n', e, N_EXP, toc(t0));
end

%% Save: same layout as the course's CSVs (one experiment per row)
writematrix(UT, fullfile(out_dir, 'x_in_uT.csv'));
writematrix(UX, fullfile(out_dir, 'x_in_ux.csv'));
for name = {'w_dot_y', 'wy', 'ax', 'vx', 'x', 'az', 'y', 'psi'}
    writematrix(double(YALL(:, :, col(name{1}))), fullfile(out_dir, ['x_out_' name{1} '.csv']));
end
save(fullfile(out_dir, 'x_dataset.mat'), 'UT', 'UX', 'YALL', 'NAMES', 'S_X', 'Ts', 'G_NOISE', ...
     'th_probe', '-v7');
fprintf('Saved in %s\n', out_dir);

%% Quick look: tilt recovered by integrating wy (the theta output is noise)
theta_hat = cumsum(double(YALL(:, :, col('wy'))), 2) * Ts;
fprintf('max |theta_hat| over all experiments: %.2f rad   max |y|: %.3f m   max |psi|: %.3f rad\n', ...
        max(abs(theta_hat), [], 'all'), max(abs(YALL(:,:,col('y'))), [], 'all'), ...
        max(abs(YALL(:,:,col('psi'))), [], 'all'));
t = (0:n_t-1)' * Ts;
figure('Name', 'x data, experiment 1');
subplot(4,1,1); stairs(t, [UT(1,:)' UX(1,:)']); grid on; ylabel('command'); legend('u_T', 'u_x');
subplot(4,1,2); plot(t, YALL(1,:,col('w_dot_y'))); grid on; ylabel('w\_dot\_y');
subplot(4,1,3); plot(t, theta_hat(1,:)); grid on; ylabel('\theta (\int wy)');
subplot(4,1,4); plot(t, YALL(1,:,col('ax'))); grid on; ylabel('ax'); xlabel('t [s]');
close_system(mdl, 0);


%% ------------------------------------------------------------------------------------------------
function build_model(mdl)
% Open-loop test bench (same as generate_yaw_data.m):
%   From Workspace (4 motor commands, held during Ts) -> UAV_model -> Mux (18 outputs) -> To Workspace
if bdIsLoaded(mdl), close_system(mdl, 0); end
new_system(mdl);
add_block('simulink/Sources/From Workspace', [mdl '/U'], 'VariableName', 'U_in', ...
          'SampleTime', 'Ts', 'Interpolate', 'off', 'OutputAfterFinalValue', 'Holding final value');
add_block('simulink/Signal Routing/Demux', [mdl '/Demux'], 'Outputs', '4');
try
    add_block('simulink/Ports & Subsystems/Subsystem Reference', [mdl '/UAV']);
catch   % older library layout: copy the block of the course's example model
    load_system('EINT_system_simulation');
    add_block('EINT_system_simulation/Subsystem Reference', [mdl '/UAV']);
end
set_param([mdl '/UAV'], 'ReferencedSubsystem', 'UAV_model');
add_block('simulink/Signal Routing/Mux', [mdl '/Mux'], 'Inputs', '18');
add_block('simulink/Sinks/To Workspace', [mdl '/Y'], 'VariableName', 'Y_out', ...
          'SaveFormat', 'Array', 'SampleTime', 'Ts');
add_block('simulink/Sinks/Terminator', [mdl '/Term']);
add_line(mdl, 'U/1', 'Demux/1');
for i = 1:4,  add_line(mdl, sprintf('Demux/%d', i), sprintf('UAV/%d', i)); end
for i = 1:18, add_line(mdl, sprintf('UAV/%d', i), sprintf('Mux/%d', i)); end
add_line(mdl, 'UAV/19', 'Term/1');          % all_outputs: not used (order not documented)
add_line(mdl, 'Mux/1', 'Y/1');
set_param(mdl, 'SolverType', 'Fixed-step', 'Solver', 'FixedStepAuto', 'FixedStep', 'Ts');
try, Simulink.BlockDiagram.arrangeSystem(mdl); catch, end
end

function U = mixer(uT, ux, S)
% The 4 motor commands. Columns: u1 u2 u3 u4.
U = uT(:) + ux(:) * S;
assert(all(abs(U(:)) <= 1 + 1e-12), 'a motor command leaves [-1, 1]');
end

function Y = run_one(mdl, U, Ts, g_noise)
% One open-loop flight. U: n x 4 motor commands (U(k,:) held during [k-1, k) * Ts).
% Y: n x 18 outputs sampled at the same instants (row k = time (k-1) * Ts).
n = size(U, 1);
in = Simulink.SimulationInput(mdl);
in = in.setVariable('U_in', [(0:n-1)' * Ts, U]);
in = in.setVariable('Ts', Ts);
in = in.setVariable('g_noise', g_noise);
for s = {'x','y','z','vx','vy','vz','ax','ay','az','phi','psi','theta', ...
         'wx','wy','wz','w_dot_x','w_dot_y','w_dot_z'}
    in = in.setVariable(['seed_' s{1}], randi(10000));     % new sensor noise at each flight
end
in = in.setModelParameter('StopTime', num2str((n - 1) * Ts));
out = sim(in);
if ~isempty(out.ErrorMessage), error(out.ErrorMessage); end
Y = out.Y_out;
assert(size(Y, 1) == n, 'expected %d samples, got %d', n, size(Y, 1));
end

function [uT, ux] = excitation(n_t, n_rest, ut_range, pause_l, quad_l, quad_h, x_max, theta_max, ...
                               k_theta, p_ut_in)
% Blocks, u_T constant inside each one (except with probability p_ut_in, see below):
%   settle  : u_x = 0, u_T steps to a new level at its start (the motors settle before the tilt)
%   quad(v) : u_x = +v (L), -v (L), 0 (H, tilted hold), -v (L), +v (L)  -> wy and theta back to 0
%   gap     : u_x = 0 (the motor lag of the last pulse dies out: ~3 tau = 12 samples)
%   quad(-v): the mirror quadruplet -> the velocity vx gained by the first one is given back
%   tail    : u_x = 0, 12 samples, so that the next u_T step does not hit a pulse still in the motors
% A u_T step during a pulse (or its tail) changes the gain half-way, theta and vx no longer come back
% to 0: the pairing keeps the drone near level and slow, so the data stays in the flight envelope.
% With probability p_ut_in, u_T steps at a random sample of the block (data on the thrust / tilt
% coupling, which the closed loop will see; costs some drift).
% |v| <= min(x_max, theta_max / (k_theta L^2)), and clipped to the margin 1 - |u_T|.
N_TAIL = 12;
uT = zeros(n_t, 1);  ux = zeros(n_t, 1);
k = n_rest + 1;
while k <= n_t
    lvl = ut_range(1) + diff(ut_range) * rand;
    L = randi(quad_l);  H = randi([quad_h(1) quad_h(2)]);
    v = min(x_max, theta_max / (k_theta * L^2)) * (2 * rand - 1);
    quad = [ones(L, 1); -ones(L, 1); zeros(H, 1); -ones(L, 1); ones(L, 1)];
    seg = [zeros(randi(pause_l), 1); v * quad; zeros(randi([N_TAIL pause_l(2)]), 1); -v * quad; ...
           zeros(N_TAIL, 1)];
    idx = k : min(k + numel(seg) - 1, n_t);
    ux(idx) = seg(1:numel(idx));
    uT(idx) = lvl;
    if rand < p_ut_in
        j = idx(randi(numel(idx)));
        uT(j:idx(end)) = ut_range(1) + diff(ut_range) * rand;
    end
    k = k + numel(seg);
end
uT(1:n_rest) = 0;
margin = 1 - abs(uT);
ux = max(min(ux, margin), -margin);
end
