%% GENERATE_YAW_DATA  Identification data for the yaw (rotation) of the drone.
%
%  The course's dataset only excites the collective thrust (the same u on the 4 motors), so it says
%  nothing about rotation. This script flies UAV_model.slx (protected quadcopter_model.p) in OPEN LOOP
%  with DIFFERENT commands on the 4 motors, built by a mixer:
%
%      u_i = u_T + S(i) * u_yaw        S = [+1 -1 +1 -1] or another pairing (found by the probe)
%
%      u_T   : collective thrust (what the altitude policy controls)
%      u_yaw : differential command; speeding one pair of motors up and the other down turns the drone
%
%  Every u_i must stay in [-1, 1]  =>  |u_yaw| <= 1 - |u_T|   (the "yaw margin" left by the thrust).
%  In practice the yaw is VERY sensitive (w_dot_z ~ 150 rad/s^2 per unit of u_yaw, no yaw drag: under a
%  constant u_yaw, wz grows without limit). A 90 deg turn in ~1 s needs |u_yaw| ~ 0.04, so the excitation
%  stays in |u_yaw| <= YAW_MAX = 0.1 (always far inside the margin), and uses DOUBLETS (+v then -v for
%  the same time) so that wz comes back near 0 after each pulse instead of drifting away.
%
%  PART 1 (probe)   : finds which pair of motors turns the drone without tilting it, and the sign.
%  PART 2 (data)    : N_EXP experiments of T_EXP seconds, same format as bdd_*_mat_05.csv
%                     (one experiment per row, one sample per column, Ts = 0.05 s).
%
%  Note: the theta output of UAV_model is pure sensor noise with a huge, seed-dependent variance (its
%  noise block uses seed_theta as covariance): do not read it as a real tilt; phi is the clean check.
%
%  Output: data/yaw/*.csv and data/yaw/yaw_dataset.mat (all 18 outputs of the drone).
%  Start with N_EXP = 5 to check that everything runs, then raise it.

clear; close all; clc;
here = fileparts(mfilename('fullpath'));
root = fileparts(here);
addpath(fullfile(root, 'Matlab files'));
out_dir = fullfile(root, 'data', 'yaw');
if ~isfolder(out_dir), mkdir(out_dir); end

%% Settings
N_EXP    = 1000;          % number of experiments (5 to test, then ~300)
T_EXP    = 100;        % duration of one experiment [s] (same as the course's dataset)
G_NOISE  = 0.01;       % sensor noise gain, the value used for the course's dataset
N_REST   = 40;         % first samples at rest (u = 0, as in the course's dataset: rest check az = -g)
UT_RANGE = [0.0 0.8];  % collective thrust levels (hover is about 0.39)
HOLD_T   = [20 40];    % hold of u_T     [samples]  (1 to 2 s)
YAW_MAX  = 0.1;        % max |u_yaw| (see the header: 0.04 is already a fast 90 deg turn)
HOLD_YAW = [5 30];     % duration of each half of a doublet [samples] (0.25 to 1.5 s; motor tau ~0.2 s)
P_ZERO   = 0.3;        % probability of a pause (u_yaw = 0) between two doublets
SEED     = 1;          % excitation seed (the sensor-noise seeds are drawn at random)

random_init;                       % defines Ts and the seed_* variables used by UAV_model
Ts = 0.05;
n_t = round(T_EXP / Ts);           % samples per experiment (2000, as in the course's dataset)
NAMES = {'x','y','z','vx','vy','vz','phi','theta','psi','wx','wy','wz', ...
         'ax','ay','az','w_dot_x','w_dot_y','w_dot_z'};
col = @(name) find(strcmp(NAMES, name));

mdl = 'yaw_data_model';
build_model(mdl);

%% PART 1 - probe: which pair of motors gives a pure rotation?
% u_T = 0.4 (about hover) on all motors, plus u_yaw = 0.2 with a candidate pattern from 1 s to 3 s,
% no noise. A pure yaw pattern changes wz and leaves phi / theta at 0; the others tilt the drone.
PATTERNS = {[1 -1 1 -1], [1 1 -1 -1], [1 -1 -1 1]};
LABELS   = {'(1,3) vs (2,4)', '(1,2) vs (3,4)', '(1,4) vs (2,3)'};
n_p = round(6 / Ts);
uT_p = 0.4 * ones(n_p, 1);
uy_p = zeros(n_p, 1);  uy_p(round(1/Ts)+1 : round(3/Ts)) = 0.2;
fprintf('PART 1 - probe (u_T = 0.4, u_yaw = 0.2 from 1 s to 3 s, no noise)\n');
fprintf('%-16s %12s %12s %12s\n', 'pattern', 'max|wz|', 'max|phi|', 'max|theta|');
score = zeros(1, numel(PATTERNS));  wz_end = zeros(1, numel(PATTERNS));
for p = 1:numel(PATTERNS)
    Y = run_one(mdl, mixer(uT_p, uy_p, PATTERNS{p}), Ts, 0);
    wz = Y(:, col('wz'));  tilt = max(abs(Y(:, col('phi')))) + max(abs(Y(:, col('theta'))));
    fprintf('%-16s %12.4f %12.4f %12.4f\n', LABELS{p}, max(abs(wz)), ...
            max(abs(Y(:, col('phi')))), max(abs(Y(:, col('theta')))));
    score(p) = max(abs(wz)) / (tilt + 1e-6);
    wz_end(p) = wz(round(3/Ts));
end
[~, best] = max(score);
S = PATTERNS{best} * sign(wz_end(best));      % convention: u_yaw > 0  =>  wz > 0
fprintf('-> yaw pattern: %s, S = [%s]  (u_yaw > 0 turns the drone with wz > 0)\n', ...
        LABELS{best}, num2str(S));
Yb = run_one(mdl, mixer(uT_p, uy_p, PATTERNS{best}), Ts, 0);
if max(abs(Yb(:, col('phi')))) + max(abs(Yb(:, col('theta')))) > 0.1 * max(abs(Yb(:, col('wz'))))
    warning(['Even the best pattern tilts the drone: the roll/pitch are not decoupled from the ' ...
             'yaw in open loop. Look at the probe before generating data.']);
end

%% PART 2 - data generation
rng(SEED);
UT = zeros(N_EXP, n_t);  UY = zeros(N_EXP, n_t);
YALL = zeros(N_EXP, n_t, numel(NAMES), 'single');
fprintf('\nPART 2 - %d experiments of %g s (g_noise = %g)\n', N_EXP, T_EXP, G_NOISE);
t0 = tic;
for e = 1:N_EXP
    [uT, uy] = excitation(n_t, N_REST, UT_RANGE, HOLD_T, YAW_MAX, HOLD_YAW, P_ZERO);
    Y = run_one(mdl, mixer(uT, uy, S), Ts, G_NOISE);
    UT(e, :) = uT';  UY(e, :) = uy';  YALL(e, :, :) = single(Y);
    fprintf('  experiment %3d / %d  (%.0f s)\n', e, N_EXP, toc(t0));
end

%% Save: same layout as the course's CSVs (one experiment per row)
writematrix(UT, fullfile(out_dir, 'yaw_in_uT.csv'));
writematrix(UY, fullfile(out_dir, 'yaw_in_uyaw.csv'));
for name = {'w_dot_z', 'wz', 'psi', 'az', 'phi', 'theta'}
    writematrix(double(YALL(:, :, col(name{1}))), fullfile(out_dir, ['yaw_out_' name{1} '.csv']));
end
save(fullfile(out_dir, 'yaw_dataset.mat'), 'UT', 'UY', 'YALL', 'NAMES', 'S', 'Ts', 'G_NOISE', '-v7');
fprintf('Saved in %s\n', out_dir);

%% Quick look at the first experiment
t = (0:n_t-1)' * Ts;
figure('Name', 'yaw data, experiment 1');
subplot(4,1,1); stairs(t, [UT(1,:)' UY(1,:)']); grid on; ylabel('command'); legend('u_T', 'u_{yaw}');
subplot(4,1,2); plot(t, YALL(1,:,col('w_dot_z'))); grid on; ylabel('w\_dot\_z');
subplot(4,1,3); plot(t, YALL(1,:,col('wz')));      grid on; ylabel('wz');
subplot(4,1,4); plot(t, YALL(1,:,col('phi'))); grid on; ylabel('phi [rad]'); xlabel('t [s]');
fprintf('max |wz| over all experiments: %.2f rad/s   max |phi|: %.4f rad\n', ...
        max(abs(YALL(:,:,col('wz'))), [], 'all'), max(abs(YALL(:,:,col('phi'))), [], 'all'));
close_system(mdl, 0);


%% ------------------------------------------------------------------------------------------------
function build_model(mdl)
% Open-loop test bench, built in the running MATLAB version:
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

function U = mixer(uT, uy, S)
% The 4 motor commands. Columns: u1 u2 u3 u4.
U = uT(:) + uy(:) * S;
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

function [uT, uy] = excitation(n_t, n_rest, ut_range, hold_t, yaw_max, hold_yaw, p_zero)
% u_T : staircase in ut_range (independent switching times from u_yaw, so their effects can be told
%       apart and the dependence of the yaw gain on the thrust is visible).
% u_yaw: doublets +v / -v (|v| <= yaw_max, random duration), with random pauses at 0 between them.
%       Clipped to the yaw margin 1 - |u_T| so that no motor command leaves [-1, 1].
uT = zeros(n_t, 1);  uy = zeros(n_t, 1);
k = n_rest + 1;
while k <= n_t
    L = randi(hold_t);  uT(k:min(k+L-1, n_t)) = ut_range(1) + diff(ut_range) * rand;  k = k + L;
end
k = n_rest + 1;
while k <= n_t
    L = randi(hold_yaw);
    if rand < p_zero                                   % pause
        uy(k:min(k+L-1, n_t)) = 0;  k = k + L;
    else                                               % doublet
        v = yaw_max * (2 * rand - 1);
        uy(k:min(k+L-1, n_t)) = v;            k = k + L;
        uy(k:min(k+L-1, n_t)) = -v;           k = k + L;
    end
end
margin = 1 - abs(uT);
uy = max(min(uy, margin), -margin);
end
