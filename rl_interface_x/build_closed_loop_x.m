function build_closed_loop_x(mdl, ctrl_fn)
%BUILD_CLOSED_LOOP_X  Builds (and saves) the closed loop of advanced task 2 (land at x = 1 m, y held at 0),
%   in the running MATLAB version, around the protected drone UAV_model.slx. Same wiring principle as
%   build_closed_loop_yaw.m, plus the measurements of the two horizontal axes:
%
%     Clock / z / vz / psi / wz / x / vx / wy / y / vy / wx ---- ZOH --------+
%     az / w_dot_z / w_dot_y / ax / w_dot_x / ay -------------- Unit Delay -+--> rl_controller_full_x --> U --> UAV u1..u4
%     u_T / u_yaw / u_x / u_y (k-1) <-------------------------- Unit Delay -+      (commands fed back)
%
%   ctrl_fn: the controller function (default 'rl_controller_full_x'; 'rl_controller_follow' for task 3),
%   any function with the inputs and outputs of rl_controller_full_x.
%
%   Logged in rl_log (one row per step):
%     t z vz az psi wz x y vx wy ax u_T u_yaw u_x z_ref psi_ref x_ref u_y vy wx y_ref u1 u2 u3 u4
if nargin < 1, mdl = 'EINT_RL_closed_loop_x'; end
if nargin < 2, ctrl_fn = 'rl_controller_full_x'; end
here = fileparts(mfilename('fullpath'));
if bdIsLoaded(mdl), close_system(mdl, 0); end
new_system(mdl);

% drone (outputs: 1 x 2 y 3 z 4 vx 5 vy 6 vz 7 phi 8 theta 9 psi 10 wx 11 wy 12 wz
%                 13 ax 14 ay 15 az 16 w_dot_x 17 w_dot_y 18 w_dot_z 19 all_outputs)
try
    add_block('simulink/Ports & Subsystems/Subsystem Reference', [mdl '/UAV']);
catch
    load_system('EINT_system_simulation');
    add_block('EINT_system_simulation/Subsystem Reference', [mdl '/UAV']);
end
set_param([mdl '/UAV'], 'ReferencedSubsystem', 'UAV_model');

% controller: MATLAB Function calling ctrl_fn (extrinsic: it loads .mat files)
args = ['t, z, vz, az_prev, u_T_prev, psi, wz, w_dot_z_prev, u_yaw_prev, x, vx, wy, w_dot_y_prev, ' ...
        'ax_prev, u_x_prev, y, vy, wx, w_dot_x_prev, ay_prev, u_y_prev'];
outs = 'U, z_ref, psi_ref, x_ref, y_ref, u_T, u_yaw, u_x, u_y';
ctrl = [mdl '/controller'];
add_block('simulink/User-Defined Functions/MATLAB Function', ctrl);
chart = find(sfroot, '-isa', 'Stateflow.EMChart', 'Path', ctrl);
chart.Script = sprintf(['function [%s] = controller(%s)\n' ...
    'coder.extrinsic(''%s'');\n' ...
    'U = zeros(4, 1); z_ref = 0; psi_ref = 0; x_ref = 0; y_ref = 0; u_T = 0; u_yaw = 0; u_x = 0; u_y = 0;\n' ...
    '[%s] = %s(%s);\n'], outs, args, ctrl_fn, outs, ctrl_fn, args);

% measurement blocks
add_block('simulink/Sources/Clock', [mdl '/Clock']);
for name = {'ZOH_t', 'ZOH_z', 'ZOH_vz', 'ZOH_psi', 'ZOH_wz', 'ZOH_x', 'ZOH_vx', 'ZOH_wy', 'ZOH_y', 'ZOH_vy', 'ZOH_wx'}
    add_block('simulink/Discrete/Zero-Order Hold', [mdl '/' name{1}], 'SampleTime', 'Ts');
end
for name = {'Delay_az', 'Delay_wdz', 'Delay_wdy', 'Delay_ax', 'Delay_wdx', 'Delay_ay', ...
            'Delay_uT', 'Delay_uyaw', 'Delay_ux', 'Delay_uy'}
    add_block('simulink/Discrete/Unit Delay', [mdl '/' name{1}], 'SampleTime', 'Ts', 'InitialCondition', '0');
end
add_block('simulink/Signal Routing/Demux', [mdl '/Demux'], 'Outputs', '4');
add_block('simulink/Signal Routing/Mux', [mdl '/Mux'], 'Inputs', '22');
add_block('simulink/Sinks/To Workspace', [mdl '/rl_log'], 'VariableName', 'rl_log', ...
          'SaveFormat', 'Array', 'SampleTime', 'Ts');

% controller inputs: 1 t, 2 z, 3 vz, 4 az_prev, 5 u_T_prev, 6 psi, 7 wz, 8 w_dot_z_prev, 9 u_yaw_prev,
%   10 x, 11 vx, 12 wy, 13 w_dot_y_prev, 14 ax_prev, 15 u_x_prev,
%   16 y, 17 vy, 18 wx, 19 w_dot_x_prev, 20 ay_prev, 21 u_y_prev
% controller outputs: 1 U, 2 z_ref, 3 psi_ref, 4 x_ref, 5 y_ref, 6 u_T, 7 u_yaw, 8 u_x, 9 u_y
L = @(a, b) add_line(mdl, a, b, 'autorouting', 'on');
L('Clock/1', 'ZOH_t/1');      L('ZOH_t/1', 'controller/1');
L('UAV/3', 'ZOH_z/1');        L('ZOH_z/1', 'controller/2');
L('UAV/6', 'ZOH_vz/1');       L('ZOH_vz/1', 'controller/3');
L('UAV/15', 'Delay_az/1');    L('Delay_az/1', 'controller/4');
L('controller/6', 'Delay_uT/1');   L('Delay_uT/1', 'controller/5');
L('UAV/9', 'ZOH_psi/1');      L('ZOH_psi/1', 'controller/6');
L('UAV/12', 'ZOH_wz/1');      L('ZOH_wz/1', 'controller/7');
L('UAV/18', 'Delay_wdz/1');   L('Delay_wdz/1', 'controller/8');
L('controller/7', 'Delay_uyaw/1'); L('Delay_uyaw/1', 'controller/9');
L('UAV/1', 'ZOH_x/1');        L('ZOH_x/1', 'controller/10');
L('UAV/4', 'ZOH_vx/1');       L('ZOH_vx/1', 'controller/11');
L('UAV/11', 'ZOH_wy/1');      L('ZOH_wy/1', 'controller/12');
L('UAV/17', 'Delay_wdy/1');   L('Delay_wdy/1', 'controller/13');
L('UAV/13', 'Delay_ax/1');    L('Delay_ax/1', 'controller/14');
L('controller/8', 'Delay_ux/1');   L('Delay_ux/1', 'controller/15');
L('UAV/2', 'ZOH_y/1');        L('ZOH_y/1', 'controller/16');
L('UAV/5', 'ZOH_vy/1');       L('ZOH_vy/1', 'controller/17');
L('UAV/10', 'ZOH_wx/1');      L('ZOH_wx/1', 'controller/18');
L('UAV/16', 'Delay_wdx/1');   L('Delay_wdx/1', 'controller/19');
L('UAV/14', 'Delay_ay/1');    L('Delay_ay/1', 'controller/20');
L('controller/9', 'Delay_uy/1');   L('Delay_uy/1', 'controller/21');
L('controller/1', 'Demux/1');
for i = 1:4, L(sprintf('Demux/%d', i), sprintf('UAV/%d', i)); end

% log: t z vz az psi wz x y vx wy ax u_T u_yaw u_x z_ref psi_ref x_ref u_y vy wx y_ref U(4)
src = {'ZOH_t/1', 'ZOH_z/1', 'ZOH_vz/1', 'UAV/15', 'ZOH_psi/1', 'ZOH_wz/1', 'ZOH_x/1', 'ZOH_y/1', ...
       'ZOH_vx/1', 'ZOH_wy/1', 'UAV/13', 'controller/6', 'controller/7', 'controller/8', ...
       'controller/2', 'controller/3', 'controller/4', 'controller/9', 'ZOH_vy/1', 'ZOH_wx/1', ...
       'controller/5', 'controller/1'};
for i = 1:numel(src), L(src{i}, sprintf('Mux/%d', i)); end
L('Mux/1', 'rl_log/1');
for p = [7 8 19]                              % unused drone outputs
    add_block('simulink/Sinks/Terminator', sprintf('%s/T%d', mdl, p));
    L(sprintf('UAV/%d', p), sprintf('T%d/1', p));
end

set_param(mdl, 'SolverType', 'Fixed-step', 'Solver', 'FixedStepAuto', 'FixedStep', 'Ts', ...
          'StopTime', 'T_end', 'ReturnWorkspaceOutputs', 'on');
try, Simulink.BlockDiagram.arrangeSystem(mdl); catch, end
save_system(mdl, fullfile(here, [mdl '.slx']));
end
