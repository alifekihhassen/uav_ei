function build_closed_loop_yaw(mdl)
%BUILD_CLOSED_LOOP_YAW  Builds (and saves) the closed loop of the full mission, in the running MATLAB
%   version, around the protected drone UAV_model.slx. Same wiring principle as the course's
%   EINT_RL_closed_loop.slx, with 4 different motor commands and the yaw measurements:
%
%     Clock ----------- ZOH -----------+
%     UAV z ----------- ZOH -----------+
%     UAV vz ---------- ZOH -----------+
%     UAV az ---------- Unit Delay ----+            +--> U (4 motors) --> Demux --> UAV u1..u4
%     u_T (k-1) <------ Unit Delay ----+--> rl_controller_full --> u_T, u_yaw (fed back via Unit Delays)
%     UAV psi --------- ZOH -----------+            +--> z_ref, psi_ref (logged)
%     UAV wz ---------- ZOH -----------+
%     UAV w_dot_z ----- Unit Delay ----+
%     u_yaw (k-1) <---- Unit Delay ----+
%
%   Logged in rl_log (one row per step): t z vz az psi wz w_dot_z phi x y u_T u_yaw z_ref psi_ref u1 u2 u3 u4
if nargin < 1, mdl = 'EINT_RL_closed_loop_yaw'; end
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

% controller: MATLAB Function calling rl_controller_full (extrinsic: it loads .mat files)
ctrl = [mdl '/controller'];
add_block('simulink/User-Defined Functions/MATLAB Function', ctrl);
chart = find(sfroot, '-isa', 'Stateflow.EMChart', 'Path', ctrl);
chart.Script = sprintf(['function [U, z_ref, psi_ref, u_T, u_yaw] = controller(t, z, vz, az_prev, u_T_prev, psi, wz, w_dot_z_prev, u_yaw_prev)\n' ...
    'coder.extrinsic(''rl_controller_full'');\n' ...
    'U = zeros(4, 1); z_ref = 0; psi_ref = 0; u_T = 0; u_yaw = 0;\n' ...
    '[U, z_ref, psi_ref, u_T, u_yaw] = rl_controller_full(t, z, vz, az_prev, u_T_prev, psi, wz, w_dot_z_prev, u_yaw_prev);\n']);

% measurement blocks
add_block('simulink/Sources/Clock', [mdl '/Clock']);
for name = {'ZOH_t', 'ZOH_z', 'ZOH_vz', 'ZOH_psi', 'ZOH_wz'}
    add_block('simulink/Discrete/Zero-Order Hold', [mdl '/' name{1}], 'SampleTime', 'Ts');
end
for name = {'Delay_az', 'Delay_wdz', 'Delay_uT', 'Delay_uyaw'}
    add_block('simulink/Discrete/Unit Delay', [mdl '/' name{1}], 'SampleTime', 'Ts', 'InitialCondition', '0');
end
add_block('simulink/Signal Routing/Demux', [mdl '/Demux'], 'Outputs', '4');
add_block('simulink/Signal Routing/Mux', [mdl '/Mux'], 'Inputs', '15');
add_block('simulink/Sinks/To Workspace', [mdl '/rl_log'], 'VariableName', 'rl_log', ...
          'SaveFormat', 'Array', 'SampleTime', 'Ts');

% controller inputs: 1 t, 2 z, 3 vz, 4 az_prev, 5 u_T_prev, 6 psi, 7 wz, 8 w_dot_z_prev, 9 u_yaw_prev
L = @(a, b) add_line(mdl, a, b, 'autorouting', 'on');
L('Clock/1', 'ZOH_t/1');      L('ZOH_t/1', 'controller/1');
L('UAV/3', 'ZOH_z/1');        L('ZOH_z/1', 'controller/2');
L('UAV/6', 'ZOH_vz/1');       L('ZOH_vz/1', 'controller/3');
L('UAV/15', 'Delay_az/1');    L('Delay_az/1', 'controller/4');
L('controller/4', 'Delay_uT/1');   L('Delay_uT/1', 'controller/5');
L('UAV/9', 'ZOH_psi/1');      L('ZOH_psi/1', 'controller/6');
L('UAV/12', 'ZOH_wz/1');      L('ZOH_wz/1', 'controller/7');
L('UAV/18', 'Delay_wdz/1');   L('Delay_wdz/1', 'controller/8');
L('controller/5', 'Delay_uyaw/1'); L('Delay_uyaw/1', 'controller/9');
% controller outputs: 1 U, 2 z_ref, 3 psi_ref, 4 u_T, 5 u_yaw
L('controller/1', 'Demux/1');
for i = 1:4, L(sprintf('Demux/%d', i), sprintf('UAV/%d', i)); end

% log: t z vz az psi wz w_dot_z phi x y u_T u_yaw z_ref psi_ref U(4)
src = {'ZOH_t/1', 'ZOH_z/1', 'ZOH_vz/1', 'UAV/15', 'ZOH_psi/1', 'ZOH_wz/1', 'UAV/18', 'UAV/7', ...
       'UAV/1', 'UAV/2', 'controller/4', 'controller/5', 'controller/2', 'controller/3', 'controller/1'};
for i = 1:numel(src), L(src{i}, sprintf('Mux/%d', i)); end
L('Mux/1', 'rl_log/1');
for p = [4 5 8 10 11 13 14 16 17 19]          % unused drone outputs
    add_block('simulink/Sinks/Terminator', sprintf('%s/T%d', mdl, p));
    L(sprintf('UAV/%d', p), sprintf('T%d/1', p));
end

set_param(mdl, 'SolverType', 'Fixed-step', 'Solver', 'FixedStepAuto', 'FixedStep', 'Ts', ...
          'StopTime', 'T_end', 'ReturnWorkspaceOutputs', 'on');
try, Simulink.BlockDiagram.arrangeSystem(mdl); catch, end
save_system(mdl, fullfile(here, [mdl '.slx']));
end
