function generate_phc_golden()
%GENERATE_PHC_GOLDEN Build deterministic MATLAB PhC RCWA fixtures.
%
% Run from the repository root with:
%   /Applications/MATLAB_R2024b.app/bin/matlab -batch "addpath('matlab_oracle'); generate_phc_golden"

script_path = mfilename('fullpath');
oracle_dir = fileparts(script_path);
repo_root = fileparts(oracle_dir);
source_dir = fullfile(repo_root, 'ZENSCAT_MAIN');
output_path = fullfile(repo_root, 'tests', 'golden', 'zenscat_matlab_phc_golden.mat');

addpath(source_dir);
old_dir = pwd;
cleanup = onCleanup(@() cd(old_dir));
cd(source_dir);

cases = [
    fixture_config('rectangle_e_layer3_legacy_bug', 'PhC_rec_square', 'E', 3)
    fixture_config('ellipse_h_layer2', 'PhC_rec_circ', 'H', 2)
    fixture_config('hex_e_layer3_legacy_bug', 'PhC_hex_columns', 'E', 3)
    fixture_config('hex_h_layer2', 'PhC_hex_columns', 'H', 2)
];

fixtures = repmat(empty_fixture(), numel(cases), 1);
for idx = 1:numel(cases)
    fixtures(idx) = run_fixture(cases(idx));
end

manifest = struct();
manifest.schema_version = '1.0';
manifest.generator = 'matlab_oracle/generate_phc_golden.m';
manifest.source_commit = strtrim(system_text('git rev-parse HEAD'));
manifest.matlab_version = version;
manifest.matlab_release = ['R' version('-release')];
manifest.fixture_file = 'tests/golden/zenscat_matlab_phc_golden.mat';
manifest.fixture_count = numel(fixtures);
manifest.fixture_names = {fixtures.name};
manifest.notes = "Legacy Device_3.m and Launch_RCWA_S_PhC.m fixtures; angles are radians.";

save(output_path, 'manifest', 'fixtures', '-v7');
fprintf('Wrote %s\n', output_path);
end

function cfg = fixture_config(name, interface, mode, layer_count)
cfg = struct();
cfg.name = name;
cfg.interface = interface;
cfg.mode = mode;
cfg.layer_count = layer_count;
cfg.NH = 1;
cfg.params = [0.32 1.781 1.2];
cfg.Nx = 32;
cfg.Nz = 8;
cfg.wavelengths_m = 1e-9 * [510 530];
cfg.angles_rad = [0 0.04];
cfg.n_sup = 1.0;
cfg.n_sub = 1.516;
cfg.calc_fresnel = false;
cfg.rec_2D_wx = 0.2;
cfg.rec_2D_wy = 0.25;
cfg.rec_rot_angle = 0.0;
cfg.ax = 1.0;
cfg.ay = 0.5;
cfg.ellipse_rot_angle = 0.0;
cfg.radius_star_ellipse = 0.22;
end

function fixture = empty_fixture()
fixture = struct();
fixture.name = '';
fixture.config = struct();
fixture.grid_solver = struct();
fixture.device = struct();
fixture.TRN = struct('minus_1', [], 'plus_1', [], 'TRN0', [], 'sum', []);
fixture.REF = struct('minus_1', [], 'plus_1', [], 'REF0', [], 'sum', []);
fixture.checksums = struct();
end

function fixture = run_fixture(cfg)
P = Parameters(1, 'all');
P.Params = cfg.params;
P.Lam0 = cfg.wavelengths_m;
P.Theta = cfg.angles_rad;
P.Nx = cfg.Nx;
P.Nz = cfg.Nz;
P.n_sup = cfg.n_sup;
P.n_sub = cfg.n_sub;
P.layer_num = 1;

interface_params = interface_params_from_config(cfg);
grid = Grid(cfg.params, cfg.interface, P);
device = Device_3(cfg.NH, grid, cfg.interface, interface_params);
[TRN, REF] = Launch_RCWA_S_PhC(cfg.layer_count, cfg.NH, grid, device, cfg.mode, cfg.calc_fresnel);

fixture = empty_fixture();
fixture.name = cfg.name;
fixture.config = cfg;
fixture.grid_solver = solver_grid(grid);
fixture.device = device_payload(device);
fixture.TRN = TRN;
fixture.REF = REF;
fixture.checksums = checksums(grid, device, TRN, REF);

fprintf('%s: TRN0 %.16g REF0 %.16g energy %.16g\n', ...
    cfg.name, TRN.TRN0(1), REF.REF0(1), TRN.sum(1) + REF.sum(1));
end

function params = interface_params_from_config(cfg)
params = struct();
params.rec_2D_wx = cfg.rec_2D_wx;
params.rec_2D_wy = cfg.rec_2D_wy;
params.rec_rot_angle = cfg.rec_rot_angle;
params.ax = cfg.ax;
params.ay = cfg.ay;
params.ellipse_rot_angle = cfg.ellipse_rot_angle;
params.radius_star_ellipse = cfg.radius_star_ellipse;
end

function grid_solver = solver_grid(grid)
grid_solver = struct();
grid_solver.Lam0 = grid.Lam0;
grid_solver.Theta = grid.Theta;
grid_solver.Lx = grid.Lx;
grid_solver.Length = grid.Length;
grid_solver.Nx = grid.Nx;
grid_solver.Nz = grid.Nz;
grid_solver.erIdx = grid.erIdx;
grid_solver.erR = grid.erR;
grid_solver.urR = grid.urR;
grid_solver.erT = grid.erT;
grid_solver.urT = grid.urT;
grid_solver.erSub = grid.erSub;
grid_solver.layer_num = grid.layer_num;
end

function out = device_payload(device)
out = struct();
out.ER = device.ER;
out.ERC = device.ERC;
out.sub_L = device.sub_L;
out.is_top = device.is_top;
out.is_bot = device.is_bot;
if device.is_top
    out.ER_top = device.ER_top;
    out.ERC_top = device.ERC_top;
    out.sub_L_top = device.sub_L_top;
else
    out.ER_top = [];
    out.ERC_top = [];
    out.sub_L_top = [];
end
if device.is_bot
    out.ER_bot = device.ER_bot;
    out.ERC_bot = device.ERC_bot;
    out.sub_L_bot = device.sub_L_bot;
else
    out.ER_bot = [];
    out.ERC_bot = [];
    out.sub_L_bot = [];
end
end
function sums = checksums(grid, device, TRN, REF)
sums = struct();
sums.grid_Lam0_sha256 = sha256_numeric(grid.Lam0);
sums.grid_Theta_sha256 = sha256_numeric(grid.Theta);
sums.grid_Length_sha256 = sha256_numeric(grid.Length);
sums.grid_erIdx_sha256 = sha256_numeric(grid.erIdx);
sums.device_ER_sha256 = sha256_numeric(device.ER);
sums.device_ERC_sha256 = sha256_numeric(device.ERC);
sums.device_sub_L_sha256 = sha256_numeric(device.sub_L);
if device.is_top
    sums.device_ER_top_sha256 = sha256_numeric(device.ER_top);
    sums.device_ERC_top_sha256 = sha256_numeric(device.ERC_top);
    sums.device_sub_L_top_sha256 = sha256_numeric(device.sub_L_top);
else
    sums.device_ER_top_sha256 = '';
    sums.device_ERC_top_sha256 = '';
    sums.device_sub_L_top_sha256 = '';
end
if device.is_bot
    sums.device_ER_bot_sha256 = sha256_numeric(device.ER_bot);
    sums.device_ERC_bot_sha256 = sha256_numeric(device.ERC_bot);
    sums.device_sub_L_bot_sha256 = sha256_numeric(device.sub_L_bot);
else
    sums.device_ER_bot_sha256 = '';
    sums.device_ERC_bot_sha256 = '';
    sums.device_sub_L_bot_sha256 = '';
end
sums.TRN_minus_1_sha256 = sha256_numeric(TRN.minus_1);
sums.TRN_plus_1_sha256 = sha256_numeric(TRN.plus_1);
sums.TRN_TRN0_sha256 = sha256_numeric(TRN.TRN0);
sums.TRN_sum_sha256 = sha256_numeric(TRN.sum);
sums.REF_minus_1_sha256 = sha256_numeric(REF.minus_1);
sums.REF_plus_1_sha256 = sha256_numeric(REF.plus_1);
sums.REF_REF0_sha256 = sha256_numeric(REF.REF0);
sums.REF_sum_sha256 = sha256_numeric(REF.sum);
end

function digest = sha256_numeric(value)
payload = double(value(:));
if ~isreal(value)
    payload = [real(payload); imag(payload)];
end
bytes = typecast(payload, 'uint8');
md = java.security.MessageDigest.getInstance('SHA-256');
md.update(bytes);
hash = typecast(md.digest(), 'uint8');
digest = lower(reshape(dec2hex(hash, 2).', 1, []));
end

function text = system_text(command)
[status, text] = system(command);
if status ~= 0
    error('Command failed: %s\n%s', command, text);
end
end
