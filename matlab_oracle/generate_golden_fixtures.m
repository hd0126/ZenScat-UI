function generate_golden_fixtures()
%GENERATE_GOLDEN_FIXTURES Build deterministic MATLAB oracle fixtures.
%
% Run from the repository root with:
%   /Applications/MATLAB_R2024b.app/bin/matlab -batch "addpath('matlab_oracle'); generate_golden_fixtures"

script_path = mfilename('fullpath');
oracle_dir = fileparts(script_path);
repo_root = fileparts(oracle_dir);
source_dir = fullfile(repo_root, 'ZENSCAT_MAIN');
output_path = fullfile(repo_root, 'tests', 'golden', 'zenscat_matlab_golden.mat');

addpath(source_dir);
old_dir = pwd;
cleanup = onCleanup(@() cd(old_dir));
cd(source_dir);

cases = [
    fixture_config( ...
        'casual_default_s_e_41x21', 'S', 'E', 5, ...
        [0.182 0.120 1.781 1.650], 0.32, 0.154, 1028, 11, ...
        linspace(480, 540, 41), linspace(0, 5, 21))
    fixture_config( ...
        'small_s_e_3x3', 'S', 'E', 2, ...
        [0.182 0.120 1.781 1.650], 0.32, 0.154, 128, 5, ...
        [480 510 540], [0 2.5 5])
    fixture_config( ...
        'small_s_h_3x3', 'S', 'H', 2, ...
        [0.182 0.120 1.781 1.650], 0.32, 0.154, 128, 5, ...
        [480 510 540], [0 2.5 5])
    fixture_config( ...
        'small_t_e_3x3', 'T', 'E', 2, ...
        [0.182 0.120 1.781 1.650], 0.32, 0.154, 128, 5, ...
        [480 510 540], [0 2.5 5])
    fixture_config( ...
        'small_t_h_3x3', 'T', 'H', 2, ...
        [0.182 0.120 1.781 1.650], 0.32, 0.154, 128, 5, ...
        [480 510 540], [0 2.5 5])
];

fixtures = repmat(empty_fixture(), numel(cases), 1);
for idx = 1:numel(cases)
    fixtures(idx) = run_fixture(cases(idx));
end

manifest = struct();
manifest.schema_version = '1.0';
manifest.generator = 'matlab_oracle/generate_golden_fixtures.m';
manifest.source_commit = strtrim(system_text('git rev-parse HEAD'));
manifest.matlab_version = version;
manifest.matlab_release = ['R' version('-release')];
manifest.fixture_file = 'tests/golden/zenscat_matlab_golden.mat';
manifest.fixture_count = numel(fixtures);
manifest.fixture_names = {fixtures.name};
manifest.notes = [ ...
    "All wavelengths are stored in meters and angles in radians. " + ...
    "Checksums are SHA-256 of MATLAB column-major double payloads."];

save(output_path, 'manifest', 'fixtures', '-v7');
fprintf('Wrote %s\n', output_path);
end

function cfg = fixture_config(name, method, polarization, NH, params, Lx_um, h_um, Nx, Nz, wavelengths_nm, angles_deg)
cfg = struct();
cfg.name = name;
cfg.method = method;
cfg.polarization = polarization;
cfg.NH = NH;
cfg.interface = 'sin';
cfg.distribution = 'all';
cfg.params = params;
cfg.Lx_um = Lx_um;
cfg.h_um = h_um;
cfg.Nx = Nx;
cfg.Nz = Nz;
cfg.wavelengths_nm = wavelengths_nm;
cfg.angles_deg = angles_deg;
cfg.n_sup = 1.0;
cfg.n_sub = 1.516;
cfg.calc_fresnel = false;
cfg.flat_substrate = false;
end

function fixture = empty_fixture()
fixture = struct();
fixture.name = '';
fixture.config = struct();
fixture.Lam0 = [];
fixture.Theta = [];
fixture.Output = [];
fixture.TRN = struct('minus_1', [], 'plus_1', [], 'TRN0', [], 'sum', []);
fixture.REF = struct('minus_1', [], 'plus_1', [], 'REF0', [], 'sum', []);
fixture.field_shapes = struct();
fixture.checksums = struct();
fixture.energy = struct();
fixture.device = struct();
fixture.grid_solver = struct();
end

function fixture = run_fixture(cfg)
layer_num = ceil(numel(cfg.params) / 2);
P = Parameters(layer_num, cfg.distribution);
P.Params = cfg.params;
P.Lx = cfg.Lx_um;
P.h = cfg.h_um;
P.Nx = cfg.Nx;
P.Nz = cfg.Nz;
P.Lam0 = 1e-9 * cfg.wavelengths_nm;
P.Theta = (pi / 180) * cfg.angles_deg;
P.n_sup = cfg.n_sup;
P.n_sub = cfg.n_sub;
P.layer_num = layer_num;

interface_params = Int_Params();
grid = Grid(cfg.params, cfg.interface, P);
if cfg.flat_substrate
    device = Device(cfg.NH, grid, cfg.interface, interface_params, 1);
else
    device = Device(cfg.NH, grid, cfg.interface, interface_params);
end

if strcmp(cfg.method, 'S')
    [TRN, REF] = Launch_RCWA_S(cfg.NH, grid, device, cfg.polarization, cfg.calc_fresnel);
elseif strcmp(cfg.method, 'T')
    [TRN, REF] = Launch_RCWA_T(cfg.NH, grid, device, cfg.polarization, cfg.calc_fresnel);
else
    error('Unsupported method: %s', cfg.method);
end

fixture = empty_fixture();
fixture.name = cfg.name;
fixture.config = cfg;
fixture.Lam0 = grid.Lam0;
fixture.Theta = grid.Theta;
fixture.Output = cfg.params;
fixture.TRN = TRN;
fixture.REF = REF;
fixture.grid_solver = solver_grid(grid);
fixture.field_shapes = field_shapes(TRN, REF, grid);
fixture.checksums = checksums(TRN, REF, grid, cfg.params, device);
energy = TRN.sum + REF.sum;
fixture.energy = struct( ...
    'sum_min', min(energy(:)), ...
    'sum_max', max(energy(:)), ...
    'max_abs_error', max(abs(energy(:) - 1)));
fixture.device = struct( ...
    'ER', device.ER, ...
    'ERC', device.ERC, ...
    'sub_L', device.sub_L, ...
    'ER_shape', size(device.ER), ...
    'ERC_shape', size(device.ERC), ...
    'sub_L_shape', size(device.sub_L), ...
    'sub_L_sum_m', sum(device.sub_L), ...
    'ER_checksum_sha256', sha256_numeric(device.ER), ...
    'ERC_checksum_sha256', sha256_numeric(device.ERC), ...
    'sub_L_checksum_sha256', sha256_numeric(device.sub_L));
fprintf('%s: energy max_abs_error %.3g\n', cfg.name, fixture.energy.max_abs_error);
end

function grid_solver = solver_grid(grid)
grid_solver = struct();
grid_solver.Lam0 = grid.Lam0;
grid_solver.Theta = grid.Theta;
grid_solver.Lx = grid.Lx;
grid_solver.layer_num = grid.layer_num;
grid_solver.erR = grid.erR;
grid_solver.urR = grid.urR;
grid_solver.erT = grid.erT;
grid_solver.urT = grid.urT;
grid_solver.erSub = grid.erSub;
end

function shapes = field_shapes(TRN, REF, grid)
shapes = struct();
shapes.Lam0 = size(grid.Lam0);
shapes.Theta = size(grid.Theta);
shapes.TRN_minus_1 = size(TRN.minus_1);
shapes.TRN_plus_1 = size(TRN.plus_1);
shapes.TRN_TRN0 = size(TRN.TRN0);
shapes.TRN_sum = size(TRN.sum);
shapes.REF_minus_1 = size(REF.minus_1);
shapes.REF_plus_1 = size(REF.plus_1);
shapes.REF_REF0 = size(REF.REF0);
shapes.REF_sum = size(REF.sum);
end

function sums = checksums(TRN, REF, grid, params, device)
sums = struct();
sums.Lam0_sha256 = sha256_numeric(grid.Lam0);
sums.Theta_sha256 = sha256_numeric(grid.Theta);
sums.Output_sha256 = sha256_numeric(params);
sums.TRN_minus_1_sha256 = sha256_numeric(TRN.minus_1);
sums.TRN_plus_1_sha256 = sha256_numeric(TRN.plus_1);
sums.TRN_TRN0_sha256 = sha256_numeric(TRN.TRN0);
sums.TRN_sum_sha256 = sha256_numeric(TRN.sum);
sums.REF_minus_1_sha256 = sha256_numeric(REF.minus_1);
sums.REF_plus_1_sha256 = sha256_numeric(REF.plus_1);
sums.REF_REF0_sha256 = sha256_numeric(REF.REF0);
sums.REF_sum_sha256 = sha256_numeric(REF.sum);
sums.device_ER_sha256 = sha256_numeric(device.ER);
sums.device_ERC_sha256 = sha256_numeric(device.ERC);
sums.device_sub_L_sha256 = sha256_numeric(device.sub_L);
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
