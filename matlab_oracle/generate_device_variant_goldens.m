function generate_device_variant_goldens()
%GENERATE_DEVICE_VARIANT_GOLDENS Build Device.m interface variant fixtures.
%
% Run from the repository root with:
%   /Applications/MATLAB_R2024b.app/bin/matlab -batch "addpath('matlab_oracle'); generate_device_variant_goldens"

script_path = mfilename('fullpath');
oracle_dir = fileparts(script_path);
repo_root = fileparts(oracle_dir);
source_dir = fullfile(repo_root, 'ZENSCAT_MAIN');
output_path = fullfile(repo_root, 'tests', 'golden', 'zenscat_device_variant_goldens.mat');

addpath(source_dir);
old_dir = pwd;
cleanup = onCleanup(@() cd(old_dir));
cd(source_dir);

cases = [
    fixture_config('de1_all_ui', 'DE1', 'all', false, false)
    fixture_config('de1_two_periodic_ui', 'DE1', 'two', true, false)
    fixture_config('de4_all_ui_smooth', 'DE4', 'all', false, true)
    fixture_config('de4_two_periodic_ui_smooth', 'DE4', 'two', true, true)
    fixture_config('tri_all_ui', 'tri', 'all', false, false)
    fixture_config('tri_two_periodic_ui', 'tri', 'two', true, false)
];

fixtures = repmat(empty_fixture(), numel(cases), 1);
for idx = 1:numel(cases)
    fixtures(idx) = run_fixture(cases(idx));
end

manifest = struct();
manifest.schema_version = '1.0';
manifest.generator = 'matlab_oracle/generate_device_variant_goldens.m';
manifest.source_commit = strtrim(system_text('git rev-parse HEAD'));
manifest.matlab_version = version;
manifest.matlab_release = ['R' version('-release')];
manifest.fixture_file = 'tests/golden/zenscat_device_variant_goldens.mat';
manifest.fixture_count = numel(fixtures);
manifest.fixture_names = {fixtures.name};
manifest.notes = "Analytic Device.m variants only; PhC/import/FDFD builders excluded.";

save(output_path, 'manifest', 'fixtures', '-v7');
fprintf('Wrote %s\n', output_path);
end

function cfg = fixture_config(name, interface, distribution, is_periodic, smooth)
cfg = struct();
cfg.name = name;
cfg.interface = interface;
cfg.distribution = distribution;
cfg.is_periodic = is_periodic;
cfg.period_num = 33;
cfg.smooth = smooth;
cfg.NH = 2;
cfg.params = [0.182 0.120 1.781 1.650];
cfg.Lx_um = 0.32;
cfg.h_um = 0.154;
cfg.Nx = 128;
cfg.Nz = 5;
cfg.wavelengths_nm = [500 520];
cfg.angles_deg = [0 3];
cfg.n_sup = 1.0;
cfg.n_sub = 1.516;
cfg.flat_substrate = false;
cfg.trapz_w_bot = 0.195;
cfg.trapz_w_top = 0.375;
cfg.supergauss_sigma = 0.30615 / 2.355;
cfg.supergauss_m = 2;
cfg.triangle_w1 = 0.25;
cfg.triangle_w2 = 0.3;
cfg.triangle_w3 = 0.1;
end

function fixture = empty_fixture()
fixture = struct();
fixture.name = '';
fixture.config = struct();
fixture.grid_solver = struct();
fixture.grid_geometry = struct();
fixture.device = struct();
fixture.checksums = struct();
end

function fixture = run_fixture(cfg)
if strcmp(cfg.distribution, 'all')
    layer_num = ceil(numel(cfg.params) / 2);
else
    layer_num = numel(cfg.params) - 2;
    if layer_num == 0
        layer_num = 1;
    end
end

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
P.is_periodic = cfg.is_periodic;
if cfg.is_periodic
    P.distribution = 'two';
end

interface_params = interface_params_from_config(cfg);
grid = Grid(cfg.params, cfg.interface, P);
if cfg.flat_substrate
    device = Device(cfg.NH, grid, cfg.interface, interface_params, 1);
else
    device = Device(cfg.NH, grid, cfg.interface, interface_params);
end

fixture = empty_fixture();
fixture.name = cfg.name;
fixture.config = cfg;
fixture.grid_solver = solver_grid(grid);
fixture.grid_geometry = geometry_grid(grid);
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
fixture.checksums = struct( ...
    'device_ER_sha256', sha256_numeric(device.ER), ...
    'device_ERC_sha256', sha256_numeric(device.ERC), ...
    'device_sub_L_sha256', sha256_numeric(device.sub_L), ...
    'grid_Length_sha256', sha256_numeric(grid.Length), ...
    'grid_erIdx_sha256', sha256_numeric(grid.erIdx), ...
    'grid_x_sha256', sha256_numeric(grid.x));

fprintf('%s: ER %dx%d, ERC %dx%dx%d, sub_L sum %.12g\n', ...
    cfg.name, size(device.ER, 1), size(device.ER, 2), ...
    size(device.ERC, 1), size(device.ERC, 2), size(device.ERC, 3), ...
    sum(device.sub_L));
end

function params = interface_params_from_config(cfg)
params = struct();
params.smooth = cfg.smooth;
params.trapz_w_bot = cfg.trapz_w_bot;
params.trapz_w_top = cfg.trapz_w_top;
params.supergauss_sigma = cfg.supergauss_sigma;
params.supergauss_m = cfg.supergauss_m;
params.triangle_w1 = cfg.triangle_w1;
params.triangle_w2 = cfg.triangle_w2;
params.triangle_w3 = cfg.triangle_w3;
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

function grid_geometry = geometry_grid(grid)
grid_geometry = struct();
grid_geometry.distribution = grid.distribution;
grid_geometry.h = grid.h;
grid_geometry.Lx = grid.Lx;
grid_geometry.Lz = grid.Lz;
grid_geometry.Nx = grid.Nx;
grid_geometry.Nz = grid.Nz;
grid_geometry.dz = grid.dz;
grid_geometry.qx = grid.qx;
grid_geometry.qz = grid.qz;
grid_geometry.delta = grid.delta;
grid_geometry.layer_num = grid.layer_num;
grid_geometry.Length = grid.Length;
grid_geometry.erIdx = grid.erIdx;
grid_geometry.x = grid.x;
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
