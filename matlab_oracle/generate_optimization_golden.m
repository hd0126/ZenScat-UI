function generate_optimization_golden()
%GENERATE_OPTIMIZATION_GOLDEN Build deterministic MATLAB merit fixtures.
%
% Run from the repository root with:
%   /Applications/MATLAB_R2024b.app/bin/matlab -batch "addpath('matlab_oracle'); generate_optimization_golden"

script_path = mfilename('fullpath');
oracle_dir = fileparts(script_path);
repo_root = fileparts(oracle_dir);
source_dir = fullfile(repo_root, 'ZENSCAT_MAIN');
output_path = fullfile(repo_root, 'tests', 'golden', 'zenscat_matlab_optimization_golden.mat');

addpath(source_dir);
old_dir = pwd;
cleanup = onCleanup(@() cd(old_dir));
cd(source_dir);

cases = {
    analytic_config('analytic_e_t0', 'E', 'T(0)')
    analytic_config('analytic_h_absorption', 'H', 'Absorption')
    import_config('import_e_gain', 'E', 'Gain')
    import_config('import_h_r0', 'H', 'R(0)')
};

fixtures = repmat(empty_fixture(), numel(cases), 1);
for idx = 1:numel(cases)
    fixtures(idx) = run_fixture(cases{idx});
end

manifest = struct();
manifest.schema_version = '1.0';
manifest.generator = 'matlab_oracle/generate_optimization_golden.m';
manifest.source_commit = strtrim(system_text('git rev-parse HEAD'));
manifest.matlab_version = version;
manifest.matlab_release = ['R' version('-release')];
manifest.fixture_file = 'tests/golden/zenscat_matlab_optimization_golden.mat';
manifest.fixture_count = numel(fixtures);
manifest.fixture_names = {fixtures.name};
manifest.notes = "Scalar Merit_Function3 and Merit_Function_Import objective values.";

save(output_path, 'manifest', 'fixtures', '-v7');
fprintf('Wrote %s\n', output_path);
end

function cfg = analytic_config(name, mode, objective)
cfg = struct();
cfg.name = name;
cfg.kind = 'analytic';
cfg.mode = mode;
cfg.objective = objective;
cfg.NH = 2;
cfg.interface = 'sin';
cfg.distribution = 'all';
cfg.params = [0.182 0.120 1.781 1.650];
cfg.X = [0.32 0.154 0.182];
cfg.Lx_um = 0.32;
cfg.h_um = 0.154;
cfg.Nx = 96;
cfg.Nz = 5;
cfg.wavelengths_nm = [500 520];
cfg.angles_deg = 0;
cfg.n_sup = 1.0;
cfg.n_sub = 1.516;
end

function cfg = import_config(name, mode, objective)
cfg = struct();
cfg.name = name;
cfg.kind = 'imported';
cfg.mode = mode;
cfg.objective = objective;
cfg.NH = 2;
cfg.X = [0.32 0.050 0.070 0.090];
cfg.Lx_um = 0.32;
cfg.Nx = 96;
cfg.Lam0 = 1e-9 * [500 520];
cfg.Theta = 0;
cfg.n_sup = 1.0;
cfg.n_sub = 1.516;
cfg.sub_L_m = [0.050 0.070 0.090] * 1e-6;
end

function fixture = empty_fixture()
fixture = struct();
fixture.name = '';
fixture.kind = '';
fixture.mode = '';
fixture.objective = '';
fixture.flavor = '';
fixture.config = struct();
fixture.fitness = [];
fixture.python_expected = [];
fixture.TRN = struct('minus_1', [], 'plus_1', [], 'TRN0', [], 'sum', []);
fixture.REF = struct('minus_1', [], 'plus_1', [], 'REF0', [], 'sum', []);
fixture.checksums = struct();
end

function fixture = run_fixture(cfg)
if strcmp(cfg.kind, 'analytic')
    fixture = run_analytic_fixture(cfg);
elseif strcmp(cfg.kind, 'imported')
    fixture = run_import_fixture(cfg);
else
    error('Unsupported fixture kind: %s', cfg.kind);
end
fprintf('%s: fitness %.16g expected %.16g\n', ...
    fixture.name, fixture.fitness, fixture.python_expected);
end

function fixture = run_analytic_fixture(cfg)
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
device = Device(cfg.NH, grid, cfg.interface, interface_params);
[TRN, REF] = Launch_RCWA_S(cfg.NH, grid, device, cfg.mode, false);
fitness = Merit_Function3(P, cfg.NH, cfg.X, cfg.params, cfg.mode, ...
    cfg.interface, cfg.objective, interface_params);

fixture = base_fixture(cfg, 'analytic', fitness, TRN, REF);
end

function fixture = run_import_fixture(cfg)
grid = struct();
grid.Lx = cfg.Lx_um;
grid.x = linspace(-cfg.Lx_um / 2, cfg.Lx_um / 2, cfg.Nx);
grid.Nx = cfg.Nx;
grid.Nz = numel(cfg.sub_L_m);
grid.Lam0 = cfg.Lam0;
grid.Theta = cfg.Theta;
grid.urR = 1;
grid.erR = cfg.n_sup.^2;
grid.urT = 1;
grid.erT = cfg.n_sub.^2;
grid.erSub = cfg.n_sub.^2;
grid.layer_num = 1;

row1 = 1.8 + 0.12 * cos(2*pi*(grid.x / cfg.Lx_um));
row2 = 2.2 + 0.08 * sin(2*pi*(grid.x / cfg.Lx_um));
row3 = 2.6 + 0.10 * cos(4*pi*(grid.x / cfg.Lx_um));
device = struct();
device.ER = [row1; row2; row3];
device.ERC = convmat1D(device.ER, cfg.NH);
device.sub_L = cfg.sub_L_m;

[TRN, REF] = Launch_RCWA_S(cfg.NH, grid, device, cfg.mode, false);
fitness = Merit_Function_Import(cfg.NH, cfg.X, cfg.mode, cfg.objective, grid, device);

fixture = base_fixture(cfg, 'imported', fitness, TRN, REF);
fixture.config.initial_sub_L_m = cfg.sub_L_m;
fixture.config.initial_ER = device.ER;
end

function fixture = base_fixture(cfg, flavor, fitness, TRN, REF)
fixture = empty_fixture();
fixture.name = cfg.name;
fixture.kind = cfg.kind;
fixture.mode = cfg.mode;
fixture.objective = cfg.objective;
fixture.flavor = flavor;
fixture.config = cfg;
fixture.fitness = fitness;
fixture.python_expected = python_expected_value(TRN, REF, cfg.objective, flavor);
fixture.TRN = TRN;
fixture.REF = REF;
fixture.checksums = struct( ...
    'TRN_minus_1_sha256', sha256_numeric(TRN.minus_1), ...
    'TRN_plus_1_sha256', sha256_numeric(TRN.plus_1), ...
    'TRN_TRN0_sha256', sha256_numeric(TRN.TRN0), ...
    'TRN_sum_sha256', sha256_numeric(TRN.sum), ...
    'REF_minus_1_sha256', sha256_numeric(REF.minus_1), ...
    'REF_plus_1_sha256', sha256_numeric(REF.plus_1), ...
    'REF_REF0_sha256', sha256_numeric(REF.REF0), ...
    'REF_sum_sha256', sha256_numeric(REF.sum));
end

function expected = python_expected_value(TRN, REF, objective, flavor)
if strcmp(objective, 'R(-1)')
    calc = REF.minus_1;
elseif strcmp(objective, 'R(0)')
    calc = REF.REF0;
elseif strcmp(objective, 'R(+1)')
    calc = REF.plus_1;
elseif strcmp(objective, 'T(-1)')
    calc = TRN.minus_1;
elseif strcmp(objective, 'T(0)')
    calc = TRN.TRN0;
elseif strcmp(objective, 'T(+1)')
    calc = TRN.plus_1;
elseif strcmp(objective, 'Absorption')
    if strcmp(flavor, 'analytic')
        calc = -1 * (TRN.sum + REF.sum);
    else
        calc = 1 - (TRN.sum + REF.sum);
    end
elseif strcmp(objective, 'Gain')
    calc = TRN.sum + REF.sum;
else
    error('Unsupported objective: %s', objective);
end

if strcmp(flavor, 'imported')
    expected = -sum(abs(calc(:)));
else
    expected = -sum(calc(:));
end
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
