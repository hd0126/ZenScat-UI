function generate_fdfd_golden()
%GENERATE_FDFD_GOLDEN Build small MATLAB FDFD oracle fixtures.
%
% Run from the repository root with:
%   /Applications/MATLAB_R2024b.app/bin/matlab -batch "addpath('matlab_oracle'); generate_fdfd_golden"

script_path = mfilename('fullpath');
oracle_dir = fileparts(script_path);
repo_root = fileparts(oracle_dir);
source_dir = fullfile(repo_root, 'ZENSCAT_MAIN');
output_path = fullfile(repo_root, 'tests', 'golden', 'zenscat_matlab_fdfd_golden.mat');

addpath(source_dir);
old_dir = pwd;
cleanup = onCleanup(@() cd(old_dir));
cd(source_dir);

base = fixture_config();
cases = [
    with_mode(base, 'fdfd_small_sin_e_2x2', 'E')
    with_mode(base, 'fdfd_small_sin_h_2x2', 'H')
    with_interface(base, 'fdfd_small_de1_e_2x2', 'DE1')
    with_interface(base, 'fdfd_small_de4_e_2x2', 'DE4')
    with_interface(base, 'fdfd_small_tri_e_2x2', 'tri')
    periodic_config('fdfd_small_sin_two_periodic_e_2x2')
    material_palette_config('fdfd_small_sin_material_palette_e_2x2')
];

fixtures = repmat(empty_fixture(), numel(cases), 1);
for idx = 1:numel(cases)
    fixtures(idx) = run_fixture(cases(idx));
end

manifest = struct();
manifest.schema_version = '1.0';
manifest.generator = 'matlab_oracle/generate_fdfd_golden.m';
manifest.source_commit = strtrim(system_text('git rev-parse HEAD'));
manifest.matlab_version = version;
manifest.matlab_release = ['R' version('-release')];
manifest.fixture_file = 'tests/golden/zenscat_matlab_fdfd_golden.mat';
manifest.fixture_count = numel(fixtures);
manifest.fixture_names = {fixtures.name};
manifest.exclusions = 'PhC FDFD branch intentionally excluded: legacy Device_FDFD_PhC path is separate and known-broken for this parity slice.';
manifest.dispersion_legacy_mode = 'Dispersion.m uses 1e6 * P.Lam0(mid). In FDFD, P.Lam0 is already micrometers, so Python also exposes corrected_um as an explicit opt-in mode.';

save(output_path, 'manifest', 'fixtures', '-v7');
fprintf('Wrote %s\n', output_path);
end

function cfg = fixture_config()
cfg = struct();
cfg.name = '';
cfg.mode = '';
cfg.interface = 'sin';
cfg.distribution = 'all';
cfg.is_periodic = false;
cfg.period_num = 33;
cfg.refractive_idx = false;
cfg.dispersion_mode = 'explicit_index';
cfg.params = [0.182 0.120 1.781 1.650];
cfg.layer_num = 2;
cfg.theta_deg = [0 5];
cfg.lam0_um = [0.50 0.54];
cfg.Lx_um = 0.32;
cfg.h_um = 0.154;
cfg.n_sup = 1.0;
cfg.n_sub = 1.516;
cfg.NRES = 4;
cfg.SPACER = [0.20 0.20];
cfg.NPML = [2 2];
cfg.trapz_w_bot = 0.10;
cfg.trapz_w_top = 0.24;
cfg.supergauss_sigma = 0.08;
cfg.supergauss_m = 2;
cfg.triangle_w1 = 0.25;
cfg.triangle_w2 = 0.3;
cfg.triangle_w3 = 0.1;
end

function cfg = with_mode(base, name, mode)
cfg = base;
cfg.name = name;
cfg.mode = mode;
end

function cfg = with_interface(base, name, interface)
cfg = base;
cfg.name = name;
cfg.mode = 'E';
cfg.interface = interface;
end

function cfg = periodic_config(name)
cfg = fixture_config();
cfg.name = name;
cfg.mode = 'E';
cfg.interface = 'sin';
cfg.distribution = 'two';
cfg.is_periodic = true;
cfg.period_num = 4;
cfg.params = [0.080 0.100 1.700 2.050];
cfg.layer_num = 2;
end

function cfg = material_palette_config(name)
cfg = fixture_config();
cfg.name = name;
cfg.mode = 'E';
cfg.interface = 'sin';
cfg.distribution = 'all';
cfg.refractive_idx = true;
cfg.dispersion_mode = 'legacy_fdfd';
cfg.params = [0.182 0.120 1 2];
cfg.layer_num = 2;
end

function fixture = empty_fixture()
fixture = struct();
fixture.name = '';
fixture.config = struct();
fixture.grid = struct();
fixture.device = struct();
fixture.TRN = struct();
fixture.REF = struct();
fixture.f = [];
fixture.dispersion = struct();
fixture.checksums = struct();
end

function fixture = run_fixture(cfg)
interface_params = interface_params_from_config(cfg);
P = Params_FDFD(cfg.theta_deg, cfg.lam0_um, cfg.layer_num, cfg.distribution);
P.Params = cfg.params;
P.params = cfg.params;
P.Lx = cfg.Lx_um;
P.h = cfg.h_um;
P.n_sup = cfg.n_sup;
P.n_sub = cfg.n_sub;
P.is_periodic = cfg.is_periodic;
P.period_num = cfg.period_num;
P.NRES = cfg.NRES;
P.SPACER = cfg.SPACER;
P.NPML = cfg.NPML;
P.nmax = max(abs(cfg.params(cfg.layer_num + 1:end)));
P.erSup = cfg.n_sup^2;
P.erSub = cfg.n_sub^2;

if cfg.refractive_idx
    dispersion_data = load('Dispersion.mat');
    dispersion_coeffs = dispersion_data.Dispersion;
    dispersion_n = Dispersion(P, dispersion_coeffs);
else
    dispersion_coeffs = [];
    dispersion_n = [];
end

grid = Grid_FDFD(cfg.params, P, dispersion_coeffs, cfg.refractive_idx);
device = Device_FDFD(cfg.params, P, grid, cfg.interface, interface_params);
[TRN, REF, f] = FDFD_2D(grid, device, cfg.mode);

fixture = empty_fixture();
fixture.name = cfg.name;
fixture.config = cfg;
fixture.dispersion = struct( ...
    'coefficients', dispersion_coeffs, ...
    'legacy_n', dispersion_n);
fixture.grid = grid;
fixture.device = device;
fixture.TRN = TRN;
fixture.REF = REF;
fixture.f = f;
fixture.checksums = struct( ...
    'ER2_sha256', sha256_numeric(device.ER2), ...
    'UR2_sha256', sha256_numeric(device.UR2), ...
    'f_sha256', sha256_numeric(f), ...
    'TRN0_sha256', sha256_numeric(TRN.TRN0), ...
    'REF0_sha256', sha256_numeric(REF.REF0));
fprintf('%s: grid %dx%d, field norm %.6g\n', cfg.name, grid.Nx, grid.Ny, norm(f(:)));
end

function params = interface_params_from_config(cfg)
params = struct();
params.smooth = false;
params.trapz_w_bot = cfg.trapz_w_bot;
params.trapz_w_top = cfg.trapz_w_top;
params.supergauss_sigma = cfg.supergauss_sigma;
params.supergauss_m = cfg.supergauss_m;
params.triangle_w1 = cfg.triangle_w1;
params.triangle_w2 = cfg.triangle_w2;
params.triangle_w3 = cfg.triangle_w3;
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
