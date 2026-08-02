# ZenScat Studio

ZenScat Studio is a Python desktop successor to the original MATLAB ZenScat
application for one-periodic-axis RCWA and 2D FDFD analysis. The numerical core
is independent of the GUI and runs on Apple Silicon without MATLAB, MATLAB
Coder, or the bundled Windows-only MEX files.

The original MATLAB implementation remains under `ZENSCAT_MAIN/` as the oracle
and compatibility reference. Python parity is locked by MATLAB R2024b fixtures
under `tests/golden/`; see `compatibility/manifest-v1.json` for the exact
workflow, field, tolerance, and legacy-quirk contract.

## Implemented compatibility surface

- 1D RCWA S-matrix and T-matrix methods with legacy E/H (`s/p`) conventions.
- Order-resolved `TRN` and `REF` fields for `-1`, zero order, `+1`, and sums.
  Modern/corrected result paths also carry `-2` and `+2` orders when the
  harmonic count is high enough to compute them.
- Analytic `sin`, trapezium (`DE1`), super-Gaussian (`DE4`), and triangle
  device generation with MATLAB indexing and rounding behavior, plus explicit
  interface parameters for repaired modern geometry variants.
- Legacy `RCWA_DATA.mat` import, trusted `.npy` dictionary import, and
  MATLAB-compatible result bundle export with additional CSV and plot assets.
- Casual 2D/PhC S-matrix RCWA for rectangle, ellipse/circle, hex-column,
  honeycomb, rotated hex-column, and hex-polygon devices. Legacy repeat
  behavior remains selectable, while corrected modes use explicit geometry
  repairs.
- Direct NumPy/SciPy sparse FDFD port with legacy public fields and raw field
  `f` for the verified mainline `sin`, `DE1`, `DE4`, and `tri` device paths,
  including material-palette dispersion in legacy and corrected unit modes.
- Corrected PhC-to-FDFD rasterization for the implemented PhC geometries,
  including dual E/H field solves and export helpers.
- Harmonic-convergence sweeps for analytic, imported, and PhC RCWA requests,
  with zero-order stability, energy-error scoring, cancellation, and optional
  recommended harmonic count.
- Legacy optimization-objective semantics plus deterministic, cancellable
  SciPy differential evolution. Corrected/modern optimization supports `±2`
  order objectives, arbitrary imported-layer bounds, integer-index constraints,
  sum constraints, seeds, time limits, checkpoint writing, checkpoint resume,
  and live merit-history callbacks. The `ga_compat` profile keeps GA-like
  sizing semantics without claiming MATLAB's private random trajectory.
- Explicit local and external backends. External solver commands use a JSON
  file-envelope protocol (`--zenscat-request` and `--zenscat-result`) and never
  fall back silently to the local Python solver.
- Versioned `.zenscat` project documents and reproducibility manifests,
  including persisted backend, compatibility, convergence, optimization, FDFD,
  PhC-FDFD, result-view, and export-related settings.

Known exclusions and policies:

- The broken legacy `Device_FDFD_PhC.m` branch is excluded because it references
  undefined identifiers and has no stable result contract. Python provides a
  corrected PhC-FDFD path instead and deliberately rejects `legacy_exact` for
  that broken branch.
- The legacy optimization dropdown exposed `±2` diffraction-order choices, but
  `Merit_Function3.m` and `Merit_Function_Import.m` never implemented them.
  Python therefore rejects `±2` objectives in `legacy_exact` optimization mode.
  Use `corrected` or `modern` mode for implemented `R(-2)`, `R(+2)`, `T(-2)`,
  and `T(+2)` objectives.
- The legacy live rectangle PhC GUI zeroed one rectangle width. Python requires
  positive rectangle widths and records that repair in the compatibility tests.
- FDFD keeps the legacy final-sweep scalar sum and the original
  material-palette wavelength behavior by default. `REF_minus1` is the primary
  negative-order reflection field; the historical `REF.TRN_minus1` alias is
  retained only for compatibility with legacy readers.
- The proprietary MATLAB GA trajectory is unavailable as a public reproducible
  contract. Python verifies objective semantics, deterministic optimizer
  controls, checkpoint/resume state, and final physical RCWA results instead.

## Desktop workflows

The GUI has seven pages:

| Page | Purpose |
| --- | --- |
| Project | Open/save `.zenscat` projects, inspect the current stack, load legacy import files, choose local or external backend execution, and track solver readiness. |
| Device | Edit the analytic layer table, interface type and parameters, stack distribution, media indices, period, height, and grid resolution. |
| RCWA Sweep | Run analytic, imported, PhC RCWA, or harmonic-convergence sweeps. PhC mode exposes rectangle, ellipse, hex-column, honeycomb, rotated hex, and hex-polygon devices. |
| Optimize | Run deterministic bounded optimization against analytic or imported RCWA templates with compatibility modes, `±2` objectives outside `legacy_exact`, seed/time/sum/integer constraints, checkpoint/resume, and live merit history. |
| FDFD Fields | Run sparse 2D FDFD, dual E/H FDFD, or corrected PhC-FDFD solves and inspect `abs(f)`, `real(f)`, `imag(f)`, `phase(f)`, `log10(abs(f))`, or `ER2` contours. |
| Results | Switch between `TRN`, `REF`, energy, and energy-error views, select diffraction orders or sums, inspect 2D heatmaps, start/mid/end wavelength or angle slices, and export tables. |
| Jobs | Inspect run logs, progress, and cancellation state. |

Export is enabled only after a solver run succeeds. RCWA, import, PhC,
harmonic-convergence final runs, and optimization final physical runs use
`zenscat.result-bundle`:

Choose a new or empty export directory. ZenScat refuses to write into a
non-empty directory so an earlier result bundle cannot be overwritten silently.

```text
TRN.mat
REF.mat
Lam.mat      # Lam0 in metres
Theta.mat    # radians
manifest.json
Params.mat   # optional optimized or supplied parameters
Output.mat   # optional legacy parameter alias
Data.txt
RCWA_plot_data.csv
RCWA_axes.csv
Selected_result.png
Selected_result.svg
```

`Data.txt` and `RCWA_plot_data.csv` are flat wavelength/angle tables. They
include `-2` and `+2` columns when those result fields are present.

FDFD, dual FDFD, and corrected PhC-FDFD exports use
`zenscat.fdfd-result-bundle` because the original FDFD plotting path did not
define a durable file schema:

```text
TRN.mat
REF.mat
Lam.mat      # Lam0 in micrometres
Theta.mat    # radians
Field.mat    # raw final field f
ER2.mat      # device permittivity field
manifest.json
Params.mat   # optional supplied parameters
FDFD_metrics.csv
Field_real.csv
Field_imag.csv
Field_abs.csv
ER2.csv
Selected_result.png
Selected_result.svg
```

Dual E/H exports write one FDFD bundle per polarization. Result-view plot
assets are deterministic snapshots of the current GUI plot widgets.

External CLI backends receive one JSON request file and must write one JSON
result file:

```text
<configured command> --zenscat-request request.json --zenscat-result result.json
```

The response can inline an RCWA, FDFD, or optimization result, or point to an
absolute result-bundle directory. Non-zero exits, timeouts, missing result
files, relative bundle paths, and malformed JSON are reported as external
solver errors.

## Run on Apple Silicon

Python 3.12 or 3.13 is required. With [`uv`](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12
uv pip install -e '.[gui,test]'
ZENSCAT_PYTHON=.venv/bin/python ./scripts/run_macos_arm64.sh
```

The production GUI prefers PySide6. A PyQt6 fallback exists for development
environments that already provide it, but release bundles use PySide6 so the
MIT application can be distributed under Qt for Python's LGPL terms.

The macOS launcher copies the Qt platform plugins into a clean temporary
directory before startup. This avoids an APFS/FileProvider case where a
virtual environment below `Documents` immediately regains Finder `hidden`
metadata and Qt skips its otherwise-valid Cocoa plugin.

Headless usage does not install Qt:

```python
from zenscat.workflows import AnalyticRCWARequest, run_analytic_rcwa

request = AnalyticRCWARequest(
    params=[0.182, 0.120, 1.781, 1.650],
    layer_num=2,
    wavelengths_m=[510e-9],
    angles_rad=[0.0],
    harmonic_count=5,
    matrix_method="S",
    polarization="E",
)
result = run_analytic_rcwa(request)
print(result.transmission.TRN0, result.reflection.REF0)
```

FDFD headless usage uses micrometre wavelengths. The example below uses a
coarse grid so it runs quickly as a smoke test:

```python
from zenscat.workflows import FDFDRequest, run_fdfd

request = FDFDRequest(
    params=[0.182, 0.120, 1.781, 1.650],
    layer_num=2,
    wavelengths_um=[0.5106],
    angles_rad=[0.0],
    interface="sin",
    polarization="E",
    nres=4,
    period_num=1,
    npml=(1, 1),
    spacer_um=(0.1, 0.1),
)
run = run_fdfd(request)
print(run.result.TRN["TRN0"], run.result.f.shape)
```

## Verify

```bash
uv run --extra test pytest -q
```

MATLAB is not needed to run the committed tests or application. It is needed
only to regenerate oracle fixtures:

```bash
/Applications/MATLAB_R2024b.app/bin/matlab -batch \
  "addpath('matlab_oracle'); generate_golden_fixtures"
```

Additional oracle generators cover device variants, optimization objectives,
FDFD fields, and PhC:

```bash
/Applications/MATLAB_R2024b.app/bin/matlab -batch \
  "addpath('matlab_oracle'); generate_device_variant_goldens; generate_optimization_golden; generate_fdfd_golden; generate_phc_golden"
```

## Build the native macOS app

On an Apple Silicon Mac with an arm64 Python environment:

```bash
uv pip install -e '.[gui,build]'
ZENSCAT_PYTHON=.venv/bin/python ./scripts/build_macos_arm64.sh
```

The script creates `dist/ZenScat.app` for local use and
`dist/ZenScat-arm64.zip` as the stable signed archive. It rejects PyQt release
fallback, verifies the executable and packaged notices, and strictly verifies
an extracted copy of the archive. FileProvider may attach Finder metadata to
the `.app` left directly under `Documents/GitHub`; the ZIP preserves the clean
ad-hoc signature across that boundary. Public notarization requires an Apple
Developer identity and credentials and is intentionally not automated here.

## Legacy data

The example directories (`Spatial Filter`, `LIDT coupler`, `Lumerical
Comparison`, and `DFB files`) contain the original `RCWA_DATA.mat` inputs. They
load directly through `zenscat.legacy_io.load_imported_device`. Result bundles
contain the original `TRN.mat`, `REF.mat`, `Lam.mat`, and `Theta.mat` files plus
a checksum-bearing `manifest.json`. NumPy dictionary imports can contain pickle
payloads, so they are trusted only after an explicit Import action; reopening a
project preserves their path but never loads or trusts them automatically.

## License

ZenScat source is MIT licensed. Release GUI bundles use dynamically linked
PySide6/Qt libraries under their applicable LGPL/GPL/commercial terms; consult
`THIRD_PARTY_NOTICES.md` before redistribution.
