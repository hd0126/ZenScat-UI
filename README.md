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
- Order-resolved `TRN` and `REF` fields, including `-1`, `+1`, zero order,
  and sums.
- Analytic `sin`, trapezium (`DE1`), super-Gaussian (`DE4`), and triangle
  device generation with MATLAB indexing and rounding behavior.
- Legacy `RCWA_DATA.mat` import, trusted `.npy` dictionary import, and
  MATLAB-compatible result bundle export.
- Casual 2D/PhC S-matrix RCWA for rectangle, ellipse/circle, and hex-column
  devices. `repeat_mode="legacy"` preserves the original non-power-of-two
  layer repetition behavior; `repeat_mode="corrected"` repeats exactly.
- Direct NumPy/SciPy sparse FDFD port with legacy public fields and raw field
  `f` for the verified mainline `sin`, `DE1`, `DE4`, and `tri` device paths,
  including material-palette dispersion in legacy and corrected unit modes.
- Legacy optimization-objective semantics plus a deterministic, cancellable
  SciPy optimizer. The proprietary MATLAB GA trajectory is not treated as a
  reproducible public result; objective values and final physical results are.
- Versioned `.zenscat` project documents and reproducibility manifests.

Known exclusions and policies:

- The broken legacy `Device_FDFD_PhC.m` branch is excluded because it references
  undefined identifiers and has no stable result contract.
- The legacy optimization dropdown exposed `±2` diffraction-order choices, but
  `Merit_Function3.m` and `Merit_Function_Import.m` never implemented them.
  The Python GUI exposes only the implemented objectives: `R(-1)`, `R(0)`,
  `R(+1)`, `T(-1)`, `T(0)`, `T(+1)`, `Absorption`, and `Gain`.
- The legacy live rectangle PhC GUI zeroed one rectangle width. Python requires
  positive rectangle widths and records that repair in the compatibility tests.
- FDFD keeps the legacy final-sweep scalar sum, the `REF.TRN_minus1` alias, and
  the original material-palette wavelength behavior by default. Use
  `corrected_um` only when you want the dispersion formula evaluated with FDFD
  wavelengths treated as micrometres.

## Desktop workflows

The GUI has seven pages:

| Page | Purpose |
| --- | --- |
| Project | Open/save `.zenscat` projects, inspect the current stack, load legacy import files, and track solver readiness. |
| Device | Edit the analytic layer table, interface type, stack distribution, media indices, period, height, and grid resolution. |
| RCWA Sweep | Run analytic, imported, or PhC RCWA sweeps. PhC mode exposes rectangle, ellipse, and hex-column devices. |
| Optimize | Run the deterministic bounded optimizer against analytic or imported RCWA templates and export the final physical S-matrix result. |
| FDFD Fields | Run the sparse 2D FDFD field solver and inspect `abs(f)`, `real(f)`, or `ER2`. |
| Results | Switch between `TRN`, `REF`, and energy views and select diffraction orders or sums. |
| Jobs | Inspect run logs, progress, and cancellation state. |

Export is enabled only after a solver run succeeds. RCWA, import, PhC, and
optimization exports use `zenscat.result-bundle`:

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
```

FDFD exports use `zenscat.fdfd-result-bundle` because the original FDFD plotting
path did not define a durable file schema:

```text
TRN.mat
REF.mat
Lam.mat      # Lam0 in micrometres
Theta.mat    # radians
Field.mat    # raw final field f
ER2.mat      # device permittivity field
manifest.json
Params.mat   # optional supplied parameters
```

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

FDFD headless usage uses micrometre wavelengths:

```python
from zenscat.workflows import FDFDRequest, run_fdfd

request = FDFDRequest(
    params=[0.182, 0.120, 1.781, 1.650],
    layer_num=2,
    wavelengths_um=[0.5106],
    angles_rad=[0.0],
    interface="sin",
    polarization="E",
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
