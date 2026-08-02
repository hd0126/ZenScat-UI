# Selected physics acceptance fixtures

These fixtures lock the four advanced-physics handoff cases against the pinned
RCWA Studio reference commit in `../selected_physics_manifest.json`.

They are deliberately **reference-only** for ZenScat's local solver.  ZenScat
currently implements the legacy one-periodic-axis RCWA and 2D FDFD surface; it
does not claim local ASR, full 2D NVF, MEEP, or conical s/p support.  A future
implementation should consume these vectors and change `local_support` only
after its own solver-output comparison passes.

`w18_meep_lossy_2d.json` is a capability-boundary fixture rather than invented
numerical data.  The reference engine still blocks lossy patterned 2D MEEP
until a 3D volume-absorption monitor independently agrees with `1-R-T`.
