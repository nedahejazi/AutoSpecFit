# ASF v1.0 in-depth revision against latest ASF v2.0

This revision keeps the defining v1.0 behavior: stellar atmospheric parameters are fixed and are not refined. It ports the general abundance-workflow robustness and output conventions from the latest v2.0 where they are applicable without introducing parameter fitting.

## Main changes

- Updated abundance convergence to match v2.0:
  - iterations 2-6: all elements must have |Delta abundance| <= 0.05 dex;
  - iterations 7-8: at most one element may exceed 0.05 dex;
  - iterations 9-15: at most two may exceed 0.05 dex and no more than one may exceed 0.10 dex;
  - maximum 15 iterative abundance cycles.
- Non-finite abundance changes are treated as non-converged/oscillating species rather than causing an ambiguous global failure.
- Maximum-iteration finalization is explicitly labeled as a hard-stop result, not convergence.
- Added one dedicated final abundance determination after iterative convergence/finalization, with the fixed v1.0 atmosphere and adopted iterative abundances used as non-target seeds.
- Added cumulative vertical `ASF_Abundance_History_GJ205.txt`.
- Added cumulative `ASF_Convergence_History_GJ205.txt`. It reports abundance convergence and explicitly states that atmospheric-parameter refinement is not performed in v1.0.
- Added `ASF_Fixed_Stellar_Parameters_GJ205.txt` so the fixed atmosphere is recorded with the run.
- Added restart/checkpoint support with v1-specific checkpoint/marker names to avoid collision with v2 checkpoints.
- Switched diagnostic outputs to rolling files (latest completed abundance iteration) to match v2 output management:
  - `ASF_Current_Abundance_Results_GJ205.txt`
  - `ASF_Current_Abundance_Chi2_GJ205.txt`
  - `ASF_Current_Abundance_Line_Errors_GJ205.txt`
  - `ASF_Run_Notes_GJ205.txt`
- Final abundance table now reports both native `ASF_Offset` and physical `Final_X_H` using the fixed input [M/H] and [alpha/Fe].
- Random abundance uncertainties now match v2.0: for N>=2 accepted lines, line-to-line scatter and RMS chi-square curvature error are added in quadrature; for N=1, only the line chi-square error is used.
- Added negative-zero protection for abundance/model filename values (e.g. `-0.000` -> `+0.000`).
- Model availability handling now matches v2.0 more closely:
  - checks missing and zero-byte models;
  - fails immediately if required models are unavailable while TS generation is disabled;
  - avoids resubmitting existing non-empty models;
  - removes zero-byte model files before regeneration;
  - waits 15 minutes per check for up to 300 checks.
- Added interpolated/non-interpolated Turbospectrum runner selection using the same current v2.0 atmosphere-grid logic.
- Updated default abundance grid to v2.0 default: -0.360 to +0.360 dex in 0.020 dex steps.
- Updated abundance-line rejection rules to current v2.0 defaults: OH restricted to (-0.250,+0.250); other species reject only -0.360, -0.350, +0.350, +0.360.
- Removed Cr from the default GJ205 species set to match the latest v2.0 default species configuration.
- Preserved the normal and quiet variants. The quiet version now differs only by disabling the Python logger, so numerical behavior and output files remain identical.

## Validation performed

- Both revised scripts pass `python -m py_compile`.
- Both import successfully with the AutoSpecNorm dependency stubbed.
- Convergence-rule tests were run for early/intermediate/late stages, including NaN abundance changes.
- Negative-zero filename formatting was tested.
- Vertical abundance-history and convergence-history writers were exercised.
- The normal and quiet scripts were diff-checked; aside from logger suppression, their code is identical.

## Important scientific distinction retained

ASF v1.0 still does **not** refine Teff, log g, [M/H], [alpha/Fe], or vmic. Consequently, it does not calculate atmospheric-parameter systematic abundance uncertainties. The final table reports random abundance uncertainties only and records the fixed atmosphere used for the analysis.
