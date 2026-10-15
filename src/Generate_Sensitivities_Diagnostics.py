#!/usr/bin/env python3
"""GJ 205 — atmospheric-parameter sensitivity and diagnostic-line selection.

PURPOSE
-------
This standalone helper prepares Turbospectrum endpoint spectra and measures
the sensitivity of selected spectral lines to metallicity, surface gravity,
and effective temperature. It then identifies candidate Pass-1 diagnostic
lines for AutoSpecFit (ASF). It does not itself run the full ASF iterations.

METHOD
------
1. Read the GJ 205 fitting windows, excluding unavailable lines.
2. Construct or locate two endpoint synthetic spectra per parameter while
   holding the other atmospheric parameters fixed.
3. Measure line sensitivities from the endpoint spectra.
4. Count lines above 50% of each parameter's maximum sensitivity.
5. Keep [M/H] first by the configured benchmark policy; order log g and
   Teff by the sensitivity counts. Report ties rather than resolving them
   arbitrarily.
6. For diagnostic selection, combine the absolute-sensitivity threshold
   with differential log-sensitivity ratios and a robust median + 1.4826*MAD
   threshold, considering parameters that remain unconstrained.
7. Use all available lines for reference distributions, but restrict the
   final diagnostic lists to atomic lines.

The selections are statistical candidates: inspect blending, morphology,
and spectral fitting quality before adopting them in ASF.

EXECUTION
---------
Run from the directory containing the relevant ASF and line-range files:

  python Generate_Sensitivities_Diagnostics.py --mode plan
  python Generate_Sensitivities_Diagnostics.py --mode run
  python Generate_Sensitivities_Diagnostics.py --mode analyze

plan    Preview required endpoint models and commands without submitting.
run     Generate/reuse endpoint models, calculate sensitivities, and select.
analyze Reuse previously saved sensitivity tables to select diagnostics.

CONFIGURATION
-------------
Check ASF_FILENAME, BASE, ENDPOINTS, PARAMETER_LINES, and model directory
before running. Results are written to GJ205_Sensitivity_Diagnostics/.

Requires Python >=3.6, NumPy, pandas, SciPy, and the compatible ASF module.
"""
import argparse
from dataclasses import replace
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# Change only if the ASF filename or initial abundance offsets differ.
ASF_FILENAME = "AutoSpecFit_Abund_Param_v2_0_GJ205_No_Alpha_Variation_SW.py"
STAR = "GJ205"
BASE = dict(teff="3700", logg="+4.7", metallicity="+0.10",
            alpha="+0.00", vmic="1.00")
ENDPOINTS = {
    "metallicity": (-0.40, +0.40),
    "logg": (4.50, 5.50),
    "teff": (3500., 3900.),
}
# Empty means no _x08... abundance suffixes, i.e. the starting model mixture.
# To use nonzero offsets, supply a JSON mapping: {"x08":"+0.000", ...}
ABUNDANCE_OFFSETS_JSON = None
# Adopt [M/H] first (ASF benchmark policy); rank gravity vs temperature by counts.
# With equal counts, no ordering is asserted: user must inspect the diagnostics.
FIRST_PARAMETER = "metallicity"
SENSITIVITY_FRACTION = 0.5
SMOOTH_SIGMA_PIXELS = 2.3
# If reproducing an existing sensitivity calculation, match its flux convention.
# The two endpoint spectra are smoothed identically, then interpolated to
# their common wavelength grid. No observed-spectrum ASN normalization is used.
OUTPUT_DIR = Path("GJ205_Sensitivity_Diagnostics")
PARAMETER_LINES = Path("All_Species_Fit_Ranges_GJ205.txt")
RV_UNAVAILABLE = -999.9
# Use every available line for the reference distributions; final sets atomic only.
EXCLUDE_FROM_FINAL = ("OH", "CO", "H2O", "CN", "FEH", "TIO")


# Load the existing ASF implementation so model names and TS commands stay consistent.
def load_asf(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError("ASF module not found: {}. Set --asf.".format(path))
    spec = importlib.util.spec_from_file_location("asf_gj205_sensitivity", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# Load valid line windows; preserve the line identifiers used by ASF.
def read_lines():
    if not PARAMETER_LINES.exists():
        raise FileNotFoundError("Required line file not found: {}".format(PARAMETER_LINES))
    table = pd.read_csv(str(PARAMETER_LINES), sep=r"\s+", comment="#")
    needed = ["Line_num", "Species", "line_center", "RV", "min_fit", "max_fit"]
    if any(c not in table for c in needed):
        raise ValueError("Expected columns: {}".format(" ".join(needed)))
    table = table.copy()
    table = table[np.isfinite(table.RV) &
                  (~np.isclose(table.RV, RV_UNAVAILABLE))].sort_values("Line_num")
    if table.empty or table.Line_num.duplicated().any():
        raise ValueError("No available lines, or duplicate line numbers.")
    if not np.all(table.max_fit.to_numpy() > table.min_fit.to_numpy()):
        raise ValueError("Invalid fitting windows.")
    return table


# Optional fixed abundance mixture for all endpoint calculations.
def fixed_offsets():
    if ABUNDANCE_OFFSETS_JSON is None:
        return {}
    with open(ABUNDANCE_OFFSETS_JSON) as f:
        return json.load(f)


# Plan or request endpoint atmospheres/spectra with one parameter varied at a time.
def make_models(asf, mode, args):
    cfg = replace(asf.AutoSpecFitConfig(), model_dir=Path(args.models))
    if mode == "run":
        cfg.model_dir.mkdir(parents=True, exist_ok=True)
    base = asf.StellarParameters(**BASE)
    offsets = fixed_offsets()
    expected = {}
    for p, (lo, hi) in ENDPOINTS.items():
        grid = np.array([lo, hi], dtype=float)
        paths = asf.build_parameter_model_paths(cfg, base, offsets, p, grid)
        expected[p] = paths
        print("{} endpoint models:".format(p))
        for path in paths:
            print(" ", path)
        # Never touch existing nonempty models during a plan.
        if mode in ("plan", "run"):
            # Build command directly: no files are deleted by the planner.
            for value, path in zip(grid, paths):
                if path.is_file() and path.stat().st_size > 0:
                    print("  REUSE", path.name)
                    continue
                trial = asf.update_one_stellar_parameter(
                    base, p, float(value), use_exact_grid_format=True)
                runner = asf.select_turbospectrum_runner(trial, cfg)
                cmd = asf.build_parameter_turbospectrum_command(
                    base, offsets, p, np.array([value], dtype=float))
                full = "{} {} {}".format(
                    cfg.turbospectrum_execution_prefix, runner, cmd)
                print("  SUBMIT:", full)
                if mode == "run":
                    if path.is_file() and path.stat().st_size == 0:
                        raise RuntimeError(
                            "Zero-byte model exists: {}. Inspect its TS job "
                            "before deleting/regenerating.".format(path))
                    asf.run_turbospectrum_command(
                        cmd, runner, cfg.turbospectrum_execution_prefix)
    return expected


# Read one synthetic spectrum for subsequent line-by-line comparison.
def load_spectrum(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError("Missing/empty TS model: {}".format(path))
    arr = np.loadtxt(str(path), usecols=(0, 1), ndmin=2)
    w, f = arr[:, 0], arr[:, 1]
    good = np.isfinite(w) & np.isfinite(f)
    w, f = w[good], f[good]
    order = np.argsort(w)
    w, f = w[order], f[order]
    w, first = np.unique(w, return_index=True)
    f = f[first]
    if len(w) < 5:
        raise ValueError("Too few valid spectrum points in {}".format(path))
    return w, gaussian_filter1d(f, SMOOTH_SIGMA_PIXELS, mode="nearest")


# Quantify each line's response between the two endpoint spectra.
def sensitivities_for_parameter(lines, paths):
    wlo, flo = load_spectrum(paths[0])
    whi, fhi = load_spectrum(paths[1])
    rows = []
    for row in lines.itertuples(index=False):
        low, high = float(row.min_fit), float(row.max_fit)
        # The fitting window is in the synthetic rest frame. Do not apply
        # stellar RV to the synthetic spectra; RV is an availability flag here.
        mask = (wlo >= low) & (wlo <= high) & (wlo >= whi[0]) & (wlo <= whi[-1])
        w = wlo[mask]
        if len(w) < 2:
            raise ValueError("Insufficient overlap in line {} [{}, {}]".format(
                int(row.Line_num), low, high))
        other = np.interp(w, whi, fhi)
        sensitivity = float(np.mean((flo[mask] - other) ** 2))
        rows.append((int(row.Line_num), float(row.line_center), sensitivity))
    return np.array(rows, dtype=float)


# Use a stable output name for each parameter's sensitivity table.
def output_filename(p):
    if p == "metallicity":
        return "GJ205_Metal_Sensitivity_MetalMin-0.40_MetalMax+0.40_Teff3700_Grav+4.7.txt"
    if p == "logg":
        return "GJ205_Logg_Sensitivity_GravMin+4.5_GravMax+5.5_Teff3700_Metal+0.20.txt"
    return "GJ205_Teff_Sensitivity_TeffMin3500_TeffMax3900_Metal+0.20_Grav+4.7.txt"


# Persist sensitivity measurements for later independent analysis.
def write_sensitivities(sensitivity_arrays):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for p, arr in sensitivity_arrays.items():
        dest = OUTPUT_DIR / output_filename(p)
        # Three whitespace-delimited columns; no header, as ASF expects.
        np.savetxt(str(dest), arr, fmt=["%d", "%.3f", "%.16g"])
        print("WROTE", dest)


# Reuse previously measured sensitivities without rerunning Turbospectrum.
def read_saved_sensitivities(lines):
    results = {}
    for p in ENDPOINTS:
        file = OUTPUT_DIR / output_filename(p)
        if not file.exists():
            raise FileNotFoundError("Sensitivity file not found: {}".format(file))
        arr = np.loadtxt(str(file), ndmin=2)
        if arr.shape[1] != 3:
            raise ValueError("Expected 3 columns in {}".format(file))
        ids = arr[:, 0].astype(int)
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate line IDs in {}".format(file))
        by_id = {int(row[0]): row for row in arr}
        vals = []
        for row in lines.itertuples(index=False):
            item = by_id.get(int(row.Line_num))
            if item is None or abs(item[1] - float(row.line_center)) > 0.10:
                raise ValueError("Line mismatch {} in {}".format(row.Line_num, file))
            vals.append(float(item[2]))
        results[p] = np.asarray(vals)
    return results


# Exclude molecular lines from final Pass-1 diagnostics, not from reference statistics.
def atomic_mask(lines):
    labels = lines.Species.astype(str).str.upper().str.replace(" ", "", regex=False)
    return ~labels.str.startswith(EXCLUDE_FROM_FINAL).to_numpy()


# Compute the robust differential-response cutoff from finite values.
def robust_threshold(values):
    finite = np.asarray(values)[np.isfinite(values)]
    if len(finite) == 0:
        return float("inf")
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    return med + 1.4826 * mad


# Determine the configured parameter sequence and select Pass-1 diagnostic lines.
def select_diagnostics(lines, s):
    params = list(ENDPOINTS)
    max_s = {p: float(np.max(s[p])) for p in params}
    strong = {p: (s[p] >= SENSITIVITY_FRACTION * max_s[p])
              for p in params}
    counts = {p: int(np.sum(strong[p])) for p in params}
    print("\nINITIAL HALF-MAXIMUM SENSITIVITY ASSESSMENT")
    for p in params:
        print("  {}: {} / {} lines (Smax={:.9g})".format(
            p, counts[p], len(lines), max_s[p]))
    remaining = [p for p in params if p != FIRST_PARAMETER]
    remaining.sort(key=lambda p: counts[p], reverse=True)
    tied = counts[remaining[0]] == counts[remaining[1]]
    order = [FIRST_PARAMETER] + remaining
    if tied:
        print("  TIE between {} and {}: order is UNRESOLVED by counts.".format(
            remaining[0], remaining[1]))
        print("  The order below is a provisional display order only.")
    else:
        print("  Recommended order based on counts: {}".format(" -> ".join(order)))
    # Differential sensitivities are undefined for zero response; treat such
    # lines as non-candidates rather than adding an arbitrary epsilon.
    def differential(a, b):
        out = np.full(len(lines), np.nan)
        valid = (s[a] > 0) & (s[b] > 0)
        out[valid] = np.log(s[a][valid] / s[b][valid])
        return out

    atomic = atomic_mask(lines)
    selected = {}
    diagnostics = pd.DataFrame({
        "Line_num": lines.Line_num.astype(int).to_numpy(),
        "Species": lines.Species.astype(str).to_numpy(),
        "Wavelength": lines.line_center.astype(float).to_numpy(),
        "Atomic_eligible": atomic,
    })
    for p in params:
        diagnostics["S_" + p] = s[p]
        diagnostics["HighS_" + p] = strong[p]
    for idx, p in enumerate(order):
        competitors = order[idx+1:]
        if competitors:
            # First parameter: min of two differential log ratios.
            # Second parameter: one differential log ratio.
            all_c = np.column_stack([differential(p, q) for q in competitors])
            c = np.full(len(lines), np.nan)
            valid = np.all(np.isfinite(all_c), axis=1)
            c[valid] = np.min(all_c[valid], axis=1)
            threshold = robust_threshold(c)
            chosen = strong[p] & atomic & np.isfinite(c) & (c > 0) & (c > threshold)
            diagnostics["C_" + p] = c
            diagnostics["C_threshold_" + p] = threshold
        else:
            chosen = strong[p] & atomic
        diagnostics["Selected_" + p] = chosen
        selected[p] = diagnostics.loc[chosen, "Line_num"].astype(int).tolist()
        print("  {} diagnostics ({}): {}".format(p, len(selected[p]), selected[p]))
    diagnostics.to_csv(str(OUTPUT_DIR / "GJ205_All_Line_Sensitivity_Diagnostics.csv"),
                       index=False)
    with open(str(OUTPUT_DIR / "GJ205_Pass1_Diagnostic_Selection.txt"), "w") as f:
        f.write("Initial parameters: {}\n".format(BASE))
        f.write("Endpoint variations: {}\n".format(ENDPOINTS))
        f.write("Half-maximum counts (all available lines): {}\n".format(counts))
        f.write("Parameter sequence: {}\n".format(" -> ".join(order)))
        if tied:
            f.write("WARNING: gravity/temperature ordering unresolved by counts.\n")
        f.write("Differential threshold: median(C) + 1.4826 MAD(C)\n")
        f.write("Reference population: all available lines; final diagnostics: atomic only.\n")
        for p in order:
            f.write("{}: {}\n".format(p, selected[p]))
        f.write("\nInspect morphology and fitting quality manually; statistical "
                "selection is not a substitute for physical validation.\n")
    with open(str(OUTPUT_DIR / "GJ205_Pass1_Diagnostic_Selection.json"), "w") as f:
        json.dump({"order": order, "order_tied": tied,
                   "half_max_counts": counts, "selected_lines": selected,
                   "base": BASE, "endpoints": ENDPOINTS}, f, indent=2)
    return selected


# Command-line entry point: planning, full execution, or saved-data analysis.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "run", "analyze"),
                        default="plan")
    parser.add_argument("--asf", default=ASF_FILENAME)
    parser.add_argument("--models", default="/home/nedahejazi/Synthetic_Spectra_APOGEE_GJ205")
    parser.add_argument("--wait-seconds", type=int, default=60)
    parser.add_argument("--max-wait-minutes", type=int, default=180)
    args = parser.parse_args()
    lines = read_lines()
    if args.mode == "analyze":
        arrays = read_saved_sensitivities(lines)
    else:
        asf = load_asf(args.asf)
        expected = make_models(asf, args.mode, args)
        if args.mode == "plan":
            print("\nPLAN ONLY. No TS jobs submitted or sensitivity files written.")
            return
        cfg = replace(asf.AutoSpecFitConfig(),
                      model_dir=Path(args.models),
                      model_wait_seconds=args.wait_seconds,
                      max_model_wait_checks=max(
                          1, int(args.max_wait_minutes * 60 / args.wait_seconds)))
        asf.wait_for_model_files(list(expected.values()), cfg)
        arrays = {p: sensitivities_for_parameter(lines, paths)
                  for p, paths in expected.items()}
        write_sensitivities(arrays)
    select_diagnostics(lines, arrays)
    print("\nCompleted. Outputs:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
