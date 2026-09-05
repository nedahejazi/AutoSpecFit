#!/usr/bin/env python3

"""
AutoSpecNorm local normalization example for GJ 205.

The scientific AutoSpecNorm calculation uses the current GJ 205 inputs
and explicit Python smoothing adopted for the working K example.

The plotting follows the original GitHub three-panel example.
"""

from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# AutoSpecFit source directory
# ============================================================================

autospecfit_src = Path(
    "/Users/neda/Desktop/AutoSpecFit/src"
)

if str(autospecfit_src) not in sys.path:
    sys.path.insert(
        0,
        str(autospecfit_src),
    )


from AutoSpecNorm_Regions import AutoSpecNorm_Regions


# ============================================================================
# User configuration
# ============================================================================

observed_spectrum_file = Path(
    "IGRINS_H_K_band_Flattened_Spectrum_GJ205.txt"
)

line_list_file = Path(
    "All_Species_Fit_Ranges_GJ205.txt"
)

model_spectrum_file = Path(
    "/Users/neda/Desktop/Mdwarf_Benchmark_June_25/"
    "Files_models_codes_July_2026/"
    "Synthetic_Spectra_APOGEE/"
    "t3700_g+4.7_z+0.20_a+0.00_v1.00.mod"
)

# Zero-based line index.
selected_line_index = 11

observed_wavelength_scale = 1.0e4

# Python Gaussian smoothing value.
gaussian_sigma_pixels = 2.3

lam_diff_large = 3.0
iteration_number = 25

limit_model = 0.95
limit_star = 0.95

peak_index_model = 0.0019
peak_index_star = 0.0019


# ============================================================================
# Plot configuration
# ============================================================================

output_figure_file = Path(
    "AutoSpecNorm_Performance_GJ205.png"
)

save_figure = True
show_figure = True

# Same normalization-point size as the original GitHub plotting example.
marker_size = 70.0


# ============================================================================
# Gaussian smoothing
# ============================================================================

def gaussian_smooth_nearest(
    flux,
    sigma_pixels,
):
    """
    Explicit Gaussian smoothing used in the working Python K calculation.
    """

    flux = np.asarray(
        flux,
        dtype=float,
    ).reshape(-1)

    radius = int(
        np.ceil(
            4.0 * sigma_pixels
        )
    )

    kernel_x = np.arange(
        -radius,
        radius + 1,
        dtype=float,
    )

    kernel = np.exp(
        -0.5
        * (
            kernel_x
            / sigma_pixels
        ) ** 2
    )

    kernel = (
        kernel
        / np.sum(kernel)
    )

    padded_flux = np.concatenate(
        (
            np.full(
                radius,
                flux[0],
                dtype=float,
            ),
            flux,
            np.full(
                radius,
                flux[-1],
                dtype=float,
            ),
        )
    )

    smoothed_padded = np.convolve(
        padded_flux,
        kernel,
        mode="same",
    )

    return smoothed_padded[
        radius:-radius
    ]


# ============================================================================
# Read numerical table
# ============================================================================

def read_numeric_table(
    filename,
    minimum_columns,
):
    """
    Read a whitespace-delimited numerical text file.
    """

    data = np.genfromtxt(
        filename,
        dtype=float,
        comments="#",
    )

    if data.ndim == 1:
        data = data.reshape(
            1,
            -1,
        )

    if data.shape[1] < minimum_columns:
        raise ValueError(
            f"{filename} must contain at least "
            f"{minimum_columns} columns."
        )

    return data


# ============================================================================
# Read six-column line list
# ============================================================================

def read_line_list(
    filename,
):
    """
    Read the six-column AutoSpecFit line list.

    Columns:
        1. line number
        2. species
        3. line center
        4. RV
        5. minimum fit wavelength
        6. maximum fit wavelength

    Header rows are ignored automatically.
    """

    species = []
    line_centers = []
    radial_velocities = []
    fit_minima = []
    fit_maxima = []

    with open(
        filename,
        "r",
    ) as handle:

        for line in handle:

            stripped = line.strip()

            if not stripped:
                continue

            if stripped.startswith("#"):
                continue

            parts = stripped.split()

            if len(parts) < 6:
                continue

            try:

                line_center = float(
                    parts[2]
                )

                radial_velocity = float(
                    parts[3]
                )

                fit_minimum = float(
                    parts[4]
                )

                fit_maximum = float(
                    parts[5]
                )

            except ValueError:
                continue

            species.append(
                parts[1]
            )

            line_centers.append(
                line_center
            )

            radial_velocities.append(
                radial_velocity
            )

            fit_minima.append(
                fit_minimum
            )

            fit_maxima.append(
                fit_maximum
            )

    if not line_centers:
        raise ValueError(
            f"No valid line-list entries were found in: {filename}"
        )

    return (
        np.asarray(
            species,
            dtype=str,
        ),
        np.asarray(
            line_centers,
            dtype=float,
        ),
        np.asarray(
            radial_velocities,
            dtype=float,
        ),
        np.asarray(
            fit_minima,
            dtype=float,
        ),
        np.asarray(
            fit_maxima,
            dtype=float,
        ),
    )


# ============================================================================
# Read observed spectrum
# ============================================================================

observed_data = read_numeric_table(
    observed_spectrum_file,
    minimum_columns=3,
)

lam_star = (
    observed_data[:, 0]
    * observed_wavelength_scale
)

flux_star = observed_data[:, 1]
err_flux_star = observed_data[:, 2]

valid_observed = (
    np.isfinite(lam_star)
    & np.isfinite(flux_star)
    & np.isfinite(err_flux_star)
)

lam_star = lam_star[
    valid_observed
]

flux_star = flux_star[
    valid_observed
]

err_flux_star = err_flux_star[
    valid_observed
]


# ============================================================================
# Read line list
# ============================================================================

(
    species,
    line_centers,
    radial_velocities,
    fit_minima,
    fit_maxima,
) = read_line_list(
    line_list_file
)

if (
    selected_line_index < 0
    or selected_line_index >= len(line_centers)
):
    raise IndexError(
        "selected_line_index is outside "
        "the available line-list range."
    )

species_name = (
    species[
        selected_line_index
    ]
)

central_line = (
    line_centers[
        selected_line_index
    ]
)

radial_velocity = (
    radial_velocities[
        selected_line_index
    ]
)

fit_minimum = (
    fit_minima[
        selected_line_index
    ]
)

fit_maximum = (
    fit_maxima[
        selected_line_index
    ]
)


# ============================================================================
# Read and smooth synthetic spectrum
# ============================================================================

model_data = read_numeric_table(
    model_spectrum_file,
    minimum_columns=2,
)

lam_model = model_data[:, 0]
flux_model = model_data[:, 1]

valid_model = (
    np.isfinite(lam_model)
    & np.isfinite(flux_model)
)

lam_model = lam_model[
    valid_model
]

flux_model = flux_model[
    valid_model
]

# Keep the same smoothing used by the working Python K calculation.
flux_model_smoothed = gaussian_smooth_nearest(
    flux_model,
    gaussian_sigma_pixels,
)


# ============================================================================
# Run AutoSpecNorm
# ============================================================================

(
    lam_star_cut,
    flux_star_cut,
    flux_interp_model_cut,
    flux_star_cut_normalized_final,
    upper_flux_star_cut_normalized_final,
    lower_flux_star_cut_normalized_final,
    err_flux_star_cut_normalized_final,
    lam_cut_work,
    flux_star_cut_work,
    flux_interp_model_cut_work,
    delta,
    flag,
) = AutoSpecNorm_Regions(
    central_line,
    lam_diff_large,
    radial_velocity,
    lam_star,
    flux_star,
    err_flux_star,
    lam_model,
    flux_model_smoothed,
    limit_model,
    limit_star,
    peak_index_model,
    peak_index_star,
    iteration_number,
)

if flag == 0:

    warnings.warn(
        "AutoSpecNorm did not find reliable normalization points.",
        RuntimeWarning,
    )


# ============================================================================
# Normalization quantities
# ============================================================================

normalization_function = (
    flux_star_cut
    / flux_star_cut_normalized_final
)

normalization_difference = (
    flux_star_cut_normalized_final
    - flux_star_cut
)


# ============================================================================
# Original GitHub plotting style
# ============================================================================

# --------------------------------------------------------------------------
# Colors
# --------------------------------------------------------------------------

color_observed_raw = "0.45"

color_observed_normalized = "tab:blue"

color_model = "tab:red"

color_points = "chartreuse"

color_points_edge = "darkgreen"

color_uncertainty = "lightsteelblue"

color_normalization_function = "tab:green"

color_flux_difference = "tab:purple"

color_line_marker = "black"

color_fit_region = "orange"


# --------------------------------------------------------------------------
# Figure layout
# --------------------------------------------------------------------------

fig, (
    ax_spectrum,
    ax_correction,
    ax_difference,
) = plt.subplots(
    3,
    1,
    figsize=(
        9.2,
        7.6,
    ),
    sharex=True,
    gridspec_kw={
        "height_ratios": [
            3.0,
            1.0,
            1.0,
        ]
    },
)

fig.subplots_adjust(
    left=0.12,
    right=0.96,
    top=0.93,
    bottom=0.09,
    hspace=0.10,
)


# ============================================================================
# TOP PANEL
# ============================================================================

# --------------------------------------------------------------------------
# Chi-square fitting window
# --------------------------------------------------------------------------

ax_spectrum.axvspan(
    fit_minimum,
    fit_maximum,
    color=color_fit_region,
    alpha=0.18,
    label="Chi-square fitting window",
    zorder=0,
)


# --------------------------------------------------------------------------
# Normalization uncertainty interval
# --------------------------------------------------------------------------

ax_spectrum.fill_between(
    lam_star_cut,
    lower_flux_star_cut_normalized_final,
    upper_flux_star_cut_normalized_final,
    color=color_uncertainty,
    alpha=0.55,
    label="Normalization uncertainty interval",
)


# --------------------------------------------------------------------------
# Observed spectrum before local normalization
# --------------------------------------------------------------------------

ax_spectrum.plot(
    lam_star_cut,
    flux_star_cut,
    color=color_observed_raw,
    linewidth=1.2,
    linestyle="--",
    label="Observed spectrum before local normalization",
)


# --------------------------------------------------------------------------
# Observed spectrum after local normalization
# --------------------------------------------------------------------------

ax_spectrum.plot(
    lam_star_cut,
    flux_star_cut_normalized_final,
    color=color_observed_normalized,
    linewidth=1.5,
    label="Observed spectrum after local normalization",
)


# --------------------------------------------------------------------------
# Synthetic spectrum
# --------------------------------------------------------------------------

ax_spectrum.plot(
    lam_star_cut,
    flux_interp_model_cut,
    color=color_model,
    linewidth=1.4,
    label="Synthetic spectrum",
)


# --------------------------------------------------------------------------
# Normalization-point halo
# --------------------------------------------------------------------------

ax_spectrum.scatter(
    lam_cut_work,
    flux_interp_model_cut_work,
    s=marker_size * 2.2,
    marker="o",
    facecolors=color_points,
    edgecolors="none",
    alpha=0.25,
    zorder=4,
)


# --------------------------------------------------------------------------
# Selected normalization points
# --------------------------------------------------------------------------

ax_spectrum.scatter(
    lam_cut_work,
    flux_interp_model_cut_work,
    s=marker_size,
    marker="o",
    facecolors=color_points,
    edgecolors=color_points_edge,
    linewidths=1.2,
    label="Selected normalization points",
    zorder=5,
)


# --------------------------------------------------------------------------
# Spectral-line marker
# --------------------------------------------------------------------------

spectrum_ymin_tmp, spectrum_ymax_tmp = (
    ax_spectrum.get_ylim()
)

line_top = (
    spectrum_ymax_tmp
    - 0.18
    * (
        spectrum_ymax_tmp
        - spectrum_ymin_tmp
    )
)

ax_spectrum.vlines(
    central_line,
    ymin=spectrum_ymin_tmp,
    ymax=line_top,
    colors=color_line_marker,
    linestyles=":",
    linewidth=1.3,
    zorder=4,
)


# --------------------------------------------------------------------------
# Axis labels and title
# --------------------------------------------------------------------------

ax_spectrum.set_ylabel(
    "Flux"
)

ax_spectrum.set_title(
    "AutoSpecNorm Local Normalization"
)


# --------------------------------------------------------------------------
# X-axis limits
# --------------------------------------------------------------------------

plot_minimum = (
    np.nanmin(
        lam_cut_work
    )
    - 0.5
)

plot_maximum = (
    np.nanmax(
        lam_cut_work
    )
    + 0.5
)

ax_spectrum.set_xlim(
    plot_minimum,
    plot_maximum,
)


# --------------------------------------------------------------------------
# Legend
# --------------------------------------------------------------------------

ax_spectrum.legend(
    fontsize=8,
    loc="lower left",
    frameon=True,
)


# --------------------------------------------------------------------------
# Tick-number formatting
# --------------------------------------------------------------------------

ax_spectrum.tick_params(
    axis="both",
    which="major",
    labelsize=9,
)


# --------------------------------------------------------------------------
# Y-axis limits
# --------------------------------------------------------------------------

y_values = np.concatenate(
    [
        flux_star_cut,
        flux_star_cut_normalized_final,
        flux_interp_model_cut,
        lower_flux_star_cut_normalized_final,
        upper_flux_star_cut_normalized_final,
    ]
)

finite_y = y_values[
    np.isfinite(
        y_values
    )
]

if finite_y.size > 0:

    y_min = max(
        0.0,
        np.nanmin(
            finite_y
        )
        - 0.05,
    )

    y_max = (
        np.nanmax(
            finite_y
        )
        + 0.05
    )

    ax_spectrum.set_ylim(
        y_min,
        y_max,
    )


# --------------------------------------------------------------------------
# Species annotation
# --------------------------------------------------------------------------

spectrum_ymin, spectrum_ymax = (
    ax_spectrum.get_ylim()
)

label_y = (
    spectrum_ymax
    - 0.08
    * (
        spectrum_ymax
        - spectrum_ymin
    )
)

ax_spectrum.text(
    central_line,
    label_y,
    species_name,
    ha="center",
    va="center",
    fontsize=11,
    fontweight="bold",
    bbox=dict(
        facecolor="white",
        edgecolor="black",
        alpha=0.85,
        pad=2,
    ),
    zorder=6,
)


# ============================================================================
# MIDDLE PANEL
# ============================================================================

ax_correction.plot(
    lam_star_cut,
    normalization_function,
    color=color_normalization_function,
    linewidth=1.5,
    label="Pseudo-continuum correction factor",
)

finite_correction = (
    normalization_function[
        np.isfinite(
            normalization_function
        )
    ]
)

if finite_correction.size > 0:

    correction_padding = max(
        0.002,
        0.15
        * (
            np.nanmax(
                finite_correction
            )
            -
            np.nanmin(
                finite_correction
            )
        ),
    )

    ax_correction.set_ylim(
        np.nanmin(
            finite_correction
        )
        - correction_padding,
        np.nanmax(
            finite_correction
        )
        + correction_padding,
    )

ax_correction.set_ylabel(
    "Correction factor"
)

ax_correction.legend(
    fontsize=8,
    loc="upper left",
    frameon=True,
)

ax_correction.tick_params(
    axis="both",
    which="major",
    labelsize=9,
)


# ============================================================================
# BOTTOM PANEL
# ============================================================================

ax_difference.plot(
    lam_star_cut,
    normalization_difference,
    color=color_flux_difference,
    linewidth=1.4,
    linestyle="--",
    label="Difference (Normalized - Original)",
)

finite_difference = (
    normalization_difference[
        np.isfinite(
            normalization_difference
        )
    ]
)

if finite_difference.size > 0:

    difference_padding = max(
        0.002,
        0.15
        * (
            np.nanmax(
                finite_difference
            )
            -
            np.nanmin(
                finite_difference
            )
        ),
    )

    ax_difference.set_ylim(
        np.nanmin(
            finite_difference
        )
        - difference_padding,
        np.nanmax(
            finite_difference
        )
        + difference_padding,
    )

ax_difference.set_xlabel(
    "Wavelength (Angstrom)"
)

ax_difference.set_ylabel(
    "Normalized - Original"
)

ax_difference.legend(
    fontsize=8,
    loc="lower left",
    frameon=True,
)

ax_difference.tick_params(
    axis="both",
    which="major",
    labelsize=9,
)


# ============================================================================
# Align y-axis labels
# ============================================================================

label_x = -0.06

ax_spectrum.yaxis.set_label_coords(
    label_x,
    0.5,
)

ax_correction.yaxis.set_label_coords(
    label_x,
    0.5,
)

ax_difference.yaxis.set_label_coords(
    label_x,
    0.5,
)


# ============================================================================
# Save figure
# ============================================================================

if save_figure:

    fig.savefig(
        output_figure_file,
        dpi=300,
        bbox_inches="tight",
    )


# ============================================================================
# Show figure
# ============================================================================

if show_figure:

    plt.show()

else:

    plt.close(
        fig
    )