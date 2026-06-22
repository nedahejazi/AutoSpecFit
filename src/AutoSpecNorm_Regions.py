"""AutoSpecNorm adaptive analysis-region selection.

This module contains the adaptive analysis-region routine used by
AutoSpecFit. The routine calls ``AutoSpecNorm_Points`` to identify
local pseudo-continuum points and, when necessary, iteratively expands the
analysis region and relaxes the model/observed flux thresholds.

The goal is to obtain a sufficient number of reliable normalization points on
both sides of a selected diagnostic line before performing the local
pseudo-continuum normalization.
"""

import numpy as np

from AutoSpecNorm_Points import AutoSpecNorm_Points


def _split_points_by_line_center(wavelength_points, central_line):
    """Return normalization points located to the left and right of a line."""
    left_points = wavelength_points[wavelength_points < central_line]
    right_points = wavelength_points[wavelength_points > central_line]
    return left_points, right_points


def _select_polynomial_order(wavelength_points):
    """Choose the polynomial order from the number and spacing of points.

    The order is intentionally kept low. Sparse or unevenly distributed
    normalization points use a lower-order polynomial to avoid unstable fits,
    while denser and more uniformly distributed points can support a higher
    order.
    """
    max_gap = np.max(np.diff(wavelength_points))

    if len(wavelength_points) <= 3 or max_gap >= 15:
        return 1

    if len(wavelength_points) == 4 and max_gap >= 10:
        return 1

    if len(wavelength_points) == 4 and max_gap < 10:
        return 2

    if len(wavelength_points) > 4 and max_gap >= 3:
        return 2

    if len(wavelength_points) > 4 and max_gap < 3:
        return 3

    return 1


def _remove_deepest_point_if_safe(
    central_line,
    lam_cut_work,
    flux_star_cut_work,
    flux_interp_model_cut_work,
):
    """Remove the deepest selected point if both sides remain constrained."""
    if np.min(flux_star_cut_work) < np.min(flux_interp_model_cut_work):
        index_min_removal = flux_star_cut_work == np.min(flux_star_cut_work)
    else:
        index_min_removal = flux_interp_model_cut_work == np.min(
            flux_interp_model_cut_work
        )

    lam_cut_work_new = lam_cut_work[~index_min_removal]
    flux_interp_model_cut_work_new = flux_interp_model_cut_work[~index_min_removal]
    flux_star_cut_work_new = flux_star_cut_work[~index_min_removal]

    left_points, right_points = _split_points_by_line_center(
        lam_cut_work_new,
        central_line,
    )

    if len(left_points) > 0 and len(right_points) > 0:
        return (
            lam_cut_work_new,
            flux_interp_model_cut_work_new,
            flux_star_cut_work_new,
        )

    return lam_cut_work, flux_interp_model_cut_work, flux_star_cut_work


def _remove_boundary_point_if_safe(
    central_line,
    lam_star_cut,
    flux_star_cut,
    flux_interp_model_cut,
    lam_cut_work,
    flux_star_cut_work,
    flux_interp_model_cut_work,
):
    """Remove a selected point if it lies exactly at an analysis-region edge."""
    left_points, right_points = _split_points_by_line_center(
        lam_cut_work,
        central_line,
    )

    if (
        lam_star_cut[0] in lam_cut_work
        and len(left_points) > 1
        and len(lam_cut_work) >= 3
    ):
        lam_cut_work = lam_cut_work[lam_cut_work != lam_star_cut[0]]
        flux_interp_model_cut_work = flux_interp_model_cut_work[
            flux_interp_model_cut_work != flux_interp_model_cut[0]
        ]
        flux_star_cut_work = flux_star_cut_work[
            flux_star_cut_work != flux_star_cut[0]
        ]

    elif (
        lam_star_cut[-1] in lam_cut_work
        and len(right_points) > 1
        and len(lam_cut_work) >= 3
    ):
        lam_cut_work = lam_cut_work[lam_cut_work != lam_star_cut[-1]]
        flux_interp_model_cut_work = flux_interp_model_cut_work[
            flux_interp_model_cut_work != flux_interp_model_cut[-1]
        ]
        flux_star_cut_work = flux_star_cut_work[
            flux_star_cut_work != flux_star_cut[-1]
        ]

    return lam_cut_work, flux_interp_model_cut_work, flux_star_cut_work


def _sigma_reject_normalization_points(
    central_line,
    lam_cut_work,
    flux_star_cut_work,
    flux_interp_model_cut_work,
    n_rejection_iterations=1,
):
    """Reject normalization points using the observed/model flux ratio."""
    sigma_values = [2.0, 1.5, 1.0]

    if len(lam_cut_work) <= 3:
        return lam_cut_work, flux_interp_model_cut_work, flux_star_cut_work

    for rejection_index in range(n_rejection_iterations):
        polynomial_order = _select_polynomial_order(lam_cut_work)
        sigma = sigma_values[rejection_index]

        residual = flux_star_cut_work / flux_interp_model_cut_work
        polynomial = np.polyfit(lam_cut_work, residual, polynomial_order)
        residual_fit = np.polyval(polynomial, lam_cut_work)

        rmse = np.sqrt(np.sum((residual - residual_fit) ** 2) / residual.size)
        keep = np.abs(residual - residual_fit) < sigma * rmse

        lam_cut_work_new = lam_cut_work[keep]
        flux_interp_model_cut_work_new = flux_interp_model_cut_work[keep]
        flux_star_cut_work_new = flux_star_cut_work[keep]

        left_points, right_points = _split_points_by_line_center(
            lam_cut_work_new,
            central_line,
        )

        if len(left_points) == 0 or len(right_points) == 0 or len(lam_cut_work_new) <= 2:
            break

        lam_cut_work = lam_cut_work_new
        flux_interp_model_cut_work = flux_interp_model_cut_work_new
        flux_star_cut_work = flux_star_cut_work_new

    return lam_cut_work, flux_interp_model_cut_work, flux_star_cut_work


def _prediction_interval_delta(x_fit, x_eval, residual, polynomial_order):
    """Return the polynomial prediction uncertainty evaluated on ``x_eval``."""
    vandermonde_fit = np.vander(x_fit, polynomial_order + 1)
    _, upper_triangular = np.linalg.qr(vandermonde_fit)

    degrees_of_freedom = len(x_fit) - (polynomial_order + 1)
    fit_values = np.polyval(np.polyfit(x_fit, residual, polynomial_order), x_fit)
    residual_norm = np.linalg.norm(residual - fit_values)

    vandermonde_eval = np.vander(x_eval, polynomial_order + 1)

    try:
        projection = np.linalg.solve(upper_triangular.T, vandermonde_eval.T).T
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = np.sqrt(1 + np.sum(projection**2, axis=1)) * (
                residual_norm / np.sqrt(degrees_of_freedom)
            )
    except np.linalg.LinAlgError:
        delta = np.full(x_eval.shape, np.nan)

    return delta


def AutoSpecNorm_Regions(
    central_line,
    lam_diff_large,
    RV,
    lam_star,
    flux_star,
    err_flux_star,
    lam_model,
    flux_model,
    limit_model,
    limit_star,
    peak_index_model,
    peak_index_star,
    iteration_number,
):
    """Adaptively select normalization regions and normalize the observed flux.

    This function controls the adaptive region-selection stage of AutoSpecNorm.
    Starting from an initial analysis region centered on a diagnostic spectral
    line, it repeatedly calls ``AutoSpecNorm_Points`` to identify
    candidate pseudo-continuum points.

    If too few normalization points are found, or if points are missing on one
    or both sides of the diagnostic line, the function expands the analysis
    region. When necessary, it also relaxes ``limit_model`` and ``limit_star``
    so that additional candidate pseudo-continuum points can be considered.

    After suitable points are identified, the observed/model flux ratio is
    fitted with a low-order polynomial. This polynomial is used to normalize
    the observed spectrum locally relative to the synthetic spectrum being
    analyzed.

    In ASF, AutoSpecNorm is applied independently to every synthetic
    spectrum tested during the chi-square fitting procedure. In cool-star
    spectra, changes in elemental abundance can alter the strengths of
    nearby atomic and molecular features and therefore modify the local
    pseudo-continuum. Re-evaluating the normalization for each synthetic
    spectrum allows the pseudo-continuum placement to adapt to these
    abundance-dependent spectral changes and ensures a self-consistent
    comparison between the observed and synthetic spectra.

    The corresponding normalization uncertainty is returned and later
    propagated into the ASF chi-square calculation.

    Both ASF and ASN generally perform best when a flattened observed
    spectrum is compared with a continuum-normalized synthetic spectrum
    generated by Turbospectrum. Under these conditions, the local normalization
    procedure mainly accounts for residual pseudo-continuum differences between
     the observed and synthetic spectra, leading to more reliable chi-square
     measurements and, consequently, more accurate abundance determinations.
    While ASF can often operate successfully under less ideal conditions, the
    use of a flattened observed spectrum and continuum-normalized synthetic spectra
    is strongly recommended whenever possible.

    Parameters
    ----------
    central_line : float
        Central wavelength of the diagnostic spectral line.
    lam_diff_large : float
        Initial half-width of the analysis region around ``central_line``.
    RV : float
        Stellar radial velocity in km/s.
    lam_star, flux_star, err_flux_star : ndarray
        Observed wavelength, flux, and flux uncertainty arrays.
    lam_model, flux_model : ndarray
        Synthetic wavelength and flux arrays.
    limit_model, limit_star : float
        Flux thresholds used by ``AutoSpecNorm_Points`` when
        selecting candidate pseudo-continuum points.
    peak_index_model, peak_index_star : float
        Tolerance parameters used by the model and observed-spectrum
        normalization-point selection criteria.
    iteration_number : int
        Maximum number of adaptive region/threshold iterations.

    Returns
    -------
    tuple
        Arrays containing the extracted analysis-region spectra, normalized
        observed flux, upper/lower normalized flux estimates, normalized flux
        uncertainty, selected normalization points, normalization uncertainty
        ``delta``, and a quality flag. The flag is 1 when usable
        normalization points exist on both sides of the line and 0 otherwise.
    """

    # Start with a symmetric analysis region around the selected line.
    lam_analysis = np.array(
        [central_line - lam_diff_large, central_line + lam_diff_large],
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Adaptive Analysis-Region and Threshold Loop
    # ------------------------------------------------------------------
    #
    # At each iteration, the current analysis region is tested with
    # AutoSpecNorm_Points. If the selected normalization points are
    # insufficient, the analysis region is expanded and/or the selection
    # thresholds are relaxed.

    for norm_iter_total in range(iteration_number):
        lam_analysis_new = None

        # First pass: identify candidate normalization points in the current
        # analysis region.
        (
            lam_star_cut,
            _,
            _,
            _,
            lam_cut_work,
            _,
            _,
        ) = AutoSpecNorm_Points(
            RV,
            lam_analysis,
            lam_star,
            flux_star,
            err_flux_star,
            lam_model,
            flux_model,
            limit_model,
            limit_star,
            peak_index_model,
            peak_index_star,
        )

        # Partition the selected normalization points into the left and right
        # sides of the diagnostic line. The distribution of normalization
        # points relative to the line center is used throughout the adaptive
        # procedure to determine whether the analysis region should be expanded
        # and whether the final normalization is sufficiently constrained.
        lam_cut_work_left, lam_cut_work_right = _split_points_by_line_center(
            lam_cut_work,
            central_line,
        )

        # Evaluate the left/right wavelength coverage of the current analysis
        # region relative to the central line.
        range_analysis = lam_analysis[1] - lam_analysis[0]
        range_analysis_left = central_line - lam_analysis[0]
        range_analysis_right = lam_analysis[1] - central_line

        # Expand the analysis region when normalization points are missing
        # from one or both sides of the diagnostic line, or when too few points
        # are available overall. The algorithm attempts to maintain a balanced
        # distribution of normalization points and analysis-region coverage on
        # both sides of the diagnostic line, while preferentially expanding
        # toward the side that requires additional normalization points.
        if len(lam_cut_work_left) > 0 and len(lam_cut_work_right) == 0 and range_analysis <= 60:
            lam_diff = lam_diff_large / 2.0
            if range_analysis_left > range_analysis_right:
                lam_analysis_new = np.array([lam_analysis[0], lam_analysis[1] + lam_diff])
            else:
                lam_analysis_new = np.array(
                    [lam_analysis[0] - (lam_diff / 2.0), lam_analysis[1] + lam_diff]
                )

        elif len(lam_cut_work_left) == 0 and len(lam_cut_work_right) > 0 and range_analysis <= 60:
            lam_diff = lam_diff_large / 2.0
            if range_analysis_left < range_analysis_right:
                lam_analysis_new = np.array([lam_analysis[0] - lam_diff, lam_analysis[1]])
            else:
                lam_analysis_new = np.array(
                    [lam_analysis[0] - lam_diff, lam_analysis[1] + (lam_diff / 2.0)]
                )

        elif len(lam_cut_work_left) == 0 and len(lam_cut_work_right) == 0 and range_analysis <= 60:
            lam_diff = lam_diff_large / 2.0
            if range_analysis_left > range_analysis_right:
                lam_analysis_new = np.array(
                    [lam_analysis[0] - (lam_diff / 2.0), lam_analysis[1] + lam_diff]
                )
            elif range_analysis_left < range_analysis_right:
                lam_analysis_new = np.array(
                    [lam_analysis[0] - lam_diff, lam_analysis[1] + (lam_diff / 2.0)]
                )
            else:
                lam_analysis_new = np.array(
                    [lam_analysis[0] - lam_diff, lam_analysis[1] + lam_diff]
                )

        elif (
            len(lam_cut_work_left) > 0
            and len(lam_cut_work_right) > 0
            and len(lam_cut_work) < 3
            and range_analysis <= 50
        ):
            lam_diff = lam_diff_large / 2.0
            if range_analysis_left > range_analysis_right:
                lam_analysis_new = np.array(
                    [lam_analysis[0] - (lam_diff / 2.0), lam_analysis[1] + lam_diff]
                )
            elif range_analysis_left < range_analysis_right:
                lam_analysis_new = np.array(
                    [lam_analysis[0] - lam_diff, lam_analysis[1] + (lam_diff / 2.0)]
                )
            else:
                lam_analysis_new = np.array(
                    [lam_analysis[0] - lam_diff, lam_analysis[1] + lam_diff]
                )

        if (
            len(lam_cut_work_left) > 0
            and len(lam_cut_work_right) > 0
            and 3 <= len(lam_cut_work) <= 8
            and range_analysis <= 50
        ):
            lam_diff = lam_diff_large / 2.0

            if len(lam_cut_work_left) > len(lam_cut_work_right):
                if range_analysis_left > range_analysis_right:
                    lam_analysis_new = np.array([lam_analysis[0], lam_analysis[1] + lam_diff])
                else:
                    lam_analysis_new = np.array(
                        [lam_analysis[0] - (lam_diff / 2.0), lam_analysis[1] + lam_diff]
                    )

            elif len(lam_cut_work_right) > len(lam_cut_work_left):
                if range_analysis_left < range_analysis_right:
                    lam_analysis_new = np.array([lam_analysis[0] - lam_diff, lam_analysis[1]])
                else:
                    lam_analysis_new = np.array(
                        [lam_analysis[0] - lam_diff, lam_analysis[1] + (lam_diff / 2.0)]
                    )

            else:
                if range_analysis_left > range_analysis_right:
                    lam_analysis_new = np.array(
                        [lam_analysis[0] - (lam_diff / 2.0), lam_analysis[1] + lam_diff]
                    )
                elif range_analysis_left < range_analysis_right:
                    lam_analysis_new = np.array(
                        [lam_analysis[0] - lam_diff, lam_analysis[1] + (lam_diff / 2.0)]
                    )
                else:
                    lam_analysis_new = np.array(
                        [lam_analysis[0] - lam_diff, lam_analysis[1] + lam_diff]
                    )

        # If selected points lie at the edges of the analysis region, expand
        # the region further so that the final normalization is not anchored by
        # boundary points alone.
        if (
            norm_iter_total < iteration_number - 1
            and len(lam_cut_work_left) > 0
            and len(lam_cut_work_right) > 0
            and 3 <= len(lam_cut_work) <= 10
            and range_analysis <= 50
        ):
            lam_diff = 1.0

            if lam_star_cut[0] not in lam_cut_work_left and lam_star_cut[-1] in lam_cut_work_right:
                if lam_analysis_new is None:
                    lam_analysis_new = np.array([lam_analysis[0], lam_analysis[1] + lam_diff])
                else:
                    lam_analysis_new = lam_analysis_new + np.array([0.0, lam_diff])

            elif lam_star_cut[0] in lam_cut_work_left and lam_star_cut[-1] not in lam_cut_work_right:
                if lam_analysis_new is None:
                    lam_analysis_new = np.array([lam_analysis[0] - lam_diff, lam_analysis[1]])
                else:
                    lam_analysis_new = lam_analysis_new + np.array([-lam_diff, 0.0])

            elif lam_star_cut[0] in lam_cut_work_left and lam_star_cut[-1] in lam_cut_work_right:
                if lam_analysis_new is None:
                    lam_analysis_new = np.array(
                        [lam_analysis[0] - lam_diff, lam_analysis[1] + lam_diff]
                    )
                else:
                    lam_analysis_new = lam_analysis_new + np.array([-lam_diff, lam_diff])

        # Apply the updated analysis region, if one was defined during this
        # adaptive iteration.
        if lam_analysis_new is not None:
            lam_analysis = lam_analysis_new

        # Second pass: re-evaluate the selected points after any region
        # expansion before deciding whether to relax the flux thresholds.
        (
            _,
            _,
            _,
            _,
            lam_cut_work,
            _,
            _,
        ) = AutoSpecNorm_Points(
            RV,
            lam_analysis,
            lam_star,
            flux_star,
            err_flux_star,
            lam_model,
            flux_model,
            limit_model,
            limit_star,
            peak_index_model,
            peak_index_star,
        )

        lam_cut_work_left, lam_cut_work_right = _split_points_by_line_center(
            lam_cut_work,
            central_line,
        )

        # If the expanded region still does not provide enough normalization
        # points on both sides of the line, gradually relax the model and
        # observed flux thresholds.
        limit_model_new = None
        limit_star_new = None

        if norm_iter_total < iteration_number - 1 and limit_model >= 0.92 and limit_star >= 0.92:
            if (
                (len(lam_cut_work_left) == 0 or len(lam_cut_work_right) == 0)
                and (lam_analysis[1] - lam_analysis[0]) <= 55
            ):
                limit_model_new = limit_model - 0.01
                limit_star_new = limit_star - 0.01

            if (
                len(lam_cut_work_left) > 0
                and len(lam_cut_work_right) > 0
                and len(lam_cut_work) < 3
                and (lam_analysis[1] - lam_analysis[0]) <= 50
            ):
                limit_model_new = limit_model - 0.01
                limit_star_new = limit_star - 0.01

        if norm_iter_total < iteration_number - 1 and limit_model >= 0.95 and limit_star >= 0.95:
            if (
                len(lam_cut_work_left) > 0
                and len(lam_cut_work_right) > 0
                and 2 <= len(lam_cut_work) <= 4
                and (lam_analysis[1] - lam_analysis[0]) <= 40
            ):
                limit_model_new = limit_model - 0.005
                limit_star_new = limit_star - 0.005

        if limit_model_new is not None:
            limit_model = limit_model_new
            limit_star = limit_star_new

        # Third pass: obtain the full set of arrays using the current
        # analysis region and thresholds.
        (
            lam_star_cut,
            flux_star_cut,
            err_flux_star_cut,
            flux_interp_model_cut,
            lam_cut_work,
            flux_star_cut_work,
            flux_interp_model_cut_work,
        ) = AutoSpecNorm_Points(
            RV,
            lam_analysis,
            lam_star,
            flux_star,
            err_flux_star,
            lam_model,
            flux_model,
            limit_model,
            limit_star,
            peak_index_model,
            peak_index_star,
        )

        lam_cut_work_left, lam_cut_work_right = _split_points_by_line_center(
            lam_cut_work,
            central_line,
        )

        # Stop expanding once enough normalization points are found on both
        # sides of the line and the analysis region is sufficiently broad.
        if (
            len(lam_cut_work_left) > 0
            and len(lam_cut_work_right) > 0
            and len(lam_cut_work) >= 6
            and (lam_analysis[1] - lam_analysis[0]) >= 30
        ):
            if lam_star_cut[0] not in lam_cut_work_left and lam_star_cut[-1] not in lam_cut_work_right:
                break

    # ------------------------------------------------------------------
    # Optional Removal of the Deepest Selected Normalization Point
    # ------------------------------------------------------------------
    #
    # This safeguard can be used to exclude the deepest selected normalization
    # point when a sufficiently large number of normalization points is
    # available. In some cases, an unusually low point may adversely affect the
    # local continuum fit and the resulting normalization.
    #
    # The threshold below controls when this safeguard is activated. A
    # sufficiently large value effectively disables it. Users may adjust the
    # threshold according to their normalization strategy and data quality.
    #
    # Any removal is only performed if normalization points remain on both
    # sides of the diagnostic line.
    if len(flux_star_cut_work) > 10000:
        (
            lam_cut_work,
            flux_interp_model_cut_work,
            flux_star_cut_work,
        ) = _remove_deepest_point_if_safe(
            central_line,
            lam_cut_work,
            flux_star_cut_work,
            flux_interp_model_cut_work,
        )

    # ------------------------------------------------------------------
    # Final Quality-Control Check for Analysis-Region Boundary Points
    # ------------------------------------------------------------------
    #
    # A final quality-control check is performed to remove normalization
    # points located exactly at the boundaries of the analysis region.
    # Although possible edge points are already searched for and removed during
    # the adaptive region-selection procedure, this additional safeguard
    # provides a final verification that no boundary points remain in the
    # normalization-point set. Such points are generally less reliable because
    # no wavelength information exists beyond the analysis-region boundaries,
    # making it impossible to determine whether they are truly local maxima
    # or near maxima, or simply appear as maxima due to truncation of the
    # surrounding wavelength range. Any removal is only performed if sufficient
    # normalization points remain on both sides of the diagnostic line.
    (
        lam_cut_work,
        flux_interp_model_cut_work,
        flux_star_cut_work,
    ) = _remove_boundary_point_if_safe(
        central_line,
        lam_star_cut,
        flux_star_cut,
        flux_interp_model_cut,
        lam_cut_work,
        flux_star_cut_work,
        flux_interp_model_cut_work,
    )

    # ------------------------------------------------------------------
    # Sigma Rejection Using a Polynomial Fit to the Observed/Model Ratio
    # ------------------------------------------------------------------
    #
    # This section is executed only when more than three normalization points
    # are available. With fewer points, alternative procedures are applied
    # later in the code.
    #
    # The observed/model ratio is defined as:
    #
    #     residual = flux_star_cut_work / flux_interp_model_cut_work
    #
    # A low-order polynomial is fitted to this ratio before the final
    # normalization polynomial is computed. This preliminary fit is used only
    # to reject normalization points that deviate strongly from the local
    # observed/model trend.
    #
    # The polynomial order is selected from the number of normalization points
    # and their wavelength spacing. Large gaps indicate sparse sampling, so a
    # lower-order polynomial is preferred. When points are more numerous and
    # better distributed, a higher-order polynomial can describe local
    # curvature while maintaining fit stability.
    #
    # The adopted strategy is:
    #   - Order 1: sparse points or large wavelength gaps.
    #   - Order 2: moderate sampling with reasonably distributed points.
    #   - Order 3: more than four points with small wavelength spacing.
    #
    # One or more sigma-rejection passes may be applied by changing
    # ``n_rejection_iterations``. During each rejection pass, the
    # root-mean-square error (RMSE) of the  polynomial fit is computed.
    # Points with residuals exceeding the corresponding sigma threshold multiplied
    # by the RMSE are rejected.The sigma thresholds for successive rejection passes
    # are specified by ``sigma_values``. A rejection pass is accepted only if points
    # remain on both sides of the diagnostic line, preventing the final normalization
    # from being constrained by only one side of the spectral feature.
    #
    # This empirical scheme was developed for high-resolution IGRINS spectra
    # and balances rejection robustness against fit stability.
    (
        lam_cut_work,
        flux_interp_model_cut_work,
        flux_star_cut_work,
    ) = _sigma_reject_normalization_points(
        central_line,
        lam_cut_work,
        flux_star_cut_work,
        flux_interp_model_cut_work,
        n_rejection_iterations=1,
    )

    # ------------------------------------------------------------------
    # Final Local Normalization and Uncertainty Estimate
    # ------------------------------------------------------------------
    #
    # The final polynomial fit defines the local normalization function
    # applied to the observed spectrum. The uncertainty of this fit is
    # represented by ``delta`` and is propagated into the ASF chi-square
    # calculation as the normalization uncertainty.
    #
    # For sufficiently constrained polynomial fits, ``delta`` is derived
    # from the polynomial prediction interval and corresponds approximately
    # to a 68% confidence (1-sigma) prediction interval.
    #
    # When only two normalization points are available, the prediction
    # interval cannot be robustly estimated. In this case, a fixed
    # uncertainty value of 0.02 is adopted.
    #
    # When fewer than two usable normalization points remain, the
    # normalization is flagged as unsuccessful. A large placeholder value
    # (999999) is assigned to ``delta`` and related normalization
    # quantities. This allows the code to continue running while clearly
    # identifying cases where a reliable normalization could not be
    # obtained.
    lam_cut_work_left, lam_cut_work_right = _split_points_by_line_center(
        lam_cut_work,
        central_line,
    )

    polynomial_order = _select_polynomial_order(lam_cut_work)

    if len(lam_cut_work) >= 3 and lam_cut_work_left.size > 0 and lam_cut_work_right.size > 0:
        flag = 1

        residual = np.asarray(flux_star_cut_work) / np.asarray(
            flux_interp_model_cut_work
        )
        polynomial = np.polyfit(lam_cut_work, residual, polynomial_order)
        residual_polynomial_final = np.polyval(polynomial, lam_star_cut)

        delta = _prediction_interval_delta(
            lam_cut_work,
            lam_star_cut,
            residual,
            polynomial_order,
        )

    elif len(lam_cut_work) == 2 and lam_cut_work_left.size > 0 and lam_cut_work_right.size > 0:
        flag = 1

        residual = flux_star_cut_work / flux_interp_model_cut_work
        polynomial = np.polyfit(lam_cut_work, residual, polynomial_order)
        residual_polynomial_final = np.polyval(polynomial, lam_star_cut)

        delta = np.zeros(len(lam_star_cut)) + 0.02

    else:
        flag = 0
        delta = 999999
        residual_polynomial_final = 999999

    # Construct upper and lower normalization curves from the polynomial
    # prediction interval and normalize both the observed flux and its
    # uncertainty using the same local normalization function.
    upper_residual_polynomial_final = residual_polynomial_final + delta
    lower_residual_polynomial_final = residual_polynomial_final - delta

    flux_star_cut_normalized_final = flux_star_cut / residual_polynomial_final
    err_flux_star_cut_normalized_final = err_flux_star_cut / residual_polynomial_final
    upper_flux_star_cut_normalized_final = (
        flux_star_cut / upper_residual_polynomial_final
    )
    lower_flux_star_cut_normalized_final = (
        flux_star_cut / lower_residual_polynomial_final
    )

    return (
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
    )
