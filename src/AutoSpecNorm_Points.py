"""AutoSpecNorm wavelength-point selection.

This module contains the wavelength-point selection routine used by
AutoSpecNorm. The routine identifies suitable local pseudo-continuum
normalization points by comparing the observed spectrum with an interpolated
synthetic spectrum within a selected analysis region.

The algorithm was developed for high-resolution IGRINS spectra
(R ≈ 45,000) of cool stars, where dense molecular absorption and blended
spectral features make continuum placement challenging. The normalization
criteria implemented here were calibrated and tested for spectra of similar
resolution. Users applying AutoSpecNorm to spectra with significantly lower
or higher spectral resolution may need to modify the selection criteria and
thresholds to achieve optimal performance.
"""

import numpy as np


def _nearest_wavelength_index(wavelength_grid, target_wavelength, atol=1.0e-8):
    """Return one robust index for a target wavelength.

    The wavelength candidates used by AutoSpecNorm are selected from the
    observed wavelength grid. In rare cases, rounding or repeated wavelength
    values can make one candidate match more than one grid index. This helper
    always returns a single representative index, preventing array-shape
    broadcast errors while preserving the selected normalization wavelength.
    """
    wavelength_grid = np.asarray(wavelength_grid, dtype=float)

    if wavelength_grid.size == 0:
        return None

    close_indices = np.where(
        np.isclose(wavelength_grid, target_wavelength, rtol=0.0, atol=atol)
    )[0]

    if close_indices.size > 0:
        closest_local_index = np.argmin(
            np.abs(wavelength_grid[close_indices] - target_wavelength)
        )
        return int(close_indices[closest_local_index])

    return int(np.argmin(np.abs(wavelength_grid - target_wavelength)))


def _unique_indices_preserve_order(indices):
    """Return unique integer indices while preserving their original order."""
    unique_indices = []
    seen = set()

    for index in indices:
        index = int(index)
        if index not in seen:
            unique_indices.append(index)
            seen.add(index)

    return unique_indices
def AutoSpecNorm_Points(
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
):
    """Select local normalization wavelength points for AutoSpecNorm.

    This routine identifies wavelength points suitable for local
    pseudo-continuum normalization by comparing the observed and synthetic
    spectra within a user-defined analysis region.

    The procedure consists of:

        1. Extracting the analysis region from the observed spectrum.
        2. Interpolating the synthetic spectrum onto the observed wavelength grid.
        3. Selecting candidate normalization points near local flux maxima.
        4. Filtering the candidates using synthetic-spectrum criteria.
        5. Filtering the remaining candidates using observed-spectrum criteria.
        6. Returning the selected normalization wavelength points.

    Parameters
    ----------
    RV : float
        Stellar radial velocity in km/s.
    lam_analysis : array_like
        Two-element wavelength interval defining the analysis region.
    lam_star : ndarray
        Observed wavelength array.
    flux_star : ndarray
        Observed flux array.
    err_flux_star : ndarray
        Observed flux uncertainty array.
    lam_model : ndarray
        Synthetic wavelength array.
    flux_model : ndarray
        Synthetic flux array.
    limit_model : float
        Fraction of the local synthetic-spectrum maximum used to preselect
        candidate normalization points.
    limit_star : float
        Fraction of the local observed-spectrum maximum used to preselect
        candidate normalization points.
    peak_index_model : float
        Model-spectrum tolerance used in the normalization-point selection
        criteria.
    peak_index_star : float
        Observed-spectrum tolerance used in the normalization-point selection
        criteria.

    Returns
    -------
    tuple of ndarray
        Arrays containing the spectra in the analysis region and the final
        normalization points selected for the observed and synthetic spectra.
    """
    c_light = 300000.0
    # ------------------------------------------------------------------
    # Step 1: Extract the analysis region and interpolate the model spectrum
    # ------------------------------------------------------------------
    #
    # Analysis regions (ARs) are supplied as inputs to this function. Each AR
    # defines the wavelength interval around a selected diagnostic line where
    # suitable local pseudo-continuum points are searched for.
    #
    # The AR limits are already defined in the rest frame, whereas the observed
    # spectrum arrays (lam_star, flux_star, and err_flux_star) remain in the
    # observed stellar frame. Therefore, the two AR boundaries are first shifted
    # to the observed stellar frame using the stellar radial velocity. The
    # corresponding portion of the observed spectrum is then extracted, and the
    # selected wavelengths are shifted back to the rest frame.
    #
    # The synthetic spectrum is subsequently interpolated onto the wavelength
    # grid of the extracted observed spectrum within the AR. This ensures that
    # the observed and synthetic spectra share the same wavelength sampling,
    # allowing direct point-by-point comparisons during the normalization-point
    # selection procedure and throughout the subsequent chi-square fitting
    # performed by ASF.

    lam_analysis_unshift_1 = ((lam_analysis[0] * (-1) * RV) / c_light) + lam_analysis[0]
    lam_analysis_unshift_2 = ((lam_analysis[1] * (-1) * RV) / c_light) + lam_analysis[1]
    index = (lam_star >= lam_analysis_unshift_1) & (lam_star <= lam_analysis_unshift_2)
    lam_star_cut = lam_star[index]
    flux_star_cut = flux_star[index]
    err_flux_star_cut = err_flux_star[index]

    flux_star_cut = np.round(flux_star_cut,3)
    err_flux_star_cut = np.round(err_flux_star_cut, 3)

    # Shift the extracted observed wavelengths in the AR back to the rest frame.
    lam_star_cut = ((lam_star_cut * RV) / c_light) + lam_star_cut

    # Interpolate the synthetic spectrum onto the observed wavelength grid
    # within the AR. 
    flux_interp_model_cut = np.interp(lam_star_cut, lam_model, flux_model)
    flux_interp_model_cut = np.round(flux_interp_model_cut, 3)

    lam_cut_work = lam_star_cut.copy()
    flux_interp_model_cut_work = flux_interp_model_cut.copy()
    flux_star_cut_work = flux_star_cut.copy()

    # ------------------------------------------------------------------
    # Step 2: Preselect candidate normalization points near local maxima
    # ------------------------------------------------------------------
    #
    # The first filtering stage identifies candidate normalization points that
    # lie close to the pseudo-continuum level within the analysis region.
    #
    # The filtering is performed in two steps:
    #
    # 1. Synthetic-spectrum filter:
    #    Points are retained if their interpolated synthetic flux is greater
    #    than or equal to:
    #
    #        limit_model × max(synthetic flux)
    #
    #    where limit_model is a user-defined input parameter. The synthetic
    #    spectrum is used first because it is not affected by observational
    #    noise, detector artifacts, cosmic rays, or telluric residuals, and
    #    therefore provides a more reliable estimate of the local pseudo-
    #    continuum level.
    #
    # 2. Observed-spectrum filter:
    #    Among the points that pass the synthetic-spectrum criterion, only
    #    those with observed flux greater than or equal to:
    #
    #        limit_star × max(observed flux)
    #
    #    are retained, where limit_star is another user-defined input
    #    parameter.
    #
    # The remaining points define a preliminary set of candidate pseudo-
    # continuum points from which the final normalization points will be
    # selected using additional criteria in the following steps.
    max_flux_interp_model_cut_work = np.max(flux_interp_model_cut_work)

    index_work_model = (flux_interp_model_cut_work >= np.round(limit_model * max_flux_interp_model_cut_work, 3))
    lam_cut_work = lam_cut_work[index_work_model]
    flux_interp_model_cut_work = flux_interp_model_cut_work[index_work_model]
    flux_star_cut_work = flux_star_cut_work[index_work_model]

    max_flux_star_cut_work = np.max(flux_star_cut_work)


    index_work_star = (flux_star_cut_work >= np.round(limit_star * max_flux_star_cut_work, 3))
    lam_cut_work = lam_cut_work[index_work_star]
    flux_interp_model_cut_work = flux_interp_model_cut_work[index_work_star]
    flux_star_cut_work = flux_star_cut_work[index_work_star]

    # ------------------------------------------------------------------
    # Step 3: Identify local pseudo-continuum candidates in the synthetic spectrum
    # ------------------------------------------------------------------
    #
    # The candidate points selected in Step 2 are now examined in greater
    # detail to identify points that correspond to local maxima, or points
    # located very close to local maxima, within the analysis region.
    #
    # The synthetic spectrum is evaluated first because it provides a smoother
    # and more stable representation of the underlying stellar spectrum and is
    # not affected by observational noise, detector artifacts, cosmic rays,
    # imperfect telluric correction, or other observational effects.
    #
    # For cool-star spectra, identifying pseudo-continuum points is
    # particularly challenging because molecular absorption bands and severe
    # line blending often produce irregular, asymmetric, and non-Gaussian
    # flux maxima. Consequently, simple maximum-flux criteria are generally
    # insufficient to identify reliable normalization points.
    #
    # To address this problem, a series of neighboring wavelength windows are
    # constructed around each candidate point. The local flux distribution and
    # slope behavior are then examined on both sides of the candidate point
    # over multiple wavelength scales. The criteria below are designed to
    # identify points that behave like local pseudo-continuum points while
    # rejecting points located within absorption features, blended structures,
    # steep flux gradients, or other complex spectral morphologies.
    #
    # The points that satisfy these synthetic-spectrum criteria are retained
    # for further evaluation using the observed spectrum in the next step.
    
    n_good_points = 0
    good_indices = []

    for point_index in range(len(lam_cut_work)):

        indices = _nearest_wavelength_index(lam_star_cut, lam_cut_work[point_index])
        if indices is None:
            continue

        index_neighbor_1 = ((lam_star_cut >= lam_cut_work[point_index] - 0.2) & (lam_star_cut < lam_cut_work[point_index])) | \
                           ((lam_star_cut > lam_cut_work[point_index]) & (lam_star_cut <= lam_cut_work[point_index] + 0.2))

        index_neighbor_1_left = (lam_star_cut >= lam_cut_work[point_index] - 0.2) & (lam_star_cut < lam_cut_work[point_index])
        index_neighbor_1_right = (lam_star_cut > lam_cut_work[point_index]) & (lam_star_cut <= lam_cut_work[point_index] + 0.2)
        index_neighbor_2 = ((lam_star_cut >= lam_cut_work[point_index] - 0.4) & (lam_star_cut < lam_cut_work[point_index] - 0.2)) | \
                           ((lam_star_cut > lam_cut_work[point_index] + 0.2) & (lam_star_cut <= lam_cut_work[point_index] + 0.4))

        index_neighbor_2_left = (lam_star_cut >= lam_cut_work[point_index] - 0.4) & (lam_star_cut < lam_cut_work[point_index] - 0.2)
        index_neighbor_2_right = (lam_star_cut > lam_cut_work[point_index] + 0.2) & (lam_star_cut <= lam_cut_work[point_index] + 0.4)
        index_neighbor_3 = ((lam_star_cut >= lam_cut_work[point_index] - 0.8) & (lam_star_cut < lam_cut_work[point_index] - 0.4)) | \
                           ((lam_star_cut > lam_cut_work[point_index] + 0.4) & (lam_star_cut <= lam_cut_work[point_index] + 0.8))

        index_neighbor_3_left = (lam_star_cut >= lam_cut_work[point_index] - 0.8) & (lam_star_cut < lam_cut_work[point_index] - 0.4)
        index_neighbor_3_right = (lam_star_cut > lam_cut_work[point_index] + 0.4) & (lam_star_cut <= lam_cut_work[point_index] + 0.8)
        index_neighbor_4 = ((lam_star_cut >= lam_cut_work[point_index] - 1.2) & (lam_star_cut < lam_cut_work[point_index] - 0.8)) | \
                           ((lam_star_cut > lam_cut_work[point_index] + 0.8) & (lam_star_cut <= lam_cut_work[point_index] + 1.2))
        index_neighbor_5 = ((lam_star_cut >= lam_cut_work[point_index] - 2) & (lam_star_cut < lam_cut_work[point_index] - 1.2)) | \
                           ((lam_star_cut > lam_cut_work[point_index] + 1.2) & (lam_star_cut <= lam_cut_work[point_index] + 2))
        index_neighbor_6 = ((lam_star_cut >= lam_cut_work[point_index] - 5) & (lam_star_cut < lam_cut_work[point_index] - 2)) | \
                           ((lam_star_cut > lam_cut_work[point_index] + 2) & (lam_star_cut <= lam_cut_work[point_index] + 5))
        flux_interp_model_cut_neighbor_1 = flux_interp_model_cut[index_neighbor_1]
        flux_interp_model_cut_neighbor_2 = flux_interp_model_cut[index_neighbor_2]
        flux_interp_model_cut_neighbor_3 = flux_interp_model_cut[index_neighbor_3]
        flux_interp_model_cut_neighbor_4 = flux_interp_model_cut[index_neighbor_4]
        flux_interp_model_cut_neighbor_5 = flux_interp_model_cut[index_neighbor_5]
        flux_interp_model_cut_neighbor_6 = flux_interp_model_cut[index_neighbor_6]

        flux_interp_model_cut_neighbor_1 = np.round(flux_interp_model_cut_neighbor_1, 3)
        flux_interp_model_cut_neighbor_2 = np.round(flux_interp_model_cut_neighbor_2, 3)
        flux_interp_model_cut_neighbor_3 = np.round(flux_interp_model_cut_neighbor_3, 3)
        flux_interp_model_cut_neighbor_4 = np.round(flux_interp_model_cut_neighbor_4, 3)
        flux_interp_model_cut_neighbor_5 = np.round(flux_interp_model_cut_neighbor_5, 3)
        flux_interp_model_cut_neighbor_6 = np.round(flux_interp_model_cut_neighbor_6, 3)
        flux_interp_model_cut_neighbor_1_left = flux_interp_model_cut[index_neighbor_1_left]
        flux_interp_model_cut_neighbor_1_right = flux_interp_model_cut[index_neighbor_1_right]

        flux_interp_model_cut_neighbor_1_left = np.round(flux_interp_model_cut_neighbor_1_left, 3)
        flux_interp_model_cut_neighbor_1_right = np.round(flux_interp_model_cut_neighbor_1_right, 3)

        flux_interp_model_cut_neighbor_2_left = flux_interp_model_cut[index_neighbor_2_left]
        flux_interp_model_cut_neighbor_2_right = flux_interp_model_cut[index_neighbor_2_right]

        flux_interp_model_cut_neighbor_2_left = np.round(flux_interp_model_cut_neighbor_2_left, 3)
        flux_interp_model_cut_neighbor_2_right = np.round(flux_interp_model_cut_neighbor_2_right, 3)

        flux_interp_model_cut_neighbor_3_left = flux_interp_model_cut[index_neighbor_3_left]
        flux_interp_model_cut_neighbor_3_right = flux_interp_model_cut[index_neighbor_3_right]

        flux_interp_model_cut_neighbor_3_left = np.round(flux_interp_model_cut_neighbor_3_left, 3)
        flux_interp_model_cut_neighbor_3_right = np.round(flux_interp_model_cut_neighbor_3_right, 3)
        lam_star_cut_neighbor_1_left = lam_star_cut[index_neighbor_1_left]
        lam_star_cut_neighbor_1_right = lam_star_cut[index_neighbor_1_right]

        lam_star_cut_neighbor_2_left = lam_star_cut[index_neighbor_2_left]
        lam_star_cut_neighbor_2_right = lam_star_cut[index_neighbor_2_right]

        lam_star_cut_neighbor_3_left = lam_star_cut[index_neighbor_3_left]
        lam_star_cut_neighbor_3_right = lam_star_cut[index_neighbor_3_right]

        flux_interp_model_cut_neighbor_1_right = np.array(flux_interp_model_cut_neighbor_1_right)
        flux_interp_model_cut_neighbor_2_right = np.array(flux_interp_model_cut_neighbor_2_right)
        slope_cut_neighbor_1_right = (flux_interp_model_cut_work[point_index] - flux_interp_model_cut_neighbor_1_right) / (
                    lam_cut_work[point_index] - lam_star_cut_neighbor_1_right)
        slope_cut_neighbor_1_left = (flux_interp_model_cut_work[point_index] - flux_interp_model_cut_neighbor_1_left) / (
                    lam_cut_work[point_index] - lam_star_cut_neighbor_1_left)

        slope_cut_neighbor_1_right = np.round(slope_cut_neighbor_1_right, 3)
        slope_cut_neighbor_1_left = np.round(slope_cut_neighbor_1_left, 3)
        slope_cut_neighbor_2_left_sep = np.array([])
        slope_cut_neighbor_2_right_sep = np.array([])
        slope_cut_neighbor_3_right_sep = np.array([])
        slope_cut_neighbor_3_left_sep = np.array([])

        flux_interp_model_cut_neighbor_1_left = np.array(flux_interp_model_cut_neighbor_1_left)
        flux_interp_model_cut_neighbor_2_left = np.array(flux_interp_model_cut_neighbor_2_left)
        flux_interp_model_cut_neighbor_3_left = np.array(flux_interp_model_cut_neighbor_3_left)

        lam_star_cut_neighbor_1_left = np.array(lam_star_cut_neighbor_1_left)
        lam_star_cut_neighbor_2_left = np.array(lam_star_cut_neighbor_2_left)
        lam_star_cut_neighbor_3_left = np.array(lam_star_cut_neighbor_3_left)

        flux_interp_model_cut_neighbor_1_right = np.array(flux_interp_model_cut_neighbor_1_right)
        flux_interp_model_cut_neighbor_2_right = np.array(flux_interp_model_cut_neighbor_2_right)
        flux_interp_model_cut_neighbor_3_right = np.array(flux_interp_model_cut_neighbor_3_right)

        lam_star_cut_neighbor_1_right = np.array(lam_star_cut_neighbor_1_right)
        lam_star_cut_neighbor_2_right = np.array(lam_star_cut_neighbor_2_right)
        lam_star_cut_neighbor_3_right = np.array(lam_star_cut_neighbor_3_right)


        if len(flux_interp_model_cut_neighbor_2_left) > 0 and len(flux_interp_model_cut_neighbor_1_left) > 0:
            slope_cut_neighbor_2_left_sep = (flux_interp_model_cut_neighbor_1_left[0] - flux_interp_model_cut_neighbor_2_left) / \
                                            (lam_star_cut_neighbor_1_left[0] - lam_star_cut_neighbor_2_left)

            slope_cut_neighbor_2_left_sep = np.round(slope_cut_neighbor_2_left_sep, 3)
        if len(flux_interp_model_cut_neighbor_2_right) > 0 and len(flux_interp_model_cut_neighbor_1_right) > 0:
            slope_cut_neighbor_2_right_sep = (flux_interp_model_cut_neighbor_2_right - flux_interp_model_cut_neighbor_1_right[-1]) / \
                                             (lam_star_cut_neighbor_2_right - lam_star_cut_neighbor_1_right[-1])

            slope_cut_neighbor_2_right_sep = np.round(slope_cut_neighbor_2_right_sep, 3)
        if len(flux_interp_model_cut_neighbor_2_left) == 0 or len(flux_interp_model_cut_neighbor_1_left) == 0:
            slope_cut_neighbor_2_left_sep = np.array([])
        if len(flux_interp_model_cut_neighbor_2_right) == 0 or len(flux_interp_model_cut_neighbor_1_right) == 0:
            slope_cut_neighbor_2_right_sep = np.array([])
        if len(flux_interp_model_cut_neighbor_3_right) > 0 and len(flux_interp_model_cut_neighbor_2_right) > 0:
            slope_cut_neighbor_3_right_sep = (flux_interp_model_cut_neighbor_3_right - flux_interp_model_cut_neighbor_2_right[-1]) / \
                                             (lam_star_cut_neighbor_3_right - lam_star_cut_neighbor_2_right[-1])

            slope_cut_neighbor_3_right_sep = np.round(slope_cut_neighbor_3_right_sep, 3)
        if len(flux_interp_model_cut_neighbor_3_left) > 0 and len(flux_interp_model_cut_neighbor_2_left) > 0:
            slope_cut_neighbor_3_left_sep = (flux_interp_model_cut_neighbor_2_left[0] - flux_interp_model_cut_neighbor_3_left) / \
                                            (lam_star_cut_neighbor_2_left[0] - lam_star_cut_neighbor_3_left)

            slope_cut_neighbor_3_left_sep = np.round(slope_cut_neighbor_3_left_sep, 3)
        if len(flux_interp_model_cut_neighbor_3_right) == 0 or len(flux_interp_model_cut_neighbor_2_right) == 0:
            slope_cut_neighbor_3_right_sep = np.array([])
        if len(flux_interp_model_cut_neighbor_3_left) == 0 or len(flux_interp_model_cut_neighbor_2_left) == 0:
            slope_cut_neighbor_3_left_sep = np.array([])
        
        # Primary synthetic-spectrum criteria. These conditions test whether
        # the candidate behaves like a local pseudo-continuum point in the
        # nearest neighboring wavelength windows.
        condition1_1 = (np.all(flux_interp_model_cut_work[point_index] >= flux_interp_model_cut_neighbor_1 - 0.8 * peak_index_model))

        condition1_2 = (np.all(flux_star_cut_work[point_index] >= flux_interp_model_cut_neighbor_1 - 4 * peak_index_model) and
                        np.mean(np.abs(slope_cut_neighbor_1_left)) < 0.035 and np.mean(np.abs(slope_cut_neighbor_1_right)) < 0.035 and
                        ((np.mean(np.abs(slope_cut_neighbor_1_left)) < 0.5 * np.mean(np.abs(slope_cut_neighbor_1_right))) or
                         (np.mean(np.abs(slope_cut_neighbor_1_right)) < 0.5 * np.mean(np.abs(slope_cut_neighbor_1_left)))))

        condition1_3 = ((np.all(flux_interp_model_cut_work[point_index] >= flux_interp_model_cut_neighbor_1_left)) and
                        (np.all(np.abs(slope_cut_neighbor_1_right) <= 0.04)) and
                        (np.all(slope_cut_neighbor_1_left > 0)) and (np.all(slope_cut_neighbor_2_left_sep > 0)) and
                        (np.mean(np.abs(slope_cut_neighbor_1_right)) < 0.35 * np.mean(np.abs(slope_cut_neighbor_1_left))) and
                        (np.mean(np.abs(slope_cut_neighbor_1_right)) < 0.35 * np.mean(np.abs(slope_cut_neighbor_2_left_sep))))

        condition1_4 = ((np.all(flux_interp_model_cut_work[point_index] >= flux_interp_model_cut_neighbor_1_right)) and
                        (np.all(np.abs(slope_cut_neighbor_1_left) <= 0.04)) and (np.all(slope_cut_neighbor_1_right < 0)) and
                        (np.all(slope_cut_neighbor_2_right_sep < 0)) and
                        (np.mean(np.abs(slope_cut_neighbor_1_left)) < 0.35 * np.mean(np.abs(slope_cut_neighbor_1_right))) and
                        (np.mean(np.abs(slope_cut_neighbor_1_left)) < 0.35 * np.mean(np.abs(slope_cut_neighbor_2_right_sep))))

        condition1_5 = ((np.all(flux_interp_model_cut_work[point_index] >= flux_interp_model_cut_neighbor_1_left)) and
                        (np.all(flux_interp_model_cut_work[point_index] <= flux_interp_model_cut_neighbor_1_right)) and
                        (np.all(slope_cut_neighbor_1_right <= 0.04) and np.all(slope_cut_neighbor_1_right >= 0)) and
                        (np.all(slope_cut_neighbor_1_left <= 0.04) and np.all(slope_cut_neighbor_1_left >= 0)) and
                        ((len(slope_cut_neighbor_2_left_sep) > 0 and np.all(slope_cut_neighbor_2_left_sep > 0) and
                        np.mean(np.abs(slope_cut_neighbor_1_left)) < 0.2 * np.mean(np.abs(slope_cut_neighbor_2_left_sep))) or
                        (len(slope_cut_neighbor_2_right_sep) > 0 and np.all(slope_cut_neighbor_2_right_sep < 0) and
                        np.mean(np.abs(slope_cut_neighbor_1_right)) < 0.2 * np.mean(np.abs(slope_cut_neighbor_2_right_sep))) or
                        (len(slope_cut_neighbor_2_left_sep) > 0 and len(slope_cut_neighbor_2_right_sep) > 0 and
                        np.all(slope_cut_neighbor_2_left_sep > 0) and np.all(slope_cut_neighbor_2_right_sep < 0) and
                        np.mean(np.abs(slope_cut_neighbor_1_left)) < 0.2 * np.mean(np.abs(slope_cut_neighbor_2_left_sep)) and
                        np.mean(np.abs(slope_cut_neighbor_1_right)) < 0.2 * np.mean(np.abs(slope_cut_neighbor_2_right_sep)) and
                        np.mean(np.abs(slope_cut_neighbor_1_left)) < 0.2 * np.mean(np.abs(slope_cut_neighbor_3_left_sep)) and
                        np.mean(np.abs(slope_cut_neighbor_1_right)) < 0.2 * np.mean(np.abs(slope_cut_neighbor_3_right_sep)))))

        condition1_6 = ((np.all(flux_interp_model_cut_work[point_index] >= flux_interp_model_cut_neighbor_1_right)) and
                        (np.all(flux_interp_model_cut_work[point_index] <= flux_interp_model_cut_neighbor_1_left)) and
                        (np.all(slope_cut_neighbor_1_right >= -0.04) and np.all(slope_cut_neighbor_1_right <= 0)) and
                        (np.all(slope_cut_neighbor_1_left >= -0.04) and np.all(slope_cut_neighbor_1_left <= 0)) and
                        ((len(slope_cut_neighbor_2_left_sep) > 0 and np.all(slope_cut_neighbor_2_left_sep > 0) and
                        np.mean(np.abs(slope_cut_neighbor_1_left)) < 0.2 * np.mean(np.abs(slope_cut_neighbor_2_left_sep))) or
                        (len(slope_cut_neighbor_2_right_sep) > 0 and np.all(slope_cut_neighbor_2_right_sep < 0) and
                        np.mean(np.abs(slope_cut_neighbor_1_right)) < 0.2 * np.mean(np.abs(slope_cut_neighbor_2_right_sep))) or
                        (len(slope_cut_neighbor_2_left_sep) > 0 and len(slope_cut_neighbor_2_right_sep) > 0 and
                        np.all(slope_cut_neighbor_2_left_sep > 0) and np.all(slope_cut_neighbor_2_right_sep < 0) and
                        np.mean(np.abs(slope_cut_neighbor_1_left)) < 0.2 * np.mean(np.abs(slope_cut_neighbor_2_left_sep)) and
                        np.mean(np.abs(slope_cut_neighbor_1_right)) < 0.2 * np.mean(np.abs(slope_cut_neighbor_2_right_sep)) and
                        np.mean(np.abs(slope_cut_neighbor_1_left)) < 0.2 * np.mean(np.abs(slope_cut_neighbor_3_left_sep)) and
                        np.mean(np.abs(slope_cut_neighbor_1_right)) < 0.2 * np.mean(np.abs(slope_cut_neighbor_3_right_sep)))))
       
        # Secondary synthetic-spectrum criteria. These conditions extend the
        # local checks to the next neighboring wavelength windows.
        condition2_1 = ((len(flux_interp_model_cut_neighbor_2_left) > 0 and len(flux_interp_model_cut_neighbor_2_right) > 0) and
                        (np.all(flux_interp_model_cut_work[point_index] >= flux_interp_model_cut_neighbor_2 - 1 * peak_index_model) or
                        (np.all(flux_interp_model_cut_work[point_index] >= flux_interp_model_cut_neighbor_2_left) and
                        (np.all(np.abs(slope_cut_neighbor_1_right) <= 0.04) and
                        (np.all(slope_cut_neighbor_2_left_sep > 0) or np.all(slope_cut_neighbor_2_left_sep < 0)
                        or not np.all(slope_cut_neighbor_2_left_sep > 0)))) or
                        (np.all(flux_interp_model_cut_work[point_index] >= flux_interp_model_cut_neighbor_2_right) and
                        (np.all(np.abs(slope_cut_neighbor_1_left) <= 0.04) and
                        (np.all(slope_cut_neighbor_2_right_sep > 0) or np.all(slope_cut_neighbor_2_right_sep < 0)
                        or not np.all(slope_cut_neighbor_2_right_sep > 0))))))

        condition2_2 = ((len(flux_interp_model_cut_neighbor_2_left) > 0 and len(flux_interp_model_cut_neighbor_2_right) == 0) and
                        (np.all(flux_interp_model_cut_work[point_index] >= flux_interp_model_cut_neighbor_2_left - 1 * peak_index_model) or
                        (np.all(flux_interp_model_cut_work[point_index] >= flux_interp_model_cut_neighbor_2_left) and
                        np.all(np.abs(slope_cut_neighbor_1_right) <= 0.04) and
                        (np.all(slope_cut_neighbor_2_left_sep > 0) or np.all(slope_cut_neighbor_2_left_sep < 0)
                        or not np.all(slope_cut_neighbor_2_left_sep > 0)))))

        condition2_3 = ((len(flux_interp_model_cut_neighbor_2_left) == 0 and len(flux_interp_model_cut_neighbor_2_right) > 0) and
                        (np.all(flux_interp_model_cut_work[point_index] >= flux_interp_model_cut_neighbor_2_right - 1 * peak_index_model) or
                        (np.all(flux_interp_model_cut_work[point_index] >= flux_interp_model_cut_neighbor_2_right) and
                        np.all(np.abs(slope_cut_neighbor_1_left) <= 0.04) and
                        (np.all(slope_cut_neighbor_2_right_sep > 0) or np.all(slope_cut_neighbor_2_right_sep < 0)
                        or not np.all(slope_cut_neighbor_2_right_sep > 0)))))
        if len(flux_interp_model_cut_neighbor_1_left) > 0 and len(flux_interp_model_cut_neighbor_1_right) > 0:

            if condition1_1 or condition1_2 or condition1_3 or condition1_4 or condition1_5 or condition1_6:

                if condition2_1 or condition2_2 or condition2_3:

                    if (len(flux_interp_model_cut_neighbor_3) > 0 and
                            np.all(flux_interp_model_cut_work[point_index] >= flux_interp_model_cut_neighbor_3 - 10* peak_index_model)):

                        if (len(flux_interp_model_cut_neighbor_4) > 0 and
                                np.all(flux_interp_model_cut_work[point_index] >= flux_interp_model_cut_neighbor_4 - 12 * peak_index_model)):

                            if (len(flux_interp_model_cut_neighbor_5) > 0 and
                                    np.all(flux_interp_model_cut_work[point_index] >= flux_interp_model_cut_neighbor_5 - 15 * peak_index_model)):

                                if (len(flux_interp_model_cut_neighbor_6) > 0 and
                                        np.all(flux_interp_model_cut_work[point_index] >= flux_interp_model_cut_neighbor_6 - 20 * peak_index_model)):
                                    n_good_points = n_good_points + 1
                                    good_indices.append(indices)

    good_indices = np.asarray(_unique_indices_preserve_order(good_indices), dtype=int)
    lam_cut_work = lam_star_cut[good_indices]
    flux_star_cut_work = flux_star_cut[good_indices]
    flux_interp_model_cut_work = flux_interp_model_cut[good_indices]

    # ------------------------------------------------------------------
    # Step 4: Validate candidate pseudo-continuum points in the observed spectrum
    # ------------------------------------------------------------------
    #
    # The candidate points that satisfy the synthetic-spectrum criteria in
    # Step 3 are now evaluated using the observed spectrum. The overall goal
    # remains the same: to identify points that correspond to local maxima,
    # or points located very close to local maxima, within the analysis region.
    #
    # Unlike the synthetic spectrum, however, the observed spectrum is affected
    # by photon noise, detector artifacts, imperfect telluric correction,
    # residual wavelength-calibration errors, and other observational effects.
    # As a result, local maxima in the observed spectrum are often less smooth,
    # less symmetric, and less well defined than those in the synthetic
    # spectrum.
    #
    # For this reason, the selection criteria used here are not identical to
    # those applied in Step 3. Instead, they are designed to be more tolerant
    # of small-scale fluctuations while still preserving the fundamental
    # requirement that a candidate point behaves like a local pseudo-continuum
    # point. Multiple neighboring wavelength windows are examined to verify
    # that the candidate remains among the highest flux points in its local
    # environment and is not associated with absorption features, blended
    # structures, steep gradients, or noise-induced peaks.
    #
    # The points that satisfy these observed-spectrum criteria are adopted as
    # the final normalization points used by AutoSpecNorm.
    n_good_points = 0
    good_indices = []

    for point_index in range(len(lam_cut_work)):

        indices = _nearest_wavelength_index(lam_star_cut, lam_cut_work[point_index])
        if indices is None:
            continue
        index_neighbor_1 = ((lam_star_cut >= lam_cut_work[point_index] - 0.2) & (lam_star_cut < lam_cut_work[point_index])) | \
                           ((lam_star_cut > lam_cut_work[point_index]) & (lam_star_cut <= lam_cut_work[point_index] + 0.2))

        index_neighbor_1_left = (lam_star_cut >= lam_cut_work[point_index] - 0.2) & (lam_star_cut < lam_cut_work[point_index])
        index_neighbor_1_right = (lam_star_cut > lam_cut_work[point_index]) & (lam_star_cut <= lam_cut_work[point_index] + 0.2)
        index_neighbor_2 = ((lam_star_cut >= lam_cut_work[point_index] - 0.4) & (lam_star_cut < lam_cut_work[point_index] - 0.2)) | \
                           ((lam_star_cut > lam_cut_work[point_index] + 0.2) & (lam_star_cut <= lam_cut_work[point_index] + 0.4))

        index_neighbor_2_left = (lam_star_cut >= lam_cut_work[point_index] - 0.4) & (lam_star_cut < lam_cut_work[point_index] - 0.2)
        index_neighbor_2_right = (lam_star_cut > lam_cut_work[point_index] + 0.2) & (lam_star_cut <= lam_cut_work[point_index] + 0.4)
        index_neighbor_3 = ((lam_star_cut >= lam_cut_work[point_index] - 0.8) & (lam_star_cut < lam_cut_work[point_index] - 0.4)) | \
                           ((lam_star_cut > lam_cut_work[point_index] + 0.4) & (lam_star_cut <= lam_cut_work[point_index] + 0.8))

        index_neighbor_3_left = (lam_star_cut >= lam_cut_work[point_index] - 0.8) & (lam_star_cut < lam_cut_work[point_index] - 0.4)
        index_neighbor_3_right = (lam_star_cut > lam_cut_work[point_index] + 0.4) & (lam_star_cut <= lam_cut_work[point_index] + 0.8)
        index_neighbor_4 = ((lam_star_cut >= lam_cut_work[point_index] - 1.2) & (lam_star_cut < lam_cut_work[point_index] - 0.8)) | \
                           ((lam_star_cut > lam_cut_work[point_index] + 0.8) & (lam_star_cut <= lam_cut_work[point_index] + 1.2))
        index_neighbor_5 = ((lam_star_cut >= lam_cut_work[point_index] - 2) & (lam_star_cut < lam_cut_work[point_index] - 1.2)) | \
                           ((lam_star_cut > lam_cut_work[point_index] + 1.2) & (lam_star_cut <= lam_cut_work[point_index] + 2))
        index_neighbor_6 = ((lam_star_cut >= lam_cut_work[point_index] - 4) & (lam_star_cut < lam_cut_work[point_index] - 2)) | \
                           ((lam_star_cut > lam_cut_work[point_index] + 2) & (lam_star_cut <= lam_cut_work[point_index] + 4))
        index_neighbor_7 = ((lam_star_cut >= lam_cut_work[point_index] - 8) & (lam_star_cut < lam_cut_work[point_index] - 4)) | \
                           ((lam_star_cut > lam_cut_work[point_index] + 4) & (lam_star_cut <= lam_cut_work[point_index] + 8))
        flux_star_cut_neighbor_1 = flux_star_cut[index_neighbor_1]
        flux_star_cut_neighbor_2 = flux_star_cut[index_neighbor_2]
        flux_star_cut_neighbor_3 = flux_star_cut[index_neighbor_3]
        flux_star_cut_neighbor_4 = flux_star_cut[index_neighbor_4]
        flux_star_cut_neighbor_5 = flux_star_cut[index_neighbor_5]
        flux_star_cut_neighbor_6 = flux_star_cut[index_neighbor_6]
        flux_star_cut_neighbor_7 = flux_star_cut[index_neighbor_7]

        flux_star_cut_neighbor_1 = np.round(flux_star_cut_neighbor_1, 3)
        flux_star_cut_neighbor_2 = np.round(flux_star_cut_neighbor_2, 3)
        flux_star_cut_neighbor_3 = np.round(flux_star_cut_neighbor_3, 3)
        flux_star_cut_neighbor_4 = np.round(flux_star_cut_neighbor_4, 3)
        flux_star_cut_neighbor_5 = np.round(flux_star_cut_neighbor_5, 3)
        flux_star_cut_neighbor_6 = np.round(flux_star_cut_neighbor_6, 3)
        flux_star_cut_neighbor_7 = np.round(flux_star_cut_neighbor_7, 3)
        flux_star_cut_neighbor_1_left = flux_star_cut[index_neighbor_1_left]
        flux_star_cut_neighbor_1_right = flux_star_cut[index_neighbor_1_right]

        flux_star_cut_neighbor_1_left = np.round(flux_star_cut_neighbor_1_left, 3)
        flux_star_cut_neighbor_1_right = np.round(flux_star_cut_neighbor_1_right, 3)


        flux_star_cut_neighbor_2_left = flux_star_cut[index_neighbor_2_left]
        flux_star_cut_neighbor_2_right = flux_star_cut[index_neighbor_2_right]

        flux_star_cut_neighbor_2_left = np.round(flux_star_cut_neighbor_2_left, 3)
        flux_star_cut_neighbor_2_right = np.round(flux_star_cut_neighbor_2_right, 3)


        flux_star_cut_neighbor_3_left = flux_star_cut[index_neighbor_3_left]
        flux_star_cut_neighbor_3_right = flux_star_cut[index_neighbor_3_right]

        flux_star_cut_neighbor_3_left = np.round(flux_star_cut_neighbor_3_left, 3)
        flux_star_cut_neighbor_3_right = np.round(flux_star_cut_neighbor_3_right, 3)
        lam_star_cut_neighbor_1_left = lam_star_cut[index_neighbor_1_left]
        lam_star_cut_neighbor_1_right = lam_star_cut[index_neighbor_1_right]

        lam_star_cut_neighbor_2_left = lam_star_cut[index_neighbor_2_left]
        lam_star_cut_neighbor_2_right = lam_star_cut[index_neighbor_2_right]

        lam_star_cut_neighbor_3_left = lam_star_cut[index_neighbor_3_left]
        lam_star_cut_neighbor_3_right = lam_star_cut[index_neighbor_3_right]
        flux_star_cut_neighbor_1_left = np.array(flux_star_cut_neighbor_1_left)
        flux_star_cut_neighbor_1_right = np.array(flux_star_cut_neighbor_1_right)

        flux_star_cut_neighbor_2_left = np.array(flux_star_cut_neighbor_2_left)
        flux_star_cut_neighbor_2_right = np.array(flux_star_cut_neighbor_2_right)

        flux_star_cut_neighbor_3_left = np.array(flux_star_cut_neighbor_3_left)
        flux_star_cut_neighbor_3_right = np.array(flux_star_cut_neighbor_3_right)
        lam_star_cut_neighbor_1_left = np.array(lam_star_cut_neighbor_1_left)
        lam_star_cut_neighbor_1_right = np.array(lam_star_cut_neighbor_1_right)

        lam_star_cut_neighbor_2_left = np.array(lam_star_cut_neighbor_2_left)
        lam_star_cut_neighbor_2_right = np.array(lam_star_cut_neighbor_2_right)

        lam_star_cut_neighbor_3_left = np.array(lam_star_cut_neighbor_3_left)
        lam_star_cut_neighbor_3_right = np.array(lam_star_cut_neighbor_3_right)
        slope_cut_neighbor_1_right = (flux_star_cut_work[point_index] - flux_star_cut_neighbor_1_right) / \
            (lam_cut_work[point_index] - lam_star_cut_neighbor_1_right)
        slope_cut_neighbor_1_left = (flux_star_cut_work[point_index] - flux_star_cut_neighbor_1_left) / \
            (lam_cut_work[point_index] - lam_star_cut_neighbor_1_left)

        slope_cut_neighbor_1_right = np.round(slope_cut_neighbor_1_right, 3)
        slope_cut_neighbor_1_left = np.round(slope_cut_neighbor_1_left, 3)


        slope_cut_neighbor_2_right = (flux_star_cut_work[point_index] - flux_star_cut_neighbor_2_right) / \
            (lam_cut_work[point_index] - lam_star_cut_neighbor_2_right)
        slope_cut_neighbor_2_left = (flux_star_cut_work[point_index] - flux_star_cut_neighbor_2_left) / \
            (lam_cut_work[point_index] - lam_star_cut_neighbor_2_left)

        slope_cut_neighbor_2_right = np.round(slope_cut_neighbor_2_right, 3)
        slope_cut_neighbor_2_left = np.round(slope_cut_neighbor_2_left, 3)
        if len(flux_star_cut_neighbor_3_right) > 0 and len(flux_star_cut_neighbor_2_right) > 0:
            slope_cut_neighbor_3_right_sep = (flux_star_cut_neighbor_3_right - flux_star_cut_neighbor_2_right[-1]) / \
                                             (lam_star_cut_neighbor_3_right - lam_star_cut_neighbor_2_right[-1])

            slope_cut_neighbor_3_right_sep = np.round(slope_cut_neighbor_3_right_sep, 3)
        if len(flux_star_cut_neighbor_3_left) > 0 and len(flux_star_cut_neighbor_2_left) > 0:
            slope_cut_neighbor_3_left_sep = (flux_star_cut_neighbor_2_left[0] - flux_star_cut_neighbor_3_left) / \
                                            (lam_star_cut_neighbor_2_left[0] - lam_star_cut_neighbor_3_left)

            slope_cut_neighbor_3_left_sep = np.round(slope_cut_neighbor_3_left_sep, 3)
        if len(flux_star_cut_neighbor_3_right) == 0 or len(flux_star_cut_neighbor_2_right) == 0:
            slope_cut_neighbor_3_right_sep = np.array([])
        if len(flux_star_cut_neighbor_3_left) == 0 or len(flux_star_cut_neighbor_2_left) == 0:
            slope_cut_neighbor_3_left_sep = np.array([])
       
        # Primary observed-spectrum criteria. These conditions test whether
        # the candidate remains a suitable high point in the observed spectrum.
        condition1_1 = (np.all(flux_star_cut_work[point_index] >= flux_star_cut_neighbor_1 - 1 * peak_index_star))

        condition1_2 = (np.all(flux_star_cut_work[point_index] >= flux_star_cut_neighbor_1 - 5 * peak_index_star) and
                        np.mean(np.abs(slope_cut_neighbor_1_left)) < 0.025 and
                        np.mean(np.abs(slope_cut_neighbor_1_right)) < 0.025 and
                        ((np.mean(np.abs(slope_cut_neighbor_1_left)) < 0.3 * np.mean(np.abs(slope_cut_neighbor_1_right))) or
                        (np.mean(np.abs(slope_cut_neighbor_1_right)) < 0.3 * np.mean(np.abs(slope_cut_neighbor_1_left)))))

        condition1_3 = (np.all(flux_star_cut_work[point_index] >= flux_star_cut_neighbor_1_left) and
                        np.all(np.abs(slope_cut_neighbor_1_right) <= 0.035) and
                        (np.mean(np.abs(slope_cut_neighbor_1_right)) < 0.3 * np.mean(np.abs(slope_cut_neighbor_1_left))))

        condition1_4 = (np.all(flux_star_cut_work[point_index] >= flux_star_cut_neighbor_1_right) and
                        np.all(np.abs(slope_cut_neighbor_1_left) <= 0.035) and
                        (np.mean(np.abs(slope_cut_neighbor_1_left)) < 0.3 * np.mean(np.abs(slope_cut_neighbor_1_right))))

        condition1_5 = ((np.mean(flux_star_cut_neighbor_1_right) > np.mean(flux_star_cut_neighbor_1_left)) and
                        np.mean(np.abs(slope_cut_neighbor_1_right)) <= 0.06 and
                        np.mean(np.abs(slope_cut_neighbor_1_left)) <= 0.06 and
                        len(slope_cut_neighbor_3_left_sep) > 0 and
                        (np.mean(np.abs(slope_cut_neighbor_1_right)) < 0.7 * np.mean(np.abs(slope_cut_neighbor_3_left_sep))))

        condition1_6 = (np.mean(flux_star_cut_neighbor_1_left) > np.mean(flux_star_cut_neighbor_1_right) and
                        np.mean(np.abs(slope_cut_neighbor_1_left)) <= 0.06 and
                        np.mean(np.abs(slope_cut_neighbor_1_right)) <= 0.06 and
                        len(slope_cut_neighbor_3_right_sep) > 0 and
                        (np.mean(np.abs(slope_cut_neighbor_1_left)) < 0.7 * np.mean(np.abs(slope_cut_neighbor_3_right_sep))))
        
        # Secondary observed-spectrum criteria. These conditions examine the
        # next neighboring windows to avoid broad features and local slopes.
        condition2_1 = ((len(flux_star_cut_neighbor_2_left) > 0 and len(flux_star_cut_neighbor_2_right) > 0) and
                        (np.all(flux_star_cut_work[point_index] >= flux_star_cut_neighbor_2 - 4 * peak_index_star) or
                        (flux_star_cut_work[point_index] >= np.mean(flux_star_cut_neighbor_2_left) and
                        np.mean(np.abs(slope_cut_neighbor_2_right)) <= 0.04 and
                        np.mean(np.abs(slope_cut_neighbor_2_right)) < 0.4 * np.mean(np.abs(slope_cut_neighbor_2_left))) or
                        (flux_star_cut_work[point_index] >= np.mean(flux_star_cut_neighbor_2_right) and
                        np.mean(np.abs(slope_cut_neighbor_2_left)) <= 0.04 and
                        np.mean(np.abs(slope_cut_neighbor_2_left)) < 0.4 * np.mean(np.abs(slope_cut_neighbor_2_right)))))

        condition2_2 = ((len(flux_star_cut_neighbor_2_left) == 0 and len(flux_star_cut_neighbor_2_right) > 0) and
                        ((flux_star_cut_work[point_index] >= np.mean(flux_star_cut_neighbor_2_right)) or
                        (flux_star_cut_work[point_index] < np.mean(flux_star_cut_neighbor_2_right) and
                        np.mean(np.abs(slope_cut_neighbor_2_right)) <= 0.04)))

        condition2_3 = ((len(flux_star_cut_neighbor_2_left) > 0 and len(flux_star_cut_neighbor_2_right) == 0) and
                        ((flux_star_cut_work[point_index] >= np.mean(flux_star_cut_neighbor_2_left)) or
                        (flux_star_cut_work[point_index] < np.mean(flux_star_cut_neighbor_2_left) and
                        np.mean(np.abs(slope_cut_neighbor_2_left)) <= 0.04)))
        if len(flux_star_cut_neighbor_1_left) > 0 and len(flux_star_cut_neighbor_1_right) > 0:

            if condition1_1 or condition1_2 or condition1_3 or condition1_4 or condition1_5 or condition1_6:

                if condition2_1 or condition2_2 or condition2_3:

                    if np.all(flux_star_cut_work[point_index] >= flux_star_cut_neighbor_3 - 10 * peak_index_star):

                        if np.all(flux_star_cut_work[point_index] >= flux_star_cut_neighbor_4 - 12 * peak_index_star):

                            if np.all(flux_star_cut_work[point_index] >= flux_star_cut_neighbor_5 - 15 * peak_index_star):

                                if np.all(flux_star_cut_work[point_index] >= flux_star_cut_neighbor_6 - 20 * peak_index_star):

                                    if np.all(flux_star_cut_work[point_index] >= flux_star_cut_neighbor_7 - 30 * peak_index_star):
                                        n_good_points = n_good_points + 1
                                        good_indices.append(indices)

    good_indices = np.asarray(_unique_indices_preserve_order(good_indices), dtype=int)
    lam_cut_work = lam_star_cut[good_indices]
    flux_star_cut_work = flux_star_cut[good_indices]
    flux_interp_model_cut_work = flux_interp_model_cut[good_indices]


    return (
        lam_star_cut,
        flux_star_cut,
        err_flux_star_cut,
        flux_interp_model_cut,
        lam_cut_work,
        flux_star_cut_work,
        flux_interp_model_cut_work,
    )

