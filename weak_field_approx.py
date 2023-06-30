import numpy as np


def calculate_b_los(
    stokes_I,
    stokes_V,
    wavelength_arr,
    lambda0,
    lambda_range_min,
    lambda_range_max,
    g_eff,
    transition_skip_list=None,
    errors=None
):
    '''
    stokes_I: Array of Intensties
    stokes_V: Array of Stokes V
    wavelength_arr: Units is nm
    lambda0: Units is nm
    lambda_range_min: Units is nm
    lambda_range_max: Units is nm
    g_eff: Effective lande g factor
    '''

    indices = np.where(
        (
            np.array(wavelength_arr) >= (lambda_range_min)
        ) & (
            np.array(wavelength_arr) <= (lambda_range_max)
        )
    )[0]

    if transition_skip_list is not None:
        skip_ind = list()
        for transition in transition_skip_list:
            skip_ind += list(
                np.where(
                    (
                            np.array(wavelength_arr) >= (transition[0] - transition[1])
                    ) & (
                            np.array(wavelength_arr) <= (transition[0] + transition[1])
                    )
                )[0]
            )
        indices = np.array(list(set(indices) - set(skip_ind)))

    wavelength = np.array(wavelength_arr)[indices]

    intensity = np.array(stokes_I)[indices]

    stokes_V_cropped = np.array(stokes_V)[indices]

    derivative = np.gradient(intensity, wavelength * 10)

    constant = 4.66e-13 * g_eff * (lambda0 * 10)**2

    numerator = np.sum(derivative * stokes_V_cropped)

    denominator = np.sum(np.square(derivative))

    if errors is None or errors is False:

        blos = -numerator / (constant * denominator)

        return blos

    elif errors == 1:
        stokes_V_err = np.ones_like(stokes_V_cropped) * stokes_V_cropped.std()

        numerator2 = np.sum(derivative * stokes_V_err)

        blos_err = -numerator2 / (constant * denominator)

        return blos_err

    else:
        blos = -numerator / (constant * denominator)

        error_norm = np.sum(derivative.std() / derivative + stokes_V_cropped.std() / stokes_V_cropped + np.std(np.square(derivative)**2) / np.square(derivative))

        error = 1 * error_norm

        return error

def prepare_calculate_blos(
    obs,
    wavelength_arr,
    lambda0,
    lambda_range_min,
    lambda_range_max,
    g_eff,
    transition_skip_list=None,
    errors=False
):
    def actual_calculate_blos(i, j):
        i = int(i)
        j = int(j)
        stokes_I, stokes_V = obs[0, i, j, :, 0], obs[0, i, j, :, 3]
        return calculate_b_los(
            stokes_I,
            stokes_V,
            wavelength_arr,
            lambda0,
            lambda_range_min,
            lambda_range_max,
            g_eff,
            transition_skip_list=transition_skip_list,
            errors=errors
        )
    return actual_calculate_blos


def prepare_calculate_blos_rh15d(
    f,
    wavelength_arr,
    lambda0,
    lambda_range_min,
    lambda_range_max,
    g_eff,
    transition_skip_list=None
):
    def actual_calculate_blos(i, j):
        i = int(i)
        j = int(j)
        stokes_I, stokes_V = f['intensity'][i, j], f['stokes_V'][i, j]
        return calculate_b_los(
            stokes_I,
            stokes_V,
            wavelength_arr,
            lambda0,
            lambda_range_min,
            lambda_range_max,
            g_eff,
            transition_skip_list=transition_skip_list
        )
    return actual_calculate_blos


def prepare_calculate_blos_vlos_gradient(
    obs,
    wavelength_arr,
    lambda0,
    lambda_range,
    g_eff,
    transition_skip_list=None
):
    def actual_calculate_blos(i, j):
        i = int(i)
        j = int(j)
        stokes_I, stokes_V = obs[0, i, j, :, 0], obs[0, i, j, :, 3]
        min_wave_position = np.argmin(stokes_I)
        lambda_range_min = wavelength_arr[min_wave_position] - lambda_range
        lambda_range_max = wavelength_arr[min_wave_position] + lambda_range
        return calculate_b_los(
            stokes_I,
            stokes_V,
            wavelength_arr,
            lambda0,
            lambda_range_min,
            lambda_range_max,
            g_eff,
            transition_skip_list=transition_skip_list
        )
    return actual_calculate_blos


def calculate_b_transverse_wing(
    stokes_I,
    stokes_Q,
    stokes_U,
    wavelength_arr,
    lambda0,
    lambda_range_min,
    lambda_range_max,
    g_eff
):

    norm_Q = np.array(stokes_Q) / np.array(stokes_I)

    norm_U = np.array(stokes_U) / np.array(stokes_I)

    total_linear_polarization = np.sqrt(
        np.add(
            np.square(
                norm_Q
            ),
            np.square(
                norm_U
            )
        )
    )

    indices = np.where(
        (
            np.array(wavelength_arr) > (lambda0 + lambda_range_min)
        ) & (
            np.array(wavelength_arr) < (lambda0 + lambda_range_max)
        )
    )[0]

    wavelength = np.array(wavelength_arr)[indices]

    intensity = np.array(stokes_I)[indices]

    # norm_Q_cropped = norm_Q[indices]

    # norm_U_cropped = norm_U[indices]

    total_linear_polarization_cropped = total_linear_polarization[indices]

    derivative = np.abs(np.gradient(intensity, wavelength * 10))

    constant = (4.6686e-10 * (lambda0 * 10)**2)**2 * g_eff

    diff_lambda = 1 / (np.abs(wavelength - lambda0) * 10)

    numerator = 4 * np.sum(
        total_linear_polarization_cropped * diff_lambda * derivative
    ) / (
        3 * constant
    )

    denominator = np.sum(np.square(diff_lambda) * np.square(derivative))

    return np.sqrt(numerator / denominator)


def prepare_compare_mag_field(magc, magc_ca, ltau, ind):
    def compare_mag_field(i, j):
        i = int(i)
        j = int(j)
        mag_field_halpha = magc[i, j]
        ca_mag_field = magc_ca[i, j]

        return ltau[ind][np.argmin(np.abs(ca_mag_field[ind] - mag_field_halpha))]

    return compare_mag_field
