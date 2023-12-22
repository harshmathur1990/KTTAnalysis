import numpy as np
import scipy.ndimage
from tqdm import tqdm
# import numba as nb

def prepare_get_indice(arr):
    def get_indice(wave):
        return np.argmin(np.abs(arr - wave))
    return get_indice


def normalise_profiles(
    line_profile,
    line_wave,
    atlas_profile,
    atlas_wave,
    cont_wave
):
    indice_line = np.argmin(np.abs(line_wave - cont_wave))
    indice_atlas = np.argmin(np.abs(atlas_wave - cont_wave))

    get_indice = prepare_get_indice(atlas_wave)

    vec_get_indice = np.vectorize(get_indice)

    atlas_indice = vec_get_indice(line_wave)

    norm_atlas = atlas_profile / atlas_profile[indice_atlas]

    return line_profile / line_profile[indice_line], \
        norm_atlas[atlas_indice], atlas_wave[atlas_indice]


def mean_squarred_error(line_profile, atlas_profile):
    return np.sqrt(
        np.sum(
            np.power(
                np.subtract(
                    line_profile,
                    atlas_profile
                ),
                2
            )
        )
    ) / line_profile.size


def approximate_stray_light_and_sigma(
    line_profile,
    atlas_profile,
    continuum=1.0,
    indices=None
):
    if indices is None:
        indices = np.arange(line_profile.size)

    fwhm = np.linspace(2, 30, 50)

    sigma = fwhm / 2.355

    k_values = np.arange(0, 1, 0.01)

    result = np.zeros(shape=(sigma.size, k_values.size))

    result_atlas = np.zeros(
        shape=(
            sigma.size,
            k_values.size,
            atlas_profile.shape[0]
        )
    )

    for i, _sig in enumerate(sigma):
        for j, k_value in enumerate(k_values):
            degraded_atlas = scipy.ndimage.gaussian_filter(
                (1 - k_value) * atlas_profile,
                _sig
            ) + (k_value * continuum)
            result_atlas[i][j] = degraded_atlas
            result[i][j] = mean_squarred_error(
                degraded_atlas[indices] / degraded_atlas[indices][0],
                line_profile / line_profile[0]
            )
    
    return result, result_atlas, fwhm, sigma, k_values


def approximate_stray_light_and_sigma_multiple_lines(
        line_profile_list,
        atlas_profile_list,
):
    fwhm = np.linspace(2, 30, 50)

    sigma = fwhm / 2.355

    k_values = np.arange(0, 1, 0.01)

    dimensions = list()

    dimensions.append(sigma.size)

    for _ in line_profile_list:
        dimensions.append(k_values.size)

    result_list = list()

    result_atlas_list = list()

    total_profile_size = 0

    for index, (line_profile, atlas_profile) in enumerate(zip(line_profile_list, atlas_profile_list)):
        result, result_atlas, fwhm, sigma, k_values = approximate_stray_light_and_sigma(line_profile, atlas_profile)

        result /= result.max()

        result_list.append(result)

        result_atlas_list.append(result_atlas)

        total_profile_size += line_profile.size

        print (result.max(), result.min())
    r_i0, r_i1, r_i2, r_i3, r_fwhm, r_sigma, r_k1, r_k2, r_k3 = speeduploop(fwhm, k_values, result_list)

    print_str = ''
    for varname, var in zip(['r_i0', 'r_i1', 'r_i2', 'r_i3', 'r_fwhm', 'r_sigma', 'r_k1', 'r_k2', 'r_k3'], [r_i0, r_i1, r_i2, r_i3, r_fwhm, r_sigma, r_k1, r_k2, r_k3]):
        print_str += '{} - {}  '.format(varname, var)

    print (print_str)

    r_atlas = list()
    r_atlas.append(result_atlas_list[0][r_i0, r_i1])
    r_atlas.append(result_atlas_list[1][r_i0, r_i2])
    r_atlas.append(result_atlas_list[2][r_i0, r_i3])

    return r_atlas, r_fwhm, r_sigma, r_k1, r_k2, r_k3, result_list, result_atlas_list, fwhm, sigma, k_values
