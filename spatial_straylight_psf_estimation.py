import sys
sys.path.insert(1, '/home/harsh/CourseworkRepo/stic/example')
# sys.path.insert(2, '/home/harsh/CourseworkRepo/WFAComparison')
sys.path.insert(1, '/mnt/f/Harsh/CourseworkRepo/stic/example')
sys.path.insert(1, 'F:\\Harsh\\CourseworkRepo\\stic\\example')
import h5py
import numpy as np
import sunpy.io
import sunpy.map
from aiapy.calibrate import register
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.ndimage
from prepare_data import *


h = 6.62606957e-34
c = 2.99792458e8
kb = 1.380649e-23


def prepare_planck_function(wavelength_in_angstrom):
    wavelength_in_m = wavelength_in_angstrom * 1e-10

    f = c / wavelength_in_m

    def planck_function(temperature_in_kelvin):
        return (2 * h * f**3 / c**2) * (1 / (np.exp((h * f) / (kb * temperature_in_kelvin)) - 1))

    return planck_function


def prepare_temperature_from_intensity(wavelength_in_angstrom):
    wavelength_in_m = wavelength_in_angstrom * 1e-10

    f = c / wavelength_in_m

    def get_temperature_from_intensity(intensity_in_si):

        I = intensity_in_si
        return (h * f) / (np.log((2 * h * f**3 / (c**2 * I)) + 1) * kb)

    return get_temperature_from_intensity


def mean_squarred_error(obs_image, hmi_image):
    return np.mean(
        np.square(
            np.subtract(
                obs_image,
                hmi_image
            )
        )
    )

def approximate_stray_light_and_sigma(
        obs_image,
        hmi_image,
):

    max_fwhm = np.min(obs_image.shape)

    if max_fwhm%2 == 0:
        max_fwhm -= 1

    s1 = hmi_image.shape[0]
    s2 = hmi_image.shape[1]

    if s1 % 2 == 0:
        s1 -= 1

    if s2 % 2 == 0:
        s2 -= 1

    fwhm = np.linspace(2, max_fwhm, 100)

    sigma = fwhm / 2.355

    k_values = np.arange(0, 0.5, 0.01)

    result = np.zeros(shape=(sigma.size, k_values.size))

    result_images = np.zeros(
        shape=(
            sigma.size,
            k_values.size,
            obs_image.shape[0],
            obs_image.shape[1]
        ),
        dtype=np.float64
    )

    for i, _sig in enumerate(sigma):
        for j, k_value in enumerate(k_values):

            rev_kernel = np.zeros((s1, s2), dtype=np.float64)

            rev_kernel[s1 // 2, s2 // 2] = 1

            kernel = scipy.ndimage.gaussian_filter(rev_kernel, sigma=_sig)

            degraded_image = scipy.signal.oaconvolve(
                k_value * hmi_image,
                kernel, mode='same') + (
                    (1 - k_value) * hmi_image
            )
            result_images[i][j] = degraded_image
            result[i][j] = mean_squarred_error(
                obs_image / obs_image.max(), degraded_image / degraded_image.max()
            )

    sigma_ind, k_ind = np.unravel_index(np.argmin(result), result.shape)
    return result, result_images, fwhm[sigma_ind], k_values[k_ind], sigma_ind, k_ind


def correct_for_straylight(image, fwhm, k_value):
    s1 = image.shape[0]
    s2 = image.shape[1]

    if s1 % 2 == 0:
        s1 -= 1

    if s2 % 2 == 0:
        s2 -= 1

    rev_kernel = np.zeros((s1, s2))

    rev_kernel[s1//2, s2//2] = 1

    kernel = scipy.ndimage.gaussian_filter(rev_kernel, sigma=fwhm / 2.355)

    kernel[s1//2, s2//2] = 0

    convolved = scipy.signal.oaconvolve(image, kernel, mode='same')

    return (image - k_value * convolved) / (1 - k_value)


def estimate_alpha_and_sigma(datestring, timestring, hmi_cont_file, hmi_ref_file):

    # base_path = Path('F:\\Harsh\\CourseworkRepo\\InstrumentalUncorrectedStokes')
    #
    # base_path = Path('/mnt/f/harsh/CourseworkRepo/InstrumentalUncorrectedStokes')

    base_path = Path('C:\\Work Things\\InstrumentalUncorrectedStokes')

    datepath = base_path / datestring

    level4path = datepath / 'Level-4'

    data, header = sunpy.io.read_file(level4path / hmi_cont_file)[1]

    hmi_map = sunpy.map.Map(data, header)

    aiamap = register(hmi_map)

    planck_function_500_nm = prepare_planck_function(6173)

    intensity_6173 = planck_function_500_nm(5000)

    plt.imshow(aiamap.data, cmap='gray', origin='lower')

    points_hmi = np.array(plt.ginput(2, 600))

    points_hmi = points_hmi.astype(np.int64)

    plt.close('all')
    plt.clf()
    plt.cla()

    calibcount = np.median(aiamap.data[points_hmi[0][1]:points_hmi[1][1], points_hmi[0][0]:points_hmi[1][0]])

    calib_factor = intensity_6173 / calibcount

    data, header = sunpy.io.read_file(level4path / hmi_ref_file)[0]

    data *= calib_factor

    get_temperature_from_intensity_6173_angs = prepare_temperature_from_intensity(6173)

    temperature_map = get_temperature_from_intensity_6173_angs(data)

    filename = level4path / 'aligned_Ca_Ha_stic_profiles_{}_{}.nc_pca.nc'.format(datestring, timestring)

    f = h5py.File(filename, 'r')

    ind = np.where(f['profiles'][0, 0, 0, :, 0] != 0)[0]

    vec_correct_for_straylight = np.vectorize(
        correct_for_straylight,
        signature='(x,y),(),()->(x,y)'
    )

    planck_function_656_nm = prepare_planck_function(6563)

    halpha_image = f['profiles'][0, :, :, 32, 0]

    plt.imshow(halpha_image, cmap='gray', origin='lower')

    points_ha = np.array(plt.ginput(3, 600))

    points_ha = points_ha.astype(np.int64)

    plt.close('all')
    plt.clf()
    plt.cla()

    divide_factor_ha = halpha_image[points_ha[0][1], points_ha[0][0]]

    norm_halpha_image = halpha_image / divide_factor_ha

    intensity_6563 = planck_function_656_nm(temperature_map)

    norm_intensity_6563 = intensity_6563 / intensity_6563[points_ha[0][1], points_ha[0][0]]

    result_ha, result_images_ha, fwhm_ha, k_value_ha, sigma_ind_ha, k_ind_ha = approximate_stray_light_and_sigma(
        norm_halpha_image[points_ha[1][1]:points_ha[2][1], points_ha[1][0]:points_ha[2][0]],
        norm_intensity_6563[points_ha[1][1]:points_ha[2][1], points_ha[1][0]:points_ha[2][0]],
    )

    corrected_halpha_intensity_data = np.transpose(
        vec_correct_for_straylight(
            np.transpose(
                f['profiles'][0, :, :, ind[0:800], 0],
                axes=(2, 0, 1)
            ),
            fwhm_ha,
            k_value_ha
        ),
        axes=(1, 2, 0)
    )

    planck_function_866_nm = prepare_planck_function(8662)

    ca_image = f['profiles'][0, :, :, 3204 + 32, 0]

    plt.imshow(ca_image, cmap='gray', origin='lower')

    points_ca = np.array(plt.ginput(3, 600))

    points_ca = points_ca.astype(np.int64)

    plt.close('all')
    plt.clf()
    plt.cla()

    divide_factor_ca = ca_image[points_ca[0][1], points_ca[0][0]]

    norm_ca_image = ca_image / divide_factor_ca

    intensity_8662 = planck_function_866_nm(temperature_map)

    norm_intensity_8662 = intensity_8662 / intensity_8662[points_ca[0][1], points_ca[0][0]]

    result_ca, result_images_ca, fwhm_ca, k_value_ca, sigma_ind_ca, k_ind_ca = approximate_stray_light_and_sigma(
        norm_ca_image[points_ca[1][1]:points_ca[2][1], points_ca[1][0]:points_ca[2][0]],
        norm_intensity_8662[points_ca[1][1]:points_ca[2][1], points_ca[1][0]:points_ca[2][0]],
    )

    corrected_ca_intensity_data = np.transpose(
        vec_correct_for_straylight(
            np.transpose(
                f['profiles'][0, :, :, ind[800:], 0],
                axes=(2, 0, 1)
            ),
            fwhm_ca,
            k_value_ca
        ),
        axes=(1, 2, 0)
    )

    fo = h5py.File(level4path / 'Spatial_straylight_correction_{}_{}.h5'.format(datestring, timestring), 'w')

    fo['points_ha'] = points_ha

    fo['result_ha'] = result_ha

    fo['result_images_ha'] = result_images_ha

    fo['fwhm_ha'] = fwhm_ha

    fo['k_value_ha'] = k_value_ha

    fo['sigma_ind_ha'] = sigma_ind_ha

    fo['k_ind_ha'] = k_ind_ha

    fo['points_ca'] = points_ca

    fo['result_ca'] = result_ca

    fo['result_images_ca'] = result_images_ca

    fo['fwhm_ca'] = fwhm_ca

    fo['k_value_ca'] = k_value_ca

    fo['sigma_ind_ca'] = sigma_ind_ca

    fo['k_ind_ca'] = k_ind_ca

    fo.close()

    ha = sp.profile(
        nx=f['profiles'].shape[2], ny=f['profiles'].shape[1], ns=4,
        nw=3204
    )
    ca = sp.profile(
        nx=f['profiles'].shape[2], ny=f['profiles'].shape[1], ns=4,
        nw=2308
    )

    ha.wav[:] = f['wav'][0:3204]

    ha.weights = f['weights'][0:3204]

    ca.wav[:] = f['wav'][3204:]

    ca.weights = f['weights'][3204:]

    ha.dat[0, :, :, ind[0:800], 0] = np.transpose(
        corrected_halpha_intensity_data,
        axes=(2, 0, 1)
    )

    ha.dat[0, :, :, ind[0:800], 1:4] = np.transpose(
        f['profiles'][0, :, :, ind[0:800], 1:4],
        axes=(2, 0, 1, 3)
    )

    ca.dat[0, :, :, ind[800:] - 3204, 0] = np.transpose(
        corrected_ca_intensity_data,
        axes=(2, 0, 1)
    )

    ca.dat[0, :, :, ind[800:] - 3204, 1:4] = np.transpose(
        f['profiles'][0, :, :, ind[800:], 1:4],
        axes=(2, 0, 1, 3)
    )

    all_profiles = ha + ca

    all_profiles.write(
        level4path / '{}_spatial_straylight_corrected.nc'.format(filename.name)
    )


if __name__ == '__main__':
    datestring = '20230601'
    timestring = '081014'
    hmi_cont_file = 'hmi.Ic_720s.20230601_050000_TAI.3.continuum.fits'
    hmi_ref_file = 'HMI_reference_image_20230601_081014.fits'
    estimate_alpha_and_sigma(
        datestring,
        timestring,
        hmi_cont_file,
        hmi_ref_file
    )