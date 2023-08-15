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
from skimage.restoration.deconvolution import wiener
from collections import OrderedDict


h = 6.62606957e-34
c = 2.99792458e8
kb = 1.380649e-23


AIA_SAMPLING_ARCSEC = 0.6


def resample_data(image, new_shape):
    return sunpy.image.resample.resample(
        orig=image,
        dimensions=new_shape,
        method='linear',
        minusone=False
    )


vec_resample_data = np.vectorize(resample_data, signature='(x,y),(n)->(w,z)')


def resample_full_data(data, CDELTX, CDELTY):
    new_shape = (
        data.shape[1] * CDELTY / AIA_SAMPLING_ARCSEC,
        data.shape[2] * CDELTX / AIA_SAMPLING_ARCSEC,
    )

    return np.transpose(
        vec_resample_data(
            np.transpose(
                data,
                axes=(0, 3, 1, 2)
            ),
            new_shape
        ),
        axes=(0, 2, 3, 1)
    )



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

# def approximate_stray_light_and_sigma(
#         obs_image,
#         hmi_image,
#         kernel_size,
#         y1,
#         y2,
#         x1,
#         x2,
#         y3,
#         y4,
#         x3,
#         x4
# ):
#
#     a = 3.457142857142857
#
#     b = 0.3428571428571427
#
#     max_fwhm = np.amin(list(obs_image.shape))
#
#     # max_fwhm = np.int64(np.round((kernel_size - b) / a, 0))
#
#     # max_fwhm = 40
#
#     # max_fwhm = np.int64(np.round((np.amin(list(obs_image.shape)) - b) / a, 0))
#
#     fwhm = np.linspace(2, max_fwhm, 100)
#
#     k_values = np.arange(0, 1, 0.01)
#
#     result = np.zeros(shape=(fwhm.size, k_values.size))
#
#     result_images = np.zeros(
#         shape=(
#             fwhm.size,
#             k_values.size,
#             obs_image.shape[0],
#             obs_image.shape[1]
#         ),
#         dtype=np.float64
#     )
#
#     for i, _fwhm in enumerate(fwhm):
#         for j, k_value in enumerate(k_values):
#
#             rev_kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
#
#             rev_kernel[kernel_size // 2, kernel_size // 2] = 1
#
#             kernel = scipy.ndimage.gaussian_filter(rev_kernel, sigma=_fwhm / 2.355)
#
#             degraded_image = scipy.signal.oaconvolve(
#                 k_value * hmi_image,
#                 kernel, mode='same') + (
#                     (1 - k_value) * hmi_image
#             )
#
#             # degraded_image = k_value * scipy.signal.fftconvolve(hmi_image, kernel, mode='same') + (1 - k_value) * hmi_image
#
#             result_images[i][j] = degraded_image
#             corr_image = correct_for_straylight(obs_image, fwhm[i], k_value, kernel_size=kernel_size)
#
#             if degraded_image.min() <= 0 or corr_image.min() <= 0:
#                 result[i][j] = np.inf
#             else:
#                 # result[i][j] = mean_squarred_error(
#                 #     obs_image / obs_image.max(), degraded_image / degraded_image.max()
#                 # )
#                 # result[i][j] = mean_squarred_error(
#                 #     (obs_image - obs_image.min()) / (obs_image.max() - obs_image.min()),
#                 #     (degraded_image - degraded_image.min()) / (degraded_image.max() - degraded_image.min())
#                 # )
#
#                 norm_obs = obs_image #(obs_image - obs_image.min()) / (obs_image.max() - obs_image.min())
#                 norm_degraded = degraded_image #(degraded_image - degraded_image.min()) / (degraded_image.max() - degraded_image.min())
#                 result[i][j] = mean_squarred_error(
#                     np.mean(norm_obs[y3:y4, x3:x4]) / np.mean(norm_obs[y1:y2, x1:x2]),
#                     np.mean(norm_degraded[y3:y4, x3:x4]) / np.mean(norm_degraded[y1:y2, x1:x2])
#                 )
#                 # result[i][j] = mean_squarred_error(
#                 #     np.mean(norm_obs[y3:y4, x3:x4]),
#                 #     np.mean(norm_degraded[y3:y4, x3:x4])
#                 # ) + mean_squarred_error(
#                 #     np.mean(norm_obs[y1:y2, x1:x2]),
#                 #     np.mean(norm_degraded[y1:y2, x1:x2])
#                 # )
#
#     sigma_ind, k_ind = np.unravel_index(np.argmin(result), result.shape)
#
#     # corr_image = correct_for_straylight(obs_image, fwhm[sigma_ind], k_values[k_ind], kernel_size=kernel_size)
#     # if corr_image.min() <= 0:
#     #     import ipdb;ipdb.set_trace()
#     return result, result_images, fwhm[sigma_ind], k_values[k_ind], sigma_ind, k_ind, max_fwhm


def approximate_stray_light_and_sigma_alternate(
        obs_image,
        hmi_image,
        kernel_size,
        y1,
        y2,
        x1,
        x2,
        y3,
        y4,
        x3,
        x4
):
    a = 3.457142857142857

    b = 0.3428571428571427

    max_fwhm = np.amin(list(obs_image.shape))

    fwhm = np.linspace(2, max_fwhm, 100)

    k_values = np.arange(0, 1, 0.01)

    result = np.zeros(shape=(fwhm.size, k_values.size))

    result_images = np.zeros(
        shape=(
            fwhm.size,
            k_values.size,
            obs_image.shape[0],
            obs_image.shape[1]
        ),
        dtype=np.float64
    )

    for i, _fwhm in enumerate(fwhm):
        for j, k_value in enumerate(k_values):

            corr_image = correct_for_straylight(obs_image, _fwhm, k_value, kernel_size=kernel_size)

            if corr_image.min() <= 0:
                result[i][j] = np.inf
            else:
                result[i][j] = mean_squarred_error(
                    np.mean(corr_image[y3:y4, x3:x4]) / np.mean(corr_image[y1:y2, x1:x2]),
                    np.mean(hmi_image[y3:y4, x3:x4]) / np.mean(hmi_image[y1:y2, x1:x2])
                )
                result_images[i, j] = corr_image

    fwhm_ind, k_ind = np.unravel_index(np.argmin(result), result.shape)

    corrected_image = result_images[fwhm_ind, k_ind]

    image_contrast_old = np.mean(obs_image[y3:y4, x3:x4]) / np.mean(obs_image[y1:y2, x1:x2])

    image_contrast_new = np.mean(corrected_image[y3:y4, x3:x4]) / np.mean(corrected_image[y1:y2, x1:x2])

    hmi_contrast = np.mean(hmi_image[y3:y4, x3:x4]) / np.mean(hmi_image[y1:y2, x1:x2])

    return result, result_images, fwhm[fwhm_ind], k_values[k_ind], fwhm_ind, k_ind, max_fwhm, image_contrast_old, image_contrast_new, hmi_contrast



def correct_for_straylight(image, fwhm, k_value, kernel_size=None):

    rev_kernel = np.zeros((kernel_size, kernel_size))

    rev_kernel[kernel_size//2, kernel_size//2] = 1

    kernel = scipy.ndimage.gaussian_filter(rev_kernel, sigma=fwhm / 2.355)

    kernel[kernel_size // 2, kernel_size // 2] = 0

    convolved = scipy.signal.oaconvolve(
        image, kernel, mode='same'
    )

    return (image - k_value * convolved) / (1 - k_value)


# def correct_for_straylight_alternate(image, fwhm, k_value, kernel_size=None, nsnr=None):
#
#     rev_kernel = np.zeros((kernel_size, kernel_size))
#
#     rev_kernel[kernel_size//2, kernel_size//2] = 1
#
#     kernel = scipy.ndimage.gaussian_filter(rev_kernel, sigma=fwhm / 2.355)
#
#     kernel[kernel_size // 2, kernel_size // 2] = 0
#
#     if nsnr is None:
#         num_photons = image / 4.36 / 5
#
#         snr = np.sqrt(np.mean(num_photons))
#
#         nsnr = 1/snr
#
#     kernel_size_by_2 = kernel_size//2
#
#     addition_size = kernel_size_by_2 * 2
#
#     larger_image = np.ones(
#         (image.shape[0] + addition_size, image.shape[1] + addition_size),
#         dtype=np.float64
#     ) * np.median(image)
#
#     larger_image[kernel_size_by_2:-kernel_size_by_2, kernel_size_by_2:-kernel_size_by_2] = image
#
#     convolved = wiener(
#         image=larger_image,
#         psf=kernel,
#         balance=nsnr,
#         clip=False
#     )[kernel_size_by_2:-kernel_size_by_2, kernel_size_by_2:-kernel_size_by_2]
#
#     return (image - k_value * convolved) / (1 - k_value)


def get_image_contrast(image):
    nm = (image - image.min()) / (image.max() - image.min())

    return nm.std()


def estimate_alpha_and_sigma(
        datestring, timestring, hmi_cont_file, hmi_ref_file,
        BIN_FACTOR_Y,
        CDELTY,
        im_top_ha,
        im_bot_ha,
        t1_HA,
        t2_HA,
        t3_HA,
        t4_HA,
        cut_shape_HA,
        BIN_FACTOR_X_HA,
        CDELTX_HA,
        im_top_ca,
        im_bot_ca,
        t1_CA,
        t2_CA,
        t3_CA,
        t4_CA,
        cut_shape_CA,
        BIN_FACTOR_X_CA,
        CDELTX_CA
):

    # base_path = Path('F:\\Harsh\\CourseworkRepo\\InstrumentalUncorrectedStokes')
    #
    # base_path = Path('/mnt/f/harsh/CourseworkRepo/InstrumentalUncorrectedStokes')

    base_path = Path('C:\\Work Things\\InstrumentalUncorrectedStokes')

    # base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes/')

    datepath = base_path / datestring

    level4path = datepath / 'Level-4'

    level5path = datepath / 'Level-5'

    level0path = datepath / 'Level-0'

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
        signature='(x,y),(),(),()->(x,y)'
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

    kernel_size_ha = np.int64(
        np.amin(
            [
                np.abs(points_ha[1][1] - points_ha[2][1]),
                np.abs(points_ha[1][0] - points_ha[2][0])
            ]
        )
    )

    if kernel_size_ha % 2 == 0:
        kernel_size_ha -= 1

    plt.imshow(intensity_6563[points_ha[1][1]:points_ha[2][1], points_ha[1][0]:points_ha[2][0]], cmap='gray', origin='lower')

    compare_points_ha = np.array(plt.ginput(4, 600))

    compare_points_ha = compare_points_ha.astype(np.int64)

    plt.close('all')

    result_ha, result_images_ha, fwhm_ha, k_value_ha, sigma_ind_ha, k_ind_ha, max_fwhm_ha, image_contrast_ha_old, image_contrast_ha_new, hmi_contrast_ha = approximate_stray_light_and_sigma_alternate(
        norm_halpha_image[points_ha[1][1]:points_ha[2][1], points_ha[1][0]:points_ha[2][0]],
        norm_intensity_6563[points_ha[1][1]:points_ha[2][1], points_ha[1][0]:points_ha[2][0]],
        kernel_size=kernel_size_ha,
        y1=compare_points_ha[0][1],
        y2=compare_points_ha[1][1],
        x1=compare_points_ha[0][0],
        x2=compare_points_ha[1][0],
        y3=compare_points_ha[2][1],
        y4=compare_points_ha[3][1],
        x3=compare_points_ha[2][0],
        x4=compare_points_ha[3][0]
    )


    top_beam_ha, bottom_beam_ha = read_file_for_observations(
        level0path / 'observation_data_{}_DETECTOR_1.fits'.format(timestring),
        t1_HA, t2_HA, t3_HA, t4_HA, cut_shape_HA
    )

    y1 = 0
    y2 = (top_beam_ha.shape[1] // BIN_FACTOR_Y) * BIN_FACTOR_Y
    x1 = 0
    x2 = (top_beam_ha.shape[2] // BIN_FACTOR_X_HA) * BIN_FACTOR_X_HA
    binned_top_beam_ha = np.mean(
        top_beam_ha[:, y1:y2, x1:x2, :].reshape(
            top_beam_ha.shape[0], top_beam_ha.shape[1] // BIN_FACTOR_Y, BIN_FACTOR_Y,
            top_beam_ha.shape[2] // BIN_FACTOR_X_HA, BIN_FACTOR_X_HA, top_beam_ha.shape[3]
        ),
        (2, 4)
    )
    binned_bottom_beam_ha = np.mean(
        bottom_beam_ha[:, y1:y2, x1:x2, :].reshape(
            bottom_beam_ha.shape[0], bottom_beam_ha.shape[1] // BIN_FACTOR_Y, BIN_FACTOR_Y,
            bottom_beam_ha.shape[2] // BIN_FACTOR_X_HA, BIN_FACTOR_X_HA, bottom_beam_ha.shape[3]
        ),
        (2, 4)
    )

    resampled_top_beam_ha = resample_full_data(binned_top_beam_ha, CDELTX_HA, CDELTY)
    resampled_bottom_beam_ha = resample_full_data(binned_bottom_beam_ha, CDELTX_HA, CDELTY)

    while True:
        corrected_halpha_intensity_data_top_beam = np.transpose(
            vec_correct_for_straylight(
                np.transpose(
                   resampled_top_beam_ha,
                    axes=(0, 3, 1, 2)
                ),
                fwhm_ha,
                k_value_ha,
                kernel_size=kernel_size_ha
            ),
            axes=(0, 2, 3, 1)
        )

        corrected_halpha_intensity_data_bottom_beam = np.transpose(
            vec_correct_for_straylight(
                np.transpose(
                    resampled_bottom_beam_ha,
                    axes=(0, 3, 1, 2)
                ),
                fwhm_ha,
                k_value_ha,
                kernel_size=kernel_size_ha
            ),
            axes=(0, 2, 3, 1)
        )

        if corrected_halpha_intensity_data_top_beam.min() <= 0 or corrected_halpha_intensity_data_bottom_beam.min() <= 0:
            k_value_ha -= 0.01
            sys.stdout.write('Decreasing k_value_ha to {}\n'.format(k_value_ha))
        else:
            break

    new_stokes_top_ha, new_stokes_bot_ha = get_stokes_from_observations(corrected_halpha_intensity_data_top_beam, corrected_halpha_intensity_data_bottom_beam, im_top_ha, im_bot_ha)

    total_stokes_ha = np.zeros_like(new_stokes_top_ha)
    total_stokes_ha[0] = (new_stokes_top_ha[0] + new_stokes_bot_ha[0]) / 2
    for i in range(1, 4):
        total_stokes_ha[i] = ((new_stokes_top_ha[i] / new_stokes_top_ha[0]) + (new_stokes_bot_ha[i] / new_stokes_bot_ha[0])) / 2
        total_stokes_ha[i] *= total_stokes_ha[0]

    sunpy.io.write_file(
        fname=level5path / 'total_stokes_{}_DETECTOR_1.fits'.format(timestring),
        data=total_stokes_ha,
        header=OrderedDict({'CDELTY': AIA_SAMPLING_ARCSEC, 'CDELTX': AIA_SAMPLING_ARCSEC, 'UNIT': 'arcsec'}),
        filetype='fits',
        overwrite=True
    )

    residuals_ha = np.zeros_like(new_stokes_top_ha)
    residuals_ha[0] = (new_stokes_top_ha[0] - new_stokes_bot_ha[0]) / 2
    for i in range(1, 4):
        residuals_ha[i] = ((new_stokes_top_ha[i] / new_stokes_top_ha[0]) - (new_stokes_bot_ha[i] / new_stokes_bot_ha[0])) / 2

    sunpy.io.write_file(
        fname=level5path / 'residuals_{}_DETECTOR_1.fits'.format(timestring),
        data=residuals_ha,
        header=OrderedDict({'CDELTY': AIA_SAMPLING_ARCSEC, 'CDELTX': AIA_SAMPLING_ARCSEC, 'UNIT': 'arcsec'}),
        filetype='fits',
        overwrite=True
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

    kernel_size_ca = np.int64(
        np.amin(
            [
                np.abs(points_ca[1][1] - points_ca[2][1]),
                np.abs(points_ca[1][0] - points_ca[2][0])
            ]
        )
    )

    if kernel_size_ca % 2 == 0:
        kernel_size_ca -= 1

    plt.imshow(
        intensity_8662[points_ca[1][1]:points_ca[2][1], points_ca[1][0]:points_ca[2][0]],
        cmap='gray',
        origin='lower'
    )

    compare_points_ca = np.array(plt.ginput(4, 600))

    compare_points_ca = compare_points_ca.astype(np.int64)

    plt.close('all')

    result_ca, result_images_ca, fwhm_ca, k_value_ca, sigma_ind_ca, k_ind_ca, max_fwhm_ca, image_contrast_ca_old, image_contrast_ca_new, hmi_contrast_ca = approximate_stray_light_and_sigma_alternate(
        norm_ca_image[points_ca[1][1]:points_ca[2][1], points_ca[1][0]:points_ca[2][0]],
        norm_intensity_8662[points_ca[1][1]:points_ca[2][1], points_ca[1][0]:points_ca[2][0]],
        kernel_size=kernel_size_ca,
        y1=compare_points_ca[0][1],
        y2=compare_points_ca[1][1],
        x1=compare_points_ca[0][0],
        x2=compare_points_ca[1][0],
        y3=compare_points_ca[2][1],
        y4=compare_points_ca[3][1],
        x3=compare_points_ca[2][0],
        x4=compare_points_ca[3][0]
    )

    top_beam_ca, bottom_beam_ca = read_file_for_observations(
        level0path / 'observation_data_{}_DETECTOR_3.fits'.format(timestring),
        t1_CA, t2_CA, t3_CA, t4_CA, cut_shape_CA
    )

    y1 = 0
    y2 = (top_beam_ca.shape[1] // BIN_FACTOR_Y) * BIN_FACTOR_Y
    x1 = 0
    x2 = (top_beam_ca.shape[2] // BIN_FACTOR_X_CA) * BIN_FACTOR_X_CA
    binned_top_beam_ca = np.mean(
        top_beam_ca[:, y1:y2, x1:x2, :].reshape(
            top_beam_ca.shape[0], top_beam_ca.shape[1] // BIN_FACTOR_Y, BIN_FACTOR_Y,
                               top_beam_ca.shape[2] // BIN_FACTOR_X_CA, BIN_FACTOR_X_CA, top_beam_ca.shape[3]
        ),
        (2, 4)
    )
    binned_bottom_beam_ca = np.mean(
        bottom_beam_ca[:, y1:y2, x1:x2, :].reshape(
            bottom_beam_ca.shape[0], bottom_beam_ca.shape[1] // BIN_FACTOR_Y, BIN_FACTOR_Y,
                                  bottom_beam_ca.shape[2] // BIN_FACTOR_X_CA, BIN_FACTOR_X_CA, bottom_beam_ca.shape[3]
        ),
        (2, 4)
    )

    resampled_top_beam_ca = resample_full_data(binned_top_beam_ca, CDELTX_CA, CDELTY)
    resampled_bottom_beam_ca = resample_full_data(binned_bottom_beam_ca, CDELTX_CA, CDELTY)

    plt.imshow(resampled_top_beam_ca[0, :, :, 8], cmap='gray', origin='lower')

    points_ca_top_beam = np.array(plt.ginput(2, 600))

    points_ca_top_beam = points_ca_top_beam.astype(np.int64)

    plt.close('all')
    plt.clf()
    plt.cla()

    plt.imshow(resampled_bottom_beam_ca[0, :, :, 8], cmap='gray', origin='lower')

    points_ca_bottom_beam = np.array(plt.ginput(2, 600))

    points_ca_bottom_beam = points_ca_bottom_beam.astype(np.int64)

    plt.close('all')
    plt.clf()
    plt.cla()

    common_area_point = [
        (
            np.max([points_ca_top_beam[0][0], points_ca_bottom_beam[0][0]]),
            np.max([points_ca_top_beam[0][1], points_ca_bottom_beam[0][1]])
        ),
        (
            np.min([points_ca_top_beam[1][0], points_ca_bottom_beam[1][0]]),
            np.min([points_ca_top_beam[1][1], points_ca_bottom_beam[1][1]])
        )
    ]

    common_area_point = np.array(common_area_point).astype(np.int64)

    while True:

        corrected_ca_intensity_data_top_beam = np.transpose(
            vec_correct_for_straylight(
                np.transpose(
                    resampled_top_beam_ca,
                    axes=(0, 3, 1, 2)
                ),
                fwhm_ca,
                k_value_ca,
                kernel_size=kernel_size_ca
            ),
            axes=(0, 2, 3, 1)
        )

        corrected_ca_intensity_data_bottom_beam = np.transpose(
            vec_correct_for_straylight(
                np.transpose(
                    resampled_bottom_beam_ca,
                    axes=(0, 3, 1, 2)
                ),
                fwhm_ca,
                k_value_ca,
                kernel_size=kernel_size_ca
            ),
            axes=(0, 2, 3, 1)
        )

        if np.min(corrected_ca_intensity_data_top_beam[:, common_area_point[0][1]:common_area_point[1][1], common_area_point[0][0]:common_area_point[1][0], :]) <= 0 or np.min(corrected_ca_intensity_data_bottom_beam[:, common_area_point[0][1]:common_area_point[1][1], common_area_point[0][0]:common_area_point[1][0], :]) <= 0:
            k_value_ca -= 0.01
            sys.stdout.write('Decreasing k_value_ca to {}\n'.format(k_value_ca))
        else:
            break

    corrected_ca_intensity_data_top_beam[np.where(corrected_ca_intensity_data_top_beam <= 0)] = 0

    corrected_ca_intensity_data_bottom_beam[np.where(corrected_ca_intensity_data_bottom_beam <= 0)] = 0

    new_stokes_top_ca, new_stokes_bot_ca = get_stokes_from_observations(
        corrected_ca_intensity_data_top_beam,
        corrected_ca_intensity_data_bottom_beam,
        im_top_ca,
        im_bot_ca
    )

    total_stokes_ca = np.zeros_like(new_stokes_top_ca)
    total_stokes_ca[0] = (new_stokes_top_ca[0] + new_stokes_bot_ca[0]) / 2
    for i in range(1, 4):
        total_stokes_ca[i] = ((new_stokes_top_ca[i] / new_stokes_top_ca[0]) + (new_stokes_bot_ca[i] / new_stokes_bot_ca[0])) / 2
        total_stokes_ca[i] *= total_stokes_ca[0]

    total_stokes_ca[np.where(np.isnan(total_stokes_ca))] = 0

    sunpy.io.write_file(
        fname=level5path / 'total_stokes_{}_DETECTOR_3.fits'.format(timestring),
        data=total_stokes_ca,
        header=OrderedDict({'CDELTY': AIA_SAMPLING_ARCSEC, 'CDELTX': AIA_SAMPLING_ARCSEC, 'UNIT': 'arcsec'}),
        filetype='fits',
        overwrite=True
    )

    residuals_ca = np.zeros_like(new_stokes_top_ca)
    residuals_ca[0] = (new_stokes_top_ca[0] - new_stokes_bot_ca[0]) / 2
    for i in range(1, 4):
        residuals_ca[i] = ((new_stokes_top_ca[i] / new_stokes_top_ca[0]) - (new_stokes_bot_ca[i] / new_stokes_bot_ca[0])) / 2

    residuals_ca[np.where(np.isnan(residuals_ca))] = 0

    sunpy.io.write_file(
        fname=level5path / 'residuals_{}_DETECTOR_3.fits'.format(timestring),
        data=residuals_ca,
        header=OrderedDict({'CDELTY': AIA_SAMPLING_ARCSEC, 'CDELTX': AIA_SAMPLING_ARCSEC, 'UNIT': 'arcsec'}),
        filetype='fits',
        overwrite=True
    )

    fo = h5py.File(level5path / 'Spatial_straylight_correction_{}_{}.h5'.format(datestring, timestring), 'w')

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

    fo['image_contrast_halpha_improved'] = image_contrast_ha_new

    fo['image_contrast_halpha_original'] = image_contrast_ha_old

    fo['image_contrast_ca_improved'] = image_contrast_ca_new

    fo['image_contrast_ca_original'] = image_contrast_ca_old

    fo['max_fwhm_ha'] = max_fwhm_ha

    fo['max_fwhm_ca'] = max_fwhm_ca

    fo.close()


def get_demodulation_matrix(mod_matrix):
    return np.matmul(
        np.linalg.inv(
            np.matmul(
                mod_matrix.T,
                mod_matrix
            )
        ),
        mod_matrix.T
    )
def get_stokes_from_observations(
    top_beam, bottom_beam,
    modulation_matrix_top, modulation_matrix_bot
):
    demod_top = get_demodulation_matrix(modulation_matrix_top)

    demod_bot = get_demodulation_matrix(modulation_matrix_bot)

    stokes_top = np.einsum('ij, jklm-> iklm', demod_top, top_beam)

    stokes_bot = np.einsum('ij, jklm-> iklm', demod_bot, bottom_beam)

    return stokes_top, stokes_bot


def read_file_for_observations(filename, t1=0, t2=270, t3=530, t4=None, cut_shape=270):
    print (filename)
    data, header = sunpy.io.read_file(filename)[0]

    return np.reshape(
        data[:, t1:t2, :],
        (4, data.shape[0] // 4, cut_shape, data.shape[2]),
        order='F'
    ), np.reshape(
        data[:, t3:t4, :],
        (4, data.shape[0] // 4, cut_shape, data.shape[2]),
        order='F'
    )


if __name__ == '__main__':
    # datestring = '20230527'
    # timestring = '074428'
    # hmi_time = '044800'
    # hmi_cont_file = 'hmi.Ic_720s.{}_{}_TAI.3.continuum.fits'.format(datestring, hmi_time)
    # hmi_ref_file = 'HMI_reference_image_{}_{}.fits'.format(datestring, timestring)
    # estimate_alpha_and_sigma(
    #     datestring,
    #     timestring,
    #     hmi_cont_file,
    #     hmi_ref_file,
    # )

    # datestring = '20230603'
    # timestring = '073616'
    # hmi_time = '043600'
    # hmi_cont_file = 'hmi.Ic_720s.{}_{}_TAI.3.continuum.fits'.format(datestring, hmi_time)
    # hmi_ref_file = 'HMI_reference_image_{}_{}.fits'.format(datestring, timestring)

    # datestring = '20230603'
    # timestring = '092458'
    # hmi_time = '063600'
    # hmi_cont_file = 'hmi.Ic_720s.{}_{}_TAI.3.continuum.fits'.format(datestring, hmi_time)
    # hmi_ref_file = 'HMI_reference_image_{}_{}.fits'.format(datestring, timestring)

    # datestring = '20230601'
    # timestring = '081014'
    # hmi_time = '050000'
    # hmi_cont_file = 'hmi.Ic_720s.{}_{}_TAI.3.continuum.fits'.format(datestring, hmi_time)
    # hmi_ref_file = 'HMI_reference_image_{}_{}.fits'.format(datestring, timestring)

    datestring = '20230527'
    timestring = '074428'
    hmi_time = '044800'
    hmi_cont_file = 'hmi.Ic_720s.{}_{}_TAI.3.continuum.fits'.format(datestring, hmi_time)
    hmi_ref_file = 'HMI_reference_image_{}_{}.fits'.format(datestring, timestring)

    # im_top_ha, im_bot_ha = (
    #     np.array(
    #         [
    #             [1., 0.55195976, 0.62138789, -0.22471723],
    #             [1., -0.19442698, 0.29436138, 0.59479479],
    #             [1., 0.30880058, -0.72689111, 0.31905054],
    #             [1., -0.78777467, -0.14543414, -0.25590761]
    #         ]
    #     ), np.array(
    #         [
    #             [1., -0.5410679, -0.60912599, 0.22028286],
    #             [1., 0.21314264, -0.32269679, -0.6520501],
    #             [1., -0.30670993, 0.72196991, -0.3168905],
    #             [1., 0.77726088, 0.14349315, 0.25249222]
    #         ]
    #     )
    # )

    # im_top_ha, im_bot_ha = (
    #     np.array(
    #         [
    #             [ 1.        ,  0.61475914,  0.67701889, -0.20614493],
    #             [ 1.        , -0.3040722 ,  0.42764098,  0.90603611],
    #             [ 1.        ,  0.36038518, -0.85531618,  0.36721698],
    #             [ 1.        , -0.86275097, -0.1436074 , -0.20420006]
    #         ]
    #     ),
    #     np.array(
    #         [
    #             [ 1.        , -0.54810641, -0.6036159 ,  0.18379451],
    #             [ 1.        ,  0.27147574, -0.38179798, -0.80890928],
    #             [ 1.        , -0.32436719,  0.76983327, -0.3305162 ],
    #             [ 1.        ,  0.77374602,  0.12879226,  0.18313393]
    #         ]
    #     )
    # )

    im_top_ha, im_bot_ha = (np.array([[ 1.        ,  0.6135243 ,  0.60096842, -0.4590448 ],
        [ 1.        , -0.36108706,  0.57706807,  0.53667164],
        [ 1.        ,  0.41747739, -0.86022583,  0.19964717],
        [ 1.        , -0.81371066, -0.15353847, -0.40048565]]),
 np.array([[ 1.        , -0.64917944, -0.63589387,  0.48572231],
        [ 1.        ,  0.39408896, -0.62980976, -0.58572126],
        [ 1.        , -0.43375042,  0.89375694, -0.2074293 ],
        [ 1.        ,  0.86785859,  0.1637556 ,  0.42713574]]))

    # im_top_ca, im_bot_ca = (
    #     np.array(
    #         [
    #             [1.,  0.67750786,  0.4346143, -0.31892562],
    #             [1., -0.07086019,  0.23021352,  0.92043555],
    #             [1.,  0.01180473, -0.949532, -0.10441373],
    #             [1., -0.63053025,  0.22564667, -0.67858406]]),
    #     np.array(
    #         [
    #             [1., -0.74357133, -0.47699334,  0.35002391],
    #             [1.,  0.07005817, -0.22760786, -0.91001763],
    #             [1., -0.0108868 ,  0.87569699,  0.09629458],
    #             [1.,  0.60529755, -0.21661669,  0.65142833]
    #         ]
    #     )
    # )

    im_top_ca, im_bot_ca = (np.array([[ 1.        ,  0.57371367,  0.29576091, -0.37156289],
        [ 1.        , -0.11671003,  0.21639848,  0.72208786],
        [ 1.        ,  0.07750957, -0.7315419 , -0.10973425],
        [ 1.        , -0.43387302,  0.17671311, -0.60011508]]),
 np.array([[ 1.        , -0.49002225, -0.2526163 ,  0.31736054],
        [ 1.        ,  0.10077224, -0.18684736, -0.62348038],
        [ 1.        , -0.0657335 ,  0.62039836,  0.09306227],
        [ 1.        ,  0.36587365, -0.14901749,  0.50606118]]))

    t1_HA = 0
    t2_HA = 270
    t3_HA = 530
    t4_HA = None
    cut_shape_HA = 270

    t1_CA = 0
    t2_CA = 285
    t3_CA = 739
    t4_CA = None
    cut_shape_CA = 285

    SCAN_STEP = 0.05  # mm
    IMAGE_SCALE = 5.5  # arcsec mm-1
    BIN_FACTOR_Y = int(AIA_SAMPLING_ARCSEC // (IMAGE_SCALE * SCAN_STEP))
    CDELTY = BIN_FACTOR_Y * IMAGE_SCALE * SCAN_STEP

    PIXEL_SIZE_HA = 6.5 / 1000  # mm
    BINNING = 4
    BIN_FACTOR_X_HA = int(AIA_SAMPLING_ARCSEC // (PIXEL_SIZE_HA * BINNING * IMAGE_SCALE))
    CDELTX_HA = BIN_FACTOR_X_HA * PIXEL_SIZE_HA * BINNING * IMAGE_SCALE

    PIXEL_SIZE_CA = 4.6 / 1000  # mm
    BINNING = 4
    BIN_FACTOR_X_CA = int(AIA_SAMPLING_ARCSEC // (PIXEL_SIZE_CA * BINNING * IMAGE_SCALE))
    CDELTX_CA = BIN_FACTOR_X_CA * PIXEL_SIZE_CA * BINNING * IMAGE_SCALE

    estimate_alpha_and_sigma(
        datestring,
        timestring,
        hmi_cont_file,
        hmi_ref_file,
        BIN_FACTOR_Y,
        CDELTY,
        im_top_ha,
        im_bot_ha,
        t1_HA,
        t2_HA,
        t3_HA,
        t4_HA,
        cut_shape_HA,
        BIN_FACTOR_X_HA,
        CDELTX_HA,
        im_top_ca,
        im_bot_ca,
        t1_CA,
        t2_CA,
        t3_CA,
        t4_CA,
        cut_shape_CA,
        BIN_FACTOR_X_CA,
        CDELTX_CA
    )