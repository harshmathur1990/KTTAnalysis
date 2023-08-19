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

    # max_fwhm = np.int64(np.round((kernel_size - b) / a, 0))

    # max_fwhm = 40

    max_fwhm = np.int64(np.round((np.amin(list(obs_image.shape)) - b) / a, 0))

    fwhm = np.linspace(2, max_fwhm, 100)

    sigma = fwhm / 2.355

    k_values = np.arange(0, 1, 0.01)

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

            rev_kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
            
            rev_kernel[kernel_size // 2, kernel_size // 2] = 1

            kernel = scipy.ndimage.gaussian_filter(rev_kernel, sigma=_sig)

            # degraded_image = scipy.signal.oaconvolve(
            #     k_value * hmi_image,
            #     kernel, mode='same') + (
            #         (1 - k_value) * hmi_image
            # )

            degraded_image = k_value * scipy.signal.fftconvolve(hmi_image, kernel, mode='same') + (1 - k_value) * hmi_image

            result_images[i][j] = degraded_image
            corr_image = correct_for_straylight(obs_image, fwhm[i], k_value, kernel_size=kernel_size)
            if degraded_image.min() <= 0 or corr_image.min() <= 0:
                result[i][j] = np.inf
            else:
                # result[i][j] = mean_squarred_error(
                #     obs_image / obs_image.max(), degraded_image / degraded_image.max()
                # )
                # result[i][j] = mean_squarred_error(
                #     (obs_image - obs_image.min()) / (obs_image.max() - obs_image.min()),
                #     (degraded_image - degraded_image.min()) / (degraded_image.max() - degraded_image.min())
                # )

                norm_obs = obs_image #(obs_image - obs_image.min()) / (obs_image.max() - obs_image.min())
                norm_degraded = degraded_image #(degraded_image - degraded_image.min()) / (degraded_image.max() - degraded_image.min())
                result[i][j] = mean_squarred_error(
                    np.mean(norm_obs[y3:y4, x3:x4]) / np.mean(norm_obs[y1:y2, x1:x2]),
                    np.mean(norm_degraded[y3:y4, x3:x4]) / np.mean(norm_degraded[y1:y2, x1:x2])
                )
                # result[i][j] = mean_squarred_error(
                #     np.mean(norm_obs[y3:y4, x3:x4]),
                #     np.mean(norm_degraded[y3:y4, x3:x4])
                # ) + mean_squarred_error(
                #     np.mean(norm_obs[y1:y2, x1:x2]),
                #     np.mean(norm_degraded[y1:y2, x1:x2])
                # )

    sigma_ind, k_ind = np.unravel_index(np.argmin(result), result.shape)

    return result, result_images, fwhm[sigma_ind], k_values[k_ind], sigma_ind, k_ind, max_fwhm


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
        x4,
        normalize_points,
        stic_cgs_calib_factor=1,
        gain_factor=1,
        regularization=1
):
    a = 3.457142857142857

    b = 0.3428571428571427

    max_fwhm = np.int64(np.round((kernel_size - b) / a, 0))

    # max_fwhm = np.amin(list(obs_image.shape))

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

            corr_image = correct_for_straylight(
                obs_image,
                _fwhm,
                k_value,
                kernel_size=kernel_size,
                stic_cgs_calib_factor=stic_cgs_calib_factor,
                gain_factor=gain_factor,
                regularization=regularization
            )

            if corr_image.min() <= 0:
                result[i][j] = np.inf
            else:
                norm_corr_image = corr_image / corr_image[normalize_points[0][1], normalize_points[0][0]]
                norm_hmi_image = hmi_image / hmi_image[normalize_points[0][1], normalize_points[0][0]]
                result[i][j] = mean_squarred_error(
                    np.mean(norm_corr_image[y3:y4, x3:x4]) / np.mean(norm_corr_image[y1:y2, x1:x2]),
                    np.mean(norm_hmi_image[y3:y4, x3:x4]) / np.mean(norm_hmi_image[y1:y2, x1:x2])
                )
                result_images[i, j] = corr_image

    fwhm_ind, k_ind = np.unravel_index(np.argmin(result), result.shape)

    corrected_image = result_images[fwhm_ind, k_ind]

    image_contrast_old = np.mean(obs_image[y3:y4, x3:x4]) / np.mean(obs_image[y1:y2, x1:x2])

    image_contrast_new = np.mean(corrected_image[y3:y4, x3:x4]) / np.mean(corrected_image[y1:y2, x1:x2])

    hmi_contrast = np.mean(hmi_image[y3:y4, x3:x4]) / np.mean(hmi_image[y1:y2, x1:x2])

    return result, result_images, fwhm[fwhm_ind], k_values[k_ind], fwhm_ind, k_ind, max_fwhm, image_contrast_old, image_contrast_new, hmi_contrast


def correct_for_straylight(image, fwhm, k_value, kernel_size=None, stic_cgs_calib_factor=1, gain_factor=1, regularization=1):

    # snr = np.sqrt(image.mean() / 6 / 4.36)

    # nsnr = 1 / snr

    rev_kernel = np.zeros((kernel_size, kernel_size))

    rev_kernel[kernel_size//2, kernel_size//2] = 1

    kernel = scipy.ndimage.gaussian_filter(rev_kernel, sigma=fwhm / 2.355)

    kernel[kernel_size//2, kernel_size//2] = 0

    convolved = scipy.signal.oaconvolve(image, kernel, mode='same')

    # convolved = wiener(
    #     image=image,
    #     psf=kernel,
    #     balance=nsr,
    #     clip=False
    # )

    noise = np.sqrt(
        image * stic_cgs_calib_factor / gain_factor
    ) * gain_factor

    return ((image - k_value * convolved) / (1 - k_value)) - ((regularization * noise) / ((1 - k_value) * stic_cgs_calib_factor))


def get_image_contrast(image):
    nm = (image - image.min()) / (image.max() - image.min())

    return nm.std()


def estimate_alpha_and_sigma(
        datestring, timestring, hmi_cont_file, hmi_ref_file,
):

    # base_path = Path('F:\\Harsh\\CourseworkRepo\\InstrumentalUncorrectedStokes')
    #
    # base_path = Path('/mnt/f/harsh/CourseworkRepo/InstrumentalUncorrectedStokes')

    # base_path = Path('C:\\Work Things\\InstrumentalUncorrectedStokes')

    # base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes/')

    base_path = Path('C:\\Work Things\\InstrumentalUncorrectedStokes')

    datepath = base_path / datestring

    level4path = datepath / 'Level-4'

    level3path = datepath / 'Level-3'

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
        signature='(x,y),(),(),(),(),(),()->(x,y)'
    )

    planck_function_656_nm = prepare_planck_function(6563)

    halpha_image = f['profiles'][0, :, :, 32, 0]

    plt.imshow(halpha_image, cmap='gray', origin='lower')

    points_ha = np.array(plt.ginput(3, 600))

    points_ha = points_ha.astype(np.int64)

    plt.close('all')
    plt.clf()
    plt.cla()

    intensity_6563 = planck_function_656_nm(temperature_map)

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

    # result_ha, result_images_ha, fwhm_ha, k_value_ha, sigma_ind_ha, k_ind_ha, max_fwhm_ha = approximate_stray_light_and_sigma(
    #     norm_halpha_image[points_ha[1][1]:points_ha[2][1], points_ha[1][0]:points_ha[2][0]],
    #     norm_intensity_6563[points_ha[1][1]:points_ha[2][1], points_ha[1][0]:points_ha[2][0]],
    #     kernel_size=kernel_size_ha,
    #     y1=compare_points_ha[0][1],
    #     y2=compare_points_ha[1][1],
    #     x1=compare_points_ha[0][0],
    #     x2=compare_points_ha[1][0],
    #     y3=compare_points_ha[2][1],
    #     y4=compare_points_ha[3][1],
    #     x3=compare_points_ha[2][0],
    #     x4=compare_points_ha[3][0]
    # )

    wave_straylight_file = level3path / 'Halpha_{}_{}_stray_corrected.h5'.format(datestring, timestring)

    fw = h5py.File(wave_straylight_file, 'r')

    stic_cgs_calib_factor_ha = fw['stic_cgs_calib_factor'][()]

    fw.close()

    gain_factor_ha = 6 * 4.36

    regularization_ha = 1

    result_ha, result_images_ha, fwhm_ha, k_value_ha, sigma_ind_ha, k_ind_ha, max_fwhm_ha, image_contrast_ha_old, image_contrast_ha_new, hmi_contrast_ha = approximate_stray_light_and_sigma_alternate(
        obs_image=halpha_image[points_ha[1][1]:points_ha[2][1], points_ha[1][0]:points_ha[2][0]],
        hmi_image=intensity_6563[points_ha[1][1]:points_ha[2][1], points_ha[1][0]:points_ha[2][0]],
        kernel_size=kernel_size_ha,
        y1=compare_points_ha[0][1],
        y2=compare_points_ha[1][1],
        x1=compare_points_ha[0][0],
        x2=compare_points_ha[1][0],
        y3=compare_points_ha[2][1],
        y4=compare_points_ha[3][1],
        x3=compare_points_ha[2][0],
        x4=compare_points_ha[3][0],
        normalize_points=points_ha,
        stic_cgs_calib_factor=stic_cgs_calib_factor_ha,
        gain_factor=gain_factor_ha,
        regularization=regularization_ha
    )

    while True:
        corrected_halpha_intensity_data = np.transpose(
            vec_correct_for_straylight(
                np.transpose(
                    f['profiles'][0, :, :, ind[0:800], 0],
                    axes=(2, 0, 1)
                ),
                fwhm_ha,
                k_value_ha,
                kernel_size=kernel_size_ha,
                stic_cgs_calib_factor=stic_cgs_calib_factor_ha,
                gain_factor=gain_factor_ha,
                regularization=regularization_ha
            ),
            axes=(1, 2, 0)
        )

        if corrected_halpha_intensity_data.min() <= 0:
            k_value_ha -= 0.01
            sys.stdout.write('Decreasing k_value_ha to {}\n'.format(k_value_ha))
        else:
            break

    planck_function_866_nm = prepare_planck_function(8662)

    ca_image = f['profiles'][0, :, :, 3204 + 32, 0]

    plt.imshow(ca_image, cmap='gray', origin='lower')

    points_ca = np.array(plt.ginput(3, 600))

    points_ca = points_ca.astype(np.int64)

    plt.close('all')
    plt.clf()
    plt.cla()

    intensity_8662 = planck_function_866_nm(temperature_map)

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
    plt.clf()
    plt.cla()

    wave_straylight_file = level3path / 'CaII8662_{}_{}_stray_corrected.h5'.format(datestring, timestring)

    fw = h5py.File(wave_straylight_file, 'r')

    stic_cgs_calib_factor_ca = fw['stic_cgs_calib_factor'][()]

    fw.close()

    gain_factor_ca = 2 * 9.36

    regularization_ca = 1

    # result_ca, result_images_ca, fwhm_ca, k_value_ca, sigma_ind_ca, k_ind_ca, max_fwhm_ca = approximate_stray_light_and_sigma(
    #     norm_ca_image[points_ca[1][1]:points_ca[2][1], points_ca[1][0]:points_ca[2][0]],
    #     norm_intensity_8662[points_ca[1][1]:points_ca[2][1], points_ca[1][0]:points_ca[2][0]],
    #     kernel_size=kernel_size_ca,
    #     y1=compare_points_ca[0][1],
    #     y2=compare_points_ca[1][1],
    #     x1=compare_points_ca[0][0],
    #     x2=compare_points_ca[1][0],
    #     y3=compare_points_ca[2][1],
    #     y4=compare_points_ca[3][1],
    #     x3=compare_points_ca[2][0],
    #     x4=compare_points_ca[3][0]
    # )

    result_ca, result_images_ca, fwhm_ca, k_value_ca, sigma_ind_ca, k_ind_ca, max_fwhm_ca, image_contrast_ca_old, image_contrast_ca_new, hmi_contrast_ca = approximate_stray_light_and_sigma_alternate(
        ca_image[points_ca[1][1]:points_ca[2][1], points_ca[1][0]:points_ca[2][0]],
        intensity_8662[points_ca[1][1]:points_ca[2][1], points_ca[1][0]:points_ca[2][0]],
        kernel_size=kernel_size_ca,
        y1=compare_points_ca[0][1],
        y2=compare_points_ca[1][1],
        x1=compare_points_ca[0][0],
        x2=compare_points_ca[1][0],
        y3=compare_points_ca[2][1],
        y4=compare_points_ca[3][1],
        x3=compare_points_ca[2][0],
        x4=compare_points_ca[3][0],
        normalize_points=points_ca,
        stic_cgs_calib_factor=stic_cgs_calib_factor_ca,
        gain_factor=gain_factor_ca,
        regularization=regularization_ca
    )

    plt.imshow(f['profiles'][0, :, :, ind[800:][8], 0], cmap='gray', origin='lower')

    points_ca_correct = np.array(plt.ginput(2, 600))

    points_ca_correct = points_ca_correct.astype(np.int64)

    plt.close('all')
    plt.clf()
    plt.cla()

    while True:
        corrected_ca_intensity_data = np.transpose(
            vec_correct_for_straylight(
                np.transpose(
                    f['profiles'][0, :, :, ind[800:], 0],
                    axes=(2, 0, 1)
                ),
                fwhm_ca,
                k_value_ca,
                kernel_size=kernel_size_ca,
                stic_cgs_calib_factor=stic_cgs_calib_factor_ca,
                gain_factor=gain_factor_ca,
                regularization=regularization_ca
            ),
            axes=(1, 2, 0)
        )

        # if corrected_ca_intensity_data.min() <= 0:
        if np.min(corrected_ca_intensity_data[points_ca_correct[0][1]:points_ca_correct[1][1], points_ca_correct[0][0]:points_ca_correct[1][0]]) <= 0:
            k_value_ca -= 0.01
            sys.stdout.write('Decreasing k_value_ca to {}\n'.format(k_value_ca))
        else:
            break

    corrected_ca_intensity_data[np.where(corrected_ca_intensity_data <= 0)] = 0

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

    fo['image_contrast_halpha_improved'] = image_contrast_ha_new

    fo['image_contrast_halpha_original'] = image_contrast_ha_old

    fo['image_contrast_ca_improved'] = image_contrast_ca_new

    fo['image_contrast_ca_original'] = image_contrast_ca_old

    fo['hmi_contrast_ha'] = hmi_contrast_ha

    fo['hmi_contrast_ca'] = hmi_contrast_ca

    fo['max_fwhm_ha'] = max_fwhm_ha

    fo['max_fwhm_ca'] = max_fwhm_ca

    fo['kernel_size_ha'] = kernel_size_ha

    fo['kernel_size_ca'] = kernel_size_ca

    fo['stic_cgs_calib_factor_ha'] = stic_cgs_calib_factor_ha

    fo['gain_factor_ha'] = gain_factor_ha

    fo['regularization_ha'] = regularization_ha

    fo['stic_cgs_calib_factor_ca'] = stic_cgs_calib_factor_ca

    fo['gain_factor_ca'] = gain_factor_ca

    fo['regularization_ca'] = regularization_ca

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

    datestring = '20230603'
    timestring = '073616'
    hmi_time = '043600'
    hmi_cont_file = 'hmi.Ic_720s.{}_{}_TAI.3.continuum.fits'.format(datestring, hmi_time)
    hmi_ref_file = 'HMI_reference_image_{}_{}.fits'.format(datestring, timestring)
    estimate_alpha_and_sigma(
        datestring,
        timestring,
        hmi_cont_file,
        hmi_ref_file,
    )