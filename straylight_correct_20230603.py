import sys
sys.path.insert(1, '/home/harsh/CourseworkRepo/stic/example')
sys.path.insert(2, '/home/harsh/CourseworkRepo/WFAComparison')
import h5py
import numpy as np
import sunpy.io.fits
from pathlib import Path
from prepare_data import *
from stray_light_approximation import *
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters


def prep_optimize_func(catalog):
    def optimize_func_residual(params, x, data):
        wave_0 = params['wave_0']
        disp = params['disp']
        wave = x * disp + wave_0
        int_list = list()
        for w in wave:
            wave_index = np.argmin(np.abs(catalog[:, 0] - w))
            int_list.append(
                catalog[wave_index, 1]
            )
        intensity = np.array(int_list)
        return (intensity / intensity[0]) - data

    return optimize_func_residual

base_path = Path(
    '/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes/'
)

catalog_file_8662 = '/home/harsh/CourseworkRepo/KTT_Final_Analysis/catalog_8662.txt'

catalog_file_6563 = '/home/harsh/CourseworkRepo/KTT_Final_Analysis/catalog_6563.txt'

cw = np.asarray([6563, 8662])
cont = []
for ii in cw:
    cont.append(getCont(ii))


def get_raw_data(filename):

    data, _ = sunpy.io.read_file(filename)[0]

    return data


def correct_for_straylight(data, median_profile, catalog, write_path, datestring, timestring):
    optimize_func_residual = prep_optimize_func(catalog)

    params = Parameters()

    if catalog[0, 0] < 6570:
        params.add('wave_0', brute_step=0.1, min=6560, max=6566)
        params.add('disp', brute_step=0.001, min=-0.01, max=0)
    else:
        params.add('wave_0', brute_step=0.1, min=8660, max=8665)
        params.add('disp', brute_step=0.001, min=-0.01, max=0.01)

    out = minimize(
        optimize_func_residual,
        params,
        args=(
            np.arange(median_profile.size),
            median_profile
        ),
        method='brute'
    )

    wave = np.arange(median_profile.size) * out.params['disp'].value + out.params['wave_0'].value

    if catalog[0, 0] < 6570:
        wavelength = 6562.8

        wave_indice = [0, median_profile.size]

        wave_name = 'Halpha'
    else:
        wavelength = 8662.14

        wave_indice = [0, median_profile.size]

        wave_name = 'CaII8662'

    stray_corrected_data_list = list()

    stray_corrected_median_list = list()

    stic_cgs_calib_factor_list = list()

    norm_line_list = list()

    norm_atlas_list = list()

    multiplicative_factor_list = list()

    norm_line, norm_atlas, atlas_wave = normalise_profiles(
        median_profile[wave_indice[0]:wave_indice[1]],
        wave[wave_indice[0]:wave_indice[1]],
        catalog[:, 1],
        catalog[:, 0],
        cont_wave=wave[wave_indice[0]]
    )

    a, b = np.polyfit([0, norm_atlas.size-1], [norm_atlas[0], norm_atlas[-1]], 1)

    atlas_slope = a * np.arange(norm_atlas.size) + b

    atlas_slope /= atlas_slope.max()

    a, b = np.polyfit([0, norm_line.size-1], [norm_line[0], norm_line[-1]], 1)

    line_slope = a * np.arange(norm_line.size) + b

    line_slope /= line_slope.max()

    multiplicative_factor = atlas_slope / line_slope

    multiplicative_factor /= multiplicative_factor.max()

    norm_line *= multiplicative_factor

    result, result_atlas, fwhm, sigma, k_values = approximate_stray_light_and_sigma(
        norm_line,
        norm_atlas,
    )

    f = h5py.File(write_path / 'straylight_{}_estimated_profile_{}_{}.h5'.format(wavelength, datestring, timestring), 'w')

    f['correction_factor'] = multiplicative_factor

    f['norm_atlas'] = norm_atlas

    f['median_profile'] = median_profile

    f['norm_median'] = norm_line

    f['mean_square_error'] = result

    f['result_atlas'] = result_atlas

    f['fwhm_in_pixels'] = fwhm[np.unravel_index(np.argmin(result), result.shape)[0]]

    f['sigma_in_pixels'] = sigma[np.unravel_index(np.argmin(result), result.shape)[0]]

    f['straylight_value'] = k_values[np.unravel_index(np.argmin(result), result.shape)[1]]

    f['fwhm'] = fwhm

    f['sigma'] = sigma

    f['k_values'] = k_values

    f['broadening_in_km_sec'] = np.abs(f['fwhm_in_pixels'][()] * (wave[1] - wave[0]) * 2.99792458e5 / wavelength)

    f.close()

    straylight = k_values[np.unravel_index(np.argmin(result), result.shape)[1]]

    r_sigma = sigma[np.unravel_index(np.argmin(result), result.shape)[0]]

    r_fwhm = fwhm[np.unravel_index(np.argmin(result), result.shape)[0]]

    stray_corrected_data = data.copy()

    stray_corrected_data = stray_corrected_data[:, :, :, wave_indice[0]:wave_indice[1]]

    stray_corrected_data[0, :, :, ] = stray_corrected_data[0] * multiplicative_factor

    stray_corrected_data[0] = (stray_corrected_data[0] - ((straylight) * stray_corrected_data[0, :, :, 0][:, :, np.newaxis])) / (1 - (straylight))

    stray_corrected_median = median_profile * multiplicative_factor

    stray_corrected_median = (stray_corrected_median - stray_corrected_median[0]) / (1 - (straylight))

    norm_median_stray, norm_atlas, atlas_wave = normalise_profiles(
        stray_corrected_median,
        wave[wave_indice[0]:wave_indice[1]],
        catalog[:, 1],
        catalog[:, 0],
        cont_wave=wave[wave_indice[0]]
    )

    stic_cgs_calib_factor = stray_corrected_median[0] / cont[0]

    plt.plot(wave[wave_indice[0]:wave_indice[1]], norm_median_stray, label='Stray Corrected Median')

    plt.plot(wave[wave_indice[0]:wave_indice[1]], scipy.ndimage.gaussian_filter1d(norm_atlas, sigma=r_sigma), label='Atlas')

    plt.legend()

    plt.savefig(write_path / '{}_{}_median_comparison.pdf'.format(wave_name, datestring), format='pdf', dpi=300)

    plt.show()


    return stray_corrected_data, stray_corrected_median,\
           stic_cgs_calib_factor, wave_name, wave_indice, wavelength, r_fwhm, r_sigma


def generate_stic_input_files(fits_file, median_file, catalog_file):

    data = get_raw_data(fits_file)

    median_profile = np.loadtxt(median_file)

    median_profile / median_profile[0]

    catalog = np.loadtxt(catalog_file)

    stray_corrected_data, stray_corrected_median, stic_cgs_calib_factor, wave_name, wave_indice, wavelength, r_fwhm, r_sigma = correct_for_straylight(
        data, median_profile, catalog
    )

    ca = None

    for stray_corrected_data, stray_corrected_median, stic_cgs_calib_factor, wave_name, wave_indice, wavelength in zip(stray_corrected_data_list, stray_corrected_median_list, stic_cgs_calib_factor_list, wave_names, wave_indices, wavelengths):

        f = h5py.File(
            write_path / '{}_stray_corrected_{}.h5'.format(
                Path(filename).name, wave_name
            ),
            'w'
        )

        f['stray_corrected_data'] = stray_corrected_data

        f['stray_corrected_median'] = stray_corrected_median

        f['stic_cgs_calib_factor'] = stic_cgs_calib_factor

        f['wave_ca'] = wave_ca[wave_indice[0]:wave_indice[1]]

        f.close()

        fov_data = stray_corrected_data[0:18, :, 230:290, :]

        wc8, ic8 = findgrid(wave_ca[wave_indice[0]:wave_indice[1]], (wave_ca[wave_indice[0]:wave_indice[1]][10] - wave_ca[wave_indice[0]:wave_indice[1]][9])*0.25, extra=8)

        ca_8 = sp.profile(nx=60, ny=18, ns=4, nw=wc8.size)

        ca_8.wav[:] = wc8[:]

        ca_8.dat[0, :, :, ic8, :] = np.transpose(
            fov_data,
            axes=(3, 0, 2, 1)
        ) / stic_cgs_calib_factor

        if ca is None:
            ca = ca_8
        else:
            ca += ca_8

        if wc8.size%2 == 0:
            kernel_size = wc8.size - 1
        else:
            kernel_size = wc8.size - 2
        rev_kernel = np.zeros(kernel_size)
        rev_kernel[kernel_size//2] = 1
        kernel = scipy.ndimage.gaussian_filter1d(rev_kernel, sigma=r_sigma * 4)

        broadening_filename = 'gaussian_broadening_{}_pixel_{}.h5'.format(np.round(r_fwhm*4, 1), wave_name)
        f = h5py.File(write_path / broadening_filename, 'w')
        f['iprof'] = kernel
        f['wav'] = np.zeros_like(kernel)
        f.close()
        lab = "region = {0:10.5f}, {1:8.5f}, {2:3d}, {3:e}, {4}"
        print(" ")
        print("Regions information for the input file:")
        print(lab.format(ca_8.wav[0], ca_8.wav[1] - ca_8.wav[0], ca_8.wav.size, cont[0], 'spectral, {}'.format(broadening_filename)))
        print("(w0, dw, nw, normalization, degradation_type, instrumental_profile file)")
        print(" ")

    ca.write(
        write_path / '{}_stic_profiles.nc'.format(
            Path(filename).name
        )
    )


def generate_input_atmos_file():

    f = h5py.File(falc_file_path, 'r')

    m = sp.model(nx=60, ny=19, nt=1, ndep=150)

    m.ltau[:, :, :] = f['ltau500'][0, 0, 0]

    m.pgas[:, :, :] = 1

    m.temp[:, :, :] = f['temp'][0, 0, 0]

    m.vlos[:, :, :] = f['vlos'][0, 0, 0]

    m.vturb[:, :, :] = f['vturb'][0, 0, 0]

    m.write('falc_60_19.nc')


def generate_stic_input_files_caller(datestring):
    datepath = base_path / datestring

    level2path = datepath / 'Level-2'

    all_files = level2path.glob('**/*')

    fits_files = [file for file in all_files if file.name.endswith('.fits') and file.name.startswith('v2u')]

    for fits_file in fits_files:
        name_split = fits_file.name.split('_')
        if name_split[-2] == 'halpha':
            generate_stic_input_files(fits_file, datepath / 'MedianProfile_DETECTOR_1.fits', catalog_file_6563)
        else:
            medprofpath = datepath / 'MedianProfile_DETECTOR_3.fits'
            if medprofpath.exists():
                generate_stic_input_files(fits_file, datepath / 'MedianProfile_DETECTOR_3.fits', catalog_file_8662)
            else:
                generate_stic_input_files(fits_file, datepath / 'MedianProfile_DETECTOR_2.fits', catalog_file_8662)


if __name__ == '__main__':
    datestring = '20230306'
    generate_stic_input_files_caller(datestring)
