import sys
# sys.path.insert(1, '/home/harsh/CourseworkRepo/stic/example')
# sys.path.insert(2, '/home/harsh/CourseworkRepo/WFAComparison')
sys.path.insert(1, '/mnt/f/Harsh/CourseworkRepo/stic/example')
import h5py
import numpy as np
import sunpy.io
from pathlib import Path
from prepare_data import *
from stray_light_approximation import *
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters
from copy import deepcopy
import scipy.ndimage
import sunpy.image.resample
import subprocess


AIA_SAMPLING_ARCSEC = 0.6

def resample_data(image, new_shape):
    return sunpy.image.resample.resample(
        orig=image,
        dimensions=new_shape,
        method='linear',
        minusone=False
    )


vec_resample_data = np.vectorize(resample_data, signature='(x,y),(n)->(w,z)')


def resample_full_data(data, header):
    new_shape = (
        data.shape[1] * header['CDELT3'] / AIA_SAMPLING_ARCSEC,
        data.shape[2] * header['CDELT2'] / AIA_SAMPLING_ARCSEC,
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
        try:
            intensity = np.array(int_list)
            return (intensity / intensity[0]) - data
        except Exception as e:
            import ipdb;ipdb.set_trace()

    return optimize_func_residual

cw = np.asarray([6563, 8662])
cont = []
for ii in cw:
    cont.append(getCont(ii))


def get_raw_data(filename, wave_indice=None):

    data, header = sunpy.io.read_file(filename)[0]

    if wave_indice is None:
        return data, header
    return data[:, :, :, wave_indice[0]:wave_indice[-1]], header


def correct_for_straylight(data, r_straylight, multiplicative_factor):

    stray_corrected_data = deepcopy(data)

    stray_corrected_data[0, :, :, ] = stray_corrected_data[0] * multiplicative_factor[np.newaxis, np.newaxis, :]

    stray_corrected_data[0] = (stray_corrected_data[0] - ((r_straylight) * stray_corrected_data[0, :, :, 0][:, :, np.newaxis])) / (1 - (r_straylight))

    return stray_corrected_data


def generate_stic_input_files(
        fits_file, r_fwhm, r_sigma, r_straylight,
        multiplicative_factor, wave, wave_name,
        stic_cgs_calib_factor, norm_median_stray,
        write_path, datestring, timestring, wave_indice=None):

    data, header = get_raw_data(fits_file, wave_indice)

    stray_corrected_data = correct_for_straylight(
        data, r_straylight, multiplicative_factor
    )

    resampled_stray_corrected_data = resample_full_data(data, header)

    f = h5py.File(
        write_path / '{}_{}_{}_stray_corrected.h5'.format(
            wave_name, datestring, timestring
        ),
        'w'
    )

    f['stray_corrected_data'] = stray_corrected_data

    f['resampled_stray_corrected_data'] = resampled_stray_corrected_data

    f['stray_corrected_median'] = norm_median_stray

    f['stic_cgs_calib_factor'] = stic_cgs_calib_factor

    f['wave'] = wave

    f.close()

    plt.close('all')

    plt.imshow(resampled_stray_corrected_data[0, :, :, 50], cmap='gray', origin='lower')

    plt.savefig(write_path / '{}_{}_{}_continuum_image.pdf'.format(wave_name, datestring, timestring), format='pdf', dpi=300)

    fov_data = resampled_stray_corrected_data[:, :, :, :]

    wc, ic = findgrid(wave, (wave[10] - wave[9])*0.25, extra=8)

    prof = sp.profile(nx=fov_data.shape[2], ny=fov_data.shape[1], ns=4, nw=wc.size)

    prof.wav[:] = wc

    prof.dat[0, :, :, ic, :] = np.transpose(
        fov_data,
        axes=(3, 1, 2, 0)
    ) / stic_cgs_calib_factor

    if wc.size%2 == 0:
        kernel_size = wc.size - 1
    else:
        kernel_size = wc.size - 2
    rev_kernel = np.zeros(kernel_size)
    rev_kernel[kernel_size//2] = 1
    kernel = scipy.ndimage.gaussian_filter1d(rev_kernel, sigma=r_sigma * 4)

    broadening_filename = 'gaussian_broadening_{}_pixel_{}.h5'.format(np.round(r_fwhm*4, 1), wave_name)
    f = h5py.File(write_path / broadening_filename, 'w')
    f['iprof'] = kernel
    f['wav'] = np.zeros_like(kernel)
    f.close()

    region_file = write_path / 'region_{}_{}.txt'.format(wave_name, datestring)

    fr = open(region_file, 'w')

    lab = "region = {0:10.5f}, {1:8.5f}, {2:3d}, {3:e}, {4}\n"
    # print(" ")
    # print("Regions information for the input file:")
    fr.write(lab.format(prof.wav[0], prof.wav[1] - prof.wav[0], prof.wav.size, cont[0], 'spectral, {}'.format(broadening_filename)))
    fr.write("(w0, dw, nw, normalization, degradation_type, instrumental_profile file)")
    # print(" ")
    fr.close()

    prof.write(
        write_path / '{}_{}_{}_stic_profiles.nc'.format(
            wave_name, datestring, timestring
        )
    )


def generate_input_atmos_file():

    f = h5py.File('/home/harsh/CourseworkRepo/stic/run/falc_nicole_for_stic.nc', 'r')

    m = sp.model(nx=60, ny=19, nt=1, ndep=150)

    m.ltau[:, :, :] = f['ltau500'][0, 0, 0]

    m.pgas[:, :, :] = 1

    m.temp[:, :, :] = f['temp'][0, 0, 0]

    m.vlos[:, :, :] = f['vlos'][0, 0, 0]

    m.vturb[:, :, :] = f['vturb'][0, 0, 0]

    m.write('falc_60_19.nc')


def calculate_straylight_from_median_profile(
        medianprofile_path, catalog_file, synth_path,
        write_path, datestring, timestring, wave_indice=None,
        cont_wave=0, fac_ind_s=None, fac_ind_e=None
):

    median_profile = np.loadtxt(medianprofile_path)

    if len(median_profile.shape) > 1:
        median_profile = np.median(median_profile, 0)
        np.savetxt(medianprofile_path, median_profile)

    if wave_indice is None:
        wave_indice = [0, median_profile.size]

    median_profile = median_profile[wave_indice[0]:wave_indice[-1]] / median_profile[wave_indice[0]:wave_indice[-1]][cont_wave]

    catalog = np.loadtxt(catalog_file)

    optimize_func_residual = prep_optimize_func(catalog)

    params = Parameters()

    if catalog[0, 0] < 6570:
        plt.close('all')
        plt.plot(median_profile)
        point = np.array(plt.ginput(2, 600))
        a, b = np.polyfit([point[0][0], point[1][0]], [6562.8, 6560.57], 1)
        wave = np.arange(median_profile.size) * a + b
    else:
        # params.add('wave_0', brute_step=0.1, min=8660, max=8665)
        # params.add('disp', brute_step=0.001, min=0.003, max=0.007)
        #
        # out = minimize(
        #     optimize_func_residual,
        #     params,
        #     args=(
        #         np.arange(median_profile.size),
        #         median_profile
        #     ),
        #     method='brute'
        # )

        # wave = np.arange(median_profile.size) * out.params['disp'].value + out.params['wave_0'].value

        plt.close('all')
        plt.plot(median_profile)
        point = np.array(plt.ginput(3, 600))
        a, b = np.polyfit([point[0][0], point[1][0], point[2][0]], [8661.908, 8661.994, 8662.176], 1)
        wave = np.arange(median_profile.size) * a + b

    if catalog[0, 0] < 6570:
        wavelength = 6562.8

    else:
        wavelength = 8662.14

    norm_line, norm_atlas, atlas_wave = normalise_profiles(
        median_profile,
        wave,
        catalog[:, 1],
        catalog[:, 0],
        cont_wave=wave[cont_wave]
    )

    if fac_ind_s is None:
        fac_ind_s = 0
        fac_ind_e = norm_atlas.size - 1
    else:
        fac_ind_e += norm_atlas.size

    a, b = np.polyfit([fac_ind_s, fac_ind_e], [norm_atlas[fac_ind_s], norm_atlas[fac_ind_e]], 1)

    atlas_slope = a * np.arange(norm_atlas.size) + b

    atlas_slope /= atlas_slope.max()

    a, b = np.polyfit([fac_ind_s, fac_ind_e], [norm_line[fac_ind_s], norm_line[fac_ind_e]], 1)

    line_slope = a * np.arange(norm_line.size) + b

    line_slope /= line_slope.max()

    multiplicative_factor = atlas_slope / line_slope

    multiplicative_factor /= multiplicative_factor.max()

    norm_line *= multiplicative_factor

    result, result_atlas, fwhm, sigma, k_values = approximate_stray_light_and_sigma(
        norm_line[fac_ind_s:fac_ind_e],
        norm_atlas[fac_ind_s:fac_ind_e],
    )

    r_fwhm = fwhm[np.unravel_index(np.argmin(result), result.shape)[0]]

    r_sigma = sigma[np.unravel_index(np.argmin(result), result.shape)[0]]

    r_straylight = k_values[np.unravel_index(np.argmin(result), result.shape)[1]]

    broadening_km_sec = np.abs(r_fwhm * (wave[1] - wave[0]) * 2.99792458e5 / wavelength)

    f = h5py.File(
        write_path / 'straylight_{}_estimated_profile_{}.h5'.format(wavelength, datestring), 'w')

    f['wave'] = wave

    f['correction_factor'] = multiplicative_factor

    f['norm_atlas'] = norm_atlas

    f['median_profile'] = median_profile

    f['norm_median'] = norm_line

    f['mean_square_error'] = result

    f['result_atlas'] = result_atlas

    f['fwhm_in_pixels'] = r_fwhm

    f['sigma_in_pixels'] = r_sigma

    f['straylight_value'] = r_sigma

    f['fwhm'] = fwhm

    f['sigma'] = sigma

    f['k_values'] = k_values

    f['broadening_in_km_sec'] = broadening_km_sec

    f.close()

    stray_corrected_median = median_profile * multiplicative_factor

    stray_corrected_median = (stray_corrected_median - (r_straylight * stray_corrected_median[cont_wave])) / (1 - r_straylight)

    norm_median_stray, norm_atlas, atlas_wave = normalise_profiles(
        stray_corrected_median,
        wave,
        catalog[:, 1],
        catalog[:, 0],
        cont_wave=wave[cont_wave]
    )

    fsynth = h5py.File(synth_path, 'r')

    ind_synth = np.argmin(np.abs(fsynth['wav'][()] - wave[cont_wave]))

    stic_cgs_calib_factor = stray_corrected_median[cont_wave] / fsynth['profiles'][0, 0, 0, ind_synth, 0]

    fsynth.close()

    plt.close('all')

    plt.plot(wave, norm_median_stray, label='Stray Corrected Median')

    plt.plot(wave, scipy.ndimage.gaussian_filter1d(norm_atlas, sigma=r_sigma),
             label='Atlas')

    plt.legend()

    plt.savefig(write_path / '{}_{}_median_comparison.pdf'.format(wavelength, datestring), format='pdf', dpi=300)

    plt.show()

    return r_fwhm, r_sigma, r_straylight, multiplicative_factor, wave, wavelength, broadening_km_sec, stic_cgs_calib_factor, norm_median_stray


def generate_stic_input_files_caller(datestring, cont_wave_ha=0, fac_ind_s_ha=None, fac_ind_e_ha=None, mode=None):
    # base_path = Path(
    #     '/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes/'
    # )

    base_path = Path(
        '/mnt/f/Harsh/CourseworkRepo/InstrumentalUncorrectedStokes'
    )

    # catalog_file_8662 = '/home/harsh/CourseworkRepo/KTTAnalysis/catalog_8662.txt'
    #
    # catalog_file_6563 = '/home/harsh/CourseworkRepo/KTTAnalysis/catalog_6563.txt'
    #
    # synth_file_6563 = '/home/harsh/CourseworkRepo/KTTAnalysis/falc_output_halpha_catalog.nc'
    #
    # synth_file_8662 = '/home/harsh/CourseworkRepo/KTTAnalysis/falc_output_CaII8662_catalog.nc'

    synth_file_6563 = '/mnt/f/Harsh/CourseworkRepo/KTTAnalysis/falc_output_halpha_catalog.nc'

    synth_file_8662 = '/mnt/f/Harsh/CourseworkRepo/KTTAnalysis/falc_output_CaII8662_catalog.nc'

    catalog_file_8662 = '/mnt/f/Harsh/CourseworkRepo/KTTAnalysis/catalog_8662.txt'

    catalog_file_6563 = '/mnt/f/Harsh/CourseworkRepo/KTTAnalysis/catalog_6563.txt'

    datepath = base_path / datestring

    level2path = datepath / 'Level-2'

    level3path = datepath / 'Level-3'

    level2path.mkdir(parents=True, exist_ok=True)

    level3path.mkdir(parents=True, exist_ok=True)

    process = subprocess.Popen(
        'rm -rf *',
        cwd=str(level3path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True
    )

    stdout, stderr = process.communicate()

    all_files = level2path.glob('**/*')

    fits_files = [file for file in all_files if file.name.endswith('.fits') and file.name.startswith('v2u')]

    r_fwhm_ha, r_sigma_ha, r_straylight_ha, multiplicative_factor_ha, wave_ha, wavelength_ha, broadening_km_sec_ha, stic_cgs_calib_factor_ha, norm_median_stray_ha = None, None, None, None, None, None, None, None, None

    r_fwhm_ca, r_sigma_ca, r_straylight_ca, multiplicative_factor_ca, wave_ca, wavelength_ca, broadening_km_sec_ca, stic_cgs_calib_factor_ca, norm_median_stray_ca = None, None, None, None, None, None, None, None, None

    for fits_file in fits_files:
        name_split = fits_file.name.split('_')
        if name_split[-2] == 'halpha':
            if mode is not None and mode == 'ca':
                continue
            timestring = name_split[-1].split('.')[0]

            wave_indice = None

            if r_fwhm_ha is None:
                r_fwhm_ha, r_sigma_ha, r_straylight_ha, multiplicative_factor_ha, wave_ha, wavelength_ha, broadening_km_sec_ha, stic_cgs_calib_factor_ha, norm_median_stray_ha = calculate_straylight_from_median_profile(
                    datepath / 'MedianProfile_DETECTOR_1.txt',
                    catalog_file_6563,
                    synth_file_6563,
                    level3path,
                    datestring,
                    timestring,
                    wave_indice=wave_indice,
                    cont_wave=cont_wave_ha,
                    fac_ind_s=fac_ind_s_ha,
                    fac_ind_e=fac_ind_e_ha
                )

            generate_stic_input_files(fits_file, r_fwhm_ha, r_sigma_ha, r_straylight_ha, multiplicative_factor_ha, wave_ha, 'Halpha',
                                      stic_cgs_calib_factor_ha, norm_median_stray_ha, level3path, datestring, timestring, wave_indice)

        else:
            if mode is not None and mode == 'ha':
                continue
            wave_indice = None
            medprofpath = datepath / 'MedianProfile_DETECTOR_3.txt'
            if not medprofpath.exists():
                medprofpath = datepath / 'MedianProfile_DETECTOR_2.txt'

            timestring = name_split[-1].split('.')[0]

            if r_fwhm_ca is None:
                r_fwhm_ca, r_sigma_ca, r_straylight_ca, multiplicative_factor_ca, wave_ca, wavelength_ca, broadening_km_sec_ca, stic_cgs_calib_factor_ca, norm_median_stray_ca = calculate_straylight_from_median_profile(
                    medprofpath,
                    catalog_file_8662,
                    synth_file_8662,
                    level3path,
                    datestring,
                    timestring,
                    wave_indice
                )

            generate_stic_input_files(fits_file, r_fwhm_ca, r_sigma_ca, r_straylight_ca, multiplicative_factor_ca, wave_ca, 'CaII8662',
                                      stic_cgs_calib_factor_ca, norm_median_stray_ca, level3path, datestring, timestring, wave_indice)

def convert_dat_to_png(base_path):
    if isinstance(base_path, str):
        base_path = Path(base_path)

    all_files = base_path.glob('**/*')

    all_dat_files = [file for file in all_files if file.name.startswith('Curr') and file.name.endswith('.dat')]

    for dat_file in all_dat_files:
        image = np.fromfile(dat_file, dtype=np.float64).reshape(256, 256)
        plt.close('all')
        plt.imshow(image, cmap='gray', origin='lower')
        plt.savefig(base_path / '{}.pdf'.format(dat_file.name), format='pdf', dpi=300)


if __name__ == '__main__':
    datestring = '20230530'
    cont_wave_ha = 0
    fac_ind_s_ha = None
    fac_ind_e_ha = None
    generate_stic_input_files_caller(
        datestring,
        cont_wave_ha,
        fac_ind_s_ha,
        fac_ind_e_ha,
        mode=None
    )

    # merge_ca_ha_data()

    # convert_dat_to_png('/mnt/f/Harsh/CourseworkRepo/New folder/20230603_092238')