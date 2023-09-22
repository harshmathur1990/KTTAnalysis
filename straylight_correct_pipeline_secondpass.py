import sys
# sys.path.insert(1, '/home/harsh/CourseworkRepo/stic/example')
# sys.path.insert(2, '/home/harsh/CourseworkRepo/WFAComparison')
# sys.path.insert(1, '/mnt/f/Harsh/CourseworkRepo/stic/example')
# sys.path.insert(1, 'F:\\Harsh\\CourseworkRepo\\stic\\example')
sys.path.insert(1, '/Volumes/HarshHDD-Data/Documents/CourseworkRepo/stic/example')
import h5py
import numpy as np
import sunpy.io
from pathlib import Path
from prepare_data import *
from stray_light_approximation import *
import matplotlib.pyplot as plt
from copy import deepcopy
import scipy.ndimage
import sunpy.image.resample
import subprocess
from scipy.interpolate import CubicSpline


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


def actual_resample_residual_data(data, header):
    new_shape = (
        data.shape[1] * header['CDELTY'] / AIA_SAMPLING_ARCSEC,
        data.shape[2] * header['CDELTX'] / AIA_SAMPLING_ARCSEC,
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


def resample_residual_data():
    datestring_list = ['20230603', '20230601', '20230531', '20230530', '20230529', '20230527', '20230526', '20230525',
                       '20230522', '20230520', '20230519']

    # base_path = Path(
    #     '/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes/'
    # )

    # base_path = Path(
    #     '/mnt/f/Harsh/CourseworkRepo/InstrumentalUncorrectedStokes'
    # )

    base_path = Path('F:\\Harsh\\CourseworkRepo\\InstrumentalUncorrectedStokes')

    for datestring in datestring_list:

        datepath = base_path / datestring

        level3path = datepath / 'Level-3'

        all_files = datepath.glob('**/*')

        all_residual_files = [file for file in all_files if file.name.startswith('residual') and file.name.endswith('.fits') and 'Level-3' not in str(file)]

        for residual_file in all_residual_files:
            print(residual_file)
            data, header = get_raw_data(residual_file)

            resampled_data = actual_resample_residual_data(data, header)

            new_header = deepcopy(header)

            new_header['CDELTX'] = 0.6

            new_header['CDELTY'] = 0.6

            sunpy.io.write_file(level3path / residual_file.name, resampled_data, new_header, overwrite=True)


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
        nc_file, r_fwhm_ha, r_sigma_ha, r_straylight_ha,
        multiplicative_factor_ha, wave_ha,
        norm_median_stray_ha, ha_ind, r_fwhm_ca, r_sigma_ca, r_straylight_ca,
        multiplicative_factor_ca, wave_ca,
        norm_median_stray_ca, ca_ind,
        write_path, datestring, timestring):

    fo = h5py.File(nc_file, 'r')

    total_ha_ind = np.where(fo['wav'][()] < 7000)[0]

    total_ca_ind = np.where(fo['wav'][()] > 8000)[0]

    stray_corrected_data_ha = correct_for_straylight(
        np.transpose(
            fo['profiles'][0, :, :, ha_ind, :],
            axes=(3, 0, 1, 2)
        ), r_straylight_ha, multiplicative_factor_ha
    )

    stray_corrected_data_ca = correct_for_straylight(
        np.transpose(
            fo['profiles'][0, :, :, ca_ind, :],
            axes=(3, 0, 1, 2)
        ), r_straylight_ca, multiplicative_factor_ca
    )

    f = h5py.File(
        write_path / '{}_{}_stray_corrected.h5'.format(
            datestring, timestring
        ),
        'w'
    )

    f['stray_corrected_data_ha'] = stray_corrected_data_ha

    f['stray_corrected_median_ha'] = norm_median_stray_ha

    f['wave_ha'] = wave_ha

    f['stray_corrected_data_ca'] = stray_corrected_data_ca

    f['stray_corrected_median_ca'] = norm_median_stray_ca

    f['wave_ca'] = wave_ca

    f.close()

    prof_ha = sp.profile(nx=stray_corrected_data_ha.shape[2], ny=stray_corrected_data_ha.shape[1], ns=4, nw=total_ha_ind.size)

    prof_ha.wav[:] = fo['wav'][total_ha_ind]

    prof_ha.dat[0, :, :, ha_ind, :] = np.transpose(
        stray_corrected_data_ha,
        axes=(3, 1, 2, 0)
    )

    if total_ha_ind.size%2 == 0:
        kernel_size = total_ha_ind.size - 1
    else:
        kernel_size = total_ha_ind.size - 2
    rev_kernel = np.zeros(kernel_size)
    rev_kernel[kernel_size//2] = 1
    kernel = scipy.ndimage.gaussian_filter1d(rev_kernel, sigma=r_sigma_ha * 4)

    broadening_filename = 'gaussian_broadening_ha_pixel_{}.h5'.format(np.round(r_fwhm_ha*4, 1))
    f = h5py.File(write_path / broadening_filename, 'w')
    f['iprof'] = kernel
    f['wav'] = np.zeros_like(kernel)
    f.close()

    region_file = write_path / 'region_ha_{}_{}.txt'.format(datestring, timestring)

    fr = open(region_file, 'w')

    lab = "region = {0:10.5f}, {1:8.5f}, {2:3d}, {3:e}, {4}\n"
    # print(" ")
    # print("Regions information for the input file:")
    fr.write(lab.format(prof_ha.wav[0], prof_ha.wav[1] - prof_ha.wav[0], prof_ha.wav.size, cont[0], 'spectral, {}'.format(broadening_filename)))
    fr.write("(w0, dw, nw, normalization, degradation_type, instrumental_profile file)")
    # print(" ")
    fr.close()

    prof_ca = sp.profile(nx=stray_corrected_data_ca.shape[2], ny=stray_corrected_data_ca.shape[1], ns=4,
                         nw=total_ca_ind.size)

    prof_ca.wav[:] = fo['wav'][total_ca_ind]

    prof_ca.dat[0, :, :, ca_ind - total_ha_ind.size, :] = np.transpose(
        stray_corrected_data_ca,
        axes=(3, 1, 2, 0)
    )

    if total_ca_ind.size % 2 == 0:
        kernel_size = total_ca_ind.size - 1
    else:
        kernel_size = total_ca_ind.size - 2
    rev_kernel = np.zeros(kernel_size)
    rev_kernel[kernel_size // 2] = 1
    kernel = scipy.ndimage.gaussian_filter1d(rev_kernel, sigma=r_sigma_ca * 4)

    broadening_filename = 'gaussian_broadening_ca_pixel_{}.h5'.format(np.round(r_fwhm_ca * 4, 1))
    f = h5py.File(write_path / broadening_filename, 'w')
    f['iprof'] = kernel
    f['wav'] = np.zeros_like(kernel)
    f.close()

    region_file = write_path / 'region_ca_{}_{}.txt'.format(datestring, timestring)

    fr = open(region_file, 'w')

    lab = "region = {0:10.5f}, {1:8.5f}, {2:3d}, {3:e}, {4}\n"
    # print(" ")
    # print("Regions information for the input file:")
    fr.write(lab.format(prof_ca.wav[0], prof_ca.wav[1] - prof_ca.wav[0], prof_ca.wav.size, cont[1],
                        'spectral, {}'.format(broadening_filename)))
    fr.write("(w0, dw, nw, normalization, degradation_type, instrumental_profile file)")
    # print(" ")
    fr.close()

    all_profs = prof_ha + prof_ca

    all_profs.write(
        write_path / 'aligned_Ca_Ha_stic_profiles_{}_{}.nc_pca.nc_spatial_straylight_corrected_spectral_straylight_secondpass.nc'.format(
            datestring, timestring
        )
    )

    fo.close()


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
        nc_file,
        catalog_file,
        ctl_variation_file,
        write_path,
        datestring,
        timestring,
        non_zero_ind,
        cont_wave=None,
        fac_ind_s=None,
        fac_ind_e=None,
        mu=0.87
):

    f = h5py.File(nc_file, 'r')

    a, b = np.unravel_index(
        np.argmax(
            f['profiles'][0, :, :, non_zero_ind[8], 0]
        ),
        f['profiles'][0, :, :, non_zero_ind[8], 0].shape
    )

    median_profile = f['profiles'][0, a, b, non_zero_ind, 0]

    catalog = np.loadtxt(catalog_file)

    wave = f['wav'][non_zero_ind]

    if catalog[0, 0] < 6570:
        wavelength = 6562.8

    else:
        wavelength = 8662.14

    fctl = h5py.File(ctl_variation_file, 'r')

    if wavelength == 6562.8:
        mul_fac = fctl['ha_{}'.format(mu)][()]
        mul_wav = fctl['wave_ha'][()]
    else:
        mul_fac = fctl['ca_{}'.format(mu)][()]
        mul_wav = fctl['wave_ca'][()]

    cs = CubicSpline(mul_wav, mul_fac)

    catalog_indice = np.where((catalog[:, 0] >= mul_wav.min()) & (catalog[:, 0] <= mul_wav.max()))[0]

    catalog = catalog[catalog_indice]

    int_mul_fac = cs(catalog[:, 0])

    catalog[:, 1] *= int_mul_fac

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

    stray_corrected_median = median_profile * multiplicative_factor

    stray_corrected_median = (stray_corrected_median - (r_straylight * stray_corrected_median[cont_wave])) / (
                1 - r_straylight)

    # stic_cgs_calib_factor = stray_corrected_median[cont_wave] / fsynth['profiles'][0, 0, 0, ind_synth, 0]

    f = h5py.File(
        write_path / 'straylight_{}_estimated_profile_{}_timestring_{}.h5'.format(wavelength, datestring, timestring), 'w')

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

    norm_median_stray, norm_atlas, atlas_wave = normalise_profiles(
        stray_corrected_median,
        wave,
        catalog[:, 1],
        catalog[:, 0],
        cont_wave=wave[cont_wave]
    )

    plt.close('all')

    plt.plot(wave, norm_median_stray, label='Stray Corrected Median')

    plt.plot(wave, scipy.ndimage.gaussian_filter1d(norm_atlas, sigma=r_sigma),
             label='Atlas')

    plt.legend()

    plt.savefig(write_path / '{}_{}_{}_median_comparison.pdf'.format(wavelength, datestring, timestring), format='pdf', dpi=300)

    plt.show()

    return r_fwhm, r_sigma, r_straylight, multiplicative_factor, wave, wavelength, broadening_km_sec, norm_median_stray


def generate_stic_input_files_caller(datestring, cont_wave=0, fac_ind_s=None, fac_ind_e=None, timestring_only=None, mu=0.87):

    base_path = Path(
        '/Volumes/HarshHDD-Data/Documents/CourseworkRepo/InstrumentalUncorrectedStokes/'
    )

    catalog_base = Path('/Volumes/HarshHDD-Data/Documents/CourseworkRepo/KTTAnalysis')

    catalog_file_8662 = catalog_base / 'catalog_8662.txt'

    catalog_file_6563 = catalog_base / 'catalog_6563.txt'

    ctl_variation_file = catalog_base / 'center_to_limb_variation.h5'

    datepath = base_path / datestring

    level4path = datepath / 'Level-4'

    level5path = datepath / 'Level-5'

    level5path.mkdir(parents=True, exist_ok=True)

    # delete_files = True
    #
    # if timestring_only is not None or mode is not None:
    #     delete_files = False
    #
    # if delete_files:
    #     process = subprocess.Popen(
    #         'rm -rf *',
    #         cwd=str(level5path),
    #         stdout=subprocess.PIPE,
    #         stderr=subprocess.PIPE,
    #         shell=True
    #     )
    #
    #     stdout, stderr = process.communicate()

    all_files = level4path.glob('**/*')

    nc_files = [file for file in all_files if file.name.endswith('spatial_straylight_corrected.nc')]

    for nc_file in nc_files:

        name_split = nc_file.name.split('_')

        timestring = name_split[6].split('.')[0]

        if timestring_only is not None and timestring != timestring_only:
            continue

        f = h5py.File(nc_file, 'r')

        ind = np.where(f['profiles'][0, 0, 0, :, 0] != 0)[0]

        ha_ind = ind[0:800]

        r_fwhm_ha, r_sigma_ha, r_straylight_ha, multiplicative_factor_ha, wave_ha, wavelength_ha, broadening_km_sec_ha, norm_median_stray_ha = calculate_straylight_from_median_profile(
            nc_file=nc_file,
            catalog_file=catalog_file_6563,
            ctl_variation_file=ctl_variation_file,
            write_path=level5path,
            datestring=datestring,
            timestring=timestring,
            non_zero_ind=ha_ind,
            cont_wave=cont_wave,
            fac_ind_s=fac_ind_s,
            fac_ind_e=fac_ind_e,
            mu=mu
        )

        ca_ind = ind[800:]

        r_fwhm_ca, r_sigma_ca, r_straylight_ca, multiplicative_factor_ca, wave_ca, wavelength_ca, broadening_km_sec_ca, norm_median_stray_ca = calculate_straylight_from_median_profile(
            nc_file=nc_file,
            catalog_file=catalog_file_8662,
            ctl_variation_file=ctl_variation_file,
            write_path=level5path,
            datestring=datestring,
            timestring=timestring,
            non_zero_ind=ca_ind,
            cont_wave=cont_wave,
            fac_ind_s=fac_ind_s,
            fac_ind_e=fac_ind_e,
            mu=mu
        )

        generate_stic_input_files(
            nc_file, r_fwhm_ha, r_sigma_ha, r_straylight_ha,
            multiplicative_factor_ha, wave_ha,
            norm_median_stray_ha, ha_ind, r_fwhm_ca, r_sigma_ca, r_straylight_ca,
            multiplicative_factor_ca, wave_ca,
            norm_median_stray_ca, ca_ind,
            level5path, datestring, timestring
        )


def convert_dat_to_png(base_path):
    if isinstance(base_path, str):
        base_path = Path(base_path)

    all_folders = base_path.glob('**/*')

    for folder in all_folders:
        all_files = folder.glob('**/*')

        all_dat_files = [file for file in all_files if file.name.startswith('Curr') and file.name.endswith('.dat')]

        for dat_file in all_dat_files:
            image = np.fromfile(dat_file, dtype=np.float64).reshape(256, 256)
            plt.close('all')
            plt.imshow(image, cmap='gray', origin='lower')
            plt.savefig(folder / '{}.pdf'.format(dat_file.name), format='pdf', dpi=300)


if __name__ == '__main__':
    datestring = '20230603'
    cont_wave = -1
    fac_ind_s = 30
    fac_ind_e = -30
    generate_stic_input_files_caller(
        datestring=datestring,
        cont_wave=0,
        fac_ind_s=None,
        fac_ind_e=None,
        timestring_only='073616',
        mu=0.87
    )

    # merge_ca_ha_data()

    # convert_dat_to_png('/mnt/f/Harsh/CourseworkRepo/Tip Tilt Data/Closed Loop/')

    # resample_residual_data()
