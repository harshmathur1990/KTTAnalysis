import sys
sys.path.insert(1, '/home/harsh/CourseworkRepo/stic/example')
sys.path.insert(2, '/home/harsh/CourseworkRepo/WFAComparison')
sys.path.insert(3, '/mnt/f/Harsh/CourseworkRepo/stic/example')
sys.path.insert(4, 'F:\\Harsh\\CourseworkRepo\\stic\\example')
sys.path.insert(5, '/Volumes/HarshHDD-Data/Documents/CourseworkRepo/stic/example')
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


def correct_for_straylight(data, r_straylight, multiplicative_factor=None):

    stray_corrected_data = deepcopy(data)

    if multiplicative_factor is not None:
        stray_corrected_data[0, :, :, ] = stray_corrected_data[0] * multiplicative_factor[np.newaxis, np.newaxis, :]

    stray_corrected_data[0] = (stray_corrected_data[0] - ((r_straylight) * stray_corrected_data[0, :, :, 0][:, :, np.newaxis])) / (1 - (r_straylight))

    return stray_corrected_data


def generate_stic_input_files(
        nc_file, r_fwhm, r_sigma, r_straylight,
        wave,
        norm_median_stray,
        write_path, datestring, timestring, stic_cgs_calib_factor
):

    data, header = sunpy.io.read_file(nc_file)[0]

    stray_corrected_data = correct_for_straylight(
        np.transpose(
            data[0],
            axes=(3, 0, 1, 2)
        ), r_straylight
    ) / stic_cgs_calib_factor

    f = h5py.File(
        write_path / '{}_{}_hmi_stray_corrected.h5'.format(
            datestring, timestring
        ),
        'w'
    )

    f['stray_corrected_data'] = stray_corrected_data

    f['stray_corrected_median'] = norm_median_stray

    f['wave'] = wave

    f.close()

    wfe, ife = findgrid(wave, (wave[1] - wave[0]) * 0.25, extra=8)

    prof = sp.profile(nx=stray_corrected_data.shape[2], ny=stray_corrected_data.shape[1], ns=4, nw=wfe.size)

    prof.wav[:] = wfe

    prof.dat[0, :, :, ife, :] = np.transpose(
        stray_corrected_data,
        axes=(3, 1, 2, 0)
    )

    if wfe.size%2 == 0:
        kernel_size = wfe.size - 1
    else:
        kernel_size = wfe.size - 2
    rev_kernel = np.zeros(kernel_size)
    rev_kernel[kernel_size//2] = 1
    kernel = scipy.ndimage.gaussian_filter1d(rev_kernel, sigma=r_sigma)

    broadening_filename = 'gaussian_broadening_hmi_pixel_{}.h5'.format(np.round(r_fwhm, 1))
    f = h5py.File(write_path / broadening_filename, 'w')
    f['iprof'] = kernel
    f['wav'] = np.zeros_like(kernel)
    f.close()

    region_file = write_path / 'region_hmi_{}_{}.txt'.format(datestring, timestring)

    fr = open(region_file, 'w')

    lab = "region = {0:10.5f}, {1:8.5f}, {2:3d}, {3:e}, {4}\n"
    # print(" ")
    # print("Regions information for the input file:")
    fr.write(lab.format(prof.wav[0], prof.wav[1] - prof.wav[0], prof.wav.size, cont[0], 'spectral, {}'.format(broadening_filename)))
    fr.write("(w0, dw, nw, normalization, degradation_type, instrumental_profile file)")
    # print(" ")
    fr.close()


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
        falc_output,
        catalog_file,
        ctl_variation_file,
        write_path,
        datestring,
        timestring,
        cont_wave=None,
        fac_ind_s=None,
        fac_ind_e=None,
        mu=0.87
):

    print(nc_file.name)

    data, header = sunpy.io.read_file(nc_file)[0]

    plt.imshow(data[0, :, :, 0, 0], origin='lower', cmap='gray')

    point = np.array(plt.ginput(2, 600))

    point = point.astype(np.int64)

    median_profile = np.mean(data[0, point[0][1]:point[1][1], point[0][0]:point[1][0], :, 0], (0, 1))

    wavelength = 6173.34

    wave = np.array([-172, -103, -34, 34, 103, 172]) / 1000 + 6173.3352

    fal = h5py.File(falc_output, 'r')

    indices = list()

    for w in wave:
        indices.append(np.argmin(np.abs(fal['wav'][()] - w)))

    indices = np.array(indices)

    # norm_line, norm_atlas, atlas_wave = normalise_profiles(
    #     median_profile,
    #     wave,
    #     fal['profiles'][0, 0, 0, :, 0],
    #     fal['wav'][()],
    #     cont_wave=wave[cont_wave]
    # )
    #
    # print(norm_line.shape)
    # print(norm_atlas.shape)
    # if fac_ind_s is None:
    #     fac_ind_s = 0
    #     fac_ind_e = norm_atlas.size - 1
    # else:
    #     fac_ind_e += norm_atlas.size
    #
    # a, b = np.polyfit([fac_ind_s, fac_ind_e], [norm_atlas[fac_ind_s], norm_atlas[fac_ind_e]], 1)
    #
    # atlas_slope = a * np.arange(norm_atlas.size) + b
    #
    # atlas_slope /= atlas_slope.max()
    #
    # a, b = np.polyfit([fac_ind_s, fac_ind_e], [norm_line[fac_ind_s], norm_line[fac_ind_e]], 1)
    #
    # line_slope = a * np.arange(norm_line.size) + b
    #
    # line_slope /= line_slope.max()
    #
    # multiplicative_factor = atlas_slope / line_slope
    #
    # multiplicative_factor /= multiplicative_factor.max()
    #
    # norm_line *= multiplicative_factor

    result, result_atlas, fwhm, sigma, k_values = approximate_stray_light_and_sigma(
        median_profile,
        fal['profiles'][0, 0, 0, :, 0],
        indices=indices,
        continuum=fal['profiles'][0, 0, 0, :, 0][0]
    )

    r_fwhm = fwhm[np.unravel_index(np.argmin(result), result.shape)[0]]

    r_sigma = sigma[np.unravel_index(np.argmin(result), result.shape)[0]]

    r_straylight = k_values[np.unravel_index(np.argmin(result), result.shape)[1]]

    broadening_km_sec = np.abs(r_fwhm * (fal['wav'][1] - fal['wav'][0]) * 2.99792458e5 / wavelength)

    stray_corrected_median = median_profile

    stray_corrected_median = (stray_corrected_median - (r_straylight * stray_corrected_median[cont_wave])) / (
                1 - r_straylight)

    ind_synth = np.argmin(np.abs(fal['wav'][()] - wave[cont_wave]))

    stic_cgs_calib_factor = stray_corrected_median[cont_wave] / fal['profiles'][0, 0, 0, ind_synth, 0]

    f = h5py.File(
        write_path / 'straylight_{}_estimated_profile_{}_timestring_{}.h5'.format(wavelength, datestring, timestring), 'w')

    f['wave'] = wave

    f['median_profile'] = median_profile

    f['mean_square_error'] = result

    f['result_atlas'] = result_atlas

    f['fwhm_in_pixels'] = r_fwhm

    f['sigma_in_pixels'] = r_sigma

    f['straylight_value'] = r_straylight

    f['fwhm'] = fwhm

    f['sigma'] = sigma

    f['k_values'] = k_values

    f['broadening_in_km_sec'] = broadening_km_sec

    f['stic_cgs_calib_factor'] = stic_cgs_calib_factor

    f.close()

    norm_median_stray, norm_atlas, atlas_wave = normalise_profiles(
        stray_corrected_median,
        wave,
        scipy.ndimage.gaussian_filter1d(fal['profiles'][0, 0, 0, :, 0], sigma=r_sigma),
        fal['wav'][()],
        cont_wave=wave[cont_wave]
    )

    plt.close('all')

    plt.plot(wave, norm_median_stray, label='Stray Corrected Median')

    plt.plot(wave, norm_atlas,
             label='Atlas')

    plt.legend()

    plt.savefig(write_path / '{}_{}_{}_median_comparison.pdf'.format(wavelength, datestring, timestring), format='pdf', dpi=300)

    plt.show()

    return r_fwhm, r_sigma, r_straylight, wave, wavelength, broadening_km_sec, norm_median_stray, stic_cgs_calib_factor


def generate_stic_input_files_caller(datestring, cont_wave=0, fac_ind_s=None, fac_ind_e=None, timestring_only=None, mu=0.87):

    # base_path = Path(
    #     '/Volumes/HarshHDD-Data/Documents/CourseworkRepo/InstrumentalUncorrectedStokes/'
    # )

    base_path = Path(
        '/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes/'
    )
    # catalog_base = Path('/Volumes/HarshHDD-Data/Documents/CourseworkRepo/KTTAnalysis')

    catalog_base = Path('/home/harsh/CourseworkRepo/KTTAnalysis')

    catalog_file_8662 = catalog_base / 'catalog_8662.txt'

    catalog_file_6563 = catalog_base / 'catalog_6563.txt'

    ctl_variation_file = catalog_base / 'center_to_limb_variation.h5'

    datepath = base_path / datestring

    level4path = datepath / 'Level-4-alt-alt'

    level5path = datepath / 'Level-5-alt-alt'

    level5path.mkdir(parents=True, exist_ok=True)

    falc_output = level4path / 'falc_nicole_for_stic_FeI_6173_profs.nc'

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

    # nc_files = [file for file in all_files if file.name.endswith('spatial_straylight_corrected.nc')]

    nc_files = [level4path / 'combined_hmi_stokes.fits']

    full_file = level4path / 'combined_hmi_stokes_full_fov.fits'

    for nc_file in nc_files:

        # name_split = nc_file.name.split('_')

        timestring = '074428'# name_split[6].split('.')[0]

        if timestring_only is not None and timestring != timestring_only:
            continue

        r_fwhm, r_sigma, r_straylight, wave, wavelength, broadening_km_sec, norm_median_stray, stic_cgs_calib_factor = calculate_straylight_from_median_profile(
            nc_file=full_file,
            falc_output=falc_output,
            catalog_file=catalog_file_6563,
            ctl_variation_file=ctl_variation_file,
            write_path=level5path,
            datestring=datestring,
            timestring=timestring,
            cont_wave=cont_wave,
            fac_ind_s=fac_ind_s,
            fac_ind_e=fac_ind_e,
            mu=mu
        )

        generate_stic_input_files(
            nc_file, r_fwhm, r_sigma, r_straylight,
            wave,
            norm_median_stray,
            level5path, datestring, timestring, stic_cgs_calib_factor
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
    # datestring = '20230603'
    # cont_wave = -1
    # fac_ind_s = 30
    # fac_ind_e = -30
    # generate_stic_input_files_caller(
    #     datestring=datestring,
    #     cont_wave=0,
    #     fac_ind_s=None,
    #     fac_ind_e=None,
    #     timestring_only='073616',
    #     mu=0.87
    # )
    #
    # datestring = '20230601'
    # cont_wave = -1
    # fac_ind_s = 30
    # fac_ind_e = -30
    # generate_stic_input_files_caller(
    #     datestring=datestring,
    #     cont_wave=0,
    #     fac_ind_s=None,
    #     fac_ind_e=None,
    #     timestring_only='081014',
    #     mu=0.6
    # )

    datestring = '20230527'
    cont_wave = 0
    fac_ind_s = None
    fac_ind_e = None
    generate_stic_input_files_caller(
        datestring=datestring,
        cont_wave=0,
        fac_ind_s=None,
        fac_ind_e=None,
        timestring_only='074428',
        mu=0.96
    )

    # merge_ca_ha_data()

    # convert_dat_to_png('/mnt/f/Harsh/CourseworkRepo/Tip Tilt Data/Closed Loop/')

    # resample_residual_data()
