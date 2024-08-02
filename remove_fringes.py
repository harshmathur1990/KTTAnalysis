import sys
import sunpy.io
sys.path.insert(1, '/home/harsh/CourseworkRepo/rvm')
import h5py
import numpy as np
from rvm import *
from pathlib import Path
import shutil
from tqdm import tqdm
from copy import deepcopy
from scipy.signal import savgol_filter


def do_fit_fringes(N, noise_std, x, y, max_freq, widths):

    # Define the basis functions

    y_max = y.max()

    use_y = y / y_max

    # Sines and cosines at different frequencies
    freq = np.arange(max_freq) + 1.0
    for i in range(max_freq):
        tmp = np.sin(freq[i] * x)[:,None]
        if (i == 0):
            Basis = tmp
        else:
            Basis = np.hstack([Basis, tmp])

        tmp = np.cos(freq[i] * x)[:,None]
        Basis = np.hstack([Basis, tmp])

    Basis_fringe = np.copy(Basis)
    n_fringes = Basis_fringe.shape[1]

    # Gaussians with different widths centered at all points
    for i in range(len(widths)):
        tmp = np.exp(-(x - x[:,None])**2 / widths[i])
        Basis = np.hstack([Basis, tmp])

    # A continuum
    Basis = np.hstack([Basis, np.ones(N)[:,None]])

    # Instantitate the RVM object and train it
    p = rvm(Basis, use_y, noise=noise_std)
    p.iterateUntilConvergence()

    # Compute total fits, fringes and line
    fit = Basis @ p.wInferred
    fit_fringes = Basis_fringe @ p.wInferred[0:n_fringes]
    fit_line = fit - fit_fringes

    return fit[:, 0] * y_max, fit_fringes[:, 0] * y_max, fit_line[:, 0] * y_max


# def correct_fringes_for_wave(f, i, j, ind):
#     y = f['profiles'][0, i, j, ind, 0]
#
#     noise_std = 0.02
#
#     wave = f['wav'][ind]
#
#     x = wave - wave[0]
#
#     N = x.size
#
#     max_freq = 151
#
#     widths = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5, 0.7])
#
#     fit, fit_fringes, fit_line = do_fit_fringes(N, noise_std, x, y, max_freq, widths)
#
#     return fit


# def correct_fringes_for_all(filepath):
#     f = h5py.File(filepath, 'r')
#     wfile = filepath.parents[0] / '{}_fringe_corrected.nc'.format(filepath.name)
#     shutil.copy(filepath, wfile)
#     ind = np.where(f['profiles'][0, 0, 0, :, 0])[0]
#     ha_ind = ind[0:800]
#     ca_ind = ind[800:]
#     fw = h5py.File(wfile, 'r+')
#
#     t = tqdm(total=f['profiles'].shape[1] * f['profiles'].shape[2])
#     for i in range(f['profiles'].shape[1]):
#         for j in range(f['profiles'].shape[2]):
#             ca_i = correct_fringes_for_wave(f, i, j, ca_ind)
#             ha_i = correct_fringes_for_wave(f, i, j, ha_ind)
#             total_sp = f['profiles'][0, i, j]
#             total_sp[ca_ind, 0] = ca_i
#             total_sp[ha_ind, 0] = ha_i
#             fw['profiles'][0, i, j] = total_sp
#             assert np.array_equal(fw['profiles'][0, i, j], total_sp)
#             t.update(1)


def correct_fringes_for_raw_one_pixel(profiles):
    x = np.arange(profiles.shape[1]) * 0.004
    N = x.size

    noise_std = 0.02
    max_freq = 151
    widths = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5, 0.7])

    if N == 576:
        si, fit_fringes, fit_line = do_fit_fringes(N, noise_std, x, profiles[0], max_freq, widths)
    else:
        si = savgol_filter(profiles[0], 31, 3)

    sqbyi = savgol_filter(profiles[1] / profiles[0], 31, 3)
    subyi = savgol_filter(profiles[2] / profiles[0], 31, 3)
    svbyi = savgol_filter(profiles[3] / profiles[0], 31, 3)

    sq = sqbyi * si
    su = subyi * si
    sv = svbyi * si

    return si, sq, su, sv


def correct_fringes_raw_data(filepath):
    data, header = sunpy.io.read_file(filepath)[0]
    corrected_data = deepcopy(data)

    t = tqdm(total=data.shape[1] * data.shape[2])
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            profiles = data[:, i, j]
            si, sq, su, sv = correct_fringes_for_raw_one_pixel(profiles)
            profiles[0] = si
            profiles[1] = sq
            profiles[2] = su
            profiles[3] = sv
            corrected_data[:, i, j] = profiles
            assert np.array_equal(corrected_data[:, i, j], profiles)
            t.update(1)

    wfilename = '{}_fringe_corrected.fits'.format(filepath.name)
    wfilepath = filepath.parents[0] / wfilename
    sunpy.io.write_file(wfilepath, corrected_data, header, overwrite=True)


if __name__ == '__main__':
    base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes')
    datestring = '20230527'
    datepath = base_path / datestring
    # level4path = datepath / 'Level-4'
    timestring = '074428'
    # filename = 'aligned_Ca_Ha_stic_profiles_{}_{}.nc_pca.nc'.format(datestring, timestring)
    # filepath = level4path / filename
    # correct_fringes_for_all(filepath)
    filename_1 = 'total_stokes_{}_DETECTOR_1.fits_pca.fits'.format(timestring)
    filepath_1 = datepath / filename_1
    filename_2 = 'total_stokes_{}_DETECTOR_3.fits_pca.fits'.format(timestring)
    filepath_2 = datepath / filename_2
    correct_fringes_raw_data(filepath_1)
    correct_fringes_raw_data(filepath_2)
