import sys
sys.path.insert(1, '/home/harsh/CourseworkRepo/pyMilne')
sys.path.insert(2, '/home/harsh/CourseworkRepo/pyMilne/example_CRISP')
sys.path.append('/home/harsh/CourseworkRepo/pyMilne/')
import time
import MilneEddington as ME
import numpy as np
from pathlib import Path
import h5py
import scipy
import scipy.ndimage
import sunpy.io
import crisp
import scipy.ndimage


def findgrid(w, dw, extra=5):
    """
    Findgrid creates a regular wavelength grid
    with a step of dw that includes all points in
    input array w. It adds extra points at the edges
    for convolution purposes

    Returns the new array and the positions of the
    wavelengths points from w in the new array
    """
    nw = np.int32(np.rint(w / dw))
    nnw = nw[-1] - nw[0] + 1 + 2 * extra

    iw = np.arange(nnw, dtype='float64') * dw - extra * dw + w[0]

    idx = np.arange(w.size, dtype='int32')
    for ii in range(w.size):
        idx[ii] = np.argmin(np.abs(iw - w[ii]))

    return iw, idx


def do_me_inversion():
    dtype = 'float32'

    # processed_inputs = Path('/home/harsh/SpinorNagaraju/maps_1/stic/processed_inputs/')
    processed_inputs = Path('/run/media/harsh/DE52135F52133BA9')

    profile_file = processed_inputs / 'nb_6302_2022-05-19T08_57_55_08_57_55=0-59_stokes_corrected_export2022-09-20T08_30_25_im.fits'

    hdpairs = sunpy.io.read_file(profile_file)
    data, header = hdpairs[0]

    the_data = data[21]

    the_data[np.where(np.isnan(the_data))] = 0

    the_data = scipy.ndimage.rotate(the_data, axes=(2, 3), reshape=False, angle=40.5)[:, :, 214:1116, 206:1116]

    wave_data, _ = hdpairs[1]

    wave = wave_data[0][0][0, :, 0, 0, 2] * 10

    print (wave)

    iw, idx = findgrid(wave, (wave[1] - wave[0]) * 0.25,
                       extra=8)  # Fe I 6302.5

    ny, nx = the_data.shape[2], the_data.shape[3]
    obs = np.zeros((ny, nx, 4, iw.size), dtype=dtype, order='c')

    obs[:, :, :, idx] = np.transpose(the_data, axes=(2, 3, 0, 1))

    print(obs[:, :, :, idx].min())
    print(obs[:, :, :, idx].max())
    #
    # Create sigma array with the estimate of the noise for
    # each Stokes parameter at all wavelengths. The extra
    # non-observed points will have a very large noise (1.e34)
    # (zero weight) compared to the observed ones (3.e-3)
    #
    sig = np.zeros((4, iw.size), dtype=dtype) + 1.e32
    sig[:, idx] = 3.e-3
    # sig[1, idx] = 3.e-3
    # sig[2, idx] = 3.e-3
    # sig[3, idx] = 3.e-3

    #
    # Since the amplitudes of Stokes Q,U and V are very small
    # they have a low imprint in Chi2. We can artificially
    # give them more weight by lowering the noise estimate.
    #
    sig[1:3, idx] /= 10
    sig[3, idx] /= 3.5

    if iw.size % 2 == 0:
        kernel_size = iw.size - 1
    else:
        kernel_size = iw.size - 2
    # rev_kernel = np.zeros(kernel_size)
    # rev_kernel[kernel_size // 2] = 1
    # kernel = scipy.ndimage.gaussian_filter1d(rev_kernel, sigma=2 * 4 / 2.355)

    tw2 = (np.arange(kernel_size, dtype=dtype) - (kernel_size // 2)) * (wave[1] - wave[0]) * 0.25
    tr2 = crisp.crisp(6302.0).dual_fpi(tw2, erh=-0.001)

    regions = [[iw, tr2/tr2.sum()]]

    lines = [6302]
    me = ME.MilneEddington(regions, lines, nthreads=8, precision=dtype)

    #
    # Init model parameters
    #
    labels = ['B [G]', 'inc [rad]', 'azi [rad]', 'Vlos [km/s]', 'vDop [Angstroms]', 'lineop', 'damp', 'S0', 'S1']
    iPar = np.float64([1500, 1, 0, -0.5, 0.035, 50., 0.1, 0.24, 0.7])
    Imodel = me.repeat_model(iPar, ny, nx)

    #
    # Run a first cycle with 4 inversions of each pixel (1 + 3 randomizations)
    #
    t0 = time.time()
    mo, syn, chi2 = me.invert_spatially_regularized(Imodel, obs, sig, nIter=100, chi2_thres=1e-3, mu=0.9, alpha=30.,
                                                    alphas=np.float32([1, 1, 1, 0.01, 0.1, 1.0, 0.1, 0.1, 0.1]),
                                                    method=1, delay_bracket=3)
    t1 = time.time()
    print("dT = {0}s -> <Chi2> (including regularization) = {1}".format(t1 - t0, chi2))

    f = h5py.File(processed_inputs / 'me_results_6302.nc', 'w')
    f['B_abs'] = mo[:, :, 0]
    f['inclination_rad'] = mo[:, :, 1]
    f['azi_rad'] = mo[:, :, 2]
    f['vlos_kms'] = mo[:, :, 3]
    f['vdoppler_angstrom'] = mo[:, :, 4]
    f['line_opacity'] = mo[:, :, 5]
    f['damping'] = mo[:, :, 6]
    f['S0'] = mo[:, :, 7]
    f['S1'] = mo[:, :, 8]
    f['syn'] = syn
    f['chi2'] = chi2
    f.close()


if __name__ == '__main__':
    do_me_inversion()
