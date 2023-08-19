import sys
sys.path.insert(1, '/home/harsh/CourseworkRepo/stic/example')
# sys.path.insert(2, '/home/harsh/CourseworkRepo/WFAComparison')
# sys.path.insert(1, '/mnt/f/Harsh/CourseworkRepo/stic/example')
sys.path.insert(1, 'F:\\Harsh\\CourseworkRepo\\stic\\example')
from sklearn.decomposition import PCA
import h5py
import sunpy.io
import numpy as np
from pathlib import Path
from prepare_data import *


def do_pca(all_data, ind):

    stokes_V = all_data[:, :, ind, 3]

    pca_stokes_V = stokes_V.reshape(stokes_V.shape[0] * stokes_V.shape[1], stokes_V.shape[2])

    mu = np.mean(pca_stokes_V, axis=0)

    pca = PCA(n_components=50)

    nComps = 0
    nCompe = 5
    Xhat = np.dot(pca.fit_transform(pca_stokes_V)[:, nComps:nCompe], pca.components_[nComps:nCompe, :])
    Xhat += mu

    new_stokes_V = Xhat.reshape(stokes_V.shape[0], stokes_V.shape[1], stokes_V.shape[2])

    stokes_Q = all_data[:, :, ind, 1]

    pca_stokes_Q = stokes_Q.reshape(stokes_Q.shape[0] * stokes_Q.shape[1], stokes_Q.shape[2])

    mu = np.mean(pca_stokes_Q, axis=0)

    pca = PCA(n_components=50)

    nComps = 0
    nCompe = 5
    Xhat = np.dot(pca.fit_transform(pca_stokes_Q)[:, nComps:nCompe], pca.components_[nComps:nCompe, :])
    Xhat += mu

    new_stokes_Q = Xhat.reshape(stokes_Q.shape[0], stokes_Q.shape[1], stokes_Q.shape[2])

    stokes_U = all_data[:, :, ind, 2]

    pca_stokes_U = stokes_U.reshape(stokes_U.shape[0] * stokes_U.shape[1], stokes_U.shape[2])

    mu = np.mean(pca_stokes_U, axis=0)

    pca = PCA(n_components=50)

    nComps = 0
    nCompe = 5
    Xhat = np.dot(pca.fit_transform(pca_stokes_U)[:, nComps:nCompe], pca.components_[nComps:nCompe, :])
    Xhat += mu

    new_stokes_U = Xhat.reshape(stokes_U.shape[0], stokes_U.shape[1], stokes_U.shape[2])

    stokes_I = all_data[:, :, ind, 0]

    pca_stokes_I = stokes_I.reshape(stokes_I.shape[0] * stokes_I.shape[1], stokes_I.shape[2])

    mu = np.mean(pca_stokes_I, axis=0)

    pca = PCA(n_components=50)

    nComps = 0
    nCompe = 5
    Xhat = np.dot(pca.fit_transform(pca_stokes_I)[:, nComps:nCompe], pca.components_[nComps:nCompe, :])
    Xhat += mu

    new_stokes_I = Xhat.reshape(stokes_I.shape[0], stokes_I.shape[1], stokes_I.shape[2])

    new_data = np.zeros(
        (
            1,
            all_data.shape[0],
            all_data.shape[1],
            ind.shape[0],
            4
        ),
        dtype=np.float64
    )

    new_data[:, :, :, :, 0] = new_stokes_I[np.newaxis]
    new_data[:, :, :, :, 1] = new_stokes_Q[np.newaxis]
    new_data[:, :, :, :, 2] = new_stokes_U[np.newaxis]
    new_data[:, :, :, :, 3] = new_stokes_V[np.newaxis]

    return new_data

def pca_stokes_params(datestring):
    # base_path = Path('F:\\Harsh\\CourseworkRepo\\InstrumentalUncorrectedStokes')

    # base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes')

    base_path = Path('C:\\Work Things\\InstrumentalUncorrectedStokes')

    datepath = base_path / datestring

    level8path = datepath / 'Level-4'

    all_files = level8path.glob('**/*')

    all_mag_files = [file for file in all_files if file.name.startswith('aligned') and file.name.endswith('spatial_straylight_corrected.nc')]

    for a_mag_file in all_mag_files:
        fcaha = h5py.File(a_mag_file, 'r')

        ind = np.where(fcaha['profiles'][0, 0, 0, :, 0] != 0)[0]

        all_data = fcaha['profiles'][0]

        new_data_halpha = do_pca(all_data, ind[0:800])

        new_data_ca = do_pca(all_data, ind[800:])

        ha = sp.profile(
            nx=fcaha['profiles'].shape[2], ny=fcaha['profiles'].shape[1], ns=4,
            nw=3204
        )
        ca = sp.profile(
            nx=fcaha['profiles'].shape[2], ny=fcaha['profiles'].shape[1], ns=4,
            nw=2308
        )

        ha.wav[:] = fcaha['wav'][0:3204]

        ha.weights = fcaha['weights'][0:3204]

        ca.wav[:] = fcaha['wav'][3204:]

        ca.weights = fcaha['weights'][3204:]

        ha.dat[:, :, :, ind[0:800]] = new_data_halpha

        ca.dat[:, :, :, ind[800:] - 3204] = new_data_ca

        all_profiles = ha + ca

        all_profiles.write(
            level8path / '{}_pca.nc'.format(a_mag_file.name)
        )


if __name__ == '__main__':
    pca_stokes_params('20230603')
