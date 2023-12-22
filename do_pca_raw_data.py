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


def do_pca(filepath):

    all_data, header = sunpy.io.read_file(filepath)[0]

    stokes_V = all_data[3]

    pca_stokes_V = stokes_V.reshape(stokes_V.shape[0] * stokes_V.shape[1], stokes_V.shape[2])

    mu = np.mean(pca_stokes_V, axis=0)

    pca = PCA(n_components=50)

    nComps = 0
    nCompe = 5
    Xhat = np.dot(pca.fit_transform(pca_stokes_V)[:, nComps:nCompe], pca.components_[nComps:nCompe, :])
    Xhat += mu

    new_stokes_V = Xhat.reshape(stokes_V.shape[0], stokes_V.shape[1], stokes_V.shape[2])

    stokes_Q = all_data[1]

    pca_stokes_Q = stokes_Q.reshape(stokes_Q.shape[0] * stokes_Q.shape[1], stokes_Q.shape[2])

    mu = np.mean(pca_stokes_Q, axis=0)

    pca = PCA(n_components=50)

    nComps = 0
    nCompe = 5
    Xhat = np.dot(pca.fit_transform(pca_stokes_Q)[:, nComps:nCompe], pca.components_[nComps:nCompe, :])
    Xhat += mu

    new_stokes_Q = Xhat.reshape(stokes_Q.shape[0], stokes_Q.shape[1], stokes_Q.shape[2])

    stokes_U = all_data[2]

    pca_stokes_U = stokes_U.reshape(stokes_U.shape[0] * stokes_U.shape[1], stokes_U.shape[2])

    mu = np.mean(pca_stokes_U, axis=0)

    pca = PCA(n_components=50)

    nComps = 0
    nCompe = 5
    Xhat = np.dot(pca.fit_transform(pca_stokes_U)[:, nComps:nCompe], pca.components_[nComps:nCompe, :])
    Xhat += mu

    new_stokes_U = Xhat.reshape(stokes_U.shape[0], stokes_U.shape[1], stokes_U.shape[2])

    stokes_I = all_data[0]

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
            4,
            all_data.shape[1],
            all_data.shape[2],
            all_data.shape[3]
        ),
        dtype=np.float64
    )

    new_data[0] = new_stokes_I
    new_data[1] = new_stokes_Q
    new_data[2] = new_stokes_U
    new_data[3] = new_stokes_V

    wfilename = '{}_pca.fits'.format(filepath.name)
    wfilepath = filepath.parents[0] / wfilename
    sunpy.io.write_file(wfilepath, new_data, header, overwrite=True)


if __name__ == '__main__':
    base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes')
    datestring = '20230527'
    datepath = base_path / datestring
    timestring = '074428'
    filename_1 = 'total_stokes_{}_DETECTOR_1.fits'.format(timestring)
    filepath_1 = datepath / filename_1
    filename_2 = 'total_stokes_{}_DETECTOR_3.fits'.format(timestring)
    filepath_2 = datepath / filename_2
    do_pca(filepath_1)
    do_pca(filepath_2)
