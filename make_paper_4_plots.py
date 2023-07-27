import sunpy.io
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
from weak_field_approx import *
from sklearn.decomposition import PCA

def calculate_magnetic_field(datestring,):

    base_path = Path('F:\\Harsh\\CourseworkRepo\\InstrumentalUncorrectedStokes')

    datepath = base_path / datestring

    level4path = datepath / 'Level-4'

    all_files = datepath.glob('**/*')

    all_mag_files = [file for file in all_files if file.name.startswith('aligned') and file.name.endswith('nc')]

    # for a_mag_file in all_mag_files:
    #     fcaha = h5py.File(a_mag_file, 'r')
    #
    #     ind = np.where(fcaha['profiles'][0, 0, 0, :, 0] != 0)[0]
    #
    #     ind = ind[800:]
    #
    #     ca_center_wave = 8662.14 / 10
    #
    #     wave_range = 0.7 / 10
    #
    #     # transition_skip_list = np.array(
    #     #     [
    #     #         [6560.57, 0.25],
    #     #         [6561.09, 0.1],
    #     #         [6562.44, 0.05],
    #     #         [6563.51, 0.15],
    #     #         [6564.15, 0.35]
    #     #     ]
    #     # ) / 10
    #
    #
    #     actual_calculate_blos = prepare_calculate_blos(
    #         fcaha['profiles'][0][:, :, ind],
    #         fcaha['wav'][ind] / 10,
    #         ca_center_wave,
    #         8661.56 / 10,
    #         (8661.56 + 0.3) / 10,
    #         0.8,
    #         transition_skip_list=None,
    #         bin_factor=8
    #         # transition_skip_list=transition_skip_list
    #     )
    #
    #     vec_actual_calculate_blos = np.vectorize(actual_calculate_blos)
    #
    #     magca = np.fromfunction(vec_actual_calculate_blos, shape=(fcaha['profiles'].shape[1], fcaha['profiles'].shape[2]))
    #
    #     sunpy.io.write_file(level4path / '{}_mag_ca.fits'.format(a_mag_file.name), magca, dict(), overwrite=True)

    for a_mag_file in all_mag_files:
        fcaha = h5py.File(a_mag_file, 'r')

        ind = np.where(fcaha['profiles'][0, 0, 0, :, 0] != 0)[0]

        ha_center_wave = 6562.8 / 10
        wave_range = 0.35 / 10

        transition_skip_list = np.array(
            [
                [6560.57, 0.25],
                [6561.09, 0.1],
                [6562.44, 0.05],
                [6563.75, 0.35],
                [6564.15, 0.35]
            ]
        ) / 10

        stokes_V = fcaha['profiles'][0][:, :, ind[0:800]][:, :, :, 3]

        pca_stokes_V = stokes_V.reshape(stokes_V.shape[0] * stokes_V.shape[1], stokes_V.shape[2])

        mu = np.mean(pca_stokes_V, axis=0)

        pca = PCA(n_components=22)

        nComps = 0
        nCompe = 5
        Xhat = np.dot(pca.fit_transform(pca_stokes_V)[:, nComps:nCompe], pca.components_[nComps:nCompe, :])
        Xhat += mu
        # plt.imshow(Xhat, cmap='gray', origin='lower')

        new_stokes_V = Xhat.reshape(stokes_V.shape[0], stokes_V.shape[1], stokes_V.shape[2])

        new_data = np.zeros_like(fcaha['profiles'][0][:, :, ind[0:800]])

        new_data[:, :, :, 3] = new_stokes_V

        actual_calculate_blos = prepare_calculate_blos(
            new_data,
            fcaha['wav'][ind[0:800]] / 10,
            ha_center_wave,
            ha_center_wave - wave_range,
            ha_center_wave + wave_range,
            1.048,
            transition_skip_list=transition_skip_list,
            bin_factor=16
        )

        vec_actual_calculate_blos = np.vectorize(actual_calculate_blos)

        magha = np.fromfunction(vec_actual_calculate_blos, shape=(fcaha['profiles'].shape[1], fcaha['profiles'].shape[2]))

        sunpy.io.write_file(level4path / '{}_mag_ha.fits'.format(a_mag_file.name), magha, dict(), overwrite=True)

    for a_mag_file in all_mag_files:
        fcaha = h5py.File(a_mag_file, 'r')

        ind = np.where(fcaha['profiles'][0, 0, 0, :, 0] != 0)[0]

        ha_center_wave = 6562.8 / 10
        wave_range = 1.5 / 10

        transition_skip_list = np.array(
            [
                [6560.57, 0.25],
                [6561.09, 0.1],
                [6562.44, 0.05],
                [6563.75, 0.35],
                [6564.15, 0.35]
            ]
        ) / 10

        actual_calculate_blos = prepare_calculate_blos(
            fcaha['profiles'][0][:, :, ind[0:800]],
            fcaha['wav'][ind[0:800]] / 10,
            ha_center_wave,
            ha_center_wave - wave_range,
            ha_center_wave + wave_range,
            1.048,
            transition_skip_list=transition_skip_list,
            bin_factor=16
        )

        vec_actual_calculate_blos = np.vectorize(actual_calculate_blos)

        magfha = np.fromfunction(vec_actual_calculate_blos, shape=(fcaha['profiles'].shape[1], fcaha['profiles'].shape[2]))

        sunpy.io.write_file(level4path / '{}_mag_ha_full_line.fits'.format(a_mag_file.name), magfha, dict(), overwrite=True)


if __name__ == '__main__':
    calculate_magnetic_field('20230603')
