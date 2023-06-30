from weak_field_approx import *
import h5py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def make_compare_plots():
    base_path_ca = Path('/home/harsh/SpinorNagaraju/maps_1/stic/fulldata_inversions/')
    base_path_ha = Path('/home/harsh/SpinorNagaraju/maps_1/stic/processed_inputs/')

    ca_output_filename = base_path_ca / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles_cycle_1_t_6_vl_3_vt_4_blos_3_atmos.nc'
    ha_input_filename = base_path_ha / 'alignedspectra_scan1_map01_Ha.fits_stic_profiles.nc'

    fha = h5py.File(ha_input_filename, 'r')
    fca = h5py.File(ca_output_filename, 'r')

    ha_ind = np.where(fha['profiles'][0, 0, 0, :, 0] != 0)[0]

    ha_center_wave = 6562.8 / 10
    wave_range = 0.9 / 10

    actual_calculate_blos = prepare_calculate_blos(
        fha['profiles'][()],
        fha['wav'][ha_ind] / 10,
        ha_center_wave,
        ha_center_wave - wave_range,
        ha_center_wave + wave_range,
        1.048,
        ha_ind
    )

    vec_actual_calculate_blos = np.vectorize(actual_calculate_blos)

    magha = np.fromfunction(vec_actual_calculate_blos, shape=(17, 60))

    plt.close()

    plt.clf()

    plt.cla()

    fig, axs = plt.subplots(2, 4, figsize=(7, 3.5))

    im00 = axs[0][0].imshow(fca['blong'][0, 2:, :, 123].T, cmap='gray', origin='lower')
    im01 = axs[0][1].imshow(fca['blong'][0, 2:, :, 102].T, cmap='gray', origin='lower')
    im02 = axs[0][2].imshow(fca['blong'][0, 2:, :, 50].T, cmap='gray', origin='lower')
    im03 = axs[0][3].imshow(magha.T, cmap='gray', origin='lower')

    im10 = axs[1][0].imshow(np.abs(fca['blong'][0, 2:, :, 123].T) - np.abs(fca['blong'][0, 2:, :, 131].T), cmap='bwr', origin='lower', vmin=-200, vmax=200)
    im11 = axs[1][1].imshow(np.abs(fca['blong'][0, 2:, :, 102].T) - np.abs(fca['blong'][0, 2:, :, 123].T), cmap='bwr', origin='lower', vmin=-200, vmax=200)
    im12 = axs[1][2].imshow(np.abs(fca['blong'][0, 2:, :, 50].T) - np.abs(fca['blong'][0, 2:, :, 102].T), cmap='bwr', origin='lower', vmin=-200, vmax=200)
    im13 = axs[1][3].imshow(np.abs(magha.T) - np.abs(fca['blong'][0, 2:, :, 50].T), cmap='bwr', origin='lower', vmin=-200, vmax=200)

    fig.colorbar(im00, ax=axs[0][0])
    fig.colorbar(im01, ax=axs[0][1])
    fig.colorbar(im02, ax=axs[0][2])
    fig.colorbar(im03, ax=axs[0][3])

    fig.colorbar(im10, ax=axs[1][0])
    fig.colorbar(im11, ax=axs[1][1])
    fig.colorbar(im12, ax=axs[1][2])
    fig.colorbar(im13, ax=axs[1][3])

    fig.tight_layout()

    fig.savefig(base_path_ca / 'FieldCompare.pdf', format='pdf', dpi=300)

    fig.savefig(base_path_ca / 'FieldCompare.png', format='png', dpi=300)

    plt.show()

    fca.close()

    fha.close()

    plt.close()

    plt.clf()

    plt.cla()


if __name__=='__main__':
    make_compare_plots()
