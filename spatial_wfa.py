import sys
sys.path.insert(1, '/home/harsh/CourseworkRepo/spatial_WFA')
import numpy as np
import h5py
import matplotlib.pyplot as plt
import wfa_spatial as wfa
import sunpy.io
from pathlib import Path


def calculate_wfa(filepath, wave_range_0, wave_range_1, line):

    f = h5py.File(filepath, 'r')

    ind = np.where(f['profiles'][0, 0, 0, :, 0] != 0)[0]

    relevant_ind = np.where((f['wav'][ind] >= wave_range_0) & (f['wav'][ind] >= wave_range_1))[0]

    w = f['wav'][ind][relevant_ind]

    ny, nx, nStokes, nWav = f['profiles'].shape[1], f['profiles'].shape[2], f['profiles'].shape[4], w.shape[0]

    intensity_level = np.median(f['profiles'][0, :, :, ind[relevant_ind], 0])

    sig = np.zeros((4, w.size), dtype='float64', order='c')

    sig[:, :] = 3.0e-3 * intensity_level

    lin = wfa.line(line)

    w -= lin.cw

    alpha_Blos = 1

    d = np.transpose(f['profiles'][0, :, :, ind[relevant_ind]], axes=(0, 1, 3, 2))

    Blos = wfa.getBlos(w, d, sig, lin, alpha_Blos)

    return Blos


def calculate_magnetic_field_for_all(datestring):
    base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes')

    # base_path = Path('C:\\Work Things\\InstrumentalUncorrectedStokes')

    # base_path = Path('C:\\Work Things\\InstrumentalUncorrectedStokes')

    datepath = base_path / datestring

    level8path = datepath / 'Level-4-alt'

    all_files = level8path.glob('**/*')

    all_mag_files = [file for file in all_files if file.name.startswith('aligned_Ca') and file.name.endswith(
        '.nc') and 'mag' not in file.name and 'stic_file' not in file.name]

    for a_mag_file in all_mag_files:
        print(a_mag_file.name)

        magca = calculate_wfa(a_mag_file, 8661.7, 8661.8, 8661)

        sunpy.io.write_file(
            level8path / '{}_mag_ca_fe_spatial.fits'.format(a_mag_file.name),
            magca, dict(),
            overwrite=True
        )

        magca = calculate_wfa(a_mag_file, 8662.17, 8662.17 + 0.4, 8662)

        sunpy.io.write_file(
            level8path / '{}_mag_ca_core_spatial.fits'.format(a_mag_file.name),
            magca, dict(),
            overwrite=True
        )

if __name__ == '__main__':
    calculate_magnetic_field_for_all('20230527')