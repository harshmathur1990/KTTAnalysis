import sys
sys.path.insert(1, '/home/harsh/CourseworkRepo/spatial_WFA')
import numpy as np
import h5py
import matplotlib.pyplot as plt
import wfa_spatial as wfa
import sunpy.io
from pathlib import Path


def calculate_wfa(filepath, wave_range_0, wave_range_1, line, transition_skip_list=None, bin_factor=None):

    f = h5py.File(filepath, 'r')

    ind = np.where(f['profiles'][0, 0, 0, :, 0] != 0)[0]

    profiles = f['profiles'][0][:, :, ind]

    wave = f['wav'][ind]

    f.close()

    if bin_factor is not None:
        if line == 6562:
            line_indice = np.where(wave < 7000)[0]

            profiles = profiles[:, :, line_indice]

            wave = wave[line_indice]

            wave = np.mean(wave.reshape(wave.shape[0] // bin_factor, bin_factor), 1)

            profiles = np.mean(profiles.reshape(profiles.shape[0], profiles.shape[1], profiles.shape[2] // bin_factor, bin_factor, profiles.shape[3]), 3)

    relevant_ind = np.where((wave >= wave_range_0) & (wave <= wave_range_1))[0]

    if transition_skip_list is not None:
        skip_ind = list()
        for transition in transition_skip_list:
            skip_ind += list(
                np.where(
                    (
                            np.array(wave[relevant_ind]) >= (transition[0] - transition[1])
                    ) & (
                            np.array(wave[relevant_ind]) <= (transition[0] + transition[1])
                    )
                )[0]
            )
        relevant_ind = np.array(list(set(relevant_ind) - set(relevant_ind[skip_ind])))

    w = wave[relevant_ind]

    print(w.shape)

    intensity_level = np.median(profiles[ :, :, relevant_ind, 0])

    sig = np.zeros((4, w.size), dtype='float64', order='c')

    sig[:, :] = 3.0e-3 * intensity_level

    lin = wfa.line(line)

    w -= lin.cw

    alpha_Blos = 1

    d = np.transpose(profiles[:, :, relevant_ind], axes=(0, 1, 3, 2))

    Blos = wfa.getBlos(w, d, sig, lin, alpha_Blos)

    return Blos


def calculate_magnetic_field_for_all(datestring):
    base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes')

    # base_path = Path('C:\\Work Things\\InstrumentalUncorrectedStokes')

    # base_path = Path('C:\\Work Things\\InstrumentalUncorrectedStokes')

    datepath = base_path / datestring

    level5path = datepath / 'Level-5-alt-alt'

    all_mag_files = [level5path / 'aligned_Ca_Ha_stic_profiles_20230527_074428.nc_straylight_secondpass.nc']

    for a_mag_file in all_mag_files:
        print(a_mag_file.name)

        # transition_skip_list = np.array(
        #     [
        #         [6560.84, 0.25],
        #         [6561.09, 0.1],
        #         [6562.1, 0.25],
        #         [6563.645, 0.3],
        #         [6564.15, 0.35]
        #     ]
        # )

        # magha_full_line = calculate_wfa(a_mag_file, 6562.8-1.5, 6562.8+1.5, 6562, transition_skip_list=transition_skip_list, bin_factor=None)
        #
        # sunpy.io.write_file(
        #     level5path / '{}_magha_full_line_spatial.fits'.format(a_mag_file.name),
        #     magha_full_line, dict(),
        #     overwrite=True
        # )
        #
        # magha_core = calculate_wfa(a_mag_file, 6562.8-0.15, 6562.8+0.15, 6562, transition_skip_list=transition_skip_list, bin_factor=None)
        #
        # sunpy.io.write_file(
        #     level5path / '{}_magha_core_spatial.fits'.format(a_mag_file.name),
        #     magha_core, dict(),
        #     overwrite=True
        # )

        # transition_skip_list = np.array(
        #     [
        #         [6560.84, 0.25],
        #         [6561.09, 0.1],
        #         [6562.1, 0.25],
        #         [6562.8, 0.15],
        #         [6563.645, 0.3],
        #         [6564.15, 0.35]
        #     ]
        # )
        #
        # magha_wing = calculate_wfa(a_mag_file, 6562.8 - 1.5, 6562.8 + 1.5, 6562,
        #                                 transition_skip_list=transition_skip_list, bin_factor=None)
        #
        # sunpy.io.write_file(
        #     level5path / '{}_magha_wing_spatial.fits'.format(a_mag_file.name),
        #     magha_wing, dict(),
        #     overwrite=True
        # )

        # ca_wing = calculate_wfa(a_mag_file, 8662.17 + 0.13, 8662.17 + 0.23, 8662, transition_skip_list=None,
        #                         bin_factor=None)
        #
        # sunpy.io.write_file(
        #     level5path / '{}_ca_wing_red_spatial.fits'.format(a_mag_file.name),
        #     ca_wing, dict(),
        #     overwrite=True
        # )
        #
        # ca_wing = calculate_wfa(a_mag_file, 8662.17-0.45, 8662.17-0.35, 8662, transition_skip_list=None,
        #                         bin_factor=None)
        #
        # sunpy.io.write_file(
        #     level5path / '{}_ca_wing_blue_spatial.fits'.format(a_mag_file.name),
        #     ca_wing, dict(),
        #     overwrite=True
        # )
        #
        # ca_wing = calculate_wfa(a_mag_file, 8661.96-0.05, 8661.96+0.05, 8662, transition_skip_list=None, bin_factor=None)
        #
        # sunpy.io.write_file(
        #     level5path / '{}_ca_wing_middle_lobe_spatial.fits'.format(a_mag_file.name),
        #     ca_wing, dict(),
        #     overwrite=True
        # )
        #
        # ca_wing = calculate_wfa(a_mag_file, 8663 - 0.05, 8663 + 0.05, 8662, transition_skip_list=None,
        #                         bin_factor=None)
        #
        # sunpy.io.write_file(
        #     level5path / '{}_ca_far_wing_spatial.fits'.format(a_mag_file.name),
        #     ca_wing, dict(),
        #     overwrite=True
        # )

        # ca_wing = calculate_wfa(a_mag_file, 8662.17 + 0.43, 8662.17 + 0.53, 8662, transition_skip_list=None,
        #                         bin_factor=None)
        #
        # sunpy.io.write_file(
        #     level5path / '{}_ca_0p43_0p53_spatial.fits'.format(a_mag_file.name),
        #     ca_wing, dict(),
        #     overwrite=True
        # )

        # ca_wing = calculate_wfa(a_mag_file, 8662.17 + 0.23, 8662.17 + 0.33, 8662, transition_skip_list=None,
        #                         bin_factor=None)
        #
        # sunpy.io.write_file(
        #     level5path / '{}_ca_0p23_0p33_spatial.fits'.format(a_mag_file.name),
        #     ca_wing, dict(),
        #     overwrite=True
        # )

        # ca_wing = calculate_wfa(a_mag_file, 8662.17 + 0.33, 8662.17 + 0.43, 8662, transition_skip_list=None,
        #                         bin_factor=None)
        #
        # sunpy.io.write_file(
        #     level5path / '{}_ca_0p33_0p43_spatial.fits'.format(a_mag_file.name),
        #     ca_wing, dict(),
        #     overwrite=True
        # )

        # ca_wing = calculate_wfa(a_mag_file, 8662.17 + 0.27, 8662.17 + 0.37, 8662, transition_skip_list=None,
        #                         bin_factor=None)
        #
        # sunpy.io.write_file(
        #     level5path / '{}_ca_0p27_0p37_spatial.fits'.format(a_mag_file.name),
        #     ca_wing, dict(),
        #     overwrite=True
        # )

        # ca_wing = calculate_wfa(a_mag_file, 8662.17 + 0.27, 8662.17 + 0.30, 8662, transition_skip_list=None,
        #                         bin_factor=None)
        #
        # sunpy.io.write_file(
        #     level5path / '{}_ca_0p27_0p30_spatial.fits'.format(a_mag_file.name),
        #     ca_wing, dict(),
        #     overwrite=True
        # )
        #
        # ca_core = calculate_wfa(a_mag_file, 8662.17, 8662.17 + 0.4, 8662, transition_skip_list=None, bin_factor=None)
        #
        # sunpy.io.write_file(
        #     level5path / '{}_ca_core_spatial.fits'.format(a_mag_file.name),
        #     ca_core, dict(),
        #     overwrite=True
        # )

        ca_fe = calculate_wfa(a_mag_file, 8661.7, 8661.8, 8661, transition_skip_list=None, bin_factor=None)

        sunpy.io.write_file(
            level5path / '{}_ca_fe_spatial.fits'.format(a_mag_file.name),
            ca_fe, dict(),
            overwrite=True
        )

if __name__ == '__main__':
    calculate_magnetic_field_for_all('20230527')