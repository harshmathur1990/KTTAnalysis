import sys
sys.path.insert(1, 'F:\\Harsh\\CourseworkRepo\\stic\\example')
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.gridspec as gridspec
from prepare_data import *
from scipy.interpolate import CubicSpline


# kmeans_output_dir = Path(
#     '/home/harsh/SpinorNagaraju/maps_1/stic/kmeans_output'
# )

# atmos_rp_write_path = Path(
#     '/home/harsh/SpinorNagaraju/maps_1/stic/'
# )

# atmos_rp_write_path = Path(
#     '/home/harsh/SpinorNagaraju/maps_2_scan10/stic/'
# )

datestring = '20230603'

timestring = '073616'

crop_indice = np.array(
    [
        [3, 25],
        [None, None]
    ]
)

base_path = Path(
    'C:\\WorkThings\\InstrumentalUncorrectedStokes\\{}\\Level-4'.format(datestring)
)

stic_path = base_path / 'stic'

input_file = base_path / 'aligned_Ca_Ha_stic_profiles_{}_{}.nc_pca.nc_spatial_straylight_corrected.nc'.format(datestring, timestring)

kmeans_file = base_path / 'out_30_{}_{}.h5'.format(datestring, timestring)

rps_plot_write_dir = base_path / 'PCA_RPs_Plots'

falc_file_path = Path(
    'F:\\Harsh\\CourseworkRepo\\stic\\example\\falc_nicole_for_stic.nc'
)

cw = np.asarray([8662.])
cont = []
for ii in cw:
    cont.append(getCont(ii))


def resample_grid(line_center, min_val, max_val, num_points):
    grid_wave = list()

    # grid_wave.append(line_center)

    separation = (max_val - min_val) / num_points

    for w in np.arange(min_val, max_val, separation):
        grid_wave.append(w + line_center)

    if line_center not in grid_wave:
        grid_wave.append(line_center)

    grid_wave.sort()

    return np.array(grid_wave)


def make_rps():
    f = h5py.File(kmeans_file, 'r+')

    fi = h5py.File(input_file, 'r')

    ind = np.where(fi['profiles'][0, 0, 0, :, 0] != 0)[0]

    profiles = fi['profiles'][0, :, :, ind]

    keys = ['rps', 'final_labels']

    for key in keys:
        if key in list(f.keys()):
            del f[key]

    labels = f['labels_'][()].reshape(profiles.shape[0], profiles.shape[1]).astype(np.int64)

    f['final_labels'] = labels

    total_labels = labels.max() + 1

    rps = np.zeros(
        (total_labels, ind.size, 4),
        dtype=np.float64
    )

    for i in range(total_labels):
        a, b = np.where(labels == i)
        rps[i] = np.mean(profiles[a, b], axis=0)

    f['rps'] = rps

    fi.close()

    f.close()


def get_farthest(whole_data, a, center, r):
    all_profiles = whole_data[a, :, r]
    difference = np.sqrt(
        np.sum(
            np.square(
                np.subtract(
                    all_profiles,
                    center
                )
            ),
            axis=1
        )
    )
    index = np.argsort(difference)[-1]
    return all_profiles[index]


def get_max_min(whole_data, a, r):
    all_profiles = whole_data[a, :, r]
    return all_profiles.max(), all_profiles.min()


def get_data(get_data=True, get_labels=True, get_rps=True, ca=True, crop_indice=None):
    whole_data, labels, rps, wave = None, None, None, None

    ind = None

    if get_data:
        f = h5py.File(input_file, 'r')

        ind = np.where(f['profiles'][0, 0, 0, :, 0] != 0)[0]

        if ca:
            ind = ind[800:]
        else:
            ind = ind[0:800]

        wave = f['wav'][ind]

        whole_data = f['profiles'][0, :, :, ind, :]

        if crop_indice is not None:
            whole_data = whole_data[crop_indice[0][1]:crop_indice[1][1], crop_indice[0][0]:crop_indice[1][0]]

        whole_data[:, :, :, 1:4] /= whole_data[:, :, :, 0][:, :, :, np.newaxis]

        whole_data[np.where(np.isnan(whole_data))] = 0

        whole_data[np.where(np.isinf(whole_data))] = 0

        whole_data = whole_data.reshape(whole_data.shape[0] * whole_data.shape[1], ind.size, 4)

        f.close()

    f = h5py.File(kmeans_file, 'r')

    if get_labels:
        labels = f['final_labels'][()]
        if crop_indice is not None:
            labels = labels[crop_indice[0][1]:crop_indice[1][1], crop_indice[0][0]:crop_indice[1][0]]
        labels = labels.reshape(labels.shape[0] * labels.shape[1])

    if get_rps:
        if ca:
            rps = f['rps'][:, 800:]
        else:
            rps = f['rps'][:, 0:800]

    f.close()

    return whole_data, labels, rps, wave


def make_rps_plots(name='RPs'):
    whole_data, labels, rps, wave = get_data(crop_indice=crop_indice)

    k = 0

    color = 'black'

    cm = 'Greys'

    wave_x = np.arange(wave.size)

    xticks = list()

    # xticks.append(np.argmin(np.abs(wave - 6562.8)))
    #
    xticks.append(np.argmin(np.abs(wave - 8662.14)))

    for m in range(2):

        plt.close('all')

        plt.clf()

        plt.cla()

        fig = plt.figure(figsize=(8.27, 11.69))

        subfigs = fig.subfigures(5, 3)

        for i in range(5):

            for j in range(3):

                gs = gridspec.GridSpec(2, 2)

                gs.update(left=0, right=1, top=1, bottom=0, wspace=0.0, hspace=0.0)

                r = 0

                sys.stdout.write('{}\n'.format(k))

                subfig = subfigs[i][j]

                a = np.where(labels == k)[0]

                for p in range(2):
                    for q in range(2):

                        ax1 = subfig.add_subplot(gs[r])

                        center = rps[k, :, r]

                        # farthest_profile = get_farthest(whole_data, a, center, r)

                        ax1.plot(
                            wave_x,
                            center,
                            color=color,
                            linewidth=0.5,
                            linestyle='solid'
                        )

                        if a.size > 0:
                            c, f = get_max_min(whole_data, a, r)

                            max_8542, min_8542 = c, f

                            min_8542 = min_8542 * 0.9
                            max_8542 = max_8542 * 1.1

                            in_bins_8542 = np.linspace(min_8542, max_8542, 1000)

                            H1, xedge1, yedge1 = np.histogram2d(
                                np.tile(wave_x, a.shape[0]),
                                whole_data[a, :, r].flatten(),
                                bins=(wave_x, in_bins_8542)
                            )

                            # ax1.plot(
                            #     wave_8542,
                            #     farthest_profile,
                            #     color=color,
                            #     linewidth=0.5,
                            #     linestyle='dotted'
                            # )

                            ymesh = H1.T

                            # ymeshmax = np.max(ymesh, axis=0)

                            ymeshnorm = ymesh / ymesh.max()

                            X1, Y1 = np.meshgrid(xedge1, yedge1)

                            ax1.pcolormesh(X1, Y1, ymeshnorm, cmap=cm)

                        else:
                            max_8542, min_8542 = np.min(center), np.max(center)
                            min_8542 = min_8542 * 0.9
                            max_8542 = max_8542 * 1.1
                            
                        ax1.set_ylim(min_8542, max_8542)

                        if r == 0:
                            ax1.text(
                                0.2,
                                0.6,
                                'n = {}'.format(
                                    a.size
                                ),
                                transform=ax1.transAxes,
                                fontsize=8
                            )

                            ax1.text(
                                0.3,
                                0.8,
                                'RP {}'.format(k),
                                transform=ax1.transAxes,
                                fontsize=8
                            )

                        ax1.set_xticks(xticks)
                        ax1.set_xticklabels([])

                        if r == 0:
                            y_ticks = [
                                np.round(
                                    min_8542 + (max_8542 - min_8542) * 0.1,
                                    2
                                ),
                                np.round(
                                    min_8542 + (max_8542 - min_8542) * 0.8,
                                    2
                                )
                            ]
                        else:
                            y_ticks = [
                                np.round(
                                    min_8542 + (max_8542 - min_8542) * 0.1,
                                    4
                                ),
                                np.round(
                                    min_8542 + (max_8542 - min_8542) * 0.8,
                                    4
                                )
                            ]

                        ax1.set_yticks(y_ticks)
                        ax1.set_yticklabels(y_ticks)

                        ax1.tick_params(axis="y", direction="in", pad=-30)

                        r += 1

                k += 1

        fig.savefig(
            rps_plot_write_dir / 'RPs_{}.png'.format(k),
            format='png',
            dpi=300
        )

        plt.close('all')

        plt.clf()

        plt.cla()


def make_halpha_rps_plots(name='RPs'):
    whole_data, labels, rps, wave = get_data(ca=False)

    k = 0

    color = 'black'

    cm = 'Greys'

    wave_x = np.arange(wave.size)

    xticks = list()

    xticks.append(np.argmin(np.abs(wave - 6562.8)))

    # xticks.append(np.argmin(np.abs(wave - 8542.09)))

    for m in range(2):

        plt.close('all')

        plt.clf()

        plt.cla()

        fig = plt.figure(figsize=(8.27, 11.69))

        subfigs = fig.subfigures(5, 3)

        for i in range(5):

            for j in range(3):

                gs = gridspec.GridSpec(2, 2)

                gs.update(left=0, right=1, top=1, bottom=0, wspace=0.0, hspace=0.0)

                r = 0

                sys.stdout.write('{}\n'.format(k))

                subfig = subfigs[i][j]

                a = np.where(labels == k)[0]

                for p in range(2):
                    for q in range(2):

                        ax1 = subfig.add_subplot(gs[r])

                        center = rps[k, :, r]

                        # farthest_profile = get_farthest(whole_data, a, center, r)

                        c, f = get_max_min(whole_data, a, r)

                        max_8542, min_8542 = c, f

                        min_8542 = min_8542 * 0.9
                        max_8542 = max_8542 * 1.1

                        in_bins_8542 = np.linspace(min_8542, max_8542, 1000)

                        H1, xedge1, yedge1 = np.histogram2d(
                            np.tile(wave_x, a.shape[0]),
                            whole_data[a, :, r].flatten(),
                            bins=(wave_x, in_bins_8542)
                        )

                        ax1.plot(
                            wave_x,
                            center,
                            color=color,
                            linewidth=0.5,
                            linestyle='solid'
                        )

                        # ax1.plot(
                        #     wave_8542,
                        #     farthest_profile,
                        #     color=color,
                        #     linewidth=0.5,
                        #     linestyle='dotted'
                        # )

                        ymesh = H1.T

                        # ymeshmax = np.max(ymesh, axis=0)

                        ymeshnorm = ymesh / ymesh.max()

                        X1, Y1 = np.meshgrid(xedge1, yedge1)

                        ax1.pcolormesh(X1, Y1, ymeshnorm, cmap=cm)

                        ax1.set_ylim(min_8542, max_8542)

                        if r == 0:
                            ax1.text(
                                0.2,
                                0.6,
                                'n = {} %'.format(
                                    np.round(a.size * 100 / labels.size, 2)
                                ),
                                transform=ax1.transAxes,
                                fontsize=8
                            )

                            ax1.text(
                                0.3,
                                0.8,
                                'RP {}'.format(k),
                                transform=ax1.transAxes,
                                fontsize=8
                            )

                        ax1.set_xticks(xticks)
                        ax1.set_xticklabels([])

                        if r == 0:
                            y_ticks = [
                                np.round(
                                    min_8542 + (max_8542 - min_8542) * 0.1,
                                    2
                                ),
                                np.round(
                                    min_8542 + (max_8542 - min_8542) * 0.8,
                                    2
                                )
                            ]
                        else:
                            y_ticks = [
                                np.round(
                                    min_8542 + (max_8542 - min_8542) * 0.1,
                                    4
                                ),
                                np.round(
                                    min_8542 + (max_8542 - min_8542) * 0.8,
                                    4
                                )
                            ]

                        ax1.set_yticks(y_ticks)
                        ax1.set_yticklabels(y_ticks)

                        ax1.tick_params(axis="y", direction="in", pad=-30)

                        r += 1

                k += 1

        fig.savefig(
            rps_plot_write_dir / 'Ha_RPs_{}.png'.format(k),
            format='png',
            dpi=300
        )

        plt.close('all')

        plt.clf()

        plt.cla()


def get_data_for_label_polarisation_map():
    ind_photosphere = np.array(list(range(0, 58)) + list(range(58, 97)))

    ind_chromosphere = np.array(list(range(97, 306)))

    f = h5py.File(input_file, 'r')

    ind = np.where(f['profiles'][0, 0, 0, :, 0] != 0)[0]

    whole_data = f['profiles'][0, :, :, ind, :]

    intensity = whole_data[:, :, 0, 0]

    linpol_p = np.mean(
        np.sqrt(
            np.sum(
                np.square(
                    whole_data[:, :, ind_photosphere, 1:3]
                ),
                3
            )
        ) / whole_data[:, :, ind_photosphere, 0],
        2
    )

    linpol_c = np.mean(
        np.sqrt(
            np.sum(
                np.square(
                    whole_data[:, :, ind_chromosphere, 1:3]
                ),
                3
            )
        ) / whole_data[:, :, ind_chromosphere, 0],
        2
    )

    circpol_p = np.mean(
        np.divide(
            np.abs(whole_data[:, :, ind_photosphere, 3]),
            whole_data[:, :, ind_photosphere, 0]
        ),
        2
    )

    circpol_c = np.mean(
        np.divide(
            np.abs(whole_data[:, :, ind_chromosphere, 3]),
            whole_data[:, :, ind_chromosphere, 0]
        ),
        2
    )

    f.close()

    f = h5py.File(kmeans_file, 'r')

    labels = f['final_labels'][()]

    f.close()

    return intensity, linpol_p, linpol_c, circpol_p, circpol_c, labels


def plot_rp_map_fov():
    intensity, linpol_p, linpol_c, circpol_p, circpol_c, labels = get_data_for_label_polarisation_map()

    plt.close('all')

    plt.clf()

    plt.cla()

    fig, axs = plt.subplots(3, 2, figsize=(6, 9))

    im00 = axs[0][0].imshow(intensity, cmap='gray', origin='lower')

    im01 = axs[0][1].imshow(labels, cmap='gray', origin='lower')

    im10 = axs[1][0].imshow(linpol_p, cmap='gray', origin='lower')

    im11 = axs[1][1].imshow(circpol_p, cmap='gray', origin='lower')

    im20 = axs[2][0].imshow(linpol_c, cmap='gray', origin='lower')

    im21 = axs[2][1].imshow(circpol_c, cmap='gray', origin='lower')

    fig.colorbar(im00, ax=axs[0][0], orientation='horizontal')

    fig.colorbar(im01, ax=axs[0][1], orientation='horizontal')

    fig.colorbar(im10, ax=axs[1][0], orientation='horizontal')

    fig.colorbar(im11, ax=axs[1][1], orientation='horizontal')

    fig.colorbar(im20, ax=axs[2][0], orientation='horizontal')

    fig.colorbar(im21, ax=axs[2][1], orientation='horizontal')

    fig.tight_layout()

    fig.savefig(
        rps_plot_write_dir / 'FoV_RPs_pol_map.pdf',
        format='pdf',
        dpi=300
    )

    plt.close('all')

    plt.clf()

    plt.cla()


def make_stic_inversion_files(rps=None):

    f = h5py.File(kmeans_file, 'r')

    fi = h5py.File(input_file, 'r')

    if rps is None:
        rps = range(f['rps'].shape[0])

    rps = np.array(rps)

    ind_ca = np.where(fi['wav'][()] >= 8600)[0]

    non_zero_ind = np.where(fi['profiles'][0, fi['profiles'].shape[1]//2, fi['profiles'].shape[2]//2, ind_ca, 0] != 0)[0]

    ca = sp.profile(
        nx=30, ny=1, ns=4,
        nw=ind_ca.shape[0]
    )

    ca.wav[:] = fi['wav'][ind_ca]

    ca.dat[0, 0, :, non_zero_ind] = np.transpose(
        f['rps'][rps, 800:],
        axes=(1, 0, 2)
    )

    weights = np.zeros((ind_ca.shape[0], 4))

    weights[non_zero_ind, 0] = 0.004

    weights[non_zero_ind[100:300], 0] = 0.001

    ca.weights = weights

    if rps.size != f['rps'].shape[0]:
        writefilename = 'ca_rps_stic_profiles_x_{}_y_1.nc'.format('_'.join([str(_rp) for _rp in rps]))
    else:
        writefilename = 'ca_rps_stic_profiles_x_{}_y_1.nc'.format(rps.size)
    ca.write(
        stic_path/ writefilename
    )

def make_stic_inversion_files_halpha_ca_both(rps=None):
    wave_indices = [[95, 153], [155, 194], [196, 405]]

    line_indices = [[0, 58], [58, 97], [97, 306]]

    core_indices = [[19, 36], [18, 27], [85, 120]]

    wave_names = ['SiI_8536', 'FeI_8538', 'CaII_8542']

    f = h5py.File(kmeans_file, 'r')

    ca = None

    if rps is None:
        rps = range(f['rps'].shape[0])

    rps = np.array(rps)

    for wave_indice, line_indice, core_indice, wave_name in zip(wave_indices, line_indices, core_indices, wave_names):
        wc8, ic8 = findgrid(wave_8542_orig[wave_indice[0]:wave_indice[1]], (
                    wave_8542_orig[wave_indice[0]:wave_indice[1]][10] - wave_8542_orig[wave_indice[0]:wave_indice[1]][
                9]) * 0.25, extra=8)

        ca_8 = sp.profile(nx=rps.size, ny=1, ns=4, nw=wc8.size)

        ca_8.wav[:] = wc8[:]

        ca_8.dat[0, 0, :, ic8, :] = np.transpose(
            f['rps'][rps][:, line_indice[0]:line_indice[1]],
            axes=(1, 0, 2)
        )

        ca_8.weights[:, :] = 1.e16  # Very high value means weight zero
        ca_8.weights[ic8, 0] = 0.004
        ca_8.weights[ic8[core_indice[0]:core_indice[1]], 0] = 0.002
        # ca_8.weights[ic8, 3] = ca_8.weights[ic8, 0] / 2

        if ca is None:
            ca = ca_8
        else:
            ca += ca_8

        broadening_filename = 'gaussian_broadening_{}_pixel_{}.h5'.format(21.7, wave_name)

        lab = "region = {0:10.5f}, {1:8.5f}, {2:3d}, {3:e}, {4}"
        print(" ")
        print("Regions information for the input file:")
        print(lab.format(ca_8.wav[0], ca_8.wav[1] - ca_8.wav[0], ca_8.wav.size, cont[0],
                         'spectral, {}'.format(broadening_filename)))
        print("(w0, dw, nw, normalization, degradation_type, instrumental_profile file)")
        print(" ")

    wha, iha = findgrid(wave_ha, (wave_ha[1] - wave_ha[0]) * 0.25, extra=8)

    ha = sp.profile(nx=rps.size, ny=1, ns=4, nw=wha.size)

    ha.wav[:] = wha[:]

    ha.dat[0, 0, :, iha, :] = np.transpose(
        f['halpha_rps'][rps],
        axes=(1, 0, 2)
    )

    ha.weights[:, :] = 1.e16  # Very high value means weight zero
    ha.weights[iha, 0] = 0.008
    ha.weights[iha[18:46], 0] = 0.004
    ha.weights[iha[69:186], 0] = 0.002
    ha.weights[iha[405:432], 0] = 0.004
    # ha.weights[iha, 3] = ca_8.weights[iha, 0] / 2

    all_profiles = ca + ha
    if rps.size != f['rps'].shape[0]:
        writefilename = 'ha_ca_rps_stic_profiles_x_{}_y_1.nc'.format('_'.join([str(_rp) for _rp in rps]))
    else:
        writefilename = 'ha_ca_rps_stic_profiles_x_{}_y_1.nc'.format(rps.size)
    all_profiles.write(
        atmos_rp_write_path / writefilename
    )


def generate_input_atmos_file(length=30, temp=None, vlos=None, blong=0, name=''):
    f = h5py.File(falc_file_path, 'r')

    m = sp.model(nx=length, ny=1, nt=1, ndep=150)

    m.ltau[:, :, :] = f['ltau500'][0, 0, 0]

    m.pgas[:, :, :] = 1

    if temp is not None:
        cs = CubicSpline(temp[0], temp[1])
        tp = cs(f['ltau500'][0, 0, 0])
    else:
        tp = f['temp'][0, 0, 0]
    if vlos is not None:
        cs = CubicSpline(vlos[0], vlos[1])
        vl = cs(f['ltau500'][0, 0, 0])
    else:
        vl = f['vlos'][0, 0, 0]
    if isinstance(blong, tuple):
        a, b = np.polyfit(blong[0], blong[1], 1)
        bl = a * f['ltau500'][0, 0, 0] + b
    else:
        bl = blong

    m.temp[:, :, :] = tp

    m.vlos[:, :, :] = vl

    m.vturb[:, :, :] = 0

    m.Bln[:, :, :] = bl

    m.write(
        stic_path / 'atmos_{}_{}.nc'.format(length, name)
    )


def generate_input_atmos_file_from_previous_result(result_filename=None, rps=None):
    if result_filename is None:
        print('Give input atmos file')
        sys.exit(1)

    result_filename = Path(result_filename)

    f = h5py.File(result_filename, 'r')

    if not rps:
        rps = range(f['ltau500'].shape[2])

    rps = np.array(rps)

    m = sp.model(nx=rps.size, ny=1, nt=1, ndep=150)

    m.ltau[:, :, :] = f['ltau500'][0, 0, rps]

    m.pgas[:, :, :] = f['pgas'][0, 0, rps]

    m.temp[:, :, :] = f['temp'][0, 0, rps]

    m.vlos[:, :, :] = f['vlos'][0, 0, rps]

    m.vturb[:, :, :] = f['vturb'][0, 0, rps]

    m.Bln[:, :, :] = f['blong'][0, 0, rps]

    m.write(
        atmos_rp_write_path / '{}_{}.nc'.format(result_filename.name, '_'.join([str(_rp) for _rp in rps]))
    )


def make_rps_inversion_result_plots(nodes_temp=None, nodes_vlos=None, nodes_vturb=None, nodes_blos=None):
    # rps_atmos_result = Path(
    #     '/home/harsh/SpinorNagaraju/maps_1/stic/RPs_plots/inversions/only_Stokes_I/rps_stic_profiles_x_30_y_1_cycle_1_t_6_vl_3_vt_4_atmos.nc'
    # )
    #
    # rps_profs_result = Path(
    #     '/home/harsh/SpinorNagaraju/maps_1/stic/RPs_plots/inversions/only_Stokes_I/rps_stic_profiles_x_30_y_1_cycle_1_t_6_vl_3_vt_4_profs.nc'
    # )

    rps_atmos_result = Path(
        '/home/harsh/SpinorNagaraju/maps_2_scan10/stic/run_nagaraju/alignedspectra_scan2_map10_Ca.fits_stic_profiles.nc_opposite_polarity_mean_13_51_1_total_1_cycle_1_t_6_vl_5_vt_4_blong_0_atmos.nc'
    )

    rps_profs_result = Path(
        '/home/harsh/SpinorNagaraju/maps_2_scan10/stic/run_nagaraju/alignedspectra_scan2_map10_Ca.fits_stic_profiles.nc_opposite_polarity_mean_13_51_1_total_1_cycle_1_t_6_vl_5_vt_4_blong_0_profs.nc'
    )

    rps_input_profs = Path(
        '/home/harsh/SpinorNagaraju/maps_2_scan10/stic/run_nagaraju/alignedspectra_scan2_map10_Ca.fits_stic_profiles.nc_opposite_polarity_mean_13_51_1_total_1.nc'
    )

    rps_plot_write_dir = Path(
        '/home/harsh/SpinorNagaraju/maps_2_scan10/stic/run_nagaraju/'
    )

    finputprofs = h5py.File(rps_input_profs, 'r')

    fatmosresult = h5py.File(rps_atmos_result, 'r')

    fprofsresult = h5py.File(rps_profs_result, 'r')

    ind = np.where(finputprofs['profiles'][0, 0, 0, :, 0] != 0)[0]

    for i, k in enumerate(range(1)):
        print(i)
        plt.close('all')

        plt.clf()

        plt.cla()

        fig, axs = plt.subplots(3, 2, figsize=(12, 18))

        axs[0][0].plot(finputprofs['wav'][ind], finputprofs['profiles'][0, 0, i, ind, 0], color='orange', linewidth=0.5)

        axs[0][0].plot(fprofsresult['wav'][ind], fprofsresult['profiles'][0, 0, i, ind, 0], color='brown',
                       linewidth=0.5)

        axs[0][1].plot(
            finputprofs['wav'][ind],
            finputprofs['profiles'][0, 0, i, ind, 3] / finputprofs['profiles'][0, 0, i, ind, 0],
            color='orange',
            linewidth=0.5
        )

        axs[0][1].plot(
            fprofsresult['wav'][ind],
            fprofsresult['profiles'][0, 0, i, ind, 3] / fprofsresult['profiles'][0, 0, i, ind, 0],
            color='brown',
            linewidth=0.5
        )

        axs[1][0].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['temp'][0, 0, i] / 1e3, color='brown')

        axs[1][1].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['vlos'][0, 0, i] / 1e5, color='brown')

        axs[2][0].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['vturb'][0, 0, i] / 1e5, color='brown')

        axs[2][1].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['blong'][0, 0, i], color='brown')

        if nodes_temp is not None:
            for nn in nodes_temp:
                axs[1][0].axvline(nn, linestyle='--', linewidth=0.5)

        if nodes_vlos is not None:
            for nn in nodes_vlos:
                axs[1][1].axvline(nn, linestyle='--', linewidth=0.5)

        if nodes_vturb is not None:
            for nn in nodes_vturb:
                axs[2][0].axvline(nn, linestyle='--', linewidth=0.5)

        if nodes_blos is not None:
            for nn in nodes_blos:
                axs[2][1].axvline(nn, linestyle='--', linewidth=0.5)

        axs[0][0].set_xlabel(r'$\lambda(\AA)$')
        axs[0][0].set_ylabel(r'$I/I_{c}$')

        axs[0][1].set_xlabel(r'$\lambda(\AA)$')
        axs[0][1].set_ylabel(r'$V/I$')

        axs[1][0].set_xlabel(r'$\log(\tau_{500})$')
        axs[1][0].set_ylabel(r'$T[kK]$')

        axs[1][1].set_xlabel(r'$\log(\tau_{500})$')
        axs[1][1].set_ylabel(r'$V_{LOS}[Kms^{-1}]$')

        axs[2][0].set_xlabel(r'$\log(\tau_{500})$')
        axs[2][0].set_ylabel(r'$V_{turb}[Kms^{-1}]$')

        axs[2][1].set_xlabel(r'$\log(\tau_{500})$')
        axs[2][1].set_ylabel(r'$B_{long}[G]$')

        fig.tight_layout()

        fig.savefig(rps_plot_write_dir / 'RPs_{}_blong_2_vlos_2.pdf'.format(k), format='pdf', dpi=300)

        plt.close('all')

        plt.clf()

        plt.cla()
    fprofsresult.close()

    fatmosresult.close()

    finputprofs.close()


def make_pixels_inversion_result_plots(nodes_temp=None, nodes_vlos=None, nodes_vturb=None, nodes_blos=None):
    # rps_atmos_result = Path(
    #     '/home/harsh/SpinorNagaraju/maps_1/stic/RPs_plots/inversions/only_Stokes_I/rps_stic_profiles_x_30_y_1_cycle_1_t_6_vl_3_vt_4_atmos.nc'
    # )
    #
    # rps_profs_result = Path(
    #     '/home/harsh/SpinorNagaraju/maps_1/stic/RPs_plots/inversions/only_Stokes_I/rps_stic_profiles_x_30_y_1_cycle_1_t_6_vl_3_vt_4_profs.nc'
    # )

    base_path = Path(
        '/home/harsh/SpinorNagaraju/maps_1/stic/new_fulldata_inversions/'
    )
    median_atmos_result = Path(
        '/home/harsh/SpinorNagaraju/maps_1/stic/new_fulldata_inversions/median_profile_cycle_1_t_6_vl_2_vt_4_blong_0_atmos.nc'
    )

    median_profs_result = Path(
        '/home/harsh/SpinorNagaraju/maps_1/stic/new_fulldata_inversions/median_profile_cycle_1_t_6_vl_2_vt_4_blong_0_profs.nc'
    )

    median_prof = Path(
        '/home/harsh/SpinorNagaraju/maps_1/stic/new_fulldata_inversions/median_profile.nc'
    )

    rps_atmos_result = base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_opposite_polarity_total_24_cycle_1_t_6_vl_2_vt_4_blong_2_atmos.nc'

    rps_profs_result = base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_opposite_polarity_total_24_cycle_1_t_6_vl_2_vt_4_blong_2_profs.nc'

    rps_input_profs = base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_opposite_polarity_total_24.nc'

    rps_plot_write_dir = base_path / 'plots'

    finputprofs = h5py.File(rps_input_profs, 'r')

    fatmosresult = h5py.File(rps_atmos_result, 'r')

    fprofsresult = h5py.File(rps_profs_result, 'r')

    fmedianprofs = h5py.File(median_profs_result, 'r')

    fmedianatmos = h5py.File(median_atmos_result, 'r')

    fmedian = h5py.File(median_prof, 'r')

    ind = np.where(finputprofs['profiles'][0, 0, 0, :, 0] != 0)[0]

    # for i, (xx, yy) in enumerate(zip([12, 12], [49, 31])):
    for i, k in enumerate(range(24)):
        print(i)
        plt.close('all')

        plt.clf()

        plt.cla()

        fig, axs = plt.subplots(3, 2, figsize=(12, 18))

        axs[0][0].plot(finputprofs['wav'][ind], finputprofs['profiles'][0, 0, i, ind, 0], color='orange', linewidth=0.5)

        axs[0][0].plot(fprofsresult['wav'][ind], fprofsresult['profiles'][0, 0, i, ind, 0], color='brown',
                       linewidth=0.5)

        axs[0][0].plot(fmedianprofs['wav'][ind], fmedianprofs['profiles'][0, 0, 0, ind, 0], color='grey', linewidth=0.5)

        axs[0][0].plot(fmedian['wav'][ind], fmedian['profiles'][0, 0, 0, ind, 0], color='grey', linewidth=0.5,
                       linestyle='--')

        axs[0][1].plot(
            finputprofs['wav'][ind],
            finputprofs['profiles'][0, 0, i, ind, 3] / finputprofs['profiles'][0, 0, i, ind, 0],
            color='orange',
            linewidth=0.5
        )

        axs[0][1].plot(
            fprofsresult['wav'][ind],
            fprofsresult['profiles'][0, 0, i, ind, 3] / fprofsresult['profiles'][0, 0, i, ind, 0],
            color='brown',
            linewidth=0.5
        )

        axs[1][0].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['temp'][0, 0, i] / 1e3, color='brown')

        axs[1][1].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['vlos'][0, 0, i] / 1e5, color='brown')

        axs[2][0].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['vturb'][0, 0, i] / 1e5, color='brown')

        axs[2][1].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['blong'][0, 0, i], color='brown')

        axs[1][0].plot(fmedianatmos['ltau500'][0, 0, 0], fmedianatmos['temp'][0, 0, 0] / 1e3, color='grey',
                       linestyle='--')

        axs[1][1].plot(fmedianatmos['ltau500'][0, 0, 0], fmedianatmos['vlos'][0, 0, 0] / 1e5, color='grey',
                       linestyle='--')

        axs[2][0].plot(fmedianatmos['ltau500'][0, 0, 0], fmedianatmos['vturb'][0, 0, 0] / 1e5, color='grey',
                       linestyle='--')

        if nodes_temp is not None:
            for nn in nodes_temp:
                axs[1][0].axvline(nn, linestyle='--', linewidth=0.5)

        if nodes_vlos is not None:
            for nn in nodes_vlos:
                axs[1][1].axvline(nn, linestyle='--', linewidth=0.5)

        if nodes_vturb is not None:
            for nn in nodes_vturb:
                axs[2][0].axvline(nn, linestyle='--', linewidth=0.5)

        if nodes_blos is not None:
            for nn in nodes_blos:
                axs[2][1].axvline(nn, linestyle='--', linewidth=0.5)

        axs[0][0].set_xlabel(r'$\lambda(\AA)$')
        axs[0][0].set_ylabel(r'$I/I_{c}$')

        axs[0][1].set_xlabel(r'$\lambda(\AA)$')
        axs[0][1].set_ylabel(r'$V/I$')

        axs[1][0].set_xlabel(r'$\log(\tau_{500})$')
        axs[1][0].set_ylabel(r'$T[kK]$')

        axs[1][1].set_xlabel(r'$\log(\tau_{500})$')
        axs[1][1].set_ylabel(r'$V_{LOS}[Kms^{-1}]$')

        axs[2][0].set_xlabel(r'$\log(\tau_{500})$')
        axs[2][0].set_ylabel(r'$V_{turb}[Kms^{-1}]$')

        axs[2][1].set_xlabel(r'$\log(\tau_{500})$')
        axs[2][1].set_ylabel(r'$B_{long}[G]$')

        fig.tight_layout()

        # fig.savefig(rps_plot_write_dir /'Pixels_{}_{}.pdf'.format(xx, yy), format='pdf', dpi=300)
        fig.savefig(rps_plot_write_dir / 'opposite_polarity_t_6_{}.pdf'.format(k), format='pdf', dpi=300)

        plt.close('all')

        plt.clf()

        plt.cla()
    fprofsresult.close()

    fatmosresult.close()

    finputprofs.close()


def make_ca_rps_inversion_result_plots():
    # rps_atmos_result = Path(
    #     '/home/harsh/SpinorNagaraju/maps_1/stic/RPs_plots/inversions/only_Stokes_I/rps_stic_profiles_x_30_y_1_cycle_1_t_6_vl_3_vt_4_atmos.nc'
    # )
    #
    # rps_profs_result = Path(
    #     '/home/harsh/SpinorNagaraju/maps_1/stic/RPs_plots/inversions/only_Stokes_I/rps_stic_profiles_x_30_y_1_cycle_1_t_6_vl_3_vt_4_profs.nc'
    # )

    rps_atmos_result = Path(
        'C:\\WorkThings\\InstrumentalUncorrectedStokes\\20230603\\Level-4\\stic\\stic_run_1\\ca_rps_stic_profiles_x_30_y_1_t_6_vl_2_vt_4_atmos.nc'
    )

    rps_profs_result = Path(
        'C:\\WorkThings\\InstrumentalUncorrectedStokes\\20230603\\Level-4\\stic\\stic_run_1\\ca_rps_stic_profiles_x_30_y_1_t_6_vl_2_vt_4_profs.nc'
    )

    rps_input_profs = Path(
        'C:\\WorkThings\\InstrumentalUncorrectedStokes\\20230603\\Level-4\\stic\\stic_run_1\\ca_rps_stic_profiles_x_30_y_1.nc'
    )

    rps_plot_write_dir = Path(
        'C:\\WorkThings\\InstrumentalUncorrectedStokes\\20230603\\Level-4\\stic\\stic_run_1\\Plots'
    )

    finputprofs = h5py.File(rps_input_profs, 'r')

    fatmosresult = h5py.File(rps_atmos_result, 'r')

    fprofsresult = h5py.File(rps_profs_result, 'r')

    ind = np.where(finputprofs['profiles'][0, 0, 0, :, 0] != 0)[0]

    for i, k in enumerate(range(30)):
        print(i)
        plt.close('all')

        plt.clf()

        plt.cla()

        fig, axs = plt.subplots(3, 2, figsize=(12, 18))

        axs[0][0].plot(finputprofs['wav'][ind], finputprofs['profiles'][0, 0, i, ind, 0], color='orange',
                       linewidth=0.5)

        axs[0][0].plot(fprofsresult['wav'][ind], fprofsresult['profiles'][0, 0, i, ind, 0], color='brown',
                       linewidth=0.5)

        axs[0][1].plot(
            finputprofs['wav'][ind],
            finputprofs['profiles'][0, 0, i, ind, 3] / finputprofs['profiles'][0, 0, i, ind, 0],
            color='orange',
            linewidth=0.5
        )

        axs[0][1].plot(
            fprofsresult['wav'][ind],
            fprofsresult['profiles'][0, 0, i, ind, 3] / fprofsresult['profiles'][0, 0, i, ind, 0],
            color='brown',
            linewidth=0.5
        )

        axs[1][0].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['temp'][0, 0, i] / 1e3, color='brown')

        axs[1][1].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['vlos'][0, 0, i] / 1e5, color='brown')

        axs[2][0].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['vturb'][0, 0, i] / 1e5, color='brown')

        axs[2][1].plot(fatmosresult['ltau500'][0, 0, 0], fatmosresult['blong'][0, 0, i], color='brown')

        axs[0][0].set_xlabel(r'$\lambda(\AA)$')
        axs[0][0].set_ylabel(r'$I/I_{c}$')

        axs[0][1].set_xlabel(r'$\lambda(\AA)$')
        axs[0][1].set_ylabel(r'$V/I$')

        axs[1][0].set_xlabel(r'$\log(\tau_{500})$')
        axs[1][0].set_ylabel(r'$T[kK]$')

        axs[1][1].set_xlabel(r'$\log(\tau_{500})$')
        axs[1][1].set_ylabel(r'$V_{LOS}[Kms^{-1}]$')

        axs[2][0].set_xlabel(r'$\log(\tau_{500})$')
        axs[2][0].set_ylabel(r'$V_{turb}[Kms^{-1}]$')

        axs[2][1].set_xlabel(r'$\log(\tau_{500})$')
        axs[2][1].set_ylabel(r'$B_{long}[G]$')

        fig.tight_layout()

        fig.savefig(rps_plot_write_dir / 'HA_CA_RPs_{}.pdf'.format(k), format='pdf', dpi=300)

        plt.close('all')

        plt.clf()

        plt.cla()
    fprofsresult.close()

    fatmosresult.close()

    finputprofs.close()


def combine_rps_atmos():
    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/RPs_plots/inversions/full_stokes_blos_2/')

    file_1 = 'rps_stic_profiles_x_30_y_1_cycle_1_t_0_vl_0_vt_0_blos_2_atmos.nc'

    file_2 = 'rps_stic_profiles_x_3_12_25_y_1_cycle_1_t_0_vl_0_vt_0_blos_2_atmos.nc'

    rps_1 = list(set(range(30)) - set([3, 12, 25]))

    # rps_2 = [3, 12, 25]

    m = sp.model(nx=30, ny=1, nt=1, ndep=150)

    f = h5py.File(base_path / file_1, 'r')
    m.ltau[:, :, :] = f['ltau500'][0, 0, 0]
    f.close()

    m.pgas[:, :, :] = 1

    for i in range(30):

        if i in rps_1:
            f = h5py.File(base_path / file_1, 'r')
            file = 1
        else:
            f = h5py.File(base_path / file_2, 'r')
            file = 2

        if file == 1:
            mj = i
        else:
            if i == 3:
                mj = 0
            elif i == 12:
                mj = 1
            else:
                mj = 2

        m.temp[0, 0, i] = f['temp'][0, 0, mj]

        m.vlos[0, 0, i] = f['vlos'][0, 0, mj]

        m.vturb[0, 0, i] = f['vturb'][0, 0, mj]

        m.Bln[0, 0, i] = f['blong'][0, 0, mj]

        f.close()

    m.write(
        base_path / 'combined_rps_stic_profiles_x_30_y_1_cycle_1_t_0_vl_0_vt_0_blos_2_atmos.nc'
    )


def prepare_get_params(param):
    def get_params(rp):
        return param[rp]

    return get_params


def full_map_generate_input_atmos_file_from_previous_result():
    result_filename = Path(
        '/home/harsh/SpinorNagaraju/maps_1/stic/RPs_plots/inversions/full_stokes_6343/rps_stic_profiles_x_30_y_1_cycle_1_t_6_vl_3_vt_4_blos_3_atmos.nc')

    f = h5py.File(result_filename, 'r')

    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic')

    fc = h5py.File(base_path / 'pac_out_30.h5', 'r')

    get_temp = prepare_get_params(f['temp'][0, 0])

    get_vlos = prepare_get_params(f['vlos'][0, 0])

    get_vturb = prepare_get_params(f['vturb'][0, 0])

    get_blos = prepare_get_params(f['blong'][0, 0])

    vec_get_temp = np.vectorize(get_temp, signature='(x,y)->(x,y,z)')

    vec_get_vlos = np.vectorize(get_vlos, signature='(x,y)->(x,y,z)')

    vec_get_vturb = np.vectorize(get_vturb, signature='(x,y)->(x,y,z)')

    vec_get_blos = np.vectorize(get_blos, signature='(x,y)->(x,y,z)')

    labels = fc['final_labels'][()]

    m = sp.model(nx=60, ny=19, nt=1, ndep=150)

    m.ltau[:, :, :] = f['ltau500'][0, 0, 0]

    m.pgas[:, :, :] = 1

    m.temp[0, :, :] = vec_get_temp(labels)

    m.vlos[0, :, :] = vec_get_vlos(labels)

    m.vturb[0, :, :] = vec_get_vturb(labels)

    m.Bln[0, :, :] = vec_get_blos(labels)

    m.write(
        base_path / 'input_atmos_19_60_from_6343_rps.nc'
    )

    fc.close()

    f.close()


'''
Quiet Profiles: 0, 1, 2, 4, 5, 7, 9, 10, 11, 13, 14, 17, 18, 28, 29
Interesting Profiles No Vlos: 3, 6, 8, 12, 15, 16, 19, 20, 22, 23, 25
Emission Red RP: 21, 24
Emission Blue RP: 26, 27
'''


def get_rp_atmos():
    quiet_profiles = [0, 1, 4, 5, 6, 7, 8, 9, 11, 14, 16, 17, 19, 22, 23, 24, 25, 26, 27, 28, 29]
    quiet_profiles_in_file = [0, 1, 4, 5, 6, 7, 8, 9, 11, 14, 16, 17, 19, 22, 23, 24, 25, 26, 27, 28, 29]
    opp_polarity_profiles = [20]
    emission_profiles = [2, 3, 10, 12, 13, 15, 18, 21]

    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/')
    input_path = base_path / 'PCA_RPs_Plots/inversions'

    quiet_file = input_path / 'rps_stic_profiles_x_0_1_4_5_6_7_8_9_11_14_16_17_19_22_23_24_25_26_27_28_29_y_1_cycle_1_t_6_vl_2_vt_4_blong_2_atmos.nc'
    opp_polarity_file = input_path / 'rps_stic_profiles_x_20_y_1_cycle_1_t_6_vl_2_vt_4_blong_2_atmos.nc'
    emission_file = input_path / 'rps_stic_profiles_x_2_3_10_12_13_15_18_21_y_1_cycle_1_t_6_vl_4_vt_4_blong_2_atmos.nc'

    temp = np.zeros((30, 150))
    vlos = np.zeros((30, 150))
    vturb = np.zeros((30, 150))
    blong = np.zeros((30, 150))

    f = h5py.File(quiet_file, 'r')
    for index, prof in enumerate(quiet_profiles_in_file):
        if prof in quiet_profiles:
            temp[prof] = f['temp'][0, 0, index]
            vlos[prof] = f['vlos'][0, 0, index]
            vturb[prof] = f['vturb'][0, 0, index]
            blong[prof] = f['blong'][0, 0, index]
    f.close()

    f = h5py.File(opp_polarity_file, 'r')
    for index, i in enumerate(opp_polarity_profiles):
        temp[i] = f['temp'][0, 0, index]
        vlos[i] = f['vlos'][0, 0, index]
        vturb[i] = f['vturb'][0, 0, index]
        blong[i] = f['blong'][0, 0, index]
    f.close()

    f = h5py.File(emission_file, 'r')
    for index, i in enumerate(emission_profiles):
        temp[i] = f['temp'][0, 0, index]
        vlos[i] = f['vlos'][0, 0, index]
        vturb[i] = f['vturb'][0, 0, index]
        blong[i] = f['blong'][0, 0, index]
    f.close()

    return temp, vlos, vturb, blong


def generate_actual_inversion_files_mean():
    line_indices = [[0, 58], [58, 97], [97, 306]]

    core_indices = [[19, 36], [18, 27], [85, 120]]

    quiet_profiles = [[21, 24]]

    temp, vlos, vturb, blong = get_rp_atmos()

    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/')
    actual_file = base_path / 'processed_inputs/alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc'
    write_path = base_path / 'new_fulldata_inversions/'

    kmeans_file = base_path / 'pac_out_30.h5'

    f = h5py.File(kmeans_file, 'r')

    labels = f['final_labels'][()]

    f.close()

    f = h5py.File(actual_file, 'r')

    wc8 = f['wav'][()]

    ic8 = np.where(f['profiles'][0, 0, 0, :, 0] != 0)[0]

    ca_8 = sp.profile(nx=len(quiet_profiles), ny=1, ns=4, nw=wc8.size)

    ca_8.wav[:] = wc8[:]

    for index, profiles in enumerate(quiet_profiles):
        a_arr, b_arr = list(), list()
        for prof in profiles:
            a1, b1 = np.where(labels == prof)
            a1 = list(a1)
            b1 = list(b1)
            a_arr += list(a1)
            b_arr += list(b1)
        a_arr, b_arr = np.array(a_arr), np.array(b_arr)
        ca_8.dat[0, 0, index, ic8, :] = np.mean(
            np.transpose(f['profiles'][()][0, a_arr, b_arr, :, :][:, ic8], axes=(1, 0, 2)), 1)

    ca_8.weights[:, :] = 1.e16

    ca_8.weights[ic8, 0] = 0.004
    for line_indice, core_indice in zip(line_indices, core_indices):
        ca_8.weights[ic8[line_indice[0] + core_indice[0]:line_indice[0] + core_indice[1]], 0] = 0.002

    ca_8.weights[ic8[97 + 85:97 + 120], 0] /= 2

    ca_8.weights[ic8, 3] = ca_8.weights[ic8, 0]

    ca_8.weights[ic8, 3] /= 2

    ca_8.weights[ic8[97 + 85:97 + 120], 3] /= 2

    ca_8.write(
        write_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_mean_of_rps_{}_total_{}.nc'.format(
            '_'.join([str(prof) for prof in list(np.array(quiet_profiles).flatten())]), len(quiet_profiles))
    )

    m = sp.model(nx=len(quiet_profiles), ny=1, nt=1, ndep=150)

    m.ltau[:, :, :] = ltau

    m.pgas[:, :, :] = 1.0

    for index, profiles in enumerate(quiet_profiles):
        prof = np.array(profiles)

        m.temp[0, 0, index] = np.mean(temp[prof], 0)

        m.vlos[0, 0, index] = np.mean(vlos[prof], 0)

        m.vturb[0, 0, index] = np.mean(vturb[prof], 0)

        m.Bln[0, 0, index] = np.mean(blong[prof], 0)

    write_filename = write_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_mean_of_rps_{}_total_{}_initial_atmos.nc'.format(
        '_'.join([str(prof) for prof in list(np.array(quiet_profiles).flatten())]), len(quiet_profiles))

    m.write(str(write_filename))


def generate_actual_inversion_files_median_profile():
    line_indices = [[0, 58], [58, 97], [97, 306]]

    core_indices = [[13, 45], [15, 36], [70, 135]]

    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/')
    actual_file = base_path / 'processed_inputs/alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc'
    write_path = base_path / 'pca_kmeans_fulldata_inversions'

    median_profile = np.loadtxt('/home/harsh/SpinorNagaraju/maps_1/stic/processed_inputs/all_median.txt')

    f = h5py.File(actual_file, 'r')

    wc8 = f['wav'][()]

    ic8 = np.where(f['profiles'][0, 0, 0, :, 0] != 0)[0]

    f.close()

    ca_8 = sp.profile(nx=1, ny=1, ns=4, nw=wc8.size)

    ca_8.wav[:] = wc8[:]

    ca_8.dat[0, 0, 0, ic8, 0] = median_profile[:, 1]

    ca_8.weights[:, :] = 1.e16
    ca_8.weights[ic8, 0] = 0.004
    k = 0
    for line_indice, core_indice in zip(line_indices, core_indices):
        if k < 2:
            ca_8.weights[ic8[line_indice[0] + core_indice[0]:line_indice[0] + core_indice[1]], 0] = 0.001 / 2
        else:
            ca_8.weights[ic8[line_indice[0] + core_indice[0]:line_indice[0] + core_indice[1]], 0] = 0.002
        k += 1

    ca_8.weights[ic8[97 + 85:97 + 120], 0] /= 2

    ca_8.write(
        write_path / 'median_profile.nc'
    )

    temp, vlos, vturb, blong = get_rp_atmos()

    m = sp.model(nx=1, ny=1, nt=1, ndep=150)

    m.ltau[:, :, :] = ltau

    m.pgas[:, :, :] = 1.0

    m.temp[0, 0] = temp[0]

    m.vlos[0, 0] = vlos[0]

    m.vturb[0, 0] = vturb[0]

    write_filename = write_path / 'median_profile_initial_atmos.nc'

    m.write(str(write_filename))


def get_blong(pb, cb):
    e1, f1 = np.polyfit([-4.5, -1], [cb, pb], 1)
    return ltau * e1 + f1


def generate_actual_inversion_files_quiet():
    wave_indices = [[95, 153], [155, 194], [196, 405]]

    line_indices = [[0, 58], [58, 97], [97, 306]]

    core_indices = [[13, 45], [15, 36], [70, 135]]

    quiet_profiles = [0, 1, 4, 5, 6, 7, 8, 9, 11, 14, 16, 17, 19, 22, 23, 24, 25, 26, 27, 28, 29]

    temp, vlos, vturb, _ = get_rp_atmos()

    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/')
    actual_file = base_path / 'processed_inputs/alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc'
    write_path = base_path / 'pca_kmeans_fulldata_inversions/'

    kmeans_file = base_path / 'pca_out_30.h5'

    f = h5py.File(kmeans_file, 'r')

    labels = f['final_labels'][()][0:17]

    f.close()

    a_list = list()
    b_list = list()
    rp_final = list()

    for profile in quiet_profiles:
        a, b = np.where(labels == profile)
        a_list += list(a)
        b_list += list(b)
        rp_final += list(np.ones(a.shape[0]) * profile)

    a_arr = np.array(a_list)
    b_arr = np.array(b_list)
    rp_final = np.array(rp_final)

    mask = np.zeros((17, 60), dtype=np.int64)

    mask[a_arr, b_arr] = 1

    fme = h5py.File(me_file_path, 'r')
    bme = fme['B_abs'][()] * np.cos(fme['inclination_rad'][()])
    fme.close()

    fwfa = h5py.File(wfa_file_path, 'r')
    wfa = fwfa['blos_gauss'][()]
    fwfa.close()

    a1, b1 = np.where(bme > 0)

    mask[a1, b1] = 0

    a_arr, b_arr = np.where(mask == 1)

    rp_final = labels[a_arr, b_arr]

    pixel_indices = np.zeros((3, a_arr.size), dtype=np.int64)

    pixel_indices[0] = a_arr
    pixel_indices[1] = b_arr
    pixel_indices[2] = rp_final

    fo = h5py.File(
        write_path / 'pixel_indices_quiet_total_{}.h5'.format(
            rp_final.size
        ), 'w'
    )

    fo['pixel_indices'] = pixel_indices

    fo.close()

    f = h5py.File(actual_file, 'r')

    wc8 = f['wav'][()]

    ic8 = np.where(f['profiles'][0, 0, 0, :, 0] != 0)[0]

    ca_8 = sp.profile(nx=rp_final.size, ny=1, ns=4, nw=wc8.size)

    ca_8.wav[:] = wc8[:]

    ca_8.dat[0, 0, :, ic8, :] = np.transpose(f['profiles'][()][0, a_arr, b_arr, :, :][:, ic8], axes=(1, 0, 2))

    ca_8.weights[:, :] = 1.e16
    ca_8.weights[ic8, 0] = 0.004
    for line_indice, core_indice in zip(line_indices, core_indices):
        ca_8.weights[ic8[line_indice[0] + core_indice[0]:line_indice[0] + core_indice[1]], 0] = 0.002

    ca_8.weights[ic8[line_indices[2][0] + core_indices[2][0]:line_indices[2][0] + core_indices[2][1]], 0] /= 2

    ca_8.weights[ic8, 3] = ca_8.weights[ic8, 0]

    ca_8.weights[ic8, 3] /= 2

    ca_8.weights[ic8[line_indices[0][0] + core_indices[0][0]:line_indices[0][0] + core_indices[0][1]], 3] /= 4

    ca_8.weights[ic8[line_indices[1][0] + core_indices[1][0]:line_indices[1][0] + core_indices[1][1]], 3] /= 4

    ca_8.weights[ic8[line_indices[2][0] + core_indices[2][0]:line_indices[2][0] + core_indices[2][1]], 3] /= 2

    ca_8.write(
        write_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_quiet_total_{}.nc'.format(rp_final.size)
    )

    m = sp.model(nx=a_arr.size, ny=1, nt=1, ndep=150)

    m.ltau[:, :, :] = ltau

    m.pgas[:, :, :] = 1.0

    fold = h5py.File('/home/harsh/SpinorNagaraju/maps_1/stic/pca_kmeans_fulldata_inversions/combined_output.nc', 'r')

    m.temp[0, 0] = fold['temp'][()][0, a_arr, b_arr]

    m.vlos[0, 0] = fold['vlos'][()][0, a_arr, b_arr]

    m.vturb[0, 0] = fold['vturb'][()][0, a_arr, b_arr]

    fold.close()

    vec_get_blong = np.vectorize(get_blong, signature='(),()->(n)')

    m.Bln[0, 0] = vec_get_blong(bme[a_arr, b_arr], wfa[a_arr, b_arr])

    write_filename = write_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_quiet_total_{}_initial_atmos.nc'.format(
        rp_final.size)

    m.write(str(write_filename))


def generate_actual_inversion_files_spot():
    line_indices = [[0, 58], [58, 97], [97, 306]]

    core_indices = [[19, 36], [18, 27], [85, 120]]

    spot_profiles = [3, 9, 10, 13, 20, 21]

    temp, vlos, vturb, blong = get_rp_atmos()

    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/')
    actual_file = base_path / 'processed_inputs/alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc'
    write_path = base_path / 'fulldata_inversions'

    kmeans_file = base_path / 'pac_out_30.h5'

    f = h5py.File(kmeans_file, 'r')

    labels = f['final_labels'][()]

    f.close()

    a_list = list()
    b_list = list()
    rp_final = list()

    for profile in spot_profiles:
        a, b = np.where(labels == profile)
        a_list += list(a)
        b_list += list(b)
        rp_final += list(np.ones(a.shape[0]) * profile)

    a_arr = np.array(a_list)
    b_arr = np.array(b_list)
    rp_final = np.array(rp_final)

    pixel_indices = np.zeros((3, a_arr.size), dtype=np.int64)

    pixel_indices[0] = a_arr
    pixel_indices[1] = b_arr
    pixel_indices[2] = rp_final

    fo = h5py.File(
        write_path / 'pixel_indices_spot_total_{}.h5'.format(
            rp_final.size
        ), 'w'
    )

    fo['pixel_indices'] = pixel_indices

    fo.close()

    f = h5py.File(actual_file, 'r')

    wc8 = f['wav'][()]

    ic8 = np.where(f['profiles'][0, 0, 0, :, 0] != 0)[0]

    ca_8 = sp.profile(nx=rp_final.size, ny=1, ns=4, nw=wc8.size)

    ca_8.wav[:] = wc8[:]

    ca_8.dat[0, 0, :, ic8, :] = np.transpose(f['profiles'][()][0, a_arr, b_arr, :, :][:, ic8], axes=(1, 0, 2))

    ca_8.weights[:, :] = 1.e16
    ca_8.weights[ic8, 0] = 0.004
    for line_indice, core_indice in zip(line_indices, core_indices):
        ca_8.weights[ic8[line_indice[0] + core_indice[0]:line_indice[0] + core_indice[1]], 0] = 0.002

    ca_8.weights[ic8[97 + 85:97 + 120], 0] /= 4

    ca_8.weights[ic8, 3] = ca_8.weights[ic8, 0]

    ca_8.weights[ic8, 3] /= 2

    # ca_8.weights[ic8[line_indices[2][0]+core_indices[2][0]:line_indices[2][0]+core_indices[2][1]], 3] /= 2

    ca_8.write(
        write_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_spot_total_{}.nc'.format(rp_final.size)
    )

    m = sp.model(nx=a_arr.size, ny=1, nt=1, ndep=150)

    m.ltau[:, :, :] = ltau

    m.pgas[:, :, :] = 1.0

    m.temp[0, 0] = temp[labels[a_arr, b_arr]]

    m.vlos[0, 0] = vlos[labels[a_arr, b_arr]]

    m.vturb[0, 0] = vturb[labels[a_arr, b_arr]]

    m.Bln[0, 0] = blong[labels[a_arr, b_arr]]

    write_filename = write_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_spot_total_{}_initial_atmos.nc'.format(
        rp_final.size)

    m.write(str(write_filename))


def generate_actual_inversion_files_emission():
    wave_indices = [[95, 153], [155, 194], [196, 405]]

    line_indices = [[0, 58], [58, 97], [97, 306]]

    core_indices = [[13, 45], [15, 36], [70, 135]]

    emission_profiles = [2, 3, 10, 12, 13, 15, 18, 21]

    # emission_profiles = [20, 21, 24]

    # emission_profiles = np.arange(30)

    temp, vlos, vturb, _ = get_rp_atmos()

    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/')
    actual_file = base_path / 'processed_inputs/alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc'
    write_path = base_path / 'pca_kmeans_fulldata_inversions'

    kmeans_file = base_path / 'pca_out_30.h5'

    f = h5py.File(kmeans_file, 'r')

    labels = f['final_labels'][()][0:17]

    f.close()

    a_list = list()
    b_list = list()
    rp_final = list()

    for profile in emission_profiles:
        a, b = np.where(labels == profile)
        a_list += list(a)
        b_list += list(b)
        rp_final += list(np.ones(a.shape[0]) * profile)

    a_arr = np.array(a_list)
    b_arr = np.array(b_list)
    rp_final = np.array(rp_final)

    mask = np.zeros((17, 60), dtype=np.int64)

    mask[a_arr, b_arr] = 1

    fme = h5py.File(me_file_path, 'r')
    bme = fme['B_abs'][()] * np.cos(fme['inclination_rad'][()])
    fme.close()

    fwfa = h5py.File(wfa_file_path, 'r')
    wfa = fwfa['blos_gauss'][()]
    fwfa.close()

    a1, b1 = np.where(bme > 0)

    mask[a1, b1] = 0

    a_arr, b_arr = np.where(mask == 1)

    rp_final = labels[a_arr, b_arr]

    pixel_indices = np.zeros((3, a_arr.size), dtype=np.int64)

    pixel_indices[0] = a_arr
    pixel_indices[1] = b_arr
    pixel_indices[2] = rp_final

    fo = h5py.File(
        write_path / 'pixel_indices_emission_total_{}.h5'.format(
            rp_final.size
        ), 'w'
    )

    fo['pixel_indices'] = pixel_indices

    fo.close()

    f = h5py.File(actual_file, 'r')

    wc8 = f['wav'][()]

    ic8 = np.where(f['profiles'][0, 0, 0, :, 0] != 0)[0]

    ca_8 = sp.profile(nx=rp_final.size, ny=1, ns=4, nw=wc8.size)

    ca_8.wav[:] = wc8[:]

    ca_8.dat[0, 0, :, ic8, :] = np.transpose(f['profiles'][()][0, a_arr, b_arr, :, :][:, ic8], axes=(1, 0, 2))

    ca_8.weights[:, :] = 1.e16
    ca_8.weights[ic8, 0] = 0.004

    for line_indice, core_indice in zip(line_indices, core_indices):
        ca_8.weights[ic8[line_indice[0] + core_indice[0]:line_indice[0] + core_indice[1]], 0] = 0.002

    ca_8.weights[ic8[line_indices[2][0] + core_indices[2][0]:line_indices[2][0] + core_indices[2][1]], 0] /= 2

    ca_8.weights[ic8, 3] = ca_8.weights[ic8, 0]

    ca_8.weights[ic8, 3] /= 2

    ca_8.weights[ic8[line_indices[0][0] + core_indices[0][0]:line_indices[0][0] + core_indices[0][1]], 3] /= 4

    ca_8.weights[ic8[line_indices[1][0] + core_indices[1][0]:line_indices[1][0] + core_indices[1][1]], 3] /= 4

    ca_8.weights[ic8[line_indices[2][0] + core_indices[2][0]:line_indices[2][0] + core_indices[2][1]], 3] /= 2

    ca_8.write(
        write_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_emission_total_{}.nc'.format(rp_final.size)
    )

    m = sp.model(nx=a_arr.size, ny=1, nt=1, ndep=150)

    m.ltau[:, :, :] = ltau

    m.pgas[:, :, :] = 1.0

    fold = h5py.File('/home/harsh/SpinorNagaraju/maps_1/stic/pca_kmeans_fulldata_inversions/combined_output.nc', 'r')

    m.temp[0, 0] = fold['temp'][()][0, a_arr, b_arr]

    m.vlos[0, 0] = fold['vlos'][()][0, a_arr, b_arr]

    m.vturb[0, 0] = fold['vturb'][()][0, a_arr, b_arr]

    fold.close()

    vec_get_blong = np.vectorize(get_blong, signature='(),()->(n)')

    m.Bln[0, 0] = vec_get_blong(bme[a_arr, b_arr], wfa[a_arr, b_arr])

    write_filename = write_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_emission_total_{}_initial_atmos.nc'.format(
        rp_final.size)

    m.write(str(write_filename))


def generate_actual_inversion_files_opposite_polarity():
    wave_indices = [[95, 153], [155, 194], [196, 405]]

    line_indices = [[0, 58], [58, 97], [97, 306]]

    core_indices = [[13, 45], [15, 36], [70, 135]]

    emission_profiles = [20]

    # emission_profiles = [20, 21, 24]

    # emission_profiles = np.arange(30)

    temp, vlos, vturb, _ = get_rp_atmos()

    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/')
    actual_file = base_path / 'processed_inputs/alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc'
    write_path = base_path / 'pca_kmeans_fulldata_inversions'

    kmeans_file = base_path / 'pca_out_30.h5'

    f = h5py.File(kmeans_file, 'r')

    labels = f['final_labels'][()][0:17]

    f.close()

    a_list = list()
    b_list = list()
    rp_final = list()

    for profile in emission_profiles:
        a, b = np.where(labels == profile)
        a_list += list(a)
        b_list += list(b)
        rp_final += list(np.ones(a.shape[0]) * profile)

    a_arr = np.array(a_list)
    b_arr = np.array(b_list)
    rp_final = np.array(rp_final)

    mask = np.zeros((17, 60), dtype=np.int64)

    mask[a_arr, b_arr] = 1

    fme = h5py.File(me_file_path, 'r')
    bme = fme['B_abs'][()] * np.cos(fme['inclination_rad'][()])
    fme.close()

    fwfa = h5py.File(wfa_file_path, 'r')
    wfa = fwfa['blos_gauss'][()]
    fwfa.close()

    a1, b1 = np.where(bme > 0)

    mask[a1, b1] = 1

    a_arr, b_arr = np.where(mask == 1)

    rp_final = labels[a_arr, b_arr]

    pixel_indices = np.zeros((3, a_arr.size), dtype=np.int64)

    pixel_indices[0] = a_arr
    pixel_indices[1] = b_arr
    pixel_indices[2] = rp_final

    fo = h5py.File(
        write_path / 'pixel_indices_opposite_polarity_total_{}.h5'.format(
            rp_final.size
        ), 'w'
    )

    fo['pixel_indices'] = pixel_indices

    fo.close()

    f = h5py.File(actual_file, 'r')

    wc8 = f['wav'][()]

    ic8 = np.where(f['profiles'][0, 0, 0, :, 0] != 0)[0]

    ca_8 = sp.profile(nx=rp_final.size, ny=1, ns=4, nw=wc8.size)

    ca_8.wav[:] = wc8[:]

    ca_8.dat[0, 0, :, ic8, :] = np.transpose(f['profiles'][()][0, a_arr, b_arr, :, :][:, ic8], axes=(1, 0, 2))

    ca_8.weights[:, :] = 1.e16
    ca_8.weights[ic8, 0] = 0.004
    k = 0
    for line_indice, core_indice in zip(line_indices, core_indices):
        if k < 2:
            ca_8.weights[ic8[line_indice[0] + core_indice[0]:line_indice[0] + core_indice[1]], 0] = 0.002
        else:
            ca_8.weights[ic8[line_indice[0] + core_indice[0]:line_indice[0] + core_indice[1]], 0] = 0.002
        k += 1

    ca_8.weights[ic8[line_indices[2][0] + core_indices[2][0]:line_indices[2][0] + core_indices[2][1]], 0] /= 2

    ca_8.weights[ic8, 3] = ca_8.weights[ic8, 0]

    ca_8.weights[ic8, 3] /= 2

    ca_8.weights[ic8[line_indices[0][0] + core_indices[0][0]:line_indices[0][0] + core_indices[0][1]], 3] /= 4

    ca_8.weights[ic8[line_indices[1][0] + core_indices[1][0]:line_indices[1][0] + core_indices[1][1]], 3] /= 4

    ca_8.weights[ic8[line_indices[2][0] + core_indices[2][0]:line_indices[2][0] + core_indices[2][1]], 3] /= 2

    ca_8.write(
        write_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_opposite_polarity_total_{}.nc'.format(
            rp_final.size)
    )

    m = sp.model(nx=a_arr.size, ny=1, nt=1, ndep=150)

    m.ltau[:, :, :] = ltau

    m.pgas[:, :, :] = 1.0

    fold = h5py.File('/home/harsh/SpinorNagaraju/maps_1/stic/pca_kmeans_fulldata_inversions/old/combined_output.nc',
                     'r')

    m.temp[0, 0] = fold['temp'][()][0, a_arr, b_arr]

    m.vlos[0, 0] = fold['vlos'][()][0, a_arr, b_arr]

    m.vturb[0, 0] = fold['vturb'][()][0, a_arr, b_arr]

    fold.close()

    vec_get_blong = np.vectorize(get_blong, signature='(),()->(n)')

    m.Bln[0, 0] = vec_get_blong(bme[a_arr, b_arr], wfa[a_arr, b_arr])

    write_filename = write_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_opposite_polarity_total_{}_initial_atmos.nc'.format(
        rp_final.size)

    m.write(str(write_filename))


def generate_actual_inversion_pixels(pixels):
    '''
    pixels a tuple of two 1D numpy arrays
    indicating the pixel location
    '''

    line_indices = [[0, 58], [58, 97], [97, 306]]

    core_indices = [[19, 36], [18, 27], [85, 120]]

    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/')
    actual_file = base_path / 'processed_inputs/alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc'
    write_path = base_path / 'fulldata_inversions'

    f = h5py.File(actual_file, 'r')

    wc8 = f['wav'][()]

    ic8 = np.where(f['profiles'][0, 0, 0, :, 0] != 0)[0]

    ca_8 = sp.profile(nx=pixels[0].size, ny=1, ns=4, nw=wc8.size)

    ca_8.wav[:] = wc8[:]

    ca_8.dat[0, 0, :, ic8, :] = np.transpose(f['profiles'][()][0, pixels[0], pixels[1], :, :][:, ic8], axes=(1, 0, 2))

    ca_8.weights[:, :] = 1.e16
    ca_8.weights[ic8, 0] = 0.004
    for line_indice, core_indice in zip(line_indices, core_indices):
        ca_8.weights[ic8[line_indice[0] + core_indice[0]:line_indice[0] + core_indice[1]], 0] = 0.002

    ca_8.weights[ic8[line_indices[2][0] + core_indices[2][0]:line_indices[2][0] + core_indices[2][1]], 0] /= 4

    # ca_8.weights[ic8[97+85:97+120], 0] /= 2

    ca_8.weights[ic8, 3] = ca_8.weights[ic8, 0]

    ca_8.weights[ic8, 3] /= 2

    ca_8.weights[ic8[97 + 85:97 + 120], 3] /= 2

    # ca_8.weights[ic8[line_indices[2][0]+core_indices[2][0]:line_indices[2][0]+core_indices[2][1]], 3] /= 2

    ca_8.write(
        write_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_x_{}_y_{}_total_{}.nc'.format(
            '_'.join([str(_p) for _p in pixels[0].tolist()]), '_'.join([str(_p) for _p in pixels[1].tolist()]),
            pixels[0].size
        )
    )


def merge_atmospheres():
    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/pca_kmeans_fulldata_inversions/')
    pixel_files = [
        base_path / 'pixel_indices_quiet_total_843.h5',
        base_path / 'pixel_indices_quiet_retry_total_77.h5',
        base_path / 'pixel_indices_opposite_polarity_total_67.h5',
        base_path / 'pixel_indices_emission_total_110.h5',
    ]

    atmos_files = [
        base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_quiet_total_843_cycle_1_t_6_vl_2_vt_4_blong_2_atmos.nc',
        base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_quiet_retry_total_77_cycle_1_t_6_vl_2_vt_4_blong_2_atmos.nc',
        base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_opposite_polarity_total_67_cycle_1_t_6_vl_2_vt_4_blong_2_nw_atmos.nc',
        base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_emission_total_110_cycle_1_t_6_vl_4_vt_4_blong_2_nw_atmos.nc',
    ]

    keys = [
        'temp',
        'vlos',
        'vturb',
        'blong',
        'nne',
        'z',
        'ltau500',
        'pgas',
        'rho',
    ]

    f = h5py.File(base_path / 'combined_output.nc', 'w')
    outs = dict()
    for key in keys:
        outs[key] = np.zeros((1, 17, 60, 150), dtype=np.float64)

    for pixel_file, atmos_file in zip(pixel_files, atmos_files):
        pf = h5py.File(pixel_file, 'r')
        af = h5py.File(atmos_file, 'r')
        for key in keys:
            a, b, rp = pf['pixel_indices'][0], pf['pixel_indices'][1], pf['pixel_indices'][2]
            outs[key][0, a, b] = af[key][0, 0]
        pf.close()
        af.close()

    for key in keys:
        f[key] = outs[key]
    f.close()


def merge_output_profiles():
    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/pca_kmeans_fulldata_inversions/')
    pixel_files = [
        base_path / 'pixel_indices_quiet_total_843.h5',
        base_path / 'pixel_indices_quiet_retry_total_77.h5',
        base_path / 'pixel_indices_opposite_polarity_total_67.h5',
        base_path / 'pixel_indices_emission_total_110.h5',
    ]

    atmos_files = [
        base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_quiet_total_843_cycle_1_t_6_vl_2_vt_4_blong_2_profs.nc',
        base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_quiet_retry_total_77_cycle_1_t_6_vl_2_vt_4_blong_2_profs.nc',
        base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_opposite_polarity_total_67_cycle_1_t_6_vl_2_vt_4_blong_2_nw_profs.nc',
        base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_emission_total_110_cycle_1_t_6_vl_4_vt_4_blong_2_nw_profs.nc',
    ]

    keys = [
        'profiles'
    ]

    f = h5py.File(base_path / 'combined_output_profs.nc', 'w')
    outs = dict()
    for key in keys:
        outs[key] = np.zeros((1, 17, 60, 1236, 4), dtype=np.float64)

    for pixel_file, atmos_file in zip(pixel_files, atmos_files):
        pf = h5py.File(pixel_file, 'r')
        af = h5py.File(atmos_file, 'r')
        for key in keys:
            a, b, rp = pf['pixel_indices'][0], pf['pixel_indices'][1], pf['pixel_indices'][2]
            outs[key][0, a, b] = af[key][0, 0]
        pf.close()
        af.close()

    for key in keys:
        f[key] = outs[key]

    af = h5py.File(atmos_file, 'r')
    f['wav'] = af['wav'][()]
    af.close()

    f.close()


def generate_init_atmos_from_previous_result():
    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/')
    write_path = base_path / 'fulldata_inversions'

    pixel_file = write_path / 'pixel_indices_emission_total_33.h5'

    prev_output = h5py.File(write_path / 'combined_output.nc', 'r')

    fp = h5py.File(pixel_file, 'r')

    x, y = fp['pixel_indices'][0:2]

    m = sp.model(nx=x.size, ny=1, nt=1, ndep=150)

    m.ltau[:, :, :] = ltau

    m.pgas[:, :, :] = 1.0

    m.temp[0, 0] = prev_output['temp'][()][0, x, y]

    m.vlos[0, 0] = prev_output['vlos'][()][0, x, y]

    m.vturb[0, 0] = prev_output['vturb'][()][0, x, y]

    m.Bln[0, 0] = prev_output['blong'][()][0, x, y]

    write_filename = write_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_emission_total_{}_initial_atmos.nc'.format(
        x.size)

    m.write(write_filename)

    fp.close()

    prev_output.close()


def generate_file_for_rp_response_function():
    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/')
    write_path = base_path / 'fulldata_inversions'

    temp, vlos, vturb, blong = get_rp_atmos()

    m = sp.model(nx=30, ny=1, nt=1, ndep=150)

    m.ltau[:, :, :] = ltau

    m.pgas[:, :, :] = 1.0

    m.temp[0, 0] = temp

    m.vlos[0, 0] = vlos

    m.vturb[0, 0] = vturb

    m.Bln[0, 0] = blong

    write_filename = write_path / 'rps_atmos_30'

    m.write(str(write_filename))


def make_quiet_retry(indices):
    line_indices = [[0, 58], [58, 97], [97, 306]]

    core_indices = [[13, 45], [15, 36], [70, 135]]

    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/pca_kmeans_fulldata_inversions/')

    old_pixel_file = base_path / 'pixel_indices_quiet_total_843.h5'

    fopf = h5py.File(old_pixel_file, 'r')

    new_pixel_Data = fopf['pixel_indices'][()][:, indices]

    fopf.close()

    fo = h5py.File(
        base_path / 'pixel_indices_quiet_retry_total_{}.h5'.format(
            indices.size
        ), 'w'
    )

    fo['pixel_indices'] = new_pixel_Data

    fo.close()

    old_profile_file = base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_quiet_total_843.nc'

    f = h5py.File(old_profile_file, 'r')

    wc8 = f['wav'][()]

    ic8 = np.where(f['profiles'][0, 0, 0, :, 0] != 0)[0]

    ca_8 = sp.profile(nx=indices.size, ny=1, ns=4, nw=wc8.size)

    ca_8.wav[:] = wc8[:]

    ca_8.dat[0, 0, :, ic8, :] = np.transpose(f['profiles'][()][0, 0, indices][:, ic8], axes=(1, 0, 2))

    ca_8.weights[:, :] = 1.e16
    ca_8.weights[ic8, 0] = 0.004
    for line_indice, core_indice in zip(line_indices, core_indices):
        ca_8.weights[ic8[line_indice[0] + core_indice[0]:line_indice[0] + core_indice[1]], 0] = 0.002

    ca_8.weights[ic8[line_indices[2][0] + core_indices[2][0]:line_indices[2][0] + core_indices[2][1]], 0] /= 2

    ca_8.weights[ic8, 3] = ca_8.weights[ic8, 0]

    ca_8.weights[ic8, 3] /= 2

    ca_8.weights[ic8[line_indices[0][0] + core_indices[0][0]:line_indices[0][0] + core_indices[0][1]], 3] /= 4

    ca_8.weights[ic8[line_indices[1][0] + core_indices[1][0]:line_indices[1][0] + core_indices[1][1]], 3] /= 4

    ca_8.weights[ic8[line_indices[2][0] + core_indices[2][0]:line_indices[2][0] + core_indices[2][1]], 3] /= 2

    ca_8.write(
        base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_quiet_retry_total_{}.nc'.format(indices.size)
    )

    f.close()

    old_atmos_file = base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_quiet_total_843_cycle_1_t_6_vl_2_vt_4_blong_2_nw_atmos.nc'

    f = h5py.File(old_atmos_file, 'r')

    m = sp.model(nx=indices.size, ny=1, nt=1, ndep=150)

    m.ltau[:, :, :] = ltau

    m.pgas[:, :, :] = 1.0

    m.temp[0, 0] = f['temp'][()][0, 0, indices]

    m.vlos[0, 0] = f['vlos'][()][0, 0, indices]

    m.vturb[0, 0] = f['vturb'][()][0, 0, indices]

    m.Bln[0, 0] = f['blong'][()][0, 0, indices]

    write_filename = base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_quiet_retry_total_{}_initial_atmos.nc'.format(
        indices.size)

    m.write(str(write_filename))


def make_emission_retry(indices):
    line_indices = [[0, 58], [58, 97], [97, 306]]

    core_indices = [[13, 45], [15, 36], [70, 135]]

    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/pca_kmeans_fulldata_inversions/')

    old_pixel_file = base_path / 'pixel_indices_emission_total_110.h5'

    fopf = h5py.File(old_pixel_file, 'r')

    new_pixel_Data = fopf['pixel_indices'][()][:, indices]

    fopf.close()

    fo = h5py.File(
        base_path / 'pixel_indices_emission_retry_total_{}.h5'.format(
            indices.size
        ), 'w'
    )

    fo['pixel_indices'] = new_pixel_Data

    fo.close()

    old_profile_file = base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_emission_total_110.nc'

    f = h5py.File(old_profile_file, 'r')

    wc8 = f['wav'][()]

    ic8 = np.where(f['profiles'][0, 0, 0, :, 0] != 0)[0]

    ca_8 = sp.profile(nx=indices.size, ny=1, ns=4, nw=wc8.size)

    ca_8.wav[:] = wc8[:]

    ca_8.dat[0, 0, :, ic8, :] = np.transpose(f['profiles'][()][0, 0, indices][:, ic8], axes=(1, 0, 2))

    ca_8.weights[:, :] = 1.e16
    ca_8.weights[ic8, 0] = 0.004
    for line_indice, core_indice in zip(line_indices, core_indices):
        ca_8.weights[ic8[line_indice[0] + core_indice[0]:line_indice[0] + core_indice[1]], 0] = 0.002

    ca_8.weights[ic8[97 + 85:97 + 120], 0] /= 2

    ca_8.weights[ic8, 3] = ca_8.weights[ic8, 0]

    ca_8.weights[ic8, 3] /= 2

    ca_8.weights[ic8[97 + 85:97 + 120], 3] /= 2

    ca_8.write(
        base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_emission_retry_total_{}.nc'.format(
            indices.size)
    )

    f.close()

    old_atmos_file = base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_emission_total_110_cycle_1_t_6_vl_4_vt_4_blong_2_atmos.nc'

    f = h5py.File(old_atmos_file, 'r')

    m = sp.model(nx=indices.size, ny=1, nt=1, ndep=150)

    m.ltau[:, :, :] = ltau

    m.pgas[:, :, :] = 1.0

    m.temp[0, 0] = f['temp'][()][0, 0, indices]

    m.vlos[0, 0] = f['vlos'][()][0, 0, indices]

    m.vturb[0, 0] = f['vturb'][()][0, 0, indices]

    m.Bln[0, 0] = f['blong'][()][0, 0, indices]

    write_filename = base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_emission_retry_total_{}_initial_atmos.nc'.format(
        indices.size)

    m.write(str(write_filename))


def make_emission_retry_retry(indices):
    line_indices = [[0, 58], [58, 97], [97, 306]]

    core_indices = [[13, 45], [15, 36], [70, 135]]

    base_path = Path('/home/harsh/SpinorNagaraju/maps_1/stic/pca_kmeans_fulldata_inversions/')

    old_pixel_file = base_path / 'pixel_indices_emission_retry_total_34.h5'

    fopf = h5py.File(old_pixel_file, 'r')

    new_pixel_Data = fopf['pixel_indices'][()][:, indices]

    fopf.close()

    fo = h5py.File(
        base_path / 'pixel_indices_emission_retry_retry_total_{}.h5'.format(
            indices.size
        ), 'w'
    )

    fo['pixel_indices'] = new_pixel_Data

    fo.close()

    old_profile_file = base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_emission_retry_total_34.nc'

    f = h5py.File(old_profile_file, 'r')

    wc8 = f['wav'][()]

    ic8 = np.where(f['profiles'][0, 0, 0, :, 0] != 0)[0]

    ca_8 = sp.profile(nx=indices.size, ny=1, ns=4, nw=wc8.size)

    ca_8.wav[:] = wc8[:]

    ca_8.dat[0, 0, :, ic8, :] = np.transpose(f['profiles'][()][0, 0, indices][:, ic8], axes=(1, 0, 2))

    ca_8.weights[:, :] = 1.e16
    ca_8.weights[ic8, 0] = 0.004
    for line_indice, core_indice in zip(line_indices, core_indices):
        ca_8.weights[ic8[line_indice[0] + core_indice[0]:line_indice[0] + core_indice[1]], 0] = 0.002

    ca_8.weights[ic8[97 + 85:97 + 120], 0] /= 2

    ca_8.weights[ic8, 3] = ca_8.weights[ic8, 0]

    ca_8.weights[ic8, 3] /= 2

    ca_8.weights[ic8[97 + 85:97 + 120], 3] /= 2

    ca_8.write(
        base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_emission_retry_retry_total_{}.nc'.format(
            indices.size)
    )

    f.close()

    old_atmos_file = base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_emission_retry_total_34_cycle_1_t_6_vl_4_vt_4_blong_2_atmos.nc'

    f = h5py.File(old_atmos_file, 'r')

    m = sp.model(nx=indices.size, ny=1, nt=1, ndep=150)

    m.ltau[:, :, :] = ltau

    m.pgas[:, :, :] = 1.0

    m.temp[0, 0] = f['temp'][()][0, 0, indices]

    m.vlos[0, 0] = f['vlos'][()][0, 0, indices]

    m.vturb[0, 0] = f['vturb'][()][0, 0, indices]

    m.Bln[0, 0] = f['blong'][()][0, 0, indices]

    write_filename = base_path / 'alignedspectra_scan1_map01_Ca.fits_stic_profiles.nc_emission_retry_retry_total_{}_initial_atmos.nc'.format(
        indices.size)

    m.write(str(write_filename))


def generate_mean_files_for_inversions_from_coordinates(x, y, radius, type):
    line_indices = [[0, 58], [58, 97], [97, 306]]

    core_indices = [[13, 45], [15, 36], [70, 135]]

    stic = Path('/home/harsh/SpinorNagaraju/maps_1/stic')

    base_path = stic / 'pca_kmeans_fulldata_inversions'

    output_atmos_file = base_path / 'combined_output.nc'

    output_prof_file = base_path / 'combined_output_profs.nc'

    stic = Path('/home/harsh/SpinorNagaraju/maps_2_scan10/stic')

    base_path = stic / 'pca_kmeans_fulldata_inversions'

    input_file = stic / 'processed_inputs' / 'alignedspectra_scan2_map10_Ca.fits_stic_profiles.nc'

    f = h5py.File(input_file, 'r')

    wc8 = f['wav'][()]

    ic8 = np.where(f['profiles'][0, 0, 0, :, 0] != 0)[0]

    ca_8 = sp.profile(nx=1, ny=1, ns=4, nw=wc8.size)

    ca_8.wav[:] = wc8[:]

    ca_8.dat[0, 0, 0, ic8, :] = np.mean(
        f['profiles'][0, x - radius:x + radius + 1, y - radius: y + radius + 1][:, :, ic8],
        axis=(0, 1)
    )

    if type == 'opposite_polarity':
        ca_8.weights[:, :] = 1.e16

        ca_8.weights[ic8, 0] = 0.004
        k = 0
        for line_indice, core_indice in zip(line_indices, core_indices):
            if k < 2:
                ca_8.weights[ic8[line_indice[0] + core_indice[0]:line_indice[0] + core_indice[1]], 0] = 0.002
            else:
                ca_8.weights[ic8[line_indice[0] + core_indice[0]:line_indice[0] + core_indice[1]], 0] = 0.002
            k += 1

        ca_8.weights[ic8[line_indices[2][0] + core_indices[2][0]:line_indices[2][0] + core_indices[2][1]], 0] /= 2

        # ca_8.weights[ic8, 3] = ca_8.weights[ic8, 0]

        # ca_8.weights[ic8, 3] /= 2

        # ca_8.weights[ic8[line_indices[0][0] + core_indices[0][0]:line_indices[0][0] + core_indices[0][1]], 3] /= 4

        # ca_8.weights[ic8[line_indices[1][0] + core_indices[1][0]:line_indices[1][0] + core_indices[1][1]], 3] /= 4

        # ca_8.weights[ic8[line_indices[2][0] + core_indices[2][0]:line_indices[2][0] + core_indices[2][1]], 3] /= 2

    else:
        ca_8.weights[:, :] = 1.e16
        ca_8.weights[ic8, 0] = 0.004

        for line_indice, core_indice in zip(line_indices, core_indices):
            ca_8.weights[ic8[line_indice[0] + core_indice[0]:line_indice[0] + core_indice[1]], 0] = 0.002

        ca_8.weights[ic8[97 + 85:97 + 120], 0] /= 2

        ca_8.weights[ic8, 3] = ca_8.weights[ic8, 0]

        ca_8.weights[ic8, 3] /= 2

        ca_8.weights[ic8[97 + 85:97 + 120], 3] /= 2

    ca_8.write(
        base_path / 'alignedspectra_scan2_map10_Ca.fits_stic_profiles.nc_{}_mean_{}_{}_{}_total_1.nc'.format(type, x, y,
                                                                                                             radius)
    )

    f.close()

    m = sp.model(nx=1, ny=1, nt=1, ndep=150)

    m.ltau[:, :, :] = ltau

    m.pgas[:, :, :] = 1.0

    f = h5py.File(output_atmos_file, 'r')

    m.temp[0, 0, 0] = np.mean(f['temp'][0, x - radius:x + radius + 1, y - radius:y + radius + 1], (0, 1))

    m.vlos[0, 0, 0] = np.mean(f['vlos'][0, x - radius:x + radius + 1, y - radius:y + radius + 1], (0, 1))

    m.vturb[0, 0, 0] = np.mean(f['vturb'][0, x - radius:x + radius + 1, y - radius:y + radius + 1], (0, 1))

    m.Bln[0, 0, 0] = np.mean(f['blong'][0, x - radius:x + radius + 1, y - radius:y + radius + 1], (0, 1))

    f.close()

    write_filename = base_path / 'alignedspectra_scan2_map10_Ca.fits_stic_profiles.nc_{}_mean_{}_{}_{}_total_1_initial_atmos.nc'.format(
        type, x, y, radius)

    m.write(str(write_filename))


if __name__ == '__main__':
    # make_rps()
    # make_halpha_rps()
    # plot_rp_map_fov()
    # make_rps_plots()
    # make_halpha_rps_plots()
    # make_stic_inversion_files()
    # make_stic_inversion_files(rps=[0, 1, 4, 5, 6, 7, 8, 9, 11, 14, 16, 17, 19, 22, 23, 24, 25, 26, 27, 28, 29])
    # make_stic_inversion_files(rps=[2, 3, 10, 12, 13, 15, 18, 21])
    # make_stic_inversion_files(rps=[20])
    # make_stic_inversion_files_halpha_ca_both(rps=[0, 1, 2, 4, 5, 7, 9, 10, 11, 13, 14, 17, 18, 28, 29])
    # make_stic_inversion_files_halpha_ca_both(rps=[3, 6, 8, 12, 15, 16, 19, 20, 22, 23, 25])
    # make_stic_inversion_files_halpha_ca_both(rps=[21, 24])
    # make_stic_inversion_files_halpha_ca_both(rps=[26, 27])
    generate_input_atmos_file(length=30, name='median')
    # generate_input_atmos_file(length=30, temp=[[-8, -6, -4.5, -3.8, -2.9, -1.8, -0.9, 0, 1.2], [9000, 5500, 4500, 4200, 4500, 5000, 5500, 6000, 8000]], vlos=[[-8, -6, -4, -2, 0, 2], [2e5, 2e5, 2e5, 2e5, 2e5, 2e5]], blong=0, name='median')
    # generate_input_atmos_file(length=5, temp=[[-8, -6, -4, -2, 0, 2], [7000, 5000, 4000, 5000, 7000, 10000]], vlos=[[-8, -6, -4, -2, 0, 2], [2e5, 2e5, 2e5, 2e5, 2e5, 2e5]], blong=100, name='emission')
    # generate_input_atmos_file(length=21, temp=[[-8, -6, -4, -2, 0, 2], [9000, 7000, 5000, 6000, 8000, 10000]], vlos=[[-8, -6, -4, -2, 0, 2], [2e5, 2e5, 2e5, 2e5, 2e5, 2e5]], blong=0, name='quiet')
    # generate_input_atmos_file(length=1, temp=[[-8, -6, -4, -2, 0, 2], [11000, 7000, 5000, 6000, 8000, 10000]], vlos=[[-8, -6, -4, -2, 0, 2], [-10e5, -5e5, -3e3, 1e5, 0, 0]], blong=-450, name='red')
    # generate_input_atmos_file(length=1, temp=[[-8, -6, -4, -2, 0, 2], [18000, 9000, 5000, 5500, 6200, 7000]], vlos=[[-8, -6, -4, -2, 0, 2], [-10e5, -5e5, -3e5, -3e5, 0, 0]], blong=([-4.5, -1], [80, -80]), name='map_2_opp_pol')
    # generate_input_atmos_file_from_previous_result(result_filename='/home/harsh/SpinorNagaraju/maps_1/stic/PCA_RPs_Plots/inversions/rps_stic_profiles_x_2_3_10_12_13_15_18_21_y_1_cycle_1_t_6_vl_4_vt_4_blong_0_atmos.nc', rps=[0, 1, 2, 4])
    # make_rps_inversion_result_plots(nodes_temp=[-4.8, -3.8, -2.9, -1.8, -0.9, 0], nodes_vlos=[-6, -4.5, -3, -1], nodes_vturb=[-5, -4, -3, -1], nodes_blos=[-4.5, -1])
    # make_rps_inversion_result_plots(nodes_temp=[-5.5, -4.5, -3.5, -2.5, -1.5, 0], nodes_vlos=[-4.5, -1], nodes_vturb=[-5, -4, -3, -1], nodes_blos=[-4.5, -1])
    # make_rps_inversion_result_plots(nodes_temp=[-5.5, -4.5, -3.5, -2.5, -1.5, 0], nodes_vlos=[-6, -4.5, -3.5, -1.5, 0],
    #                                 nodes_vturb=[-5, -4, -3, -1], nodes_blos=None)
    # make_ca_rps_inversion_result_plots()
    # combine_rps_atmos()
    # full_map_generate_input_atmos_file_from_previous_result()
    # make_quiet_retry(indices=np.array([ 13,  14,  16,  26,  41,  48,  50,  60,  64,  68,  82,  87, 104,
    #    113, 114, 119, 139, 143, 155, 162, 163, 176, 177, 182, 188, 199,
    #    202, 209, 211, 221, 225, 234, 290, 307, 311, 330, 337, 340, 384,
    #    386, 397, 422, 442, 452, 469, 472, 498, 510, 512, 535, 541, 550,
    #    555, 584, 590, 603, 621, 627, 636, 663, 681, 695, 698, 708, 718,
    #    719, 726, 730, 736, 772, 773, 794, 798, 811, 816, 827, 830]))
    # make_quiet_retry(
    #     indices=np.array(
    #         [6,  48,  50,  60,  66,  68,  77, 102, 104, 139, 143, 144, 147,
    #     166, 171, 194, 199, 204, 209, 211, 227, 234, 247, 260, 291, 309,
    #     310, 330, 344, 375, 378, 383, 384, 386, 401, 418, 424, 447, 454,
    #     469, 470, 472, 487, 498, 584, 612, 627, 658, 703, 775]
    #     )
    # )
    # make_emission_retry(indices=np.array([ 4, 21, 22, 24, 25, 29, 50, 51, 62, 91, 92, 95]))
    # make_emission_retry_retry(indices=np.array([17, 19]))
    # generate_actual_inversion_files_quiet()
    # generate_actual_inversion_files_median_profile()
    # generate_actual_inversion_files_emission()
    # generate_actual_inversion_files_opposite_polarity()
    # generate_init_atmos_from_previous_result()
    # generate_actual_inversion_files_spot()
    # generate_actual_inversion_files_mean()
    # generate_mean_files_for_inversions_from_coordinates(8, 51, 1, 'opposite_polarity')
    # generate_mean_files_for_inversions_from_coordinates(13, 51, 1, 'opposite_polarity')
    # generate_mean_files_for_inversions_from_coordinates(12, 40, 1, 'emission')
    # generate_mean_files_for_inversions_from_coordinates(13, 17, 1, 'quiet')
    # generate_mean_files_for_inversions_from_coordinates(10, 34, 1, 'spot')
    # merge_atmospheres()
    # merge_output_profiles()
    # generate_actual_inversion_pixels((np.array([12, 12]), np.array([49, 31])))
    # generate_actual_inversion_pixels((np.array([12]), np.array([40])))
    # generate_input_atmos_file(length=2, temp=[[-8, -6, -4, -2, 0, 2], [11000, 7000, 5000, 6000, 8000, 10000]], vlos=[[-8, -6, -4, -2, 0, 2], [-10e5, -5e5, -3e3, 1e5, 0, 0]], blong=-200, name='red')
    # generate_input_atmos_file(length=1, temp=[[-8, -6, -4, -2, 0, 2], [11000, 7000, 5000, 6000, 8000, 10000]], vlos=[[-8, -6, -4, -2, 0, 2], [20e5, -6e5, -3e5, -1e5, 0, 0]], blong=-200, name='blue')
    # make_pixels_inversion_result_plots(nodes_temp=[-4.5, -3.8, -2.9, -1.8, -0.9, 0], nodes_vlos=[-4.5, -1], nodes_vturb=[-5, -4, -3, -1], nodes_blos=[-4.5, -1])
    # generate_file_for_rp_response_function()
