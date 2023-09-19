import sys

import sunpy.io
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
from weak_field_approx import *
from sklearn.decomposition import PCA
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FormatStrFormatter

def calculate_magnetic_field(datestring, errors=None, bin_factor=None):

    # base_path = Path('F:\\Harsh\\CourseworkRepo\\InstrumentalUncorrectedStokes')

    # base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes')

    # base_path = Path('C:\\Work Things\\InstrumentalUncorrectedStokes')

    base_path = Path('C:\\Work Things\\InstrumentalUncorrectedStokes')

    datepath = base_path / datestring

    level8path = datepath / 'Level-4'

    all_files = level8path.glob('**/*')

    all_mag_files = [file for file in all_files if file.name.startswith('aligned') and file.name.endswith('spatial_straylight_corrected.nc')]

    # for a_mag_file in all_mag_files:
    #     fcaha = h5py.File(a_mag_file, 'r')
    #
    #     ind = np.where(fcaha['profiles'][0, 0, 0, :, 0] != 0)[0]
    #
    #     ind = ind[800:]
    #
    #     actual_calculate_blos = prepare_calculate_blos(
    #         fcaha['profiles'][0][:, :, ind],
    #         fcaha['wav'][ind] / 10,
    #         8661.8991 / 10,
    #         8661.7 / 10,
    #         8661.8 / 10,
    #         1.5,
    #         transition_skip_list=None,
    #         bin_factor=bin_factor,
    #         errors=errors
    #     )
    #
    #     vec_actual_calculate_blos = np.vectorize(actual_calculate_blos)
    #
    #     magca = np.fromfunction(vec_actual_calculate_blos, shape=(fcaha['profiles'].shape[1], fcaha['profiles'].shape[2]))
    #
    #     if errors is not None:
    #         sunpy.io.write_file(level8path / '{}_mag_ca_fe_errors.fits'.format(a_mag_file.name), magca, dict(), overwrite=True)
    #     else:
    #         sunpy.io.write_file(level8path / '{}_mag_ca_fe.fits'.format(a_mag_file.name), magca, dict(), overwrite=True)
    #
    #     sys.stdout.write('Ca fe created\n')
    #
    #     actual_calculate_blos = prepare_calculate_blos(
    #         fcaha['profiles'][0][:, :, ind],
    #         fcaha['wav'][ind] / 10,
    #         8662.17 / 10,
    #         8662.17 / 10,
    #         (8662.17 + 0.4) / 10,
    #         0.83,
    #         transition_skip_list=None,
    #         bin_factor=bin_factor,
    #         errors=errors
    #     )
    #
    #     vec_actual_calculate_blos = np.vectorize(actual_calculate_blos)
    #
    #     magca = np.fromfunction(vec_actual_calculate_blos,
    #                             shape=(fcaha['profiles'].shape[1], fcaha['profiles'].shape[2]))
    #
    #     if errors is not None:
    #         sunpy.io.write_file(level8path / '{}_mag_ca_core_errors.fits'.format(a_mag_file.name), magca, dict(),
    #                             overwrite=True)
    #     else:
    #         sunpy.io.write_file(level8path / '{}_mag_ca_core.fits'.format(a_mag_file.name), magca, dict(), overwrite=True)
    #
    #     sys.stdout.write('Ca core created\n')

    for a_mag_file in all_mag_files:
        fcaha = h5py.File(a_mag_file, 'r')

        ind = np.where(fcaha['profiles'][0, 0, 0, :, 0] != 0)[0]

        ind = ind[0:800]
        ha_center_wave = 6562.8 / 10
        wave_range = 0.35 / 10

        transition_skip_list = np.array(
            [
                [6560.57, 0.25],
                [6561.09, 0.1],
                [6562.44, 0.1],
                [6563.645, 0.3],
                [6564.15, 0.35]
            ]
        ) / 10

        actual_calculate_blos = prepare_calculate_blos(
            fcaha['profiles'][0][:, :, ind],
            fcaha['wav'][ind] / 10,
            ha_center_wave,
            ha_center_wave - wave_range,
            ha_center_wave + wave_range,
            1.048,
            transition_skip_list=transition_skip_list,
            bin_factor=bin_factor,
            errors=errors
        )

        vec_actual_calculate_blos = np.vectorize(actual_calculate_blos)

        magha = np.fromfunction(vec_actual_calculate_blos, shape=(fcaha['profiles'].shape[1], fcaha['profiles'].shape[2]))

        if errors is not None:
            sunpy.io.write_file(level8path / '{}_mag_ha_core_errors.fits'.format(a_mag_file.name), magha, dict(),
                                overwrite=True)
        else:
            sunpy.io.write_file(level8path / '{}_mag_ha_core.fits'.format(a_mag_file.name), magha, dict(), overwrite=True)

        sys.stdout.write('Ha core created\n')
        wave_range = 1.5 / 10

        actual_calculate_blos = prepare_calculate_blos(
            fcaha['profiles'][0][:, :, ind],
            fcaha['wav'][ind] / 10,
            ha_center_wave,
            ha_center_wave - wave_range,
            ha_center_wave + (0.5/10),
            1.048,
            transition_skip_list=transition_skip_list,
            bin_factor=bin_factor,
            errors=errors
        )

        vec_actual_calculate_blos = np.vectorize(actual_calculate_blos)

        magfha = np.fromfunction(vec_actual_calculate_blos,
                                 shape=(fcaha['profiles'].shape[1], fcaha['profiles'].shape[2]))

        if errors is not None:
            sunpy.io.write_file(level8path / '{}_mag_ha_full_line_errors.fits'.format(a_mag_file.name), magfha, dict(),
                                overwrite=True)
        else:
            sunpy.io.write_file(level8path / '{}_mag_ha_full_line.fits'.format(a_mag_file.name), magfha, dict(),
                            overwrite=True)

        sys.stdout.write('Ha full line created\n')


def create_fov_plots(datestring, timestring, x1, y1, x2, y2, ticks, limit, points):

    colors = ["blue", "green", "white", "darkgoldenrod", "darkred"]

    colors.reverse()

    cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)

    extent = [0, 0.6 * 50, 0, 0.6 * 50]

    # write_path = Path('/home/harsh/CourseworkRepo/KTTAnalysis/figures')

    write_path = Path('F:\\Harsh\\CourseworkRepo\\KTTAnalysis\\figures')

    # base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes')

    base_path = Path('C:\\Work Things\\InstrumentalUncorrectedStokes')

    datepath = base_path / datestring

    level4path = datepath / 'Level-4'

    f = h5py.File(
        level4path / 'aligned_Ca_Ha_stic_profiles_{}_{}.nc_pca.nc_spatial_straylight_corrected.nc'.format(
            datestring, timestring
        ),
        'r'
    )

    fe_mag, _ = sunpy.io.read_file(
        level4path / 'aligned_Ca_Ha_stic_profiles_{}_{}.nc_pca.nc_spatial_straylight_corrected.nc_mag_ca_fe.fits'.format(
            datestring, timestring
        )
    )[0]

    ca_mag, _ = sunpy.io.read_file(
        level4path / 'aligned_Ca_Ha_stic_profiles_{}_{}.nc_pca.nc_spatial_straylight_corrected.nc_mag_ca_core.fits'.format(
            datestring, timestring
        )
    )[0]

    hmi_img, _ = sunpy.io.read_file(
        level4path / 'HMI_reference_image_{}_{}.fits'.format(
            datestring, timestring
        )
    )[0]

    hmi_mag, _ = sunpy.io.read_file(
        level4path / 'HMI_reference_magnetogram_{}_{}.fits'.format(
            datestring, timestring
        )
    )[0]

    ind = np.where(f['profiles'][0, 0, 0, :, 0] != 0)[0]

    wave = f['wav'][ind]

    ind_halpha_core = ind[np.argmin(np.abs(wave - 6562.8))]

    ind_ca_core = ind[np.argmin(np.abs(wave - 8662.17))]

    font = {'size': 8}

    matplotlib.rc('font', **font)

    fig, axs = plt.subplots(2, 4, figsize=(7, 3.5))

    axs[0][0].imshow(f['profiles'][0, y1:y2, x1:x2, 3204 + 8, 0], cmap='gray', origin='lower', extent=extent)

    axs[1][0].imshow(f['profiles'][0, y1:y2, x1:x2, ind_ca_core, 0], cmap='gray', origin='lower', extent=extent)

    axs[0][1].imshow(f['profiles'][0, y1:y2, x1:x2, 32, 0], cmap='gray', origin='lower', extent=extent)

    axs[1][1].imshow(f['profiles'][0, y1:y2, x1:x2, ind_halpha_core, 0], cmap='gray', origin='lower', extent=extent)

    im02 = axs[0][2].imshow(fe_mag[y1:y2, x1:x2], cmap=cmap1, origin='lower', vmin=-limit, vmax=limit, extent=extent)

    im12 = axs[1][2].imshow(ca_mag[y1:y2, x1:x2], cmap=cmap1, origin='lower', vmin=-limit, vmax=limit, extent=extent)

    axs[0][3].imshow(hmi_img[y1:y2, x1:x2], cmap='gray', origin='lower', extent=extent)

    im13 = axs[1][3].imshow(hmi_mag[y1:y2, x1:x2], cmap=cmap1, origin='lower', vmin=-limit, vmax=limit, extent=extent)

    pcolors = ['brown', 'navy']
    for indice, point in enumerate(points):
        for i in range(2):
            for j in range(4):
                axs[i][j].scatter((point[0] - x1) * 0.6, (point[1] - y1) * 0.6, marker='x', s=16, color=pcolors[indice])

    axins02 = inset_axes(
        axs[0][2],
        width="100%",
        height="100%",
        loc="upper right",
        bbox_to_anchor=(0.05, 0.05, 0.9, 0.05),
        bbox_transform=axs[0][2].transAxes,
        borderpad=0,
    )

    cbar = fig.colorbar(im02, cax=axins02, ticks=ticks, orientation='horizontal')

    tick_values = np.array(ticks) / 100

    tick_values = tick_values.astype(np.int64)

    tick_values = [str(tick) for tick in tick_values]

    cbar.ax.set_xticklabels(tick_values)

    cbar.ax.xaxis.set_ticks_position('top')

    ticks = [0, 5, 10, 15, 20, 25]

    for i in range(2):
        for j in range(4):
            axs[i][j].set_xticks(ticks, [])
            axs[i][j].set_yticks(ticks, [])
            axs[i][j].xaxis.set_minor_locator(MultipleLocator(1))
            axs[i][j].yaxis.set_minor_locator(MultipleLocator(1))

    axs[0][0].set_yticks(ticks, ticks)
    axs[1][0].set_yticks(ticks, ticks)

    axs[1][0].set_xticks(ticks, ticks)
    axs[1][1].set_xticks(ticks, ticks)
    axs[1][2].set_xticks(ticks, ticks)
    axs[1][3].set_xticks(ticks, ticks)

    axs[1][2].text(
        -0.3, -0.25,
        'Scan direction [arcsec]',
        transform=axs[1][2].transAxes,
        color='black'
    )

    axs[1][0].text(
        -0.3, 0.6,
        'Slit direction [arcsec]',
        transform=axs[1][0].transAxes,
        color='black',
        rotation=90
    )

    axs[0][0].text(
        0.05, 0.92,
        '(a) Ca far-wing',
        transform=axs[0][0].transAxes,
        color='black'
    )

    axs[0][1].text(
        0.05, 0.92,
        r'(c) H$\alpha$ far-wing',
        transform=axs[0][1].transAxes,
        color='black'
    )

    axs[0][2].text(
        0.05, 0.92,
        r'(e) $B_{LOS}$ Fe I [x 100 G]',
        transform=axs[0][2].transAxes,
        color='black'
    )

    axs[0][3].text(
        0.05, 0.92,
        r'(g) HMI continuum',
        transform=axs[0][3].transAxes,
        color='black'
    )

    axs[1][0].text(
        0.05, 0.92,
        '(b) Ca core',
        transform=axs[1][0].transAxes,
        color='white'
    )

    axs[1][1].text(
        0.05, 0.92,
        r'(d) H$\alpha$ core',
        transform=axs[1][1].transAxes,
        color='white'
    )

    axs[1][2].text(
        0.05, 0.92,
        r'(f) $B_{LOS}$ Ca II [x 100 G]',
        transform=axs[1][2].transAxes,
        color='black'
    )

    axs[1][3].text(
        0.05, 0.92,
        r'(h) HMI magnetogram',
        transform=axs[1][3].transAxes,
        color='black'
    )

    plt.subplots_adjust(left=0.13, right=1, bottom=0.13, top=1, wspace=0.0, hspace=0.0)

    plt.savefig(
        write_path / 'FOV_{}_{}.pdf'.format(
            datestring, timestring
        ),
        format='pdf',
        dpi=300
    )

    f.close()


def get_stokes_v_map(f, wave, wave_range):

    ind_non_zero = np.where(f['profiles'][0, 0, 0, :, 0])[0]

    ind_range = np.where((f['wav'][ind_non_zero] >= wave - wave_range) & (f['wav'][ind_non_zero] <= wave + wave_range))[0]

    wave_ind = ind_non_zero[ind_range]

    sm = np.nanmean(f['profiles'][0, :, :, wave_ind, 3] / f['profiles'][0, :, :, wave_ind, 0], 2)

    sm[np.where(np.isnan(sm))] = 0

    sm[np.where(np.isinf(sm))] = 0

    return sm


def create_fov_plots_adaptive_optics(datestring, timestring, x1, y1, x2, y2, ticks1, sv_max, limit, points):

    colors = ["blue", "green", "white", "darkgoldenrod", "darkred"]

    colors.reverse()

    cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)

    extent = [0, 0.6 * 50, 0, 0.6 * 50]

    # write_path = Path('/home/harsh/CourseworkRepo/KTTAnalysis/figures')

    # write_path = Path('F:\\Harsh\\CourseworkRepo\\KTTAnalysis\\figures')

    write_path = Path('/Volumes/HarshHDD-Data/Documents/CourseworkRepo/KTTAnalysis/figures')

    # base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes')

    base_path = Path('/Volumes/HarshHDD-Data/Documents/CourseworkRepo/InstrumentalUncorrectedStokes')

    # base_path = Path('C:\\Work Things\\InstrumentalUncorrectedStokes')

    datepath = base_path / datestring

    level4path = datepath / 'Level-4'

    f = h5py.File(
        level4path / 'aligned_Ca_Ha_stic_profiles_{}_{}.nc_pca.nc_spatial_straylight_corrected.nc'.format(
            datestring, timestring
        ),
        'r'
    )

    hmi_img, _ = sunpy.io.read_file(
        level4path / 'HMI_reference_image_{}_{}.fits'.format(
            datestring, timestring
        )
    )[0]

    hmi_mag, _ = sunpy.io.read_file(
        level4path / 'HMI_reference_magnetogram_{}_{}.fits'.format(
            datestring, timestring
        )
    )[0]

    ind = np.where(f['profiles'][0, 0, 0, :, 0] != 0)[0]

    wave = f['wav'][ind]

    ind_halpha_core = ind[np.argmin(np.abs(wave - 6562.8))]

    ind_ca_core = ind[np.argmin(np.abs(wave - 8662.17))]

    plt.close('all')

    plt.clf()

    plt.cla()

    font = {'size': 8}

    matplotlib.rc('font', **font)

    fig, axs = plt.subplots(2, 5, figsize=(7, 3))

    axs[0][0].imshow(f['profiles'][0, y1:y2, x1:x2, 3204 + 8, 0], cmap='gray', origin='lower', extent=extent, vmin=f['profiles'][0, y1:y2, x1:x2, 3204 + 8, 0][5:-5, 5:-5].min(), vmax=f['profiles'][0, y1:y2, x1:x2, 3204 + 8, 0][5:-5, 5:-5].max())

    axs[1][0].imshow(f['profiles'][0, y1:y2, x1:x2, ind_ca_core, 0], cmap='gray', origin='lower', extent=extent, vmin=f['profiles'][0, y1:y2, x1:x2, ind_ca_core, 0][5:-5, 5:-5].min(), vmax=f['profiles'][0, y1:y2, x1:x2, ind_ca_core, 0][5:-5, 5:-5].max())

    axs[0][1].imshow(f['profiles'][0, y1:y2, x1:x2, 32, 0], cmap='gray', origin='lower', extent=extent, vmin=f['profiles'][0, y1:y2, x1:x2, 32, 0][5:-5, 5:-5].min(), vmax=f['profiles'][0, y1:y2, x1:x2, 32, 0][5:-5, 5:-5].max())

    axs[1][1].imshow(f['profiles'][0, y1:y2, x1:x2, ind_halpha_core, 0], cmap='gray', origin='lower', extent=extent, vmin=f['profiles'][0, y1:y2, x1:x2, ind_halpha_core, 0][5:-5, 5:-5].min(), vmax=f['profiles'][0, y1:y2, x1:x2, ind_halpha_core, 0][5:-5, 5:-5].max())

    axs[0][2].imshow(hmi_img[y1:y2, x1:x2], cmap='gray', origin='lower', extent=extent)

    im12 = axs[1][2].imshow(hmi_mag[y1:y2, x1:x2], cmap=cmap1, origin='lower', vmin=-limit, vmax=limit, extent=extent)

    axs[0][4].imshow(get_stokes_v_map(f, 6562.2405, 0.0664)[y1:y2, x1:x2] * 100, cmap=cmap1, origin='lower', vmin=-sv_max, vmax=sv_max, extent=extent)

    im14 = axs[1][4].imshow(get_stokes_v_map(f, 6563.2505, 0.0664)[y1:y2, x1:x2] * 100, cmap=cmap1, origin='lower', vmin=-sv_max, vmax=sv_max, extent=extent)

    axs[0][3].imshow(get_stokes_v_map(f, 8661.705, 0.0664)[y1:y2, x1:x2] * 100, cmap=cmap1, origin='lower', vmin=-sv_max, vmax=sv_max, extent=extent)

    axs[1][3].imshow(get_stokes_v_map(f, 8662.425, 0.0664)[y1:y2, x1:x2] * 100, cmap=cmap1, origin='lower', vmin=-sv_max, vmax=sv_max, extent=extent)

    pcolors = ['brown', 'navy']

    # for indice, point in enumerate(points):
    #     for i in range(2):
    #         for j in range(4):
    #             axs[i][j].scatter((point[0] - x1) * 0.6, (point[1] - y1) * 0.6, marker='x', s=16, color=pcolors[indice])

    axins12 = inset_axes(
        axs[1][2],
        width="100%",
        height="100%",
        loc="upper right",
        bbox_to_anchor=(0.05, 0.05, 0.9, 0.05),
        bbox_transform=axs[1][2].transAxes,
        borderpad=0,
    )

    cbar12 = fig.colorbar(im12, cax=axins12, ticks=ticks1, orientation='horizontal')

    tick_values1 = np.array(ticks1) / 100

    tick_values1 = tick_values1.astype(np.int64)

    tick_values1 = [str(tick) for tick in tick_values1]

    tick_values1 = [tt if ind%2 == 0 else '' for (ind, tt) in enumerate(tick_values1)]

    cbar12.ax.set_xticklabels(tick_values1)

    cbar12.ax.xaxis.set_ticks_position('top')

    axins14 = inset_axes(
        axs[1][4],
        width="100%",
        height="100%",
        loc="upper right",
        bbox_to_anchor=(0.05, 0.05, 0.9, 0.05),
        bbox_transform=axs[1][4].transAxes,
        borderpad=0,
    )

    ticks2 = np.arange(-sv_max, sv_max + 5, 5)

    if ticks2.size >= 7:
        ticks2 = np.arange(-sv_max, sv_max + 10, 10)

    cbar14 = fig.colorbar(im14, cax=axins14, ticks=ticks2, orientation='horizontal')

    tick_values2 = np.array(ticks2)

    tick_values2 = tick_values2.astype(np.int64)

    tick_values2 = [str(tick) for tick in tick_values2]

    tick_values2 = [tt if ind % 2 == 0 else '' for (ind, tt) in enumerate(tick_values2)]

    cbar14.ax.set_xticklabels(tick_values2)

    cbar14.ax.xaxis.set_ticks_position('top')

    ticks = [0, 5, 10, 15, 20, 25]

    for i in range(2):
        for j in range(5):
            axs[i][j].set_xticks(ticks, [])
            axs[i][j].set_yticks(ticks, [])
            axs[i][j].xaxis.set_minor_locator(MultipleLocator(1))
            axs[i][j].yaxis.set_minor_locator(MultipleLocator(1))

    axs[0][0].set_yticks(ticks, ticks)
    axs[1][0].set_yticks(ticks, ticks)

    axs[1][0].set_xticks(ticks, ticks)
    axs[1][1].set_xticks(ticks, ticks)
    axs[1][2].set_xticks(ticks, ticks)
    axs[1][3].set_xticks(ticks, ticks)
    axs[1][4].set_xticks(ticks, ticks)

    axs[1][2].text(
        0.0, -0.25,
        'Scan direction [arcsec]',
        transform=axs[1][2].transAxes,
        color='black'
    )

    axs[1][0].text(
        -0.33, 0.6,
        'Slit direction [arcsec]',
        transform=axs[1][0].transAxes,
        color='black',
        rotation=90
    )

    axs[0][0].text(
        0.03, 0.9,
        '(a) Ca far-wing',
        transform=axs[0][0].transAxes,
        color='black'
    )

    axs[0][1].text(
        0.03, 0.9,
        r'(c) H$\alpha$ far-wing',
        transform=axs[0][1].transAxes,
        color='black'
    )

    axs[0][2].text(
        0.03, 0.9,
        r'(e) HMI continuum',
        transform=axs[0][2].transAxes,
        color='black'
    )

    axs[1][0].text(
        0.03, 0.9,
        '(b) Ca core',
        transform=axs[1][0].transAxes,
        color='black'
    )

    axs[1][1].text(
        0.03, 0.9,
        r'(d) H$\alpha$ core',
        transform=axs[1][1].transAxes,
        color='black'
    )

    axs[1][2].text(
        0.03, 0.9,
        r'(f) HMI $B_{\mathrm{LOS}}$',
        transform=axs[1][2].transAxes,
        color='black'
    )

    axs[1][2].text(
        0.18, 0.8,
        r'[x 100 G]',
        transform=axs[1][2].transAxes,
        color='black'
    )

    axs[0][4].text(
        0.03, 0.9,
        r'(i) Stokes $V$ (H$\alpha$) [%]',
        transform=axs[0][4].transAxes,
        color='black'
    )

    axs[0][4].text(
        0.1, 0.8,
        r'$-0.56\pm0.06\,\mathrm{\AA}$',
        transform=axs[0][4].transAxes,
        color='black'
    )

    axs[1][4].text(
        0.03, 0.9,
        r'(j) Stokes $V$ (H$\alpha$) [%]',
        transform=axs[1][4].transAxes,
        color='black'
    )

    axs[1][4].text(
        0.1, 0.8,
        r'$+0.45\pm0.06\,\mathrm{\AA}$',
        transform=axs[1][4].transAxes,
        color='black'
    )

    axs[0][3].text(
        0.03, 0.9,
        r'(g) Stokes $V$ (Ca II) [%]',
        transform=axs[0][3].transAxes,
        color='black'
    )

    axs[0][3].text(
        0.1, 0.8,
        r'$-0.44\pm0.06\,\mathrm{\AA}$',
        transform=axs[0][3].transAxes,
        color='black'
    )

    axs[1][3].text(
        0.03, 0.9,
        r'(h) Stokes $V$ (Ca II) [%]',
        transform=axs[1][3].transAxes,
        color='black'
    )

    axs[1][3].text(
        0.1, 0.8,
        r'$+0.28\pm0.06\,\mathrm{\AA}$',
        transform=axs[1][3].transAxes,
        color='black'
    )

    plt.subplots_adjust(left=0.07, right=1, bottom=0.13, top=1, wspace=0.0, hspace=0.0)

    plt.savefig(
        write_path / 'AO_FOV_{}_{}.pdf'.format(
            datestring, timestring
        ),
        format='pdf',
        dpi=300
    )

    plt.close('all')

    plt.clf()

    plt.cla()

    f.close()


def make_profile_plots(datestring, timestring, points):
    write_path = Path('/home/harsh/CourseworkRepo/KTTAnalysis/figures')

    base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes')

    datepath = base_path / datestring

    level4path = datepath / 'Level-4'

    f = h5py.File(
        level4path / 'aligned_Ca_Ha_stic_profiles_{}_{}.nc_pca.nc_spatial_straylight_corrected.nc'.format(
            datestring, timestring
        ),
        'r'
    )

    ind = np.where(f['profiles'][0, 0, 0, :, 0] != 0)[0]

    font = {'size': 8}

    matplotlib.rc('font', **font)

    fig, axs = plt.subplots(2, 4, figsize=(7, 3))

    color = ['brown', 'navy']

    axs[0][0].set_xticklabels([])
    axs[0][1].set_xticklabels([])
    axs[0][2].set_xticklabels([])
    axs[0][3].set_xticklabels([])
    axs[0][0].xaxis.set_minor_locator(MultipleLocator(.25))
    axs[0][1].xaxis.set_minor_locator(MultipleLocator(.25))
    axs[0][2].xaxis.set_minor_locator(MultipleLocator(.25))
    axs[0][3].xaxis.set_minor_locator(MultipleLocator(.25))
    axs[1][0].xaxis.set_minor_locator(MultipleLocator(.25))
    axs[1][1].xaxis.set_minor_locator(MultipleLocator(.25))
    axs[1][2].xaxis.set_minor_locator(MultipleLocator(.25))
    axs[1][3].xaxis.set_minor_locator(MultipleLocator(.25))
    axs[0][0].yaxis.set_minor_locator(MultipleLocator(.01))
    axs[0][1].yaxis.set_minor_locator(MultipleLocator(.5))
    axs[0][2].yaxis.set_minor_locator(MultipleLocator(.01))
    axs[0][3].yaxis.set_minor_locator(MultipleLocator(.5))
    axs[1][0].yaxis.set_minor_locator(MultipleLocator(.1))
    axs[1][1].yaxis.set_minor_locator(MultipleLocator(.5))
    axs[1][2].yaxis.set_minor_locator(MultipleLocator(.1))
    axs[1][3].yaxis.set_minor_locator(MultipleLocator(.5))

    for indice, point in enumerate(points):

        axs[indice][0].plot(f['wav'][ind[0:800]], f['profiles'][0, point[1], point[0], ind[0:800], 0], color=color[indice])

        svh = f['profiles'][0, point[1], point[0], ind[0:800], 3] * 100/ f['profiles'][0, point[1], point[0], ind[0:800], 0]

        axs[indice][1].plot(f['wav'][ind[0:800]], svh, color=color[indice])
        axs[indice][2].plot(f['wav'][ind[800:]], f['profiles'][0, point[1], point[0], ind[800:], 0], color=color[indice])

        svca = f['profiles'][0, point[1], point[0], ind[800:], 3] * 100 / f['profiles'][0, point[1], point[0], ind[800:], 0]
        axs[indice][3].plot(f['wav'][ind[800:]], svca, color=color[indice])

        axs[indice][0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs[indice][2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs[indice][1].yaxis.set_major_formatter(FormatStrFormatter('%d'))
        axs[indice][3].yaxis.set_major_formatter(FormatStrFormatter('%d'))

        max_svh = np.amax(np.abs(svh)) * 1.1
        max_svca = np.amax(np.abs(svca)) * 1.1

        axs[indice][1].set_ylim(-max_svh, max_svh)
        axs[indice][3].set_ylim(-max_svca, max_svca)


    axs[1][2].text(
        -0.4, -0.4,
        r'Wavelength [$\mathrm{\AA}$]',
        transform=axs[1][2].transAxes,
        color='black'
    )

    axs[0][0].text(
        0.3, 1.1,
        r'Stokes $I$',
        transform=axs[0][0].transAxes,
        color='black'
    )
    axs[0][1].text(
        0.3, 1.1,
        r'Stokes $V/I$',
        transform=axs[0][1].transAxes,
        color='black'
    )
    axs[0][2].text(
        0.3, 1.1,
        r'Stokes $I$',
        transform=axs[0][2].transAxes,
        color='black'
    )
    axs[0][3].text(
        0.3, 1.1,
        r'Stokes $V/I$',
        transform=axs[0][3].transAxes,
        color='black'
    )
    # axs[0][0].tick_params(axis='x', which='minor', bottom=False)
    # axs[0][0].tick_params(axis='x', which='minor', bottom=False)
    # axs[0][0].tick_params(axis='x', which='minor', bottom=False)
    # axs[0][0].tick_params(axis='x', which='minor', bottom=False)
    # axs[0][0].tick_params(axis='x', which='minor', bottom=False)
    # axs[0][0].tick_params(axis='x', which='minor', bottom=False)
    # axs[0][0].tick_params(axis='x', which='minor', bottom=False)
    # axs[0][0].tick_params(axis='x', which='minor', bottom=False)

    plt.subplots_adjust(left=0.05, right=1, bottom=0.17, top=0.9, wspace=0.3, hspace=0.13)

    plt.savefig(write_path / 'Profile_plots_{}_{}.pdf'.format(datestring, timestring))

    # plt.show()


def make_halpha_magnetic_field_plots(datestring, timestring, x1, y1, x2, y2, ticks, limit, points):

    # write_path = Path('/home/harsh/CourseworkRepo/KTTAnalysis/figures')

    write_path = Path('F:\\Harsh\\CourseworkRepo\\KTTAnalysis\\figures')

    # base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes')

    base_path = Path('C:\\Work Things\\InstrumentalUncorrectedStokes')

    datepath = base_path / datestring

    level4path = datepath / 'Level-4'

    mag_ha_full_line, _ = sunpy.io.read_file(
        level4path / 'aligned_Ca_Ha_stic_profiles_{}_{}.nc_pca.nc_spatial_straylight_corrected.nc_mag_ha_full_line.fits'.format(
            datestring, timestring
        )
    )[0]

    mag_ha_core, _ = sunpy.io.read_file(
        level4path / 'aligned_Ca_Ha_stic_profiles_{}_{}.nc_pca.nc_spatial_straylight_corrected.nc_mag_ha_core.fits'.format(
            datestring, timestring
        )
    )[0]

    extent = [0, 0.6 * 50, 0, 0.6 * 50]

    colors = ["blue", "green", "white", "darkgoldenrod", "darkred"]

    colors.reverse()

    cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)

    font = {'size': 6}

    matplotlib.rc('font', **font)

    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.75))

    im0 = axs[0].imshow(mag_ha_full_line[y1:y2, x1:x2], cmap=cmap1, origin='lower', vmin=-limit, vmax=limit, extent=extent)

    axs[1].imshow(mag_ha_core[y1:y2, x1:x2], cmap=cmap1, origin='lower', vmin=-limit, vmax=limit, extent=extent)

    axins0 = inset_axes(
        axs[0],
        width="100%",
        height="100%",
        loc="upper right",
        bbox_to_anchor=(0.05, 0.05, 0.9, 0.05),
        bbox_transform=axs[0].transAxes,
        borderpad=0,
    )

    cbar = fig.colorbar(im0, cax=axins0, ticks=ticks, orientation='horizontal')

    tick_values = np.array(ticks) / 100

    tick_values = tick_values.astype(np.int64)

    tick_values = [str(tick) for tick in tick_values]

    cbar.ax.set_xticklabels(tick_values)

    cbar.ax.xaxis.set_ticks_position('top')

    pcolors = ['brown', 'navy']
    for indice, point in enumerate(points):
        for i in range(2):
                axs[i].scatter((point[0] - x1) * 0.6, (point[1] - y1) * 0.6, marker='x', s=16, color=pcolors[indice])

    ticks = [0, 5, 10, 15, 20, 25]

    for i in range(2):
        axs[i].set_xticks(ticks, [])
        axs[i].set_yticks(ticks, [])
        axs[i].xaxis.set_minor_locator(MultipleLocator(1))
        axs[i].yaxis.set_minor_locator(MultipleLocator(1))

    axs[0].set_yticks(ticks, ticks)
    axs[1].set_yticks(ticks, [])

    axs[0].set_xticks(ticks, ticks)
    axs[1].set_xticks(ticks, ticks)

    axs[1].text(
        -0.4, -0.25,
        'Scan direction [arcsec]',
        transform=axs[1].transAxes,
        color='black'
    )

    axs[0].text(
        -0.3, 0.1,
        'Slit direction [arcsec]',
        transform=axs[0].transAxes,
        color='black',
        rotation=90
    )

    axs[0].text(
        0.05, 0.9,
        r'(a) H$\alpha\pm$1.5$\mathrm{\AA}$',
        transform=axs[0].transAxes,
        color='black'
    )

    axs[1].text(
        0.05, 0.9,
        r'(b) H$\alpha\pm$0.35$\mathrm{\AA}$',
        transform=axs[1].transAxes,
        color='black'
    )

    # axs[0][0].text(
    #     0.05, 0.92,
    #     '(a) Ca far-wing',
    #     transform=axs[0][0].transAxes,
    #     color='black'
    # )

    plt.subplots_adjust(left=0.23, right=1, bottom=0.23, top=1, wspace=0.0, hspace=0.0)

    plt.savefig(
        write_path / 'Halpha_magnetic_field_{}_{}.pdf'.format(
            datestring, timestring
        ),
        format='pdf',
        dpi=300
    )


def plot_magnetic_fields_scatter_plots(datestring, timestring, x1, y1, x2, y2, ticks):

    write_path = Path('F:\\Harsh\\CourseworkRepo\\KTTAnalysis\\figures')

    # base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes')

    base_path = Path('C:\\Work Things\\InstrumentalUncorrectedStokes')

    datepath = base_path / datestring

    level4path = datepath / 'Level-4'

    mag_ha_full_line, _ = sunpy.io.read_file(
        level4path / 'aligned_Ca_Ha_stic_profiles_{}_{}.nc_pca.nc_spatial_straylight_corrected.nc_mag_ha_full_line.fits'.format(
            datestring, timestring
        )
    )[0]

    mag_ha_core, _ = sunpy.io.read_file(
        level4path / 'aligned_Ca_Ha_stic_profiles_{}_{}.nc_pca.nc_spatial_straylight_corrected.nc_mag_ha_core.fits'.format(
            datestring, timestring
        )
    )[0]

    fe_mag, _ = sunpy.io.read_file(
        level4path / 'aligned_Ca_Ha_stic_profiles_{}_{}.nc_pca.nc_spatial_straylight_corrected.nc_mag_ca_fe.fits'.format(
            datestring, timestring
        )
    )[0]

    ca_mag, _ = sunpy.io.read_file(
        level4path / 'aligned_Ca_Ha_stic_profiles_{}_{}.nc_pca.nc_spatial_straylight_corrected.nc_mag_ca_core.fits'.format(
            datestring, timestring
        )
    )[0]

    hmi_mag, _ = sunpy.io.read_file(
        level4path / 'HMI_reference_magnetogram_{}_{}.fits'.format(
            datestring, timestring
        )
    )[0]

    # plt.close('all')
    # plt.clf()
    # plt.cla()

    font = {'size': 12}

    matplotlib.rc('font', **font)

    fig, axs = plt.subplots(2, 2, figsize=(7, 7))

    hmi_mag /= 100
    fe_mag /= 100
    mag_ha_full_line /= 100
    ca_mag /= 100
    mag_ha_core /= 100

    axs[0][0].scatter(mag_ha_full_line[y1:y2, x1:x2], mag_ha_core[y1:y2, x1:x2], s=1, color='royalblue')
    axs[0][1].scatter(fe_mag[y1:y2, x1:x2], mag_ha_full_line[y1:y2, x1:x2], s=1, color='royalblue')
    axs[1][0].scatter(hmi_mag[y1:y2, x1:x2], fe_mag[y1:y2, x1:x2], s=1, color='royalblue')
    axs[1][1].scatter(ca_mag[y1:y2, x1:x2], mag_ha_core[y1:y2, x1:x2], s=1, color='royalblue')

    minn, maxx = np.amin(
        np.array(
            [
                mag_ha_full_line[y1:y2, x1:x2].min(),
                mag_ha_core[y1:y2, x1:x2].min()
            ]
        )
    ), np.amax(
        np.array(
            [
                mag_ha_full_line[y1:y2, x1:x2].max(),
                mag_ha_core[y1:y2, x1:x2].max()
            ]
        )
    )

    a, b = np.polyfit(mag_ha_full_line[y1:y2, x1:x2].flatten(), mag_ha_core[y1:y2, x1:x2].flatten(), 1)

    xxx = np.arange(minn, maxx, 0.5)

    y = a * xxx + b

    axs[0][0].plot(xxx, y, color='black', linestyle='--')

    axs[0][0].plot(xxx, xxx, color='darkorange', linestyle='--')

    axs[0][0].text(
        0.05, 0.9,
        r'(c)',
        transform=axs[0][0].transAxes
    )

    axs[0][0].text(
        0.05, 0.8,
        r'$m$ = {}'.format(np.round(a, 2)),
        transform=axs[0][0].transAxes
    )

    axs[0][0].set_xlim(
        minn, maxx
    )
    axs[0][0].set_ylim(
        minn, maxx
    )

    minn, maxx = np.amin(
        np.array(
            [
                fe_mag[y1:y2, x1:x2].min(),
                mag_ha_full_line[y1:y2, x1:x2].min()
            ]
        )
    ), np.amax(
        np.array(
            [
                fe_mag[y1:y2, x1:x2].max(),
                mag_ha_full_line[y1:y2, x1:x2].max()
            ]
        )
    )

    a, b = np.polyfit(fe_mag[y1:y2, x1:x2].flatten(), mag_ha_full_line[y1:y2, x1:x2].flatten(), 1)

    xxx = np.arange(minn, maxx, 0.5)

    y = a * xxx + b

    axs[0][1].plot(xxx, y, color='black', linestyle='--')

    axs[0][1].plot(xxx, xxx, color='darkorange', linestyle='--')

    axs[0][1].text(
        0.05, 0.9,
        r'(d)',
        transform=axs[0][1].transAxes
    )

    axs[0][1].text(
        0.05, 0.8,
        r'$m$ = {}'.format(np.round(a, 2)),
        transform=axs[0][1].transAxes
    )

    axs[0][1].set_xlim(
        minn, maxx
    )
    axs[0][1].set_ylim(
        minn, maxx
    )

    minn, maxx = np.amin(
        np.array(
            [
                hmi_mag[y1:y2, x1:x2].min(),
                fe_mag[y1:y2, x1:x2].min()
            ]
        )
    ), np.amax(
        np.array(
            [
                hmi_mag[y1:y2, x1:x2].max(),
                fe_mag[y1:y2, x1:x2].max()
            ]
        )
    )
    a, b = np.polyfit(hmi_mag[y1:y2, x1:x2].flatten(), fe_mag[y1:y2, x1:x2].flatten(), 1)

    xxx = np.arange(minn, maxx, 0.5)

    y = a * xxx + b

    axs[1][0].plot(xxx, y, color='black', linestyle='--')
    axs[1][0].plot(xxx, xxx, color='darkorange', linestyle='--')

    axs[1][0].text(
        0.05, 0.9,
        r'(e)',
        transform=axs[1][0].transAxes
    )

    axs[1][0].text(
        0.05, 0.8,
        r'$m$ = {}'.format(np.round(a, 2)),
        transform=axs[1][0].transAxes
    )

    axs[1][0].set_xlim(
        minn, maxx
    )
    axs[1][0].set_ylim(
        minn, maxx
    )

    minn, maxx = np.amin(
        np.array(
            [
                ca_mag[y1:y2, x1:x2].min(),
                mag_ha_core[y1:y2, x1:x2].min()
            ]
        )
    ), np.amax(
        np.array(
            [
                ca_mag[y1:y2, x1:x2].max(),
                mag_ha_core[y1:y2, x1:x2].max()
            ]
        )
    )

    a, b = np.polyfit(ca_mag[y1:y2, x1:x2].flatten(), mag_ha_core[y1:y2, x1:x2].flatten(), 1)

    xxx = np.arange(minn, maxx, 0.5)

    y = a * xxx + b

    axs[1][1].plot(xxx, y, color='black', linestyle='--')

    axs[1][1].plot(xxx, xxx, color='darkorange', linestyle='--')

    axs[1][1].text(
        0.05, 0.9,
        r'(f)',
        transform=axs[1][1].transAxes
    )

    axs[1][1].text(
        0.05, 0.8,
        r'$m$ = {}'.format(np.round(a, 2)),
        transform=axs[1][1].transAxes
    )

    axs[1][1].set_xlim(
        minn, maxx
    )
    axs[1][1].set_ylim(
        minn, maxx
    )

    axs[0][0].set_xticks(ticks, ticks)
    axs[0][1].set_xticks(ticks, ticks)
    axs[1][0].set_xticks(ticks, ticks)
    axs[1][1].set_xticks(ticks, ticks)

    axs[0][0].set_yticks(ticks, ticks)
    axs[0][1].set_yticks(ticks, ticks)
    axs[1][0].set_yticks(ticks, ticks)
    axs[1][1].set_yticks(ticks, ticks)

    for i in range(2):
        for j in range(2):
            axs[i][j].xaxis.set_minor_locator(MultipleLocator(1))
            axs[i][j].yaxis.set_minor_locator(MultipleLocator(1))

    axs[0][0].text(
        -0.25, 0.1,
        r'$B_{\mathrm{LOS}}$ (H$\alpha\pm$0.35$\mathrm{\AA}$) [x 100 G]',
        transform=axs[0][0].transAxes,
        rotation=90,
        color='black'
    )

    axs[0][0].text(
        0.15, -0.2,
        r'$B_{\mathrm{LOS}}$ (H$\alpha\pm$1.5$\mathrm{\AA}$) [x 100 G]',
        transform=axs[0][0].transAxes,
        color='black'
    )

    axs[0][1].text(
        -0.25, 0.1,
        r'$B_{\mathrm{LOS}}$ (H$\alpha\pm$1.5$\mathrm{\AA}$) [x 100 G]',
        transform=axs[0][1].transAxes,
        rotation=90,
        color='black'
    )

    axs[0][1].text(
        0.0, -0.2,
        r'$B_{\mathrm{LOS}}$ (Fe I 8661.8991 $\mathrm{\AA}$) [x 100 G]',
        transform=axs[0][1].transAxes,
        color='black'
    )

    axs[1][0].text(
        -0.25, 0.02,
        r'$B_{\mathrm{LOS}}$ (Fe I 8661.8991 $\mathrm{\AA}$) [x 100 G]',
        transform=axs[1][0].transAxes,
        rotation=90,
        color='black'
    )

    axs[1][0].text(
        0.25, -0.2,
        r'$B_{\mathrm{LOS}}$ (HMI) [x 100 G]',
        transform=axs[1][0].transAxes,
        color='black'
    )

    axs[1][1].text(
        -0.25, 0.1,
        r'$B_{\mathrm{LOS}}$ (H$\alpha\pm$0.35$\mathrm{\AA}$) [x 100 G]',
        transform=axs[1][1].transAxes,
        rotation=90,
        color='black'
    )

    axs[1][1].text(
        0.05, -0.2,
        r'$B_{\mathrm{LOS}}$ (Ca II 8662 $\mathrm{\AA}$) [x 100 G]',
        transform=axs[1][1].transAxes,
        color='black'
    )

    plt.subplots_adjust(left=0.1, right=0.99, bottom=0.1, top=0.99, wspace=0.3, hspace=0.3)

    plt.savefig(
        write_path / 'Halpha_magnetic_field_scatter_{}_{}.pdf'.format(
            datestring, timestring
        ),
        format='pdf',
        dpi=300
    )

    # plt.show()


if __name__ == '__main__':
    # calculate_magnetic_field('20230603', bin_factor=16)

    # datestring='20230603'
    # timestring='073616'
    # y1 = 14
    # y2 = 64
    # x1 = 4
    # x2 = 54
    # ticks = [-1500, -1000, -500, 0, 500, 1000, 1500]
    # limit=1900
    # points = [
    #     [20, 45],
    #     [46, 57]
    # ]
    # create_fov_plots(
    #     datestring=datestring,
    #     timestring=timestring,
    #     x1=x1,
    #     y1=y1,
    #     x2=x2,
    #     y2=y2,
    #     ticks=ticks,
    #     limit=limit,
    #     points=points
    # )

    # datestring = '20230601'
    # timestring = '081014'
    # y1 = 14
    # y2 = 64
    # x1 = 3
    # x2 = 53
    # ticks = [-1000, -500, 0, 500, 1000]
    # limit = 1500
    # points = [
    #     [20, 45],
    #     [10, 37]
    # ]
    # create_fov_plots(
    #     datestring=datestring,
    #     timestring=timestring,
    #     x1=x1,
    #     y1=y1,
    #     x2=x2,
    #     y2=y2,
    #     ticks=ticks,
    #     limit=limit,
    #     points=points
    # )

    # datestring = '20230527'
    # timestring = '074428'
    # y1 = 15
    # y2 = 65
    # x1 = 3
    # x2 = 53
    # ticks = [-2000, -1000, 0, 1000, 2000]
    # limit = 2500
    # points = [
    #     [10, 47],
    #     [41, 43]
    # ]
    # create_fov_plots(
    #     datestring=datestring,
    #     timestring=timestring,
    #     x1=x1,
    #     y1=y1,
    #     x2=x2,
    #     y2=y2,
    #     ticks=ticks,
    #     limit=limit,
    #     points=points
    # )

    # datestring = '20230603'
    # timestring = '073616'
    # points = [
    #     [20, 45],
    #     [46, 57]
    # ]
    # make_profile_plots(datestring, timestring, points)

    # datestring = '20230601'
    # timestring = '081014'
    # points = [
    #     [20, 45],
    #     [10, 37]
    # ]
    # make_profile_plots(datestring, timestring, points)

    # datestring = '20230527'
    # timestring = '074428'
    # points = [
    #     [10, 47],
    #     [41, 43]
    # ]
    # make_profile_plots(datestring, timestring, points)

    # datestring='20230603'
    # timestring='073616'
    # y1 = 14
    # y2 = 64
    # x1 = 4
    # x2 = 54
    # ticks = [-1500, -1000, -500, 0, 500, 1000, 1500]
    # limit=1700
    # points = [
    #     [20, 45],
    #     [46, 57]
    # ]
    # make_halpha_magnetic_field_plots(
    #     datestring=datestring,
    #     timestring=timestring,
    #     x1=x1,
    #     y1=y1,
    #     x2=x2,
    #     y2=y2,
    #     ticks=ticks,
    #     limit=limit,
    #     points=points
    # )
    #
    # datestring = '20230601'
    # timestring = '081014'
    # y1 = 14
    # y2 = 64
    # x1 = 3
    # x2 = 53
    # ticks = [-1000, -500, 0, 500, 1000]
    # limit = 1300
    # points = [
    #     [20, 45],
    #     [10, 37]
    # ]
    # make_halpha_magnetic_field_plots(
    #     datestring=datestring,
    #     timestring=timestring,
    #     x1=x1,
    #     y1=y1,
    #     x2=x2,
    #     y2=y2,
    #     ticks=ticks,
    #     limit=limit,
    #     points=points
    # )
    #
    # datestring = '20230527'
    # timestring = '074428'
    # y1 = 15
    # y2 = 65
    # x1 = 3
    # x2 = 53
    # ticks = [-1000, -500, 0, 500, 1000, 2000]
    # limit = 1200
    # points = [
    #     [10, 47],
    #     [41, 43]
    # ]
    # make_halpha_magnetic_field_plots(
    #     datestring=datestring,
    #     timestring=timestring,
    #     x1=x1,
    #     y1=y1,
    #     x2=x2,
    #     y2=y2,
    #     ticks=ticks,
    #     limit=limit,
    #     points=points
    # )

    # datestring = '20230603'
    # timestring = '073616'
    # y1 = 14
    # y2 = 64
    # x1 = 4
    # x2 = 54
    # ticks = [-20, -15, -10, -5, 0, 5]
    # plot_magnetic_fields_scatter_plots(datestring, timestring, x1, y1, x2, y2, ticks)
    #
    # datestring = '20230601'
    # timestring = '081014'
    # y1 = 14
    # y2 = 64
    # x1 = 3
    # x2 = 53
    # ticks = [-20, -15, -10, -5, 0, 5]
    # plot_magnetic_fields_scatter_plots(datestring, timestring, x1, y1, x2, y2, ticks)
    #
    # datestring = '20230527'
    # timestring = '074428'
    # y1 = 15
    # y2 = 65
    # x1 = 3
    # x2 = 53
    # ticks = [-20, -15, -10, -5, 0, 5]
    # plot_magnetic_fields_scatter_plots(datestring, timestring, x1, y1, x2, y2, ticks)

    datestring = '20230603'
    timestring = '073616'
    y1 = 14
    y2 = 64
    x1 = 4
    x2 = 54
    ticks1 = [-1500, -1000, -500, 0, 500, 1000, 1500]
    # ticks2 = [-15, -10, -5, 0, 5, 10, 15]
    sv_max = 30
    limit = 2000
    points = [
        [20, 45],
        [46, 57]
    ]
    create_fov_plots_adaptive_optics(
        datestring=datestring,
        timestring=timestring,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        ticks1=ticks1,
        sv_max=sv_max,
        limit=limit,
        points=points
    )

    datestring = '20230601'
    timestring = '081014'
    y1 = 14
    y2 = 64
    x1 = 3
    x2 = 53
    ticks1 = [-1500, -1000, -500, 0, 500, 1000, 1500]
    # ticks2 = [-15, -10, -5, 0, 5, 10, 15]
    sv_max = 30
    limit = 2000
    points = [
        [20, 45],
        [46, 57]
    ]
    create_fov_plots_adaptive_optics(
        datestring=datestring,
        timestring=timestring,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        ticks1=ticks1,
        sv_max=sv_max,
        limit=limit,
        points=points
    )

    datestring = '20230527'
    timestring = '074428'
    y1 = 15
    y2 = 65
    x1 = 3
    x2 = 53
    ticks1 = [-1500, -1000, -500, 0, 500, 1000, 1500]
    # ticks2 = [-15, -10, -5, 0, 5, 10, 15]
    sv_max = 40
    limit = 2000
    points = [
        [20, 45],
        [46, 57]
    ]
    create_fov_plots_adaptive_optics(
        datestring=datestring,
        timestring=timestring,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        ticks1=ticks1,
        sv_max=sv_max,
        limit=limit,
        points=points
    )