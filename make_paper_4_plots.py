import sys

import numpy as np
import sunpy.io
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
from weak_field_approx import *
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import pearsonr


def add_subplot_axes(ax,rect,axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    #subax = fig.add_axes([x,y,width,height],facecolor=facecolor)  # matplotlib 2.0+
    subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_ti


def calculate_magnetic_field(datestring, errors=None, bin_factor=None):

    # base_path = Path('F:\\Harsh\\CourseworkRepo\\InstrumentalUncorrectedStokes')

    base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes')

    # base_path = Path('C:\\Work Things\\InstrumentalUncorrectedStokes')

    # base_path = Path('C:\\Work Things\\InstrumentalUncorrectedStokes')

    datepath = base_path / datestring

    level5path = datepath / 'Level-5-alt-alt'

    all_files = level5path.glob('**/*')

    all_mag_files = [level5path / 'aligned_Ca_Ha_stic_profiles_20230527_074428.nc_straylight_secondpass.nc']

    # for a_mag_file in all_mag_files:
    #
    #     print(a_mag_file.name)
    #
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
    #         sunpy.io.write_file(level5path / '{}_mag_ca_fe_errors.fits'.format(a_mag_file.name), magca, dict(), overwrite=True)
    #     else:
    #         sunpy.io.write_file(level5path / '{}_mag_ca_fe.fits'.format(a_mag_file.name), magca, dict(), overwrite=True)
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
    #         sunpy.io.write_file(level5path / '{}_mag_ca_core_errors.fits'.format(a_mag_file.name), magca, dict(),
    #                             overwrite=True)
    #     else:
    #         sunpy.io.write_file(level5path / '{}_mag_ca_core.fits'.format(a_mag_file.name), magca, dict(), overwrite=True)
    #
    #     sys.stdout.write('Ca core created\n')

    for a_mag_file in all_mag_files:

        print(a_mag_file.name)

        fcaha = h5py.File(a_mag_file, 'r')

        ind = np.where(fcaha['profiles'][0, 0, 0, :, 0] != 0)[0]

        ind = ind[0:800]
        ha_center_wave = 6562.8 / 10
        wave_range = 0.15 / 10

        transition_skip_list = np.array(
            [
                [6560.84, 0.25],
                [6561.09, 0.1],
                [6562.1, 0.25],
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
            sunpy.io.write_file(level5path / '{}_mag_ha_core_errors.fits'.format(a_mag_file.name), magha, dict(),
                                overwrite=True)
        else:
            sunpy.io.write_file(level5path / '{}_mag_ha_core.fits'.format(a_mag_file.name), magha, dict(), overwrite=True)

        sys.stdout.write('Ha core created\n')

        wave_range = 1.5 / 10

        actual_calculate_blos = prepare_calculate_blos(
            fcaha['profiles'][0][:, :, ind],
            fcaha['wav'][ind] / 10,
            ha_center_wave,
            ha_center_wave - wave_range,
            ha_center_wave,
            1.048,
            transition_skip_list=transition_skip_list,
            bin_factor=bin_factor,
            errors=errors
        )

        vec_actual_calculate_blos = np.vectorize(actual_calculate_blos)

        magfha = np.fromfunction(vec_actual_calculate_blos,
                                 shape=(fcaha['profiles'].shape[1], fcaha['profiles'].shape[2]))

        if errors is not None:
            sunpy.io.write_file(level5path / '{}_mag_ha_full_line_errors.fits'.format(a_mag_file.name), magfha, dict(),
                                overwrite=True)
        else:
            sunpy.io.write_file(level5path / '{}_mag_ha_full_line.fits'.format(a_mag_file.name), magfha, dict(),
                            overwrite=True)

        sys.stdout.write('Ha full line created\n')


def create_fov_plots(datestring, timestring, x1, y1, x2, y2, ticks, limit, points, pcolors):

    colors = ["blue", "green", "white", "darkgoldenrod", "darkred"]

    colors.reverse()

    cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)

    extent = [0, 0.6 * 50, 0, 0.6 * 50]

    write_path = Path('/home/harsh/CourseworkRepo/KTTAnalysis/figures')

    # write_path = Path('F:\\Harsh\\CourseworkRepo\\KTTAnalysis\\figures')

    base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes')

    # base_path = Path('C:\\Work Things\\InstrumentalUncorrectedStokes')

    datepath = base_path / datestring

    level5path = datepath / 'Level-5-alt-alt'

    hmipath = datepath / 'Level-4-alt-alt' / 'aligned_hmi'

    f = h5py.File(
        level5path / 'aligned_Ca_Ha_stic_profiles_20230527_074428.nc_straylight_secondpass.nc'.format(
            datestring, timestring
        ),
        'r'
    )

    hmi_img, _ = sunpy.io.read_file(
        hmipath / 'hmi.Ic_720s.20230527_023600_TAI.3.continuum.fits_20230527_074428.fits'
    )[0]

    hmi_mag, _ = sunpy.io.read_file(
        hmipath / 'hmi.M_720s.20230527_023600_TAI.3.magnetogram.fits_20230527_074428.fits'
    )[0]

    ind = np.where(f['profiles'][0, 0, 0, :, 0] != 0)[0]

    wave = f['wav'][ind]

    ind_halpha_core = ind[np.argmin(np.abs(wave - 6562.8))]

    ind_ca_core = ind[np.argmin(np.abs(wave - 8662.17))]

    font = {'size': 6}

    matplotlib.rc('font', **font)

    fig, axs = plt.subplots(2, 3, figsize=(3.5, 2.33))

    axs[0][0].imshow(f['profiles'][0, y1:y2, x1:x2, 3204 + 8, 0], cmap='gray', origin='lower', extent=extent)

    axs[1][0].imshow(f['profiles'][0, y1:y2, x1:x2, ind_ca_core, 0], cmap='gray', origin='lower', extent=extent)

    axs[0][1].imshow(f['profiles'][0, y1:y2, x1:x2, 32, 0], cmap='gray', origin='lower', extent=extent)

    axs[1][1].imshow(f['profiles'][0, y1:y2, x1:x2, ind_halpha_core, 0], cmap='gray', origin='lower', extent=extent)

    axs[0][2].imshow(hmi_img[y1:y2, x1:x2], cmap='gray', origin='lower', extent=extent)

    sunspot_mask, _ = sunpy.io.read_file(level5path / 'sunspot_mask.fits')[0]

    lightbridge_mask, _ = sunpy.io.read_file(level5path / 'lightbridge_mask.fits')[0]

    emission_mask, _ = sunpy.io.read_file(level5path / 'emission_mask.fits')[0]

    X, Y = np.meshgrid(np.arange(50) * 0.6, np.arange(50) * 0.6)

    axs[0][0].contour(X, Y, sunspot_mask, levels=0, colors='black', linewidths=0.5)
    axs[0][1].contour(X, Y, sunspot_mask, levels=0, colors='black', linewidths=0.5)
    axs[0][2].contour(X, Y, sunspot_mask, levels=0, colors='black', linewidths=0.5)
    axs[1][0].contour(X, Y, sunspot_mask, levels=0, colors='black', linewidths=0.5)
    axs[1][1].contour(X, Y, sunspot_mask, levels=0, colors='black', linewidths=0.5)
    axs[1][2].contour(X, Y, sunspot_mask, levels=0, colors='black', linewidths=0.5)

    axs[0][0].contour(X, Y, lightbridge_mask, levels=0, colors='green', linewidths=0.5)
    axs[0][1].contour(X, Y, lightbridge_mask, levels=0, colors='green', linewidths=0.5)
    axs[0][2].contour(X, Y, lightbridge_mask, levels=0, colors='green', linewidths=0.5)
    axs[1][0].contour(X, Y, lightbridge_mask, levels=0, colors='green', linewidths=0.5)
    axs[1][1].contour(X, Y, lightbridge_mask, levels=0, colors='green', linewidths=0.5)
    axs[1][2].contour(X, Y, lightbridge_mask, levels=0, colors='green', linewidths=0.5)

    axs[0][0].contour(X, Y, emission_mask, levels=0, colors='cyan', linewidths=0.5)
    axs[0][1].contour(X, Y, emission_mask, levels=0, colors='cyan', linewidths=0.5)
    axs[0][2].contour(X, Y, emission_mask, levels=0, colors='cyan', linewidths=0.5)
    axs[1][0].contour(X, Y, emission_mask, levels=0, colors='cyan', linewidths=0.5)
    axs[1][1].contour(X, Y, emission_mask, levels=0, colors='cyan', linewidths=0.5)
    axs[1][2].contour(X, Y, emission_mask, levels=0, colors='cyan', linewidths=0.5)

    im12 = axs[1][2].imshow(hmi_mag[y1:y2, x1:x2], cmap=cmap1, origin='lower', vmin=-limit, vmax=limit, extent=extent)

    for indice, point in enumerate(points):
        for i in range(2):
            for j in range(3):
                axs[i][j].scatter((point[0]) * 0.6, (point[1]) * 0.6, marker='x', s=8, color=pcolors[indice])

    axins12 = inset_axes(
        axs[1][2],
        width="90%",
        height="70%",
        loc="upper left",
        bbox_to_anchor=(0.05, 0.01, 0.8, 0.05),
        bbox_transform=axs[1][2].transAxes,
        borderpad=0,
    )

    cbar = fig.colorbar(im12, cax=axins12, ticks=ticks, orientation='horizontal')

    tick_values = np.array(ticks) / 100

    tick_values = tick_values.astype(np.int64)

    tick_values = [str(tick) for tick in tick_values]

    cbar.ax.set_xticklabels(tick_values)

    cbar.ax.xaxis.set_ticks_position('top')

    ticks = [0, 5, 10, 15, 20, 25]

    for i in range(2):
        for j in range(3):
            axs[i][j].set_xticks(ticks, [])
            axs[i][j].set_yticks(ticks, [])
            axs[i][j].xaxis.set_minor_locator(MultipleLocator(1))
            axs[i][j].yaxis.set_minor_locator(MultipleLocator(1))

    axs[0][0].set_yticks(ticks, ticks)
    axs[1][0].set_yticks(ticks, ticks)

    axs[1][0].set_xticks(ticks, ticks)
    axs[1][1].set_xticks(ticks, ticks)
    axs[1][2].set_xticks(ticks, ticks)

    axs[1][1].text(
        0.03, -0.25,
        'Scan direction [arcsec]',
        transform=axs[1][1].transAxes,
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
        color='white'
    )

    axs[0][1].text(
        0.05, 0.92,
        r'(c) H$\alpha$ far-wing',
        transform=axs[0][1].transAxes,
        color='white'
    )

    axs[0][2].text(
        0.05, 0.92,
        r'(e) HMI continuum',
        transform=axs[0][2].transAxes,
        color='white'
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
        r'(f) HMI magnetogram',
        transform=axs[1][2].transAxes,
        color='white'
    )

    axs[1][2].text(
        0.05, 0.85,
        r'     [x 100 G]',
        transform=axs[1][2].transAxes,
        color='white'
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


def create_fov_plots_adaptive_optics(datestring, timestring, x1, y1, x2, y2, ticks1, sv_max, limit, points, plot_points=False):

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

    if plot_points is True:
        for indice, point in enumerate(points):
            for i in range(2):
                for j in range(5):
                    axs[i][j].scatter((point[0] - x1) * 0.6, (point[1] - y1) * 0.6, marker='x', s=16, color=pcolors[indice])

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

    plt.savefig(
        write_path / 'AO_FOV_{}_{}.png'.format(
            datestring, timestring
        ),
        format='png',
        dpi=300
    )

    plt.close('all')

    plt.clf()

    plt.cla()

    f.close()


def make_profile_plots(datestring, timestring, points, color, index, y1, x1):
    write_path = Path('/home/harsh/CourseworkRepo/KTTAnalysis/figures')

    base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes')

    datepath = base_path / datestring

    level5path = datepath / 'Level-5-alt-alt'

    f = h5py.File(
        level5path / 'aligned_Ca_Ha_stic_profiles_{}_{}.nc_straylight_secondpass.nc'.format(
            datestring, timestring
        ),
        'r'
    )

    fmca = h5py.File(
        level5path / 'straylight_8662.14_estimated_profile_20230527_timestring_074428.h5',
        'r'
    )

    fmha = h5py.File(
        level5path / 'straylight_6562.8_estimated_profile_20230527_timestring_074428.h5',
        'r'
    )

    mca = fmca['stray_corrected_median'][()]

    mha = fmha['stray_corrected_median'][()]

    ind = np.where(f['profiles'][0, 0, 0, :, 0] != 0)[0]

    font = {'size': 6}

    matplotlib.rc('font', **font)

    fig, axs = plt.subplots(2, 4, figsize=(3.5, 1.5))

    # color = ['brown', 'navy']

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
    # axs[0][0].yaxis.set_minor_locator(MultipleLocator(.1))
    axs[0][1].yaxis.set_minor_locator(MultipleLocator(.5))
    # axs[0][2].yaxis.set_minor_locator(MultipleLocator(.1))
    axs[0][3].yaxis.set_minor_locator(MultipleLocator(1))
    # axs[1][0].yaxis.set_minor_locator(MultipleLocator(.1))
    axs[1][1].yaxis.set_minor_locator(MultipleLocator(.5))
    # axs[1][2].yaxis.set_minor_locator(MultipleLocator(.1))
    axs[1][3].yaxis.set_minor_locator(MultipleLocator(1))

    for indice, point in enumerate(points):

        axs[indice][0].plot(f['wav'][ind[0:800]], f['profiles'][0, y1 + point[1], x1 + point[0], ind[0:800], 0], color=color[indice], linewidth=0.5)

        svh = f['profiles'][0, y1 + point[1], x1 + point[0], ind[0:800], 3] * 100 / f['profiles'][0, y1 + point[1], x1 + point[0], ind[0:800], 0]

        axs[indice][1].plot(f['wav'][ind[0:800]], svh, color=color[indice], linewidth=0.5)

        axs[indice][2].plot(f['wav'][ind[800:]], f['profiles'][0, y1 + point[1], x1 + point[0], ind[800:], 0], color=color[indice], linewidth=0.5)

        svca = f['profiles'][0, y1 + point[1], x1 + point[0], ind[800:], 3] * 100 / f['profiles'][0, y1 + point[1], x1 + point[0], ind[800:], 0]

        axs[indice][3].plot(f['wav'][ind[800:]], svca, color=color[indice], linewidth=0.5)

        axs[indice][0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axs[indice][2].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # axs[indice][1].yaxis.set_major_formatter(FormatStrFormatter('%d'))
        # axs[indice][3].yaxis.set_major_formatter(FormatStrFormatter('%d'))

        max_svh = np.amax(np.abs(svh)) * 1.1
        max_svca = np.amax(np.abs(svca)) * 1.1

        if max_svca > 10:
            axs[indice][3].yaxis.set_minor_locator(MultipleLocator(5))

        axs[indice][1].set_ylim(-max_svh, max_svh)
        axs[indice][3].set_ylim(-max_svca, max_svca)

        axs[indice][0].axvline(x=6562.8, color='gray', linestyle='--', linewidth=0.5)
        axs[indice][1].axvline(x=6562.8, color='gray', linestyle='--', linewidth=0.5)
        axs[indice][2].axvline(x=8662.14, color='gray', linestyle='--', linewidth=0.5)
        axs[indice][3].axvline(x=8662.14, color='gray', linestyle='--', linewidth=0.5)

        axs[indice][0].plot(f['wav'][ind[0:800]], mha, color='gray', linewidth=0.5)

        axs[indice][2].plot(f['wav'][ind[800:]], mca, color='gray', linewidth=0.5)

    if index > 0:
        axs[1][2].text(
            -0.7, -0.57,
            r'Wavelength [$\mathrm{\AA}$]',
            transform=axs[1][2].transAxes,
            color='black'
        )

    axs[0][0].text(
        0.2, 1.1,
        r'Stokes $I$',
        transform=axs[0][0].transAxes,
        color='black'
    )
    axs[0][1].text(
        0.0, 1.1,
        r'Stokes $V/I$ [%]',
        transform=axs[0][1].transAxes,
        color='black'
    )
    axs[0][2].text(
        0.2, 1.1,
        r'Stokes $I$',
        transform=axs[0][2].transAxes,
        color='black'
    )
    axs[0][3].text(
        0.0, 1.1,
        r'Stokes $V/I$ [%]',
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

    plt.subplots_adjust(left=0.065, right=1, bottom=0.2, top=0.9, wspace=0.5, hspace=0.13)

    plt.savefig(write_path / 'Profile_plots_{}_{}_{}.pdf'.format(datestring, timestring, index))

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


def create_atmospheric_param_plots(points, pcolors):

    extent = [0, 0.6 * 50, 0, 0.6 * 50]

    write_path = Path('/home/harsh/CourseworkRepo/KTTAnalysis/figures')

    base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes/20230527/Level-5-alt-alt/pca_kmeans_fulldata_inversions/')

    fa = h5py.File(base_path / 'combined_output_cycle_B_T_2_retry_all.nc', 'r')

    ltau = fa['ltau500'][0, 0, 0]

    ind_n_1 = np.argmin(np.abs(ltau + 1))

    ind_n_5 = np.argmin(np.abs(ltau + 5))

    ind_n_0 = np.argmin(np.abs(ltau))

    temp = fa['temp'][0]

    vlos = fa['vlos'][0]

    vturb = fa['vturb'][0]

    fa.close()

    fo = h5py.File('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes/20230527/Level-5-alt-alt/aligned_Ca_Ha_stic_profiles_20230527_074428.nc_straylight_secondpass_stic_file.nc', 'r')

    ind = np.where(fo['profiles'][0, 0, 0, :, 0] != 0)[0]

    a, b = np.unravel_index(
        np.argmin(
            fo['profiles'][0, :, :, ind[8], 0]
        ),
        fo['profiles'][0, :, :, ind[8], 0].shape
    )

    print(
        'shape: {}'.format(
            fo['profiles'][0, :, :, ind[8], 0].shape
        )
    )

    img = fo['profiles'][0, :, :, ind[8], 0]

    fo.close()

    k = 2

    mn = img.mean()

    sd = img.std()

    thres = mn - k * sd

    j, k = np.where(img < thres)

    print(
        'a: {}, b: {}'.format(
            a, b
        )
    )

    calib_velocity = np.mean(
        vlos[j, k][:, ind_n_1:ind_n_0]
    )

    print(
        'calib_velocity: {}'.format(
            calib_velocity
        )
    )

    font = {'size': 6}

    matplotlib.rc('font', **font)

    fig, axs = plt.subplots(2, 3, figsize=(3.5, 2.33))

    im00 = axs[0][0].imshow(temp[:, :, ind_n_1] / 1e3, cmap='hot', origin='lower', extent=extent, vmin=4, vmax=5.5)

    im10 = axs[1][0].imshow(temp[:, :, ind_n_5] / 1e3, cmap='hot', origin='lower', extent=extent, vmin=4, vmax=7)

    im01 = axs[0][1].imshow((vlos[:, :, ind_n_1] - calib_velocity) / 1e5, cmap='bwr', origin='lower', extent=extent, vmin=-5, vmax=5)

    im11 = axs[1][1].imshow((vlos[:, :, ind_n_5] - calib_velocity) / 1e5, cmap='bwr', origin='lower', extent=extent, vmin=-15, vmax=15)

    im02 = axs[0][2].imshow(vturb[:, :, ind_n_1] / 1e5, cmap='summer', origin='lower', extent=extent, vmin=0, vmax=5)

    im12 = axs[1][2].imshow(vturb[:, :, ind_n_5] / 1e5, cmap='summer', origin='lower', extent=extent, vmin=0, vmax=5)

    level5path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes/20230527/Level-5-alt-alt/')

    sunspot_mask, _ = sunpy.io.read_file(level5path / 'sunspot_mask.fits')[0]

    lightbridge_mask, _ = sunpy.io.read_file(level5path / 'lightbridge_mask.fits')[0]

    emission_mask, _ = sunpy.io.read_file(level5path / 'emission_mask.fits')[0]

    X, Y = np.meshgrid(np.arange(50) * 0.6, np.arange(50) * 0.6)

    axs[0][0].contour(X, Y, sunspot_mask, levels=0, colors='black', linewidths=0.5)
    axs[0][1].contour(X, Y, sunspot_mask, levels=0, colors='black', linewidths=0.5)
    axs[0][2].contour(X, Y, sunspot_mask, levels=0, colors='black', linewidths=0.5)
    axs[1][0].contour(X, Y, sunspot_mask, levels=0, colors='black', linewidths=0.5)
    axs[1][1].contour(X, Y, sunspot_mask, levels=0, colors='black', linewidths=0.5)
    axs[1][2].contour(X, Y, sunspot_mask, levels=0, colors='black', linewidths=0.5)

    axs[0][0].contour(X, Y, lightbridge_mask, levels=0, colors='green', linewidths=0.5)
    axs[0][1].contour(X, Y, lightbridge_mask, levels=0, colors='green', linewidths=0.5)
    axs[0][2].contour(X, Y, lightbridge_mask, levels=0, colors='green', linewidths=0.5)
    axs[1][0].contour(X, Y, lightbridge_mask, levels=0, colors='green', linewidths=0.5)
    axs[1][1].contour(X, Y, lightbridge_mask, levels=0, colors='green', linewidths=0.5)
    axs[1][2].contour(X, Y, lightbridge_mask, levels=0, colors='green', linewidths=0.5)

    axs[0][0].contour(X, Y, emission_mask, levels=0, colors='cyan', linewidths=0.5)
    axs[0][1].contour(X, Y, emission_mask, levels=0, colors='cyan', linewidths=0.5)
    axs[0][2].contour(X, Y, emission_mask, levels=0, colors='cyan', linewidths=0.5)
    axs[1][0].contour(X, Y, emission_mask, levels=0, colors='cyan', linewidths=0.5)
    axs[1][1].contour(X, Y, emission_mask, levels=0, colors='cyan', linewidths=0.5)
    axs[1][2].contour(X, Y, emission_mask, levels=0, colors='cyan', linewidths=0.5)

    for indice, point in enumerate(points):
        for i in range(2):
            for j in range(3):
                axs[i][j].scatter((point[0]) * 0.6, (point[1]) * 0.6, marker='x', s=8, color=pcolors[indice])

    axins00 = inset_axes(
        axs[0][0],
        width="80%",
        height="70%",
        loc="upper right",
        bbox_to_anchor=(0.005, 0.005, 0.7, 0.05),
        bbox_transform=axs[0][0].transAxes,
        borderpad=0,
    )

    cbar = fig.colorbar(im00, cax=axins00, ticks=[4, 5], orientation='horizontal')

    cbar.ax.xaxis.set_ticks_position('top')

    axins10 = inset_axes(
        axs[1][0],
        width="80%",
        height="70%",
        loc="upper right",
        bbox_to_anchor=(0.005, 0.005, 0.7, 0.05),
        bbox_transform=axs[1][0].transAxes,
        borderpad=0,
    )

    cbar = fig.colorbar(im10, cax=axins10, orientation='horizontal')

    cbar.ax.xaxis.set_ticks_position('top')

    axins01 = inset_axes(
        axs[0][1],
        width="80%",
        height="70%",
        loc="upper right",
        bbox_to_anchor=(0.005, 0.005, 0.7, 0.05),
        bbox_transform=axs[0][1].transAxes,
        borderpad=0,
    )

    cbar = fig.colorbar(im01, cax=axins01, orientation='horizontal')

    cbar.ax.xaxis.set_ticks_position('top')

    axins11 = inset_axes(
        axs[1][1],
        width="80%",
        height="70%",
        loc="upper right",
        bbox_to_anchor=(0.005, 0.005, 0.7, 0.05),
        bbox_transform=axs[1][1].transAxes,
        borderpad=0,
    )

    cbar = fig.colorbar(im11, cax=axins11, orientation='horizontal')

    cbar.ax.xaxis.set_ticks_position('top')

    axins02 = inset_axes(
        axs[0][2],
        width="80%",
        height="70%",
        loc="upper right",
        bbox_to_anchor=(0.005, 0.005, 0.7, 0.05),
        bbox_transform=axs[0][2].transAxes,
        borderpad=0,
    )

    cbar = fig.colorbar(im02, cax=axins02, orientation='horizontal')

    cbar.ax.xaxis.set_ticks_position('top')

    axins12 = inset_axes(
        axs[1][2],
        width="80%",
        height="70%",
        loc="upper right",
        bbox_to_anchor=(0.005, 0.005, 0.7, 0.05),
        bbox_transform=axs[1][2].transAxes,
        borderpad=0,
    )

    cbar = fig.colorbar(im12, cax=axins12, orientation='horizontal')

    cbar.ax.xaxis.set_ticks_position('top')

    cbar.ax.tick_params(colors='black')

    ticks = [0, 5, 10, 15, 20, 25]

    for i in range(2):
        for j in range(3):
            axs[i][j].set_xticks(ticks, [])
            axs[i][j].set_yticks(ticks, [])
            axs[i][j].xaxis.set_minor_locator(MultipleLocator(1))
            axs[i][j].yaxis.set_minor_locator(MultipleLocator(1))

    axs[0][0].set_yticks(ticks, ticks)
    axs[1][0].set_yticks(ticks, ticks)

    axs[1][0].set_xticks(ticks, ticks)
    axs[1][1].set_xticks(ticks, ticks)
    axs[1][2].set_xticks(ticks, ticks)

    axs[1][1].text(
        0.03, -0.25,
        'Scan direction [arcsec]',
        transform=axs[1][1].transAxes,
        color='black'
    )

    axs[1][0].text(
        -0.44, 0.6,
        'Slit direction [arcsec]',
        transform=axs[1][0].transAxes,
        color='black',
        rotation=90
    )

    axs[0][0].text(
        -0.33, 0.3,
        r'$\log\tau_{500}$ = $-$1',
        transform=axs[0][0].transAxes,
        color='black',
        rotation=90
    )

    axs[1][0].text(
        -0.33, 0.3,
        r'$\log\tau_{500}$ = $-$5',
        transform=axs[1][0].transAxes,
        color='black',
        rotation=90
    )

    axs[0][0].text(
        0.15, 0.9,
        '(a) $T$ [kK]',
        transform=axs[0][0].transAxes,
        color='black'
    )

    axs[0][1].text(
        0.15, 0.9,
        r'(c) $V_{LOS}$ [$\mathrm{km\;s^{-1}}$]',
        transform=axs[0][1].transAxes,
        color='black'
    )

    axs[0][2].text(
        0.15, 0.9,
        r'(e) $V_{turb}$ [$\mathrm{km\;s^{-1}}$]',
        transform=axs[0][2].transAxes,
        color='black'
    )

    axs[1][0].text(
        0.15, 0.9,
        '(b) $T$ [kK]',
        transform=axs[1][0].transAxes,
        color='black'
    )

    axs[1][1].text(
        0.15, 0.9,
        r'(d) $V_{LOS}$ [$\mathrm{km\;s^{-1}}$]',
        transform=axs[1][1].transAxes,
        color='black'
    )

    axs[1][2].text(
        0.15, 0.9,
        r'(f) $V_{turb}$ [$\mathrm{km\;s^{-1}}$]',
        transform=axs[1][2].transAxes,
        color='black'
    )

    plt.subplots_adjust(left=0.13, right=1, bottom=0.13, top=1, wspace=0.0, hspace=0.0)

    plt.savefig(
        write_path / 'Atmosphere.pdf',
        format='pdf',
        dpi=300
    )


def make_individual_inversion_plots(points, color, index):
    write_path = Path('/home/harsh/CourseworkRepo/KTTAnalysis/figures')

    base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes/20230527/Level-5-alt-alt/pca_kmeans_fulldata_inversions/')

    fa = h5py.File(base_path / 'combined_output_cycle_B_T_2_retry_all.nc', 'r')

    font = {'size': 6}

    matplotlib.rc('font', **font)

    fig, axs = plt.subplots(2, 4, figsize=(3.5, 1.5))

    # color = ['brown', 'navy']

    axs[0][0].set_xticklabels([])
    axs[0][1].set_xticklabels([])
    axs[0][2].set_xticklabels([])
    axs[0][3].set_xticklabels([])
    axs[0][0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0][1].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0][2].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0][3].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1][0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1][1].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1][2].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1][3].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0][0].yaxis.set_minor_locator(MultipleLocator(1))
    axs[0][1].yaxis.set_minor_locator(MultipleLocator(1))
    axs[0][2].yaxis.set_minor_locator(MultipleLocator(1))
    # axs[0][3].yaxis.set_minor_locator(MultipleLocator(1))
    axs[1][0].yaxis.set_minor_locator(MultipleLocator(1))
    axs[1][1].yaxis.set_minor_locator(MultipleLocator(1))
    axs[1][2].yaxis.set_minor_locator(MultipleLocator(1))
    # axs[1][3].yaxis.set_minor_locator(MultipleLocator(1))

    calib_velocity = 324619.99385413347

    for indice, point in enumerate(points):

        axs[indice][0].plot(fa['ltau500'][0, point[1], point[0]], fa['temp'][0, point[1], point[0]] / 1e3, color=color[indice], linewidth=0.5)

        axs[indice][1].plot(fa['ltau500'][0, point[1], point[0]], (fa['vlos'][0, point[1], point[0]] - calib_velocity) / 1e5, color=color[indice], linewidth=0.5)

        axs[indice][2].plot(fa['ltau500'][0, point[1], point[0]], fa['vturb'][0, point[1], point[0]] / 1e5, color=color[indice], linewidth=0.5)

        axs[indice][3].plot(
            fa['ltau500'][0, point[1], point[0]],
            fa['blong'][0, point[1], point[0]] / 1e2,
            color=color[indice],
            linewidth=0.5
        )

        axs[indice][3].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        axs[indice][0].set_xlim([-5.3, -0.3])
        axs[indice][1].set_xlim([-5.3, -0.3])
        axs[indice][2].set_xlim([-5.3, -0.3])
        axs[indice][3].set_xlim([-5.3, -0.3])

        axs[indice][0].set_xticks([-1, -3, -5], [-1, -3, -5])
        axs[indice][1].set_xticks([-1, -3, -5], [-1, -3, -5])
        axs[indice][2].set_xticks([-1, -3, -5], [-1, -3, -5])
        axs[indice][3].set_xticks([-1, -3, -5], [-1, -3, -5])

        if index == 2:
            axs[1][1].set_yticks([0, -2, -4], [0, -2, -4])

        if index == 1:
            axs[1][3].set_yticks([-15, -16, -17], ['', '-16', ''])

        axs[0][0].set_yticks([3, 4, 5, 6, 7], ['', '4', '', '6', ''])
        axs[1][0].set_yticks([3, 4, 5, 6, 7], ['', '4', '', '6', ''])

        axs[0][0].set_ylim(2.9, 7.1)
        axs[1][0].set_ylim(2.9, 7.1)

    if index > 0:
        axs[1][2].text(
            -0.7, -0.57,
            r'$\log\tau_{500}$',
            transform=axs[1][2].transAxes,
            color='black'
        )

    axs[0][0].text(
        0.2, 1.1,
        r'T [kK]',
        transform=axs[0][0].transAxes,
        color='black'
    )
    axs[0][1].text(
        0.0, 1.1,
        r'$V_{LOS}$ [$\mathrm{km\;s^{-1}}$]',
        transform=axs[0][1].transAxes,
        color='black'
    )
    axs[0][2].text(
        0.00, 1.1,
        r'$V_{turb}$ [$\mathrm{km\;s^{-1}}$]',
        transform=axs[0][2].transAxes,
        color='black'
    )
    axs[0][3].text(
        -0.03, 1.1,
        r'$B_{LOS}$ [x 100 G]',
        transform=axs[0][3].transAxes,
        color='black'
    )

    plt.subplots_adjust(left=0.05, right=1, bottom=0.2, top=0.9, wspace=0.6, hspace=0.13)

    plt.savefig(write_path / 'Individual_inversion_plots_{}.pdf'.format( index))


def create_magnetic_field_plots(points, pcolors, l1x, l1y, l2x, l2y, line_colors):

    colors = ["blue", "green", "white", "darkgoldenrod", "darkred"]

    colors.reverse()

    cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)

    y1 = 15
    y2 = 65
    x1 = 3
    x2 = 53

    extent = [0, 0.6 * 50, 0, 0.6 * 50]

    write_path = Path('/home/harsh/CourseworkRepo/KTTAnalysis/figures')

    inversion_base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes/20230527/Level-5-alt-alt/pca_kmeans_fulldata_inversions/')

    wfa_base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes/20230527/Level-5-alt-alt/')

    fa = h5py.File(inversion_base_path / 'combined_output_cycle_B_T_2_retry_all.nc', 'r')

    mag_full_line, _ = sunpy.io.read_file(wfa_base_path / 'aligned_Ca_Ha_stic_profiles_20230527_074428.nc_straylight_secondpass.nc_magha_full_line_spatial.fits')[0]

    mag_ha_core, _ = sunpy.io.read_file(wfa_base_path / 'aligned_Ca_Ha_stic_profiles_20230527_074428.nc_straylight_secondpass.nc_magha_core_spatial.fits')[0]

    mag_full_line = mag_full_line[y1:y2, x1:x2]

    mag_ha_core = mag_ha_core[y1:y2, x1:x2]

    ltau = fa['ltau500'][0, 0, 0]

    ind_n_1 = np.argmin(np.abs(ltau + 1))

    ind_n_5 = np.argmin(np.abs(ltau + 4.5))

    blong = fa['blong'][0]

    font = {'size': 6}

    matplotlib.rc('font', **font)

    fig, axs = plt.subplots(2, 3, figsize=(3.5, 2.33))

    im00 = axs[0][0].imshow(blong[:, :, ind_n_1] / 1e2, cmap=cmap1, origin='lower', extent=extent, vmin=-19, vmax=19)

    im10 = axs[1][0].imshow(blong[:, :, ind_n_5] / 1e2, cmap=cmap1, origin='lower', extent=extent, vmin=-19, vmax=19)

    im01 = axs[0][1].imshow(mag_full_line / 1e2, cmap=cmap1, origin='lower', extent=extent, vmin=-19, vmax=19)

    im11 = axs[1][1].imshow(mag_ha_core / 1e2, cmap=cmap1, origin='lower', extent=extent, vmin=-19, vmax=19)

    im02 = axs[0][2].imshow((np.abs(mag_full_line) - np.abs(blong[:, :, ind_n_1])) / 1e2, cmap='bwr', origin='lower', extent=extent, vmin=-15, vmax=15)

    im12 = axs[1][2].imshow((np.abs(mag_ha_core) - np.abs(blong[:, :, ind_n_5])) / 1e2, cmap='bwr', origin='lower', extent=extent, vmin=-15, vmax=15)

    level5path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes/20230527/Level-5-alt-alt/')

    sunspot_mask, _ = sunpy.io.read_file(level5path / 'sunspot_mask.fits')[0]

    lightbridge_mask, _ = sunpy.io.read_file(level5path / 'lightbridge_mask.fits')[0]

    emission_mask, _ = sunpy.io.read_file(level5path / 'emission_mask.fits')[0]

    X, Y = np.meshgrid(np.arange(50) * 0.6, np.arange(50) * 0.6)

    axs[0][0].contour(X, Y, sunspot_mask, levels=0, colors='black', linewidths=0.5)
    axs[0][1].contour(X, Y, sunspot_mask, levels=0, colors='black', linewidths=0.5)
    axs[0][2].contour(X, Y, sunspot_mask, levels=0, colors='black', linewidths=0.5)
    axs[1][0].contour(X, Y, sunspot_mask, levels=0, colors='black', linewidths=0.5)
    axs[1][1].contour(X, Y, sunspot_mask, levels=0, colors='black', linewidths=0.5)
    axs[1][2].contour(X, Y, sunspot_mask, levels=0, colors='black', linewidths=0.5)

    axs[0][0].contour(X, Y, lightbridge_mask, levels=0, colors='green', linewidths=0.5)
    axs[0][1].contour(X, Y, lightbridge_mask, levels=0, colors='green', linewidths=0.5)
    axs[0][2].contour(X, Y, lightbridge_mask, levels=0, colors='green', linewidths=0.5)
    axs[1][0].contour(X, Y, lightbridge_mask, levels=0, colors='green', linewidths=0.5)
    axs[1][1].contour(X, Y, lightbridge_mask, levels=0, colors='green', linewidths=0.5)
    axs[1][2].contour(X, Y, lightbridge_mask, levels=0, colors='green', linewidths=0.5)

    axs[0][0].contour(X, Y, emission_mask, levels=0, colors='cyan', linewidths=0.5)
    axs[0][1].contour(X, Y, emission_mask, levels=0, colors='cyan', linewidths=0.5)
    axs[0][2].contour(X, Y, emission_mask, levels=0, colors='cyan', linewidths=0.5)
    axs[1][0].contour(X, Y, emission_mask, levels=0, colors='cyan', linewidths=0.5)
    axs[1][1].contour(X, Y, emission_mask, levels=0, colors='cyan', linewidths=0.5)
    axs[1][2].contour(X, Y, emission_mask, levels=0, colors='cyan', linewidths=0.5)

    # axs[0][0].plot(
    #     l1x[np.array([0, l1x.size // 2, l1x.size - 1, (l1x.size // 2) - 1, 0])] * 0.6,
    #     l1y[np.array([0, l1y.size // 2, l1y.size - 1, (l1y.size // 2) - 1, 0])] * 0.6,
    #     linewidth=0.5,
    #     color=line_colors[0]
    # )

    axs[0][0].plot(
        l2x[np.array([0, l2x.size // 2, l2x.size - 1, (l2x.size // 2) - 1, 0])] * 0.6,
        l2y[np.array([0, l2y.size // 2, l2y.size - 1, (l2y.size // 2) - 1, 0])] * 0.6,
        linewidth=0.5,
        color=line_colors[1]
    )

    # axs[1][0].plot(
    #     l1x[np.array([0, l1x.size // 2, l1x.size - 1, (l1x.size // 2) - 1, 0])] * 0.6,
    #     l1y[np.array([0, l1y.size // 2, l1y.size - 1, (l1y.size // 2) - 1, 0])] * 0.6,
    #     linewidth=0.5,
    #     color=line_colors[0]
    # )

    axs[1][0].plot(
        l2x[np.array([0, l2x.size // 2, l2x.size - 1, (l2x.size // 2) - 1, 0])] * 0.6,
        l2y[np.array([0, l2y.size // 2, l2y.size - 1, (l2y.size // 2) - 1, 0])] * 0.6,
        linewidth=0.5,
        color=line_colors[1]
    )

    # axs[0][1].plot(
    #     l1x[np.array([0, l1x.size // 2, l1x.size - 1, (l1x.size // 2) - 1, 0])] * 0.6,
    #     l1y[np.array([0, l1y.size // 2, l1y.size - 1, (l1y.size // 2) - 1, 0])] * 0.6,
    #     linewidth=0.5,
    #     color=line_colors[0]
    # )

    axs[0][1].plot(
        l2x[np.array([0, l2x.size // 2, l2x.size - 1, (l2x.size // 2) - 1, 0])] * 0.6,
        l2y[np.array([0, l2y.size // 2, l2y.size - 1, (l2y.size // 2) - 1, 0])] * 0.6,
        linewidth=0.5,
        color=line_colors[1]
    )

    # axs[1][1].plot(
    #     l1x[np.array([0, l1x.size // 2, l1x.size - 1, (l1x.size // 2) - 1, 0])] * 0.6,
    #     l1y[np.array([0, l1y.size // 2, l1y.size - 1, (l1y.size // 2) - 1, 0])] * 0.6,
    #     linewidth=0.5,
    #     color=line_colors[0]
    # )

    axs[1][1].plot(
        l2x[np.array([0, l2x.size // 2, l2x.size - 1, (l2x.size // 2) - 1, 0])] * 0.6,
        l2y[np.array([0, l2y.size // 2, l2y.size - 1, (l2y.size // 2) - 1, 0])] * 0.6,
        linewidth=0.5,
        color=line_colors[1]
    )

    for indice, point in enumerate(points):
        for i in range(2):
            for j in range(3):
                axs[i][j].scatter((point[0]) * 0.6, (point[1]) * 0.6, marker='x', s=8, color=pcolors[indice])

    axins00 = inset_axes(
        axs[0][0],
        width="90%",
        height="70%",
        loc="upper left",
        bbox_to_anchor=(0.05, 0.01, 0.8, 0.05),
        bbox_transform=axs[0][0].transAxes,
        borderpad=0,
    )

    cbar = fig.colorbar(im00, cax=axins00, orientation='horizontal')

    cbar.ax.xaxis.set_ticks_position('top')

    axins10 = inset_axes(
        axs[1][0],
        width="90%",
        height="70%",
        loc="upper left",
        bbox_to_anchor=(0.05, 0.01, 0.8, 0.05),
        bbox_transform=axs[1][0].transAxes,
        borderpad=0,
    )

    cbar = fig.colorbar(im10, cax=axins10, orientation='horizontal')

    cbar.ax.xaxis.set_ticks_position('top')

    axins01 = inset_axes(
        axs[0][1],
        width="90%",
        height="70%",
        loc="upper left",
        bbox_to_anchor=(0.05, 0.01, 0.8, 0.05),
        bbox_transform=axs[0][1].transAxes,
        borderpad=0,
    )

    cbar = fig.colorbar(im01, cax=axins01, orientation='horizontal')

    cbar.ax.xaxis.set_ticks_position('top')

    axins11 = inset_axes(
        axs[1][1],
        width="90%",
        height="70%",
        loc="upper left",
        bbox_to_anchor=(0.05, 0.01, 0.8, 0.05),
        bbox_transform=axs[1][1].transAxes,
        borderpad=0,
    )

    cbar = fig.colorbar(im11, cax=axins11, orientation='horizontal')

    cbar.ax.xaxis.set_ticks_position('top')

    axins02 = inset_axes(
        axs[0][2],
        width="90%",
        height="70%",
        loc="upper left",
        bbox_to_anchor=(0.05, 0.01, 0.8, 0.05),
        bbox_transform=axs[0][2].transAxes,
        borderpad=0,
    )

    cbar = fig.colorbar(im02, cax=axins02, orientation='horizontal')

    cbar.ax.xaxis.set_ticks_position('top')

    axins12 = inset_axes(
        axs[1][2],
        width="90%",
        height="70%",
        loc="upper left",
        bbox_to_anchor=(0.05, 0.01, 0.8, 0.05),
        bbox_transform=axs[1][2].transAxes,
        borderpad=0,
    )

    cbar = fig.colorbar(im12, cax=axins12, orientation='horizontal')

    cbar.ax.xaxis.set_ticks_position('top')

    ticks = [0, 5, 10, 15, 20, 25]

    for i in range(2):
        for j in range(3):
            axs[i][j].set_xticks(ticks, [])
            axs[i][j].set_yticks(ticks, [])
            axs[i][j].xaxis.set_minor_locator(MultipleLocator(1))
            axs[i][j].yaxis.set_minor_locator(MultipleLocator(1))

    axs[0][0].set_yticks(ticks, ticks)
    axs[1][0].set_yticks(ticks, ticks)

    axs[1][0].set_xticks(ticks, ticks)
    axs[1][1].set_xticks(ticks, ticks)
    axs[1][2].set_xticks(ticks, ticks)

    axs[1][1].text(
        0.03, -0.25,
        'Scan direction [arcsec]',
        transform=axs[1][1].transAxes,
        color='black'
    )

    axs[1][0].text(
        -0.44, 0.6,
        'Slit direction [arcsec]',
        transform=axs[1][0].transAxes,
        color='black',
        rotation=90
    )

    # axs[0][0].text(
    #     -0.33, 0.3,
    #     r'$\log\tau_{500}$ = $-$1',
    #     transform=axs[0][0].transAxes,
    #     color='black',
    #     rotation=90
    # )
    #
    # axs[1][0].text(
    #     -0.33, 0.3,
    #     r'$\log\tau_{500}$ = $-$5',
    #     transform=axs[1][0].transAxes,
    #     color='black',
    #     rotation=90
    # )

    axs[0][0].text(
        0.05, 0.9,
        r'(a) $\log\tau_{500}$ = $-$1',
        transform=axs[0][0].transAxes,
        color='black'
    )

    axs[0][1].text(
        0.05, 0.9,
        r'(c) WFA (H$\alpha\pm1.5\mathrm{\AA}$ )',
        transform=axs[0][1].transAxes,
        color='black'
    )

    axs[0][2].text(
        0.05, 0.9,
        r'(e) |c| $-$ |a|',
        transform=axs[0][2].transAxes,
        color='black'
    )

    axs[1][0].text(
        0.05, 0.9,
        r'(b) $\log\tau_{500}$ = $-$4.5',
        transform=axs[1][0].transAxes,
        color='black'
    )

    axs[1][1].text(
        0.05, 0.9,
        r'(d) WFA (H$\alpha\pm0.15\mathrm{\AA}$ )',
        transform=axs[1][1].transAxes,
        color='black'
    )

    axs[1][2].text(
        0.05, 0.9,
        r'(f) |d| $-$ |b|',
        transform=axs[1][2].transAxes,
        color='black'
    )

    plt.subplots_adjust(left=0.13, right=1, bottom=0.13, top=1, wspace=0.0, hspace=0.0)

    plt.savefig(
        write_path / 'MagField.pdf',
        format='pdf',
        dpi=300
    )


def create_magnetic_field_scatter_plots():

    y1 = 15
    y2 = 65
    x1 = 3
    x2 = 53

    extent = [0, 0.6 * 50, 0, 0.6 * 50]

    write_path = Path('/home/harsh/CourseworkRepo/KTTAnalysis/figures')

    inversion_base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes/20230527/Level-5-alt-alt/pca_kmeans_fulldata_inversions/')

    wfa_base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes/20230527/Level-5-alt-alt/')

    fa = h5py.File(inversion_base_path / 'combined_output_cycle_B_T_2_retry_all.nc', 'r')

    # fa = h5py.File(inversion_base_path / 'aligned_Ca_Ha_stic_profiles_20230527_074428.nc_straylight_secondpass_stic_file.nc_level_5_alt_alt_cycle_B_T_3_retry_all_t_0_vl_5_vt_0_blong_2_atmos.nc', 'r')

    hmipath = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes/20230527/Level-4-alt-alt/aligned_hmi/')

    hmi_mag, _ = sunpy.io.read_file(
        hmipath / 'hmi.M_720s.20230527_023600_TAI.3.magnetogram.fits_20230527_074428.fits'
    )[0]

    hmi_mag = hmi_mag[y1:y2, x1:x2]

    mag_full_line, _ = sunpy.io.read_file(wfa_base_path / 'aligned_Ca_Ha_stic_profiles_20230527_074428.nc_straylight_secondpass.nc_magha_full_line_spatial.fits')[0]

    mag_ha_core, _ = sunpy.io.read_file(wfa_base_path / 'aligned_Ca_Ha_stic_profiles_20230527_074428.nc_straylight_secondpass.nc_magha_core_spatial.fits')[0]

    mag_full_line = mag_full_line[y1:y2, x1:x2]

    mag_ha_core = mag_ha_core[y1:y2, x1:x2]

    ltau = fa['ltau500'][0, 0, 0]

    ind_n_1 = np.argmin(np.abs(ltau + 1))

    ind_n_5 = np.argmin(np.abs(ltau + 4.5))

    blong = fa['blong'][0]

    level5path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes/20230527/Level-5-alt-alt/')

    sunspot_mask, _ = sunpy.io.read_file(level5path / 'sunspot_mask.fits')[0]

    lightbridge_mask, _ = sunpy.io.read_file(level5path / 'lightbridge_mask.fits')[0]

    emission_mask, _ = sunpy.io.read_file(level5path / 'emission_mask.fits')[0]

    active_region_mask = sunspot_mask + lightbridge_mask

    active_region_mask[np.where(active_region_mask >= 1)] = 1

    penumbra_mask = 1 - active_region_mask

    everything_minus_emission = 1 - emission_mask

    c, d = np.where(everything_minus_emission == 1)

    sa, sb = np.polyfit(blong[c, d, ind_n_1], mag_full_line[c, d], 1)

    print(sa, sb)

    total_mask = np.zeros_like(sunspot_mask, dtype=np.int64)

    total_mask[np.where(sunspot_mask == 1)] = 1

    total_mask[np.where(penumbra_mask == 1)] = 2

    total_mask[np.where(lightbridge_mask == 1)] = 3

    total_mask -= 1

    total_mask = total_mask.flatten()

    categories = np.array(["Umbra", "Penumbra", "Lightbridge",])

    total_categories = categories[total_mask]

    mag_n_1 = blong[:, :, ind_n_1].flatten()

    mag_n_5 = blong[:, :, ind_n_5].flatten()

    a, b = np.where(emission_mask == 1)

    emission_n_5 = blong[a, b, ind_n_5].flatten()

    emission_n_1 = blong[a, b, ind_n_1].flatten()

    mag_ha_core_emission = mag_ha_core[a, b]

    mag_ha_core = mag_ha_core.flatten()

    mag_full_line_emission = mag_full_line[a, b]

    mag_full_line = mag_full_line.flatten()

    hmi_mag_extended = hmi_mag[a, b]

    hmi_mag = hmi_mag.flatten()

    mag_n_5_extended = emission_n_5  # np.concatenate([mag_n_5, emission_n_5])

    mag_n_1_extended = emission_n_1  # np.concatenate([mag_n_1, emission_n_1])

    mag_ha_core_extended = mag_ha_core_emission  # np.concatenate([mag_ha_core, mag_ha_core_emission])

    mag_full_line_extended = mag_full_line_emission  # np.concatenate([mag_full_line, mag_full_line_emission])

    if total_categories.dtype.byteorder == '>':
        total_categories = total_categories.byteswap().newbyteorder()

    if mag_n_1.dtype.byteorder == '>':
        mag_n_1 = mag_n_1.byteswap().newbyteorder()

    if mag_n_5.dtype.byteorder == '>':
        mag_n_5 = mag_n_5.byteswap().newbyteorder()

    if mag_ha_core.dtype.byteorder == '>':
        mag_ha_core = mag_ha_core.byteswap().newbyteorder()

    if mag_full_line.dtype.byteorder == '>':
        mag_full_line = mag_full_line.byteswap().newbyteorder()

    if hmi_mag.dtype.byteorder == '>':
        hmi_mag = hmi_mag.byteswap().newbyteorder()

    if mag_ha_core_extended.dtype.byteorder == '>':
        mag_ha_core_extended = mag_ha_core_extended.byteswap().newbyteorder()

    if mag_n_5_extended.dtype.byteorder == '>':
        mag_n_5_extended = mag_n_5_extended.byteswap().newbyteorder()

    if mag_n_1_extended.dtype.byteorder == '>':
        mag_n_1_extended = mag_n_1_extended.byteswap().newbyteorder()

    if mag_full_line_extended.dtype.byteorder == '>':
        mag_full_line_extended = mag_full_line_extended.byteswap().newbyteorder()

    if hmi_mag_extended.dtype.byteorder == '>':
        hmi_mag_extended = hmi_mag_extended.byteswap().newbyteorder()

    data = {
        'categories': total_categories,
        r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$1)': np.abs(mag_n_1) / 100,
        r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$4.5)': np.abs(mag_n_5) / 100,
        r'$|$WFA$|$ (H$\alpha$ core)': np.abs(mag_ha_core) / 100,
        r'$|$WFA$|$ (H$\alpha \pm 1.5 \mathrm{\AA}$)': np.abs(mag_full_line) / 100,
        r'$|B|$ (HMI)': np.abs(hmi_mag) / 100
    }

    data = pd.DataFrame(data)

    data2 = {
        r'$|$WFA$|$ (H$\alpha$ core)': np.abs(mag_ha_core_extended) / 100,
        r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$4.5)': np.abs(mag_n_5_extended) / 100,
        r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$1)': np.abs(mag_n_1_extended) / 100,
        r'$|$WFA$|$ (H$\alpha \pm 1.5 \mathrm{\AA}$)': np.abs(mag_full_line_extended) / 100,
        r'$|B|$ (HMI)': np.abs(hmi_mag_extended) / 100
    }

    data2 = pd.DataFrame(data2)

    # font = {'size': 6}

    font = {'size': 8}

    matplotlib.rc('font', **font)

    plt.close('all')

    '''
    -----------------------------------------------------------------------------------
    '''
    g = sns.jointplot(
        data=data,
        x=r'$|B|$ (HMI)',
        y=r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$1)',
        hue='categories',
        kind='scatter',
        legend=False,
        palette=['darkslategray', 'coral', 'dodgerblue'],
        s=4
    )

    g.ax_marg_x.set_xlim(0, 20)

    g.ax_marg_y.set_ylim(0, 20)

    index = np.where(total_categories == 'Umbra')[0]

    a, b = np.polyfit(data[r'$|B|$ (HMI)'][index], data[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$1)'][index], 1)

    g.ax_joint.plot(np.arange(0, 20), a * np.arange(0, 20) + b, color='coral', linestyle='--', linewidth=1)

    r, p = pearsonr(data[r'$|B|$ (HMI)'][index], data[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$1)'][index])

    g.ax_joint.text(
        0.05, 0.97,
        r'$m,r,p={},{},{}$'.format(np.round(a, 2), np.round(r, 1), np.round(p, 2)),
        color='coral',
        transform=g.ax_joint.transAxes
    )

    index = np.where(total_categories == 'Penumbra')[0]

    a, b = np.polyfit(data[r'$|B|$ (HMI)'][index], data[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$1)'][index], 1)

    g.ax_joint.plot(np.arange(0, 20), a * np.arange(0, 20) + b, color='darkslategray', linestyle='--',
                    linewidth=1)

    r, p = pearsonr(data[r'$|B|$ (HMI)'][index], data[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$1)'][index])

    g.ax_joint.text(
        0.05, 0.87,
        r'$m,r,p={},{},{}$'.format(np.round(a, 2), np.round(r, 2), np.round(p, 2)),
        color='darkslategray',
        transform=g.ax_joint.transAxes
    )

    index = np.where(total_categories == 'Lightbridge')[0]

    a, b = np.polyfit(data[r'$|B|$ (HMI)'][index], data[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$1)'][index], 1)

    g.ax_joint.plot(np.arange(0, 20), a * np.arange(0, 20) + b, color='dodgerblue', linestyle='--',
                    linewidth=1)

    r, p = pearsonr(data[r'$|B|$ (HMI)'][index], data[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$1)'][index])

    g.ax_joint.text(
        0.05, 0.77,
        r'$m,r,p={},{},{}$'.format(np.round(a, 2), np.round(r, 2), np.round(p, 2)),
        color='dodgerblue',
        transform=g.ax_joint.transAxes
    )

    g.ax_joint.plot(
        np.arange(0, 20), np.arange(0, 20), color='black', linestyle='--', linewidth=1
    )

    g.ax_joint.set_xticks([0, 5, 10, 15, 20], ['0', '', '10', '', '20'])

    g.ax_joint.set_yticks([0, 5, 10, 15, 20], ['0', '', '10', '', '20'])

    # g.ax_joint.text(
    #     -0.3, 1.13,
    #     '(a)',
    #     transform=g.ax_joint.transAxes,
    #     color='black',
    # )

    g.ax_joint.text(
        -0.2, 1.13,
        '(a)',
        transform=g.ax_joint.transAxes,
        color='black',
    )

    g.ax_joint.yaxis.set_minor_locator(MultipleLocator(1))

    g.ax_joint.xaxis.set_minor_locator(MultipleLocator(1))

    # g._figure.set_size_inches(1.75, 1.4, forward=True)

    g._figure.set_size_inches(1.75 * 2, 1.75 * 2, forward=True)

    # g._figure.subplots_adjust(left=0.21, bottom=0.23, top=0.99, right=0.99)

    r, p = pearsonr(data[r'$|B|$ (HMI)'], data[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$1)'])

    print('All {}, {}'.format(r, p))

    axins = inset_axes(
        g.ax_joint,
        width="100%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(0.7, 0.05, 0.3, 0.3),
        bbox_transform=g.ax_joint.transAxes,
        borderpad=0,
    )

    sns.scatterplot(
        data=data2,
        x=r'$|B|$ (HMI)',
        y=r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$1)',
        legend=False,
        color='rosybrown',
        s=4,
        ax=axins
    )

    axins.yaxis.set_minor_locator(MultipleLocator(1))

    axins.xaxis.set_minor_locator(MultipleLocator(1))

    axins.set_xlabel('')

    axins.set_ylabel('')

    axins.set_xticks([0, 5, 10, 15, 20], ['', '', '', '', ''])

    axins.set_yticks([0, 5, 10, 15, 20], ['', '', '', '', ''])

    a, b = np.polyfit(data2[r'$|B|$ (HMI)'], data2[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$1)'], 1)

    axins.plot(np.arange(0, 20), a * np.arange(0, 20) + b, color='rosybrown', linestyle='--',
                    linewidth=1)

    r, p = pearsonr(data2[r'$|B|$ (HMI)'], data2[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$1)'])

    g.ax_joint.text(
        0.05, 0.67,
        r'$m,r,p={},{},{}$'.format(np.round(a, 2), np.round(r, 2), np.round(p, 2)),
        color='rosybrown',
        transform=g.ax_joint.transAxes
    )

    axins.plot(
        np.arange(0, 20), np.arange(0, 20), color='black', linestyle='--', linewidth=1
    )

    axins.set_xlim(0, 20)

    axins.set_ylim(0, 20)

    g._figure.subplots_adjust(left=0.15, bottom=0.15, top=0.99, right=0.99)

    g._figure.savefig(
        write_path / 'MagFieldScatter_0.pdf',
        format='pdf',
        dpi=300
    )

    plt.close('all')

    '''
    -----------------------------------------------------------------------------------
    '''

    g = sns.jointplot(
        data=data,
        x=r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$4.5)',
        y=r'$|$WFA$|$ (H$\alpha$ core)',
        hue='categories',
        kind='scatter',
        legend=False,
        palette=['darkslategray', 'coral', 'dodgerblue'],
        s=4
    )

    g.ax_marg_x.set_xlim(0, 20)

    g.ax_marg_y.set_ylim(0, 20)

    index = np.where(total_categories == 'Umbra')[0]

    a, b = np.polyfit(data[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$4.5)'][index], data[r'$|$WFA$|$ (H$\alpha$ core)'][index], 1)

    g.ax_joint.plot(np.arange(0, 20), a * np.arange(0, 20) + b, color='coral', linestyle='--', linewidth=1)

    r, p = pearsonr(data[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$4.5)'][index], data[r'$|$WFA$|$ (H$\alpha$ core)'][index])

    g.ax_joint.text(
        0.05, 0.97,
        r'$m,r,p={},{},{}$'.format(np.round(a, 2), np.round(r, 2), np.round(p, 2)),
        color='coral',
        transform=g.ax_joint.transAxes
    )

    index = np.where(total_categories == 'Penumbra')[0]

    a, b = np.polyfit(data[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$4.5)'][index], data[r'$|$WFA$|$ (H$\alpha$ core)'][index], 1)

    g.ax_joint.plot(np.arange(0, 20), a * np.arange(0, 20) + b, color='darkslategray', linestyle='--',
                    linewidth=1)

    r, p = pearsonr(data[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$4.5)'][index], data[r'$|$WFA$|$ (H$\alpha$ core)'][index])

    g.ax_joint.text(
        0.05, 0.87,
        r'$m,r,p={},{},{}$'.format(np.round(a, 2), np.round(r, 2), np.round(p, 2)),
        color='darkslategray',
        transform=g.ax_joint.transAxes
    )

    index = np.where(total_categories == 'Lightbridge')[0]

    a, b = np.polyfit(data[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$4.5)'][index], data[r'$|$WFA$|$ (H$\alpha$ core)'][index], 1)

    g.ax_joint.plot(np.arange(0, 20), a * np.arange(0, 20) + b, color='dodgerblue', linestyle='--',
                    linewidth=1)

    r, p = pearsonr(data[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$4.5)'][index], data[r'$|$WFA$|$ (H$\alpha$ core)'][index])

    g.ax_joint.text(
        0.05, 0.77,
        r'$m,r,p={},{},{}$'.format(np.round(a, 2), np.round(r, 2), np.round(p, 2)),
        color='dodgerblue',
        transform=g.ax_joint.transAxes
    )

    g.ax_joint.plot(
        np.arange(0, 20), np.arange(0, 20), color='black', linestyle='--', linewidth=1
    )

    g.ax_joint.plot(
        np.arange(0, 20), np.arange(0, 20), color='black', linestyle='--', linewidth=1
    )

    g.ax_joint.set_xticks([0, 5, 10, 15, 20], ['0', '', '10', '', '20'])

    g.ax_joint.set_yticks([0, 5, 10, 15, 20], ['0', '', '10', '', '20'])

    # g.ax_joint.text(
    #     -0.3, 1.13,
    #     '(c)',
    #     transform=g.ax_joint.transAxes,
    #     color='black',
    # )

    g.ax_joint.text(
        -0.2, 1.13,
        '(d)',
        transform=g.ax_joint.transAxes,
        color='black',
    )

    g.ax_joint.yaxis.set_minor_locator(MultipleLocator(1))

    g.ax_joint.xaxis.set_minor_locator(MultipleLocator(1))

    # g._figure.set_size_inches(1.75, 1.4, forward=True)

    g._figure.set_size_inches(1.75 * 2, 1.75 * 2, forward=True)

    # g._figure.subplots_adjust(left=0.21, bottom=0.23, top=0.99, right=0.99)

    axins = inset_axes(
        g.ax_joint,
        width="100%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(0.7, 0.7, 0.3, 0.3),
        bbox_transform=g.ax_joint.transAxes,
        borderpad=0,
    )

    sns.scatterplot(
        data=data2,
        x=r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$4.5)',
        y=r'$|$WFA$|$ (H$\alpha$ core)',
        legend=False,
        color='rosybrown',
        s=4,
        ax=axins
    )

    axins.yaxis.set_minor_locator(MultipleLocator(1))

    axins.xaxis.set_minor_locator(MultipleLocator(1))

    axins.set_xlabel('')

    axins.set_ylabel('')

    axins.set_xticks([0, 5, 10, 15, 20], ['', '', '', '', ''])

    axins.set_yticks([0, 5, 10, 15, 20], ['', '', '', '', ''])

    a, b = np.polyfit(data2[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$4.5)'], data2[r'$|$WFA$|$ (H$\alpha$ core)'], 1)

    axins.plot(np.arange(0, 20), a * np.arange(0, 20) + b, color='rosybrown', linestyle='--',
               linewidth=1)

    r, p = pearsonr(data2[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$4.5)'], data2[r'$|$WFA$|$ (H$\alpha$ core)'])

    g.ax_joint.text(
        0.05, 0.67,
        r'$m,r,p={},{},{}$'.format(np.round(a, 2), np.round(r, 2), np.round(p, 2)),
        color='rosybrown',
        transform=g.ax_joint.transAxes
    )

    axins.plot(
        np.arange(0, 20), np.arange(0, 20), color='black', linestyle='--', linewidth=1
    )

    axins.set_xlim(0, 20)

    axins.set_ylim(0, 20)

    g._figure.subplots_adjust(left=0.15, bottom=0.15, top=0.99, right=0.99)

    g._figure.savefig(
        write_path / 'MagFieldScatter_3.pdf',
        format='pdf',
        dpi=300
    )

    plt.close('all')

    '''
    -----------------------------------------------------------------------------------
    '''

    g = sns.jointplot(
        data=data,
        x=r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$1)',
        y=r'$|$WFA$|$ (H$\alpha \pm 1.5 \mathrm{\AA}$)',
        hue='categories',
        kind='scatter',
        legend=False,
        palette=['darkslategray', 'coral', 'dodgerblue'],
        s=4
    )

    g.ax_marg_x.set_xlim(0, 20)

    g.ax_marg_y.set_ylim(0, 20)

    index = np.where(total_categories == 'Umbra')[0]

    a, b = np.polyfit(data[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$1)'][index], data[r'$|$WFA$|$ (H$\alpha \pm 1.5 \mathrm{\AA}$)'][index], 1)

    g.ax_joint.plot(np.arange(0, 20), a * np.arange(0, 20) + b, color='coral', linestyle='--', linewidth=1)

    r, p = pearsonr(data[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$1)'][index], data[r'$|$WFA$|$ (H$\alpha \pm 1.5 \mathrm{\AA}$)'][index])

    g.ax_joint.text(
        0.05, 0.97,
        r'$m,r,p={},{},{}$'.format(np.round(a, 2), np.round(r, 2), np.round(p, 2)),
        color='coral',
        transform=g.ax_joint.transAxes
    )

    index = np.where(total_categories == 'Penumbra')[0]

    a, b = np.polyfit(data[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$1)'][index], data[r'$|$WFA$|$ (H$\alpha \pm 1.5 \mathrm{\AA}$)'][index], 1)

    g.ax_joint.plot(np.arange(0, 20), a * np.arange(0, 20) + b, color='darkslategray', linestyle='--',
                    linewidth=1)

    r, p = pearsonr(data[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$1)'][index], data[r'$|$WFA$|$ (H$\alpha \pm 1.5 \mathrm{\AA}$)'][index])

    g.ax_joint.text(
        0.05, 0.87,
        r'$m,r,p={},{},{}$'.format(np.round(a, 2), np.round(r, 2), np.round(p, 2)),
        color='darkslategray',
        transform=g.ax_joint.transAxes
    )

    index = np.where(total_categories == 'Lightbridge')[0]

    a, b = np.polyfit(data[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$1)'][index], data[r'$|$WFA$|$ (H$\alpha \pm 1.5 \mathrm{\AA}$)'][index], 1)

    g.ax_joint.plot(np.arange(0, 20), a * np.arange(0, 20) + b, color='dodgerblue', linestyle='--',
                    linewidth=1)

    r, p = pearsonr(data[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$1)'][index], data[r'$|$WFA$|$ (H$\alpha \pm 1.5 \mathrm{\AA}$)'][index])

    g.ax_joint.text(
        0.05, 0.77,
        r'$m,r,p={},{},{}$'.format(np.round(a, 2), np.round(r, 2), np.round(p, 2)),
        color='dodgerblue',
        transform=g.ax_joint.transAxes
    )

    g.ax_joint.plot(
        np.arange(0, 20), np.arange(0, 20), color='black', linestyle='--', linewidth=1
    )

    g.ax_joint.set_xticks([0, 5, 10, 15, 20], ['0', '', '10', '', '20'])

    g.ax_joint.set_yticks([0, 5, 10, 15, 20], ['0', '', '10', '', '20'])

    # g.ax_joint.text(
    #     -0.3, 1.13,
    #     '(d)',
    #     transform=g.ax_joint.transAxes,
    #     color='black',
    # )

    g.ax_joint.text(
        -0.2, 1.13,
        '(c)',
        transform=g.ax_joint.transAxes,
        color='black',
    )

    g.ax_joint.yaxis.set_minor_locator(MultipleLocator(1))

    g.ax_joint.xaxis.set_minor_locator(MultipleLocator(1))

    # g._figure.set_size_inches(1.75, 1.4, forward=True)

    g._figure.set_size_inches(1.75 * 2, 1.75 * 2, forward=True)

    # g._figure.subplots_adjust(left=0.21, bottom=0.23, top=0.99, right=0.99)

    axins = inset_axes(
        g.ax_joint,
        width="100%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(0.7, 0.7, 0.3, 0.3),
        bbox_transform=g.ax_joint.transAxes,
        borderpad=0,
    )

    sns.scatterplot(
        data=data2,
        x=r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$1)',
        y=r'$|$WFA$|$ (H$\alpha \pm 1.5 \mathrm{\AA}$)',
        legend=False,
        color='rosybrown',
        s=4,
        ax=axins
    )

    axins.yaxis.set_minor_locator(MultipleLocator(1))

    axins.xaxis.set_minor_locator(MultipleLocator(1))

    axins.set_xlabel('')

    axins.set_ylabel('')

    axins.set_xticks([0, 5, 10, 15, 20], ['', '', '', '', ''])

    axins.set_yticks([0, 5, 10, 15, 20], ['', '', '', '', ''])

    a, b = np.polyfit(data2[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$1)'], data2[r'$|$WFA$|$ (H$\alpha \pm 1.5 \mathrm{\AA}$)'],1)

    axins.plot(np.arange(0, 20), a * np.arange(0, 20) + b, color='rosybrown', linestyle='--',
               linewidth=1)

    r, p = pearsonr(data2[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$1)'], data2[r'$|$WFA$|$ (H$\alpha \pm 1.5 \mathrm{\AA}$)'])

    g.ax_joint.text(
        0.05, 0.67,
        r'$m,r,p={},{},{}$'.format(np.round(a, 2), np.round(r, 2), np.round(p, 2)),
        color='rosybrown',
        transform=g.ax_joint.transAxes
    )

    axins.plot(
        np.arange(0, 20), np.arange(0, 20), color='black', linestyle='--', linewidth=1
    )

    axins.set_xlim(0, 20)

    axins.set_ylim(0, 20)

    g._figure.subplots_adjust(left=0.15, bottom=0.15, top=0.99, right=0.99)

    g._figure.savefig(
        write_path / 'MagFieldScatter_2.pdf',
        format='pdf',
        dpi=300
    )

    plt.close('all')

    '''
    -----------------------------------------------------------------------------------
    '''

    # g = sns.jointplot(
    #     data=data,
    #     x=r'$|$WFA$|$ (H$\alpha \pm 1.5 \mathrm{\AA}$)',
    #     y=r'$|$WFA$|$ (H$\alpha$ core)',
    #     hue='categories',
    #     kind='scatter',
    #     legend=False,
    #     palette=['darkslategray', 'coral', 'dodgerblue'],
    #     s=1,
    # )

    g = sns.jointplot(
        data=data,
        x=r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$1)',
        y=r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$4.5)',
        hue='categories',
        kind='scatter',
        legend=False,
        palette=['darkslategray', 'coral', 'dodgerblue'],
        s=4
    )

    g.ax_marg_x.set_xlim(0, 20)

    g.ax_marg_y.set_ylim(0, 20)

    index = np.where(total_categories == 'Umbra')[0]

    # a, b = np.polyfit(data[r'$|$WFA$|$ (H$\alpha \pm 1.5 \mathrm{\AA}$)'][index], data[r'$|$WFA$|$ (H$\alpha$ core)'][index], 1)

    a, b = np.polyfit(data[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$1)'][index], data[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$4.5)'][index], 1)

    g.ax_joint.plot(np.arange(0, 20), a * np.arange(0, 20) + b, color='coral', linestyle='--', linewidth=1)

    r, p = pearsonr(data[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$1)'][index], data[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$4.5)'][index])

    g.ax_joint.text(
        0.05, 0.97,
        r'$m,r,p={},{},{}$'.format(np.round(a, 2), np.round(r, 2), np.round(p, 2)),
        color='coral',
        transform=g.ax_joint.transAxes
    )

    index = np.where(total_categories == 'Penumbra')[0]

    # a, b = np.polyfit(data[r'$|$WFA$|$ (H$\alpha \pm 1.5 \mathrm{\AA}$)'][index], data[r'$|$WFA$|$ (H$\alpha$ core)'][index], 1)

    a, b = np.polyfit(data[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$1)'][index], data[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$4.5)'][index], 1)

    g.ax_joint.plot(np.arange(0, 20), a * np.arange(0, 20) + b, color='darkslategray', linestyle='--',
                    linewidth=1)

    r, p = pearsonr(data[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$1)'][index], data[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$4.5)'][index])

    g.ax_joint.text(
        0.05, 0.87,
        r'$m,r,p={},{},{}$'.format(np.round(a, 2), np.round(r, 2), np.round(p, 2)),
        color='darkslategray',
        transform=g.ax_joint.transAxes
    )

    index = np.where(total_categories == 'Lightbridge')[0]

    # a, b = np.polyfit(data[r'$|$WFA$|$ (H$\alpha \pm 1.5 \mathrm{\AA}$)'][index], data[r'$|$WFA$|$ (H$\alpha$ core)'][index], 1)

    a, b = np.polyfit(data[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$1)'][index], data[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$4.5)'][index], 1)

    g.ax_joint.plot(np.arange(0, 20), a * np.arange(0, 20) + b, color='dodgerblue', linestyle='--',
                    linewidth=1)

    r, p = pearsonr(data[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$1)'][index], data[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$4.5)'][index])

    g.ax_joint.text(
        0.05, 0.77,
        r'$m,r,p={},{},{}$'.format(np.round(a, 2), np.round(r, 2), np.round(p, 2)),
        color='dodgerblue',
        transform=g.ax_joint.transAxes
    )

    g.ax_joint.plot(
        np.arange(0, 20), np.arange(0, 20), color='black', linestyle='--', linewidth=1
    )

    # g.ax_joint.text(
    #     -0.3, 1.13,
    #     '(b)',
    #     transform=g.ax_joint.transAxes,
    #     color='black',
    # )

    g.ax_joint.text(
        -0.2, 1.13,
        '(b)',
        transform=g.ax_joint.transAxes,
        color='black',
    )

    g.ax_joint.set_xticks([0, 5, 10, 15, 20], ['0', '', '10', '', '20'])

    g.ax_joint.set_yticks([0, 5, 10, 15, 20], ['0', '', '10', '', '20'])

    g.ax_joint.yaxis.set_minor_locator(MultipleLocator(1))

    g.ax_joint.xaxis.set_minor_locator(MultipleLocator(1))

    # g._figure.set_size_inches(1.75, 1.4, forward=True)

    g._figure.set_size_inches(1.75 * 2, 1.75 * 2, forward=True)

    # g._figure.subplots_adjust(left=0.21, bottom=0.23, top=0.99, right=0.99)

    axins = inset_axes(
        g.ax_joint,
        width="100%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(0.7, 0.05, 0.3, 0.3),
        bbox_transform=g.ax_joint.transAxes,
        borderpad=0,
    )

    sns.scatterplot(
        data=data2,
        x=r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$1)',
        y=r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$4.5)',
        legend=False,
        color='rosybrown',
        s=4,
        ax=axins
    )

    axins.yaxis.set_minor_locator(MultipleLocator(1))

    axins.xaxis.set_minor_locator(MultipleLocator(1))

    axins.set_xlabel('')

    axins.set_ylabel('')

    axins.set_xticks([0, 5, 10, 15, 20], ['', '', '', '', ''])

    axins.set_yticks([0, 5, 10, 15, 20], ['', '', '', '', ''])

    a, b = np.polyfit(data2[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$1)'], data2[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$4.5)'], 1)
    # a, b = np.polyfit(data2[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$1)'],
    #                   data2[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$4.5)'],
    #                   1)

    axins.plot(np.arange(0, 20), a * np.arange(0, 20) + b, color='rosybrown', linestyle='--',
               linewidth=1)

    r, p = pearsonr(data2[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$1)'], data2[r'$|B_{\mathrm{LOS}}|$ ($\log \tau_{500}$ = $-$4.5)'])

    g.ax_joint.text(
        0.05, 0.67,
        r'$m,r,p={},{},{}$'.format(np.round(a, 2), np.round(r, 2), np.round(p, 2)),
        color='rosybrown',
        transform=g.ax_joint.transAxes
    )

    axins.plot(
        np.arange(0, 20), np.arange(0, 20), color='black', linestyle='--', linewidth=1
    )

    axins.set_xlim(0, 20)

    axins.set_ylim(0, 20)

    g._figure.subplots_adjust(left=0.15, bottom=0.15, top=0.99, right=0.99)

    g._figure.savefig(
        write_path / 'MagFieldScatter_1.pdf',
        format='pdf',
        dpi=300
    )


def make_legend():

    color = ['coral', 'darkslategray', 'dodgerblue', 'rosybrown']
    label_list = ['Umbra', 'Penumbra', 'Lightbridge', 'Emission']

    handles = [Patch(color=c, label=l) for l, c in zip(label_list, color)]
    plt.close('all')
    plt.clf()
    plt.cla()

    # font = {'size': 6}

    font = {'size': 8}

    matplotlib.rc('font', **font)

    # fig = plt.figure(figsize=(3.5, 3.5))

    fig = plt.figure(figsize=(3.5 * 2, 3.5 * 2))

    legend = plt.legend(
        handles,
        label_list,
        ncol=4,
        bbox_to_anchor=(0., 1.02, 1., .102),
        loc='lower left',
        mode="expand",
        borderaxespad=0.
    )
    fig.canvas.draw()
    bbox = legend.get_window_extent().padded(2)
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())

    write_path = Path('/home/harsh/CourseworkRepo/KTTAnalysis/figures/')

    fig.savefig(write_path / 'legends.pdf', dpi=300, transparent=True, bbox_inches=bbox)

    plt.close('all')
    plt.clf()
    plt.cla()


def make_quality_of_fits_fov(points, pcolors):

    input_file = '/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes/20230527/Level-5-alt-alt/aligned_Ca_Ha_stic_profiles_20230527_074428.nc_straylight_secondpass_stic_file.nc'

    fit_file = '/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes/20230527/Level-5-alt-alt/pca_kmeans_fulldata_inversions/combined_output_cycle_B_retry_3_synth_profs.nc'

    level5path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes/20230527/Level-5-alt-alt/')

    write_path = Path('/home/harsh/CourseworkRepo/KTTAnalysis/figures/')

    fi = h5py.File(input_file, 'r')

    ff = h5py.File(fit_file, 'r')

    ind = np.where(fi['profiles'][0, 0, 0, :, 0] != 0)[0]

    wave = fi['wav'][ind]

    ind_ca_core = ind[np.argmin(np.abs(wave - 8662.17))]

    font = {'size': 6}

    matplotlib.rc('font', **font)

    plt.close('all')

    plt.clf()

    plt.cla()

    extent = [0, 0.6 * 50, 0, 0.6 * 50]

    fig, axs = plt.subplots(2, 2, figsize=(3.5, 2.33))

    axs[0][0].imshow(fi['profiles'][0, :, :, ind[8], 0], cmap='gray', origin='lower', aspect='auto', extent=extent)

    axs[0][1].imshow(ff['profiles'][0, :, :, ind[8], 0], cmap='gray', origin='lower', aspect='auto', extent=extent)

    axs[1][0].imshow(fi['profiles'][0, :, :, ind_ca_core, 0], cmap='gray', origin='lower', aspect='auto', extent=extent)

    axs[1][1].imshow(ff['profiles'][0, :, :, ind_ca_core, 0], cmap='gray', origin='lower', aspect='auto', extent=extent)

    sunspot_mask, _ = sunpy.io.read_file(level5path / 'sunspot_mask.fits')[0]

    lightbridge_mask, _ = sunpy.io.read_file(level5path / 'lightbridge_mask.fits')[0]

    emission_mask, _ = sunpy.io.read_file(level5path / 'emission_mask.fits')[0]

    X, Y = np.meshgrid(np.arange(50) * 0.6, np.arange(50) * 0.6)

    axs[0][0].contour(X, Y, sunspot_mask, levels=0, colors='black', linewidths=0.5)
    axs[0][1].contour(X, Y, sunspot_mask, levels=0, colors='black', linewidths=0.5)
    axs[1][0].contour(X, Y, sunspot_mask, levels=0, colors='black', linewidths=0.5)
    axs[1][1].contour(X, Y, sunspot_mask, levels=0, colors='black', linewidths=0.5)

    axs[0][0].contour(X, Y, lightbridge_mask, levels=0, colors='green', linewidths=0.5)
    axs[0][1].contour(X, Y, lightbridge_mask, levels=0, colors='green', linewidths=0.5)
    axs[1][0].contour(X, Y, lightbridge_mask, levels=0, colors='green', linewidths=0.5)
    axs[1][1].contour(X, Y, lightbridge_mask, levels=0, colors='green', linewidths=0.5)

    axs[0][0].contour(X, Y, emission_mask, levels=0, colors='cyan', linewidths=0.5)
    axs[0][1].contour(X, Y, emission_mask, levels=0, colors='cyan', linewidths=0.5)
    axs[1][0].contour(X, Y, emission_mask, levels=0, colors='cyan', linewidths=0.5)
    axs[1][1].contour(X, Y, emission_mask, levels=0, colors='cyan', linewidths=0.5)

    for indice, point in enumerate(points):
        for i in range(2):
            for j in range(2):
                axs[i][j].scatter((point[0]) * 0.6, (point[1]) * 0.6, marker='x', s=8, color=pcolors[indice])

    ticks = [0, 5, 10, 15, 20, 25]

    for i in range(2):
        for j in range(2):
            axs[i][j].set_xticks(ticks, [])
            axs[i][j].set_yticks(ticks, [])
            axs[i][j].xaxis.set_minor_locator(MultipleLocator(1))
            axs[i][j].yaxis.set_minor_locator(MultipleLocator(1))

    axs[0][0].set_yticks(ticks, ticks)
    axs[1][0].set_yticks(ticks, ticks)

    axs[1][0].set_xticks(ticks, ticks)
    axs[1][1].set_xticks(ticks, ticks)

    axs[1][1].text(
        -0.35, -0.35,
        'Scan direction [arcsec]',
        transform=axs[1][1].transAxes,
        color='black'
    )

    axs[1][0].text(
        -0.4, 0.6,
        'Slit direction [arcsec]',
        transform=axs[1][0].transAxes,
        color='black',
        rotation=90
    )

    axs[0][0].text(
        0.05, 0.92,
        '(a) Ca far-wing',
        transform=axs[0][0].transAxes,
        color='white'
    )

    axs[0][0].text(
        0.05, 0.82,
        '     [observed]',
        transform=axs[0][0].transAxes,
        color='white'
    )

    axs[0][1].text(
        0.05, 0.92,
        '(b) Ca far-wing',
        transform=axs[0][1].transAxes,
        color='white'
    )

    axs[0][1].text(
        0.05, 0.82,
        '     [synthesized]',
        transform=axs[0][1].transAxes,
        color='white'
    )

    axs[1][0].text(
        0.05, 0.92,
        '(c) Ca core',
        transform=axs[1][0].transAxes,
        color='white'
    )

    axs[1][0].text(
        0.05, 0.82,
        '     [observed]',
        transform=axs[1][0].transAxes,
        color='white'
    )

    axs[1][1].text(
        0.05, 0.92,
        '(d) Ca core',
        transform=axs[1][1].transAxes,
        color='white'
    )

    axs[1][1].text(
        0.05, 0.82,
        '     [synthesized]',
        transform=axs[1][1].transAxes,
        color='white'
    )

    plt.subplots_adjust(left=0.23, right=0.76, bottom=0.16, top=0.97, wspace=0.0, hspace=0.0)

    fig.savefig(write_path / 'QualityOfFits_FOV.pdf', dpi=300, format='pdf')


def make_quality_of_fits_profiles(points, pcolors):

    input_file = '/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes/20230527/Level-5-alt-alt/aligned_Ca_Ha_stic_profiles_20230527_074428.nc_straylight_secondpass_stic_file.nc'

    fit_file = '/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes/20230527/Level-5-alt-alt/pca_kmeans_fulldata_inversions/combined_output_cycle_B_retry_3_synth_profs.nc'

    write_path = Path('/home/harsh/CourseworkRepo/KTTAnalysis/figures/')

    fi = h5py.File(input_file, 'r')

    ff = h5py.File(fit_file, 'r')

    ind = np.where(fi['profiles'][0, 0, 0, :, 0] != 0)[0]

    wave = fi['wav'][ind]

    font = {'size': 6}

    matplotlib.rc('font', **font)

    plt.close('all')

    plt.clf()

    plt.cla()

    fig, axs = plt.subplots(3, 4, figsize=(3.5, 2.33))

    for indice, (point, pcolor) in enumerate(zip(points, pcolors)):

        k = indice // 2

        if indice%2 == 0:
            l = 0
        else:
            l = 1

        axs[k][l].plot(wave, fi['profiles'][0, point[1], point[0], ind, 0], color=pcolor, linewidth=0.5, linestyle='--')
        axs[k][l].plot(wave, ff['profiles'][0, point[1], point[0], ind, 0], color=pcolor, linewidth=0.5, linestyle='-')

        axs[k][l + 2].plot(wave, fi['profiles'][0, point[1], point[0], ind, 3] / fi['profiles'][0, point[1], point[0], ind, 0], color=pcolor, linewidth=0.5, linestyle='--')
        axs[k][l + 2].plot(wave, ff['profiles'][0, point[1], point[0], ind, 3] / ff['profiles'][0, point[1], point[0], ind, 0], color=pcolor, linewidth=0.5, linestyle='-')

        axs[k][l].set_ylim(0.05, 0.8)

        axs[k][l + 2].set_ylim(-0.3, 0.3)

        k += 1

    xticks = [8662, 8663]

    yticks1 = [0.3, 0.7]

    yticks2 = [-0.2, -0.1, 0, 0.1, 0.2]

    yticks2_labels = ['-20', '', '0', '', '20']

    for i in range(3):
        for j in range(4):
            axs[i][j].set_xticks(xticks, [])
            if j < 2:
                axs[i][j].set_yticks(yticks1, [])
                axs[i][j].yaxis.set_minor_locator(MultipleLocator(0.1))
            else:
                axs[i][j].set_yticks(yticks2, [])
                axs[i][j].yaxis.set_minor_locator(MultipleLocator(0.025))
                axs[i][j].yaxis.tick_right()
            axs[i][j].xaxis.set_minor_locator(MultipleLocator(0.1))

    axs[0][0].set_yticks(yticks1, yticks1)
    axs[1][0].set_yticks(yticks1, yticks1)
    axs[2][0].set_yticks(yticks1, yticks1)

    axs[0][3].set_yticks(yticks2, yticks2_labels)
    axs[1][3].set_yticks(yticks2, yticks2_labels)
    axs[2][3].set_yticks(yticks2, yticks2_labels)

    axs[2][0].set_xticks(xticks, xticks)
    axs[2][1].set_xticks(xticks, xticks)
    axs[2][2].set_xticks(xticks, xticks)
    axs[2][3].set_xticks(xticks, xticks)

    axs[2][1].text(
        0.6, -0.45,
        r'Wavelength [$\mathrm{\AA}$]',
        transform=axs[2][1].transAxes
    )

    axs[1][0].text(
        -0.6, 0.2,
        r'Stokes $I/I_{c}$',
        transform=axs[1][0].transAxes,
        rotation=90
    )

    axs[1][3].text(
        1.5, 0.0,
        r'Stokes $V/I$ [%]',
        transform=axs[1][3].transAxes,
        rotation=90
    )

    plt.subplots_adjust(left=0.12, right=0.87, bottom=0.13, top=0.99, wspace=0.1, hspace=0.1)

    fig.savefig(write_path / 'QualityOfFits_Profiles.pdf', dpi=300, format='pdf')

    fi.close()

    ff.close()


def make_response_function_plots(points, pcolors):
    base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes/20230527/Level-5-alt-alt/pca_kmeans_fulldata_inversions/')

    write_path = Path('/home/harsh/CourseworkRepo/KTTAnalysis/figures/')

    temp_fr = h5py.File(base_path / 'response_atmospheres_response_output.nc', 'r')

    temp_fa = h5py.File(base_path / 'response_atmospheres.nc', 'r')

    mag_fr = [
        h5py.File(base_path / 'response_atmospheres_more_0_response_output.nc', 'r'),
        h5py.File(base_path / 'response_atmospheres_1_response_output.nc', 'r'),
        h5py.File(base_path / 'response_atmospheres_more_2_response_output.nc', 'r'),
        h5py.File(base_path / 'response_atmospheres_more_3_response_output.nc', 'r'),
        h5py.File(base_path / 'response_atmospheres_4_response_output.nc', 'r'),
        h5py.File(base_path / 'response_atmospheres_5_response_output.nc', 'r'),
    ]

    mag_fa = [
        h5py.File(base_path / 'response_atmospheres_more_0.nc', 'r'),
        h5py.File(base_path / 'response_atmospheres_1.nc', 'r'),
        h5py.File(base_path / 'response_atmospheres_more_2.nc', 'r'),
        h5py.File(base_path / 'response_atmospheres_more_3.nc', 'r'),
        h5py.File(base_path / 'response_atmospheres_4.nc', 'r'),
        h5py.File(base_path / 'response_atmospheres_5.nc', 'r'),
    ]

    fo = h5py.File('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes/20230527/Level-5-alt-alt/aligned_Ca_Ha_stic_profiles_20230527_074428.nc_straylight_secondpass_stic_file.nc')

    ind = np.where(fo['profiles'][0, 0, 0, :, 0] != 0)[0]

    ltau = temp_fa['ltau500'][0, 0, 0]

    wave = temp_fr['wav'][()]

    X, Y = np.meshgrid(wave, ltau)

    font = {'size': 8}

    matplotlib.rc('font', **font)

    plt.close('all')

    plt.clf()

    plt.cla()

    colors = ["blue", "green", "white", "darkgoldenrod", "darkred"]

    colors.reverse()

    cmap1 = LinearSegmentedColormap.from_list("mycmap", colors, N=200)

    bounds = np.linspace(-1, 1, 200)
    norm = BoundaryNorm(bounds, cmap1.N)

    calib_velocity = 324619.99385413347

    fig, axs = plt.subplots(3, 4, figsize=(3.5 * 2, 2.33 * 2))

    mag_map = {
        0: 0,
        1: 10,
        2: 0,
        3: 4,
        4: 19,
        5: 19
    }

    for indice, (point, pcolor) in enumerate(zip(points, pcolors)):

        k = indice // 2

        if indice%2 == 0:
            l = 0
        else:
            l = 1

        temp_response = temp_fr['derivatives'][0, 0, indice, 0, :, :, 0]

        # if indice == 2:
        #     blong_response = fr[indice]['derivatives'][0, 0, 0, 0, :, :, 3]
        # else:
        blong_response = mag_fr[indice]['derivatives'][0, 0, mag_map[indice], 0, :, :, 3]

        temp_response /= np.max(np.abs(temp_response))
        blong_response /= np.max(np.abs(blong_response))

        print('{} - {} - {} - {} - {}'.format(k, l, temp_response.shape, temp_response.min(), temp_response.max()))
        print('{} - {} - {} - {} - {}'.format(k, l + 2, blong_response.shape, blong_response.min(), blong_response.max()))

        imtemp = axs[k][l].pcolormesh(X, Y, temp_response, cmap=cmap1, vmin=-1, vmax=1, rasterized=True)

        axs[k][l].set_ylim(-6.3, -0.3)

        axs[k][l].axvline(8662.14, color='black', linestyle='--', linewidth=0.5)

        if l == 0:
            axs[k][l].set_yticks([-1, -2, -3, -4, -5, -6], ['-1', '', '-3', '', '-5', ''])
        else:
            axs[k][l].set_yticks([-1, -2, -3, -4, -5, -6], [])

        axs[k][l].invert_yaxis()

        axs2 = axs[k][l].twiny()
        axs2.plot(temp_fa['temp'][0, 0, indice] / 1e3, ltau, color=pcolors[indice], linewidth=0.5)
        axs2.xaxis.set_minor_locator(MultipleLocator(1))
        axs2.tick_params(axis='x', colors='brown')
        axs2.xaxis.tick_top()
        axs2.set_xlim(3, 7.8)
        if k == 0:
            axs2.set_xticks([4, 5, 6, 7], ['4', '', '6', ''])
        else:
            axs2.set_xticks([4, 5, 6, 7], [])

        ax3 = axs[k][l].twinx()

        ax3.plot(wave, temp_fr['profiles'][0, 0, indice, :, 0], color=pcolors[indice], linewidth=0.5, linestyle='--')
        ax3.plot(wave[ind], fo['profiles'][0, point[1], point[0], ind, 0], color=pcolors[indice], linewidth=0.5, linestyle='dotted')

        ax3.set_yticks([], [])

        ax3.set_ylim(0.05, 0.8)

        if l == 1:
            ax3.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], ['', '', '0.3', '', '', '', '0.7'])
            # ax3.tick_params(axis='y', direction='in', pad=-15)
            ax3.tick_params(axis='y', direction='in', pad=-20)

        axs4 = axs[k][l].twiny()
        axs4.plot((temp_fa['vlos'][0, 0, indice]) / 1e5, ltau, color=pcolors[indice], linewidth=0.5, linestyle='dashdot')
        # axs4.xaxis.set_minor_locator(MultipleLocator(1))
        axs4.xaxis.tick_top()
        axs4.tick_params(axis='x', colors='blue')
        axs4.set_xlim(-15, 15)
        axs4.axvline(x=0, linestyle='dashdot', color='gray', linewidth=0.5)
        if k == 0:
            axs4.spines.top.set_position(("axes", 1.2))
            if l == 0:
                axs4.set_xticks([-15, -10, -5, 0, 5, 10, 15], ['', '-10', '', '0', '', '', ''])
            else:
                axs4.set_xticks([-15, -10, -5, 0, 5, 10, 15], ['', '', '', '0', '', '10', ''])
        else:
            axs4.set_xticks([], [])

        immag = axs[k][l + 2].pcolormesh(X, Y, blong_response, cmap=cmap1, vmin=-1, vmax=1, rasterized=True)

        if k == 2 and l == 1:
            cbaxes = inset_axes(
                axs[k][l + 2], width="80%", height="5%",
                loc=3, borderpad=0,
                bbox_to_anchor=[0.09, 0.05, 1, 1],
                bbox_transform=axs[k][l + 2].transAxes
            )
            cbar = fig.colorbar(
                immag,
                cax=cbaxes,
                ticks=[-1, 1],
                orientation='horizontal'
            )
            cbar.ax.xaxis.set_ticks_position('top')

            cbar.ax.tick_params(colors='black')

            cbaxes = inset_axes(
                axs[k][l], width="80%", height="5%",
                loc=3, borderpad=0,
                bbox_to_anchor=[0.09, 0.05, 1, 1],
                bbox_transform=axs[k][l].transAxes
            )
            cbar = fig.colorbar(
                imtemp,
                cax=cbaxes,
                ticks=[-1, 1],
                orientation='horizontal'
            )
            cbar.ax.xaxis.set_ticks_position('top')

            cbar.ax.tick_params(colors='black')

        axs[k][l + 2].set_ylim(-6.3, -0.3)

        if l == 1:
            axs[k][l + 2].set_yticks([-1, -2, -3, -4, -5, -6], [])
        else:
            axs[k][l + 2].set_yticks([], [])

        axs[k][l + 2].invert_yaxis()

        axs2 = axs[k][l + 2].twiny()
        axs2.plot(mag_fa[indice]['blong'][0, 0, mag_map[indice]] / 1e2, ltau, color=pcolors[indice], linewidth=0.5)
        axs2.set_xlim(-17, 2)
        if k == 0:
            axs2.set_xticks([-15, -10, -5, 0], ['', '-10', '', '0'])
        else:
            axs2.set_xticks([-15, -10, -5, 0], [])

        axs2.xaxis.set_minor_locator(MultipleLocator(1))
        axs2.xaxis.tick_top()

        ax3 = axs[k][l + 2].twinx()

        ax3.plot(wave, mag_fr[indice]['profiles'][0, 0, mag_map[indice], :, 3] / mag_fr[indice]['profiles'][0, 0, mag_map[indice], :, 0], color=pcolors[indice], linewidth=0.5, linestyle='--')
        ax3.plot(wave[ind], fo['profiles'][0, point[1], point[0], ind, 3] / fo['profiles'][0, point[1], point[0], ind, 0], color=pcolors[indice], linewidth=0.5, linestyle='dotted')

        ax3.set_yticks([], [])

        ax3.set_ylim(-0.3, 0.3)

        if l == 1:
            ax3.set_yticks([-0.2, -0.1, 0, 0.1, 0.2], ['-20', '', '0', '', '20'])
            ax3.yaxis.set_minor_locator(MultipleLocator(0.025))

        axs[k][l].xaxis.set_minor_locator(MultipleLocator(0.1))
        axs[k][l + 2].xaxis.set_minor_locator(MultipleLocator(0.1))

    # axs[0][0].text(
    #     0.8, 1.4,
    #     r'$T$ [kK]',
    #     transform=axs[0][0].transAxes
    # )
    #
    # axs[0][2].text(
    #     0.6, 1.4,
    #     r'$B_{\mathrm{LOS}}$ [x 100 G]',
    #     transform=axs[0][2].transAxes
    # )

    axs[0][0].text(
        0.9, 1.07,
        r'$T$ [kK]',
        transform=axs[0][0].transAxes,
        color='brown'
    )

    axs[0][0].text(
        0.8, 1.27,
        r'$V_{\mathrm{LOS}}$ [$\mathrm{km\;s^{-1}}$]',
        transform=axs[0][0].transAxes,
        color='blue'
    )

    axs[0][2].text(
        0.8, 1.27,
        r'$B_{\mathrm{LOS}}$ [x 100 G]',
        transform=axs[0][2].transAxes
    )

    # axs[1][0].text(
    #     -0.43, 0.2,
    #     r'$\log \tau_{500}$',
    #     transform=axs[1][0].transAxes,
    #     rotation=90
    # )

    axs[1][0].text(
        -0.28, 0.4,
        r'$\log \tau_{500}$',
        transform=axs[1][0].transAxes,
        rotation=90
    )

    # axs[2][1].text(
    #     0.6, -0.46,
    #     r'Wavelength [$\mathrm{\AA}$]',
    #     transform=axs[2][1].transAxes
    # )

    axs[2][1].text(
        0.7, -0.31  ,
        r'Wavelength [$\mathrm{\AA}$]',
        transform=axs[2][1].transAxes
    )

    # axs[1][3].text(
    #     1.4, 0.0,
    #     r'Stokes $V/I$ [%]',
    #     transform=axs[1][3].transAxes,
    #     rotation=90
    # )

    axs[1][3].text(
        1.3, 0.2,
        r'Stokes $V/I$ [%]',
        transform=axs[1][3].transAxes,
        rotation=90
    )

    # plt.subplots_adjust(left=0.09, right=0.89, bottom=0.12, top=0.86, wspace=0.0, hspace=0.0)

    plt.subplots_adjust(left=0.06, right=0.92, bottom=0.09, top=0.9, wspace=0.0, hspace=0.0)

    fig.savefig(write_path / 'Response.pdf', dpi=300, format='pdf')

    # for _fa in fa:
    #     _fa.close()
    #
    # for _fr in fr:
    #     _fr.close()

    temp_fr.close()

    temp_fa.close()


def make_vertical_cut(y=30):
    colors = ["blue", "green", "white", "darkgoldenrod", "darkred"]

    colors.reverse()

    cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)

    write_path = Path('/home/harsh/CourseworkRepo/KTTAnalysis/figures/')

    inversion_base_path = Path(
        '/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes/20230527/Level-5-alt-alt/pca_kmeans_fulldata_inversions/'
    )

    # fa = h5py.File(inversion_base_path / 'combined_output_cycle_B_retry_3.nc', 'r')

    fa = h5py.File(inversion_base_path / 'aligned_Ca_Ha_stic_profiles_20230527_074428.nc_straylight_secondpass_stic_file.nc_level_5_alt_alt_cycle_B_T_3_retry_all_t_0_vl_5_vt_0_blong_2_atmos.nc', 'r')

    calib_velocity = 324098.3630889062

    ltau = fa['ltau500'][0, 0, 0]

    xx = np.arange(50) * 0.6

    fig, axs = plt.subplots(2, 2, figsize=(3.5, 2.33))

    X, Y = np.meshgrid(xx, ltau)

    axs[0][0].pcolormesh(X, Y, fa['temp'][0, y, :, :].T / 1e3, cmap='hot', vmin=4, vmax=7.5)

    axs[0][1].pcolormesh(X, Y, (fa['vlos'][0, y, :, :].T - calib_velocity) / 1e5, cmap='bwr', vmin=-10, vmax=10)

    axs[1][0].pcolormesh(X, Y, fa['vturb'][0, y, :, :].T / 1e5, cmap='summer', vmin=0, vmax=8)

    axs[1][1].pcolormesh(X, Y, fa['blong'][0, y, :, :].T / 1e2, cmap=cmap1, vmin=-16, vmax=16)

    for i in range(2):
        for j in range(2):
            axs[i][j].set_ylim(-5.3, -0.3)
            axs[i][j].invert_yaxis()

    plt.subplots_adjust(left=0.09, right=0.89, bottom=0.12, top=0.86, wspace=0.0, hspace=0.0)

    fig.savefig(write_path / 'VerticalCut.pdf', dpi=300, format='pdf')

    fa.close()


def make_line_cuts(l1x, l1y, l2x, l2y, line_colors):
    y1 = 15
    y2 = 65
    x1 = 3
    x2 = 53

    write_path = Path('/home/harsh/CourseworkRepo/KTTAnalysis/figures')

    inversion_base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes/20230527/Level-5-alt-alt/pca_kmeans_fulldata_inversions/')

    wfa_base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes/20230527/Level-5-alt-alt/')

    mag_full_line, _ = sunpy.io.read_file(wfa_base_path / 'aligned_Ca_Ha_stic_profiles_20230527_074428.nc_straylight_secondpass.nc_magha_full_line_spatial.fits')[0]

    mag_ha_core, _ = sunpy.io.read_file(wfa_base_path / 'aligned_Ca_Ha_stic_profiles_20230527_074428.nc_straylight_secondpass.nc_magha_core_spatial.fits')[0]

    mag_full_line = mag_full_line[y1:y2, x1:x2]

    mag_ha_core = mag_ha_core[y1:y2, x1:x2]

    fa = h5py.File(inversion_base_path / 'combined_output_cycle_B_T_2_retry_all.nc', 'r')

    ltau = fa['ltau500'][0, 0, 0]

    ind_n_1 = np.argmin(np.abs(ltau + 1))

    ind_n_5 = np.argmin(np.abs(ltau + 4.5))

    blong = fa['blong'][0]

    f = h5py.File(
        wfa_base_path / 'aligned_Ca_Ha_stic_profiles_20230527_074428.nc_straylight_secondpass.nc',
        'r'
    )

    ind_ca_wing = 3204 + 8

    ind_ha_wing = 32

    # ca_intensity_1 = np.mean(f['profiles'][0][y1:y2, x1:x2, ind_ca_wing, 0][l1y, l1x], 0)

    ca_intensity_2 = np.mean(f['profiles'][0 ,y1:y2, x1:x2, ind_ca_wing, 0][l2y, l2x], 0)

    # ha_intensity_1 = np.mean(f['profiles'][0][y1:y2, x1:x2, ind_ha_wing, 0][l1y, l1x], 0)

    ha_intensity_2 = np.mean(f['profiles'][0, y1:y2, x1:x2, ind_ha_wing, 0][l2y, l2x], 0)

    # ca_1_photo_mag = np.mean(np.abs(blong[l1y, l1x, ind_n_1] / 1e3), 0)

    ca_2_photo_mag = np.mean(np.abs(blong[l2y, l2x, ind_n_1] / 1e3), 0)

    # ca_1_chromo_mag = np.mean(np.abs(blong[l1y, l1x, ind_n_5] / 1e3), 0)

    ca_2_chromo_mag = np.mean(np.abs(blong[l2y, l2x, ind_n_5] / 1e3), 0)

    # ha_1_photo_mag = np.mean(np.abs(mag_full_line[l1y, l1x] / 1e3), 0)

    ha_2_photo_mag = np.mean(np.abs(mag_full_line[l2y, l2x] / 1e3), 0)

    # ha_1_chromo_mag = np.mean(np.abs(mag_ha_core[l1y, l1x] / 1e3), 0)

    ha_2_chromo_mag = np.mean(np.abs(mag_ha_core[l2y, l2x] / 1e3), 0)

    font = {'size': 6}

    matplotlib.rc('font', **font)

    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.75))

    # axs[0][0].plot(ca_intensity_1, linewidth=0.5, linestyle='dotted', color='black')

    # ax2 = axs[0][0].twinx()

    # ax2.plot(ca_1_photo_mag, linewidth=0.5, linestyle='solid', color=line_colors[0])

    # ax2.plot(ca_1_chromo_mag, linewidth=0.5, linestyle='dashed', color=line_colors[0])

    # ax2.set_yticks([0.5, 1], [0.5, 1])

    # axs[0][1].plot(ha_intensity_1, linewidth=0.5, linestyle='dotted', color='black')

    # ax2 = axs[0][1].twinx()

    # ax2.plot(ha_1_photo_mag, linewidth=0.5, linestyle='solid', color=line_colors[0])

    # ax2.plot(ha_1_chromo_mag, linewidth=0.5, linestyle='dashed', color=line_colors[0])

    # ax2.set_yticks([0.5, 1], [0.5, 1])

    axs[0].plot(np.arange(ca_intensity_2.shape[0]) * 0.6 / np.cos(63 * np.pi / 180), ca_intensity_2, linewidth=0.5, linestyle='dotted', color='black')

    ax2 = axs[0].twinx()

    ax2.plot(np.arange(ca_2_photo_mag.shape[0]) * 0.6 / np.cos(63 * np.pi / 180), ca_2_photo_mag, linewidth=0.5, linestyle='solid', color=line_colors[1])

    ax2.plot(np.arange(ca_2_chromo_mag.shape[0]) * 0.6 / np.cos(63 * np.pi / 180), ca_2_chromo_mag, linewidth=0.5, linestyle='dashed', color=line_colors[1])

    ax2.set_yticks([0.5, 1], [0.5, 1])
    ax2.set_ylim(0.4, 1.3)
    ax2.yaxis.set_minor_locator(MultipleLocator(0.1))

    # ax2.set_ylabel(r'B [kG]')

    axs[1].plot(np.arange(ha_intensity_2.shape[0]) * 0.6 / np.cos(63 * np.pi / 180), ha_intensity_2, linewidth=0.5, linestyle='dotted', color='black')

    ax2 = axs[1].twinx()

    ax2.plot(np.arange(ha_2_photo_mag.shape[0]) * 0.6 / np.cos(63 * np.pi / 180), ha_2_photo_mag, linewidth=0.5, linestyle='solid', color=line_colors[1])

    ax2.plot(np.arange(ha_2_chromo_mag.shape[0]) * 0.6 / np.cos(63 * np.pi / 180), ha_2_chromo_mag, linewidth=0.5, linestyle='dashed', color=line_colors[1])

    ax2.set_yticks([0.5, 1], [0.5, 1])
    ax2.set_ylim(0.4, 1.3)
    ax2.yaxis.set_minor_locator(MultipleLocator(0.1))

    ax2.set_ylabel(r'$|$B$|$ [kG]')

    for i in range(1):
        for j in range(2):
            axs[j].set_yticks([0.4, 0.8], [0.4, 0.8])

    axs[0].set_title(r'Ca II 8662 $\mathrm{\AA}$')

    axs[1].set_title(r'H$\alpha$')

    axs[0].set_xlabel(
        'Distance [arcsec]'
    )

    axs[0].set_ylabel(r'$I/I_{\mathrm{c}}$')

    axs[0].set_xticks(
        [0, 3, 6, 9],
        [0, 3, 6, 9]
    )

    axs[1].set_xlabel(
        'Distance [arcsec]'
    )

    # axs[1].set_ylabel(r'$I/I_{\mathrm{c}}$')

    axs[1].set_xticks(
        [0, 3, 6, 9],
        [0, 3, 6, 9]
    )

    axs[0].xaxis.set_minor_locator(MultipleLocator(1))

    axs[1].xaxis.set_minor_locator(MultipleLocator(1))

    axs[1].yaxis.set_minor_locator(MultipleLocator(0.1))

    axs[0].yaxis.set_minor_locator(MultipleLocator(0.1))

    axs[0].set_ylim(0.4, 0.9)

    axs[1].set_ylim(0.4, 0.9)

    fig.tight_layout()

    plt.savefig(write_path / 'Linecuts.pdf', format='pdf', dpi=300)


if __name__ == '__main__':
    # calculate_magnetic_field('20230527', bin_factor=None)

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
    # ticks = [-2000, -1000, 0, 1000, 2000]
    # limit = 2100
    # points = [
    #     [5, 37],
    #     [8, 25],
    #     [23, 29],
    #     [40, 30],
    #     [31, 24],
    #     [40, 4]
    # ]
    # y1 = 15
    # y2 = 65
    # x1 = 3
    # x2 = 53
    # pcolors = ['blueviolet', 'blue', 'dodgerblue', 'orange', 'brown', 'green', 'darkslateblue']
    # create_fov_plots(
    #     datestring=datestring,
    #     timestring=timestring,
    #     x1=x1,
    #     y1=y1,
    #     x2=x2,
    #     y2=y2,
    #     ticks=ticks,
    #     limit=limit,
    #     points=points,
    #     pcolors=pcolors
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

    # y1 = 15
    # y2 = 65
    # x1 = 3
    # x2 = 53
    # colors = ['blueviolet', 'blue', 'dodgerblue', 'orange', 'brown', 'green', 'darkslateblue']
    #
    # datestring = '20230527'
    # timestring = '074428'
    # points = [
    #     [5, 37],
    #     [8, 25]
    # ]
    #
    # make_profile_plots(datestring, timestring, points, color=colors[0:2], index=0, y1=y1, x1=x1)
    #
    # datestring = '20230527'
    # timestring = '074428'
    # points = [
    #     [23, 29],
    #     [40, 30]
    # ]
    # make_profile_plots(datestring, timestring, points, color=colors[2:4], index=1, y1=y1, x1=x1)

    # datestring = '20230527'
    # timestring = '074428'
    # points = [
    #     [31, 24],
    #     [40, 4]
    # ]
    # make_profile_plots(datestring, timestring, points, color=colors[4:6], index=2, y1=y1, x1=x1)

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

    # datestring = '20230603'
    # timestring = '073616'
    # y1 = 14
    # y2 = 64
    # x1 = 4
    # x2 = 54
    # ticks1 = [-1500, -1000, -500, 0, 500, 1000, 1500]
    # # ticks2 = [-15, -10, -5, 0, 5, 10, 15]
    # sv_max = 30
    # limit = 2000
    # points = [
    #     [20, 45],
    #     [46, 57]
    # ]
    # create_fov_plots_adaptive_optics(
    #     datestring=datestring,
    #     timestring=timestring,
    #     x1=x1,
    #     y1=y1,
    #     x2=x2,
    #     y2=y2,
    #     ticks1=ticks1,
    #     sv_max=sv_max,
    #     limit=limit,
    #     points=points
    # )

    # datestring = '20230603'
    # timestring = '073616'
    # y1 = 14
    # y2 = 64
    # x1 = 4
    # x2 = 54
    # ticks1 = [-1500, -1000, -500, 0, 500, 1000, 1500]
    # # ticks2 = [-15, -10, -5, 0, 5, 10, 15]
    # sv_max = 30
    # limit = 2000
    # points = [
    #     [20, 45],
    #     [46, 57]
    # ]
    # create_fov_plots_adaptive_optics(
    #     datestring=datestring,
    #     timestring=timestring,
    #     x1=x1,
    #     y1=y1,
    #     x2=x2,
    #     y2=y2,
    #     ticks1=ticks1,
    #     sv_max=sv_max,
    #     limit=limit,
    #     points=points,
    #     plot_points=True
    # )


    # datestring = '20230601'
    # timestring = '081014'
    # y1 = 14
    # y2 = 64
    # x1 = 3
    # x2 = 53
    # ticks1 = [-1500, -1000, -500, 0, 500, 1000, 1500]
    # # ticks2 = [-15, -10, -5, 0, 5, 10, 15]
    # sv_max = 30
    # limit = 2000
    # points = [
    #     [20, 45],
    #     [46, 57]
    # ]
    # create_fov_plots_adaptive_optics(
    #     datestring=datestring,
    #     timestring=timestring,
    #     x1=x1,
    #     y1=y1,
    #     x2=x2,
    #     y2=y2,
    #     ticks1=ticks1,
    #     sv_max=sv_max,
    #     limit=limit,
    #     points=points
    # )

    # datestring = '20230527'
    # timestring = '074428'
    # y1 = 15
    # y2 = 65
    # x1 = 3
    # x2 = 53
    # ticks1 = [-1500, -1000, -500, 0, 500, 1000, 1500]
    # # ticks2 = [-15, -10, -5, 0, 5, 10, 15]
    # sv_max = 40
    # limit = 2000
    # points = [
    #     [20, 45],
    #     [46, 57]
    # ]
    # create_fov_plots_adaptive_optics(
    #     datestring=datestring,
    #     timestring=timestring,
    #     x1=x1,
    #     y1=y1,
    #     x2=x2,
    #     y2=y2,
    #     ticks1=ticks1,
    #     sv_max=sv_max,
    #     limit=limit,
    #     points=points
    # )

    # points = [
    #     [5, 37],
    #     [8, 25],
    #     [23, 29],
    #     [40, 30],
    #     [31, 24],
    #     [40, 4]
    # ]
    # pcolors = ['blueviolet', 'blue', 'dodgerblue', 'orange', 'brown', 'green']
    # create_atmospheric_param_plots(points, pcolors)
    #
    # colors = ['blueviolet', 'blue', 'dodgerblue', 'orange', 'brown', 'green']
    #
    # points = [
    #     [5, 37],
    #     [8, 25]
    # ]
    # make_individual_inversion_plots(points=points, color=colors[0:2], index=0)
    #
    # points = [
    #     [23, 29],
    #     [40, 30]
    # ]
    # make_individual_inversion_plots(points=points, color=colors[2:4], index=1)
    #
    # points = [
    #     [31, 24],
    #     [40, 4]
    # ]
    # make_individual_inversion_plots(points=points, color=colors[4:6], index=2)

    # points = [
    #     [5, 37],
    #     [8, 25],
    #     [23, 29],
    #     [40, 30],
    #     [31, 24],
    #     [40, 4]
    # ]
    # pcolors = ['blueviolet', 'blue', 'dodgerblue', 'orange', 'brown', 'green', 'darkslateblue']
    # line_points = np.array(
    #     [
    #         [22, 34],
    #         [38, 14],
    #         [38, 35],
    #         [26, 10]
    #     ]
    # )
    # x1 = np.arange(line_points[0][0], line_points[1][0], 1)
    # l1s, l1c = np.polyfit([line_points[0][0], line_points[1][0]], [line_points[0][1], line_points[1][1]], 1)
    # y1 = np.round(x1 * l1s + l1c, 0).astype(np.int64)
    # x1d = x1 - 1
    # y1d = y1 - 1
    # l1x = np.concatenate([x1, x1d])
    # l1y = np.concatenate([y1, y1d])
    # x2 = np.arange(line_points[2][0], line_points[3][0], -1)
    # l2s, l2c = np.polyfit([line_points[2][0], line_points[3][0]], [line_points[2][1], line_points[3][1]], 1)
    # y2 = np.round(x2 * l2s + l2c, 0).astype(np.int64)
    # x2d = x2 - 1
    # y2d = y2 + 1
    # l2x = np.concatenate([x2[3:], x2d[3:]])
    # l2y = np.concatenate([y2[3:], y2d[3:]])
    # line_colors = ['red', 'blue']
    # create_magnetic_field_plots(points, pcolors, l1x, l1y, l2x, l2y, line_colors)

    create_magnetic_field_scatter_plots()
    # make_legend()

    # points = [
    #     [5, 37],
    #     [8, 25],
    #     [23, 29],
    #     [40, 30],
    #     [31, 24],
    #     [40, 4]
    # ]
    # pcolors = ['blueviolet', 'blue', 'dodgerblue', 'orange', 'brown', 'green']
    # make_quality_of_fits_fov(points, pcolors)
    # make_quality_of_fits_profiles(points, pcolors)
    #
    # make_response_function_plots(points, pcolors)

    # y=30
    #
    # make_vertical_cut(y=y)

    # line_points = np.array(
    #     [
    #         [22, 34],
    #         [38, 14],
    #         [38, 35],
    #         [26, 10]
    #     ]
    # )
    #
    # x1 = np.arange(line_points[0][0], line_points[1][0], 1)
    # l1s, l1c = np.polyfit([line_points[0][0], line_points[1][0]], [line_points[0][1], line_points[1][1]], 1)
    # y1 = np.round(x1 * l1s + l1c, 0).astype(np.int64)
    # x1d = x1 - 1
    # y1d = y1 - 1
    # l1x = np.concatenate([x1[np.newaxis, :], x1d[np.newaxis, :]])
    # l1y = np.concatenate([y1[np.newaxis, :], y1d[np.newaxis, :]])
    # x2 = np.arange(line_points[2][0], line_points[3][0], -1)
    # l2s, l2c = np.polyfit([line_points[2][0], line_points[3][0]], [line_points[2][1], line_points[3][1]], 1)
    # print(l2s)
    # y2 = np.round(x2 * l2s + l2c, 0).astype(np.int64)
    # x2d = x2 - 1
    # y2d = y2 + 1
    # l2x = np.concatenate([x2[np.newaxis, 3:], x2d[np.newaxis, 3:]])
    # l2y = np.concatenate([y2[np.newaxis, 3:], y2d[np.newaxis, 3:]])
    # line_colors = ['red', 'blue']
    #
    # make_line_cuts(l1x, l1y, l2x, l2y, line_colors)