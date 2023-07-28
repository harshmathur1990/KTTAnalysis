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

def calculate_magnetic_field(datestring):

    # base_path = Path('F:\\Harsh\\CourseworkRepo\\InstrumentalUncorrectedStokes')

    base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes')

    # base_path = Path('C:\\Work Things\\InstrumentalUncorrectedStokes')

    datepath = base_path / datestring

    level4path = datepath / 'Level-4'

    all_files = datepath.glob('**/*')

    all_mag_files = [file for file in all_files if file.name.startswith('aligned') and file.name.endswith('straylight_corrected.nc')]

    for a_mag_file in all_mag_files:
        fcaha = h5py.File(a_mag_file, 'r')

        ind = np.where(fcaha['profiles'][0, 0, 0, :, 0] != 0)[0]

        ind = ind[800:]

        # actual_calculate_blos = prepare_calculate_blos(
        #     fcaha['profiles'][0][:, :, ind],
        #     fcaha['wav'][ind] / 10,
        #     8661.8991 / 10,
        #     8661.5 / 10,
        #     8661.9 / 10,
        #     1.5,
        #     transition_skip_list=None,
        #     bin_factor=16
        # )
        #
        # vec_actual_calculate_blos = np.vectorize(actual_calculate_blos)
        #
        # magca = np.fromfunction(vec_actual_calculate_blos, shape=(fcaha['profiles'].shape[1], fcaha['profiles'].shape[2]))
        #
        # sunpy.io.write_file(level4path / '{}_mag_ca_fe.fits'.format(a_mag_file.name), magca, dict(), overwrite=True)

        actual_calculate_blos = prepare_calculate_blos(
            fcaha['profiles'][0][:, :, ind],
            fcaha['wav'][ind] / 10,
            8662.17 / 10,
            (8662.17) / 10,
            (8662.17 + 0.4) / 10,
            0.83,
            transition_skip_list=None,
            bin_factor=16
        )

        vec_actual_calculate_blos = np.vectorize(actual_calculate_blos)

        magca = np.fromfunction(vec_actual_calculate_blos,
                                shape=(fcaha['profiles'].shape[1], fcaha['profiles'].shape[2]))

        sunpy.io.write_file(level4path / '{}_mag_ca_core.fits'.format(a_mag_file.name), magca, dict(), overwrite=True)

    # for a_mag_file in all_mag_files:
    #     fcaha = h5py.File(a_mag_file, 'r')
    #
    #     ind = np.where(fcaha['profiles'][0, 0, 0, :, 0] != 0)[0]
    #
    #     ind = ind[0:800]
    #     ha_center_wave = 6562.8 / 10
    #     wave_range = 0.35 / 10
    #
    #     transition_skip_list = np.array(
    #         [
    #             [6560.57, 0.25],
    #             [6561.09, 0.1],
    #             [6562.44, 0.1],
    #             [6563.5, 0.25],
    #             [6564.15, 0.35]
    #         ]
    #     ) / 10
    #
    #     actual_calculate_blos = prepare_calculate_blos(
    #         fcaha['profiles'][0][:, :, ind],
    #         fcaha['wav'][ind] / 10,
    #         ha_center_wave,
    #         ha_center_wave - wave_range,
    #         ha_center_wave + wave_range,
    #         1.048,
    #         transition_skip_list=transition_skip_list,
    #         bin_factor=16
    #     )
    #
    #     vec_actual_calculate_blos = np.vectorize(actual_calculate_blos)
    #
    #     magha = np.fromfunction(vec_actual_calculate_blos, shape=(fcaha['profiles'].shape[1], fcaha['profiles'].shape[2]))
    #
    #     sunpy.io.write_file(level4path / '{}_mag_ha_core.fits'.format(a_mag_file.name), magha, dict(), overwrite=True)
    #
    #     wave_range = 1.5 / 10
    #
    #     actual_calculate_blos = prepare_calculate_blos(
    #         fcaha['profiles'][0][:, :, ind],
    #         fcaha['wav'][ind] / 10,
    #         ha_center_wave,
    #         ha_center_wave - wave_range,
    #         ha_center_wave + wave_range,
    #         1.048,
    #         transition_skip_list=transition_skip_list,
    #         bin_factor=16
    #     )
    #
    #     vec_actual_calculate_blos = np.vectorize(actual_calculate_blos)
    #
    #     magfha = np.fromfunction(vec_actual_calculate_blos,
    #                              shape=(fcaha['profiles'].shape[1], fcaha['profiles'].shape[2]))
    #
    #     sunpy.io.write_file(level4path / '{}_mag_ha_full_line.fits'.format(a_mag_file.name), magfha, dict(),
    #                         overwrite=True)


def create_fov_plots(datestring, timestring, x1, y1, x2, y2, ticks, limit, points):

    colors = ["blue", "green", "white", "darkgoldenrod", "darkred"]

    colors.reverse()

    cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)

    extent = [0, 0.6 * 50, 0, 0.6 * 50]

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

    ml = MultipleLocator(1)

    for i in range(2):
        for j in range(4):
            axs[i][j].set_xticks(ticks, [])
            axs[i][j].set_yticks(ticks, [])
            axs[i][j].xaxis.set_minor_locator(ml)
            axs[i][j].yaxis.set_minor_locator(ml)

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

        max_svh = np.max(np.abs(svh)) * 1.1
        max_svca = np.max(np.abs(svca)) * 1.1

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


if __name__ == '__main__':
    # calculate_magnetic_field('20230527')

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

    datestring = '20230527'
    timestring = '074428'
    y1 = 15
    y2 = 65
    x1 = 3
    x2 = 53
    ticks = [-2000, -1000, 0, 1000, 2000]
    limit = 2500
    points = [
        [10, 47],
        [41, 43]
    ]
    create_fov_plots(
        datestring=datestring,
        timestring=timestring,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        ticks=ticks,
        limit=limit,
        points=points
    )

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

    datestring = '20230527'
    timestring = '074428'
    points = [
        [10, 47],
        [41, 43]
    ]
    make_profile_plots(datestring, timestring, points)