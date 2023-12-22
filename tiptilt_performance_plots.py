import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
import sunpy.io
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import h5py


A0 = 0.54348


def make_histogram_tiptilt_plots():

    base_path = Path('F:\\Harsh\\CourseworkRepo\\Tip Tilt Data\\Closed Loop')

    write_path = Path('F:\\Harsh\\CourseworkRepo\\KTTAnalysis\\figures')

    data_tiptilt = np.loadtxt(
        base_path / '20230603_073530' / 'Shifts_Uncorrected.csv',
        delimiter=','
    )[:, 2:4]

    data_no_tiptilt = np.genfromtxt(
        base_path / '20230520_074727' / 'Shifts_Uncorrected.csv',
        delimiter=',',
        invalid_raise=False
    )[:, 2:4]

    bins = np.arange(-99.5, 99.5, 1)

    sns.set_style("ticks")

    font = {'size': 12}

    matplotlib.rc('font', **font)

    fig, axs = plt.subplots(2, 2, figsize=(7, 7))

    element = 'bars'

    kde = False

    sns.histplot(
        x=data_no_tiptilt[:, 0],
        bins=bins, kde=kde,
        element=element,
        ax=axs[0][0],
        weights=np.ones(
            len(data_no_tiptilt[:, 0])
        ) / len(data_no_tiptilt[:, 0])
    )

    sns.histplot(
        x=data_tiptilt[:, 0],
        bins=bins, kde=kde,
        element=element,
        ax=axs[0][1],
        weights=np.ones(
            len(data_tiptilt[:, 0])
        ) / len(data_tiptilt[:, 0])
    )

    sns.histplot(
        x=data_no_tiptilt[:, 1],
        bins=bins, kde=kde,
        element=element,
        ax=axs[1][0],
        weights=np.ones(
            len(data_no_tiptilt[:, 1])
        ) / len(data_no_tiptilt[:, 1])
    )

    sns.histplot(
        x=data_tiptilt[:, 1],
        bins=bins, kde=kde,
        element=element,
        ax=axs[1][1],
        weights=np.ones(
            len(data_tiptilt[:, 1])
        ) / len(data_tiptilt[:, 1])
    )

    axs[0][0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0][1].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1][0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1][1].xaxis.set_minor_locator(MultipleLocator(1))

    axs[0][0].yaxis.set_minor_locator(MultipleLocator(0.01))
    axs[0][1].yaxis.set_minor_locator(MultipleLocator(0.01))
    axs[1][0].yaxis.set_minor_locator(MultipleLocator(0.01))
    axs[1][1].yaxis.set_minor_locator(MultipleLocator(0.01))

    axs[0][0].xaxis.set_minor_locator(MultipleLocator(0.5))
    axs[0][1].xaxis.set_minor_locator(MultipleLocator(0.5))
    axs[1][0].xaxis.set_minor_locator(MultipleLocator(0.5))
    axs[1][1].xaxis.set_minor_locator(MultipleLocator(0.5))

    # axs[0][0].yaxis.set_major_formatter(PercentFormatter(1))
    # axs[0][1].yaxis.set_major_formatter(PercentFormatter(1))
    # axs[1][0].yaxis.set_major_formatter(PercentFormatter(1))
    # axs[1][1].yaxis.set_major_formatter(PercentFormatter(1))

    axs[0][0].set_ylim(0, 1)
    axs[0][1].set_ylim(0, 1)
    axs[1][0].set_ylim(0, 1)
    axs[1][1].set_ylim(0, 1)

    axs[0][0].set_xlim(-10, 10)
    axs[0][1].set_xlim(-5, 5)
    axs[1][0].set_xlim(-10, 10)
    axs[1][1].set_xlim(-5, 5)

    axs[0][0].set_xlabel('Shifts [arcsec]')
    axs[0][0].set_ylabel('Counts [%]')

    axs[0][1].set_xlabel('Shifts [arcsec]')
    axs[0][1].set_ylabel('Counts [%]')

    axs[1][0].set_xlabel('Shifts [arcsec]')
    axs[1][0].set_ylabel('Counts [%]')

    axs[1][1].set_xlabel('Shifts [arcsec]')
    axs[1][1].set_ylabel('Counts [%]')

    axs[0][0].text(
        0.3, 1.15,
        'Without tip-tilt',
        transform=axs[0][0].transAxes,
        color='black'
    )
    axs[0][1].text(
        0.3, 1.15,
        'With tip-tilt',
        transform=axs[0][1].transAxes,
        color='black'
    )

    yticks = np.arange(0, 1.1, 0.1)
    xticks_no_tiptilt = np.arange(-10, 10, 1)
    xticks_tiptilt = np.arange(-5, 5, 1)

    ytick_labels = [ str(np.int64(ytick * 100)) if np.int64(ytick * 100)%20 == 0 else '' for ytick in yticks]
    xticks_no_tiptilt_labels = [str(np.round(xtick * 0.22, 2)) if np.int64(xtick) % 4 == 0 else '' for xtick in xticks_no_tiptilt]
    xticks_tiptilt_labels = [str(np.round(xtick * 0.22, 2)) if np.int64(xtick) % 2 == 0 else '' for xtick in xticks_tiptilt]

    axs[0][0].set_yticks(yticks, ytick_labels)
    axs[0][1].set_yticks(yticks, ytick_labels)
    axs[1][0].set_yticks(yticks, ytick_labels)
    axs[1][1].set_yticks(yticks, ytick_labels)

    axs[0][0].set_xticks(xticks_no_tiptilt, xticks_no_tiptilt_labels)
    axs[1][0].set_xticks(xticks_no_tiptilt, xticks_no_tiptilt_labels)

    axs[0][1].set_xticks(xticks_tiptilt, xticks_tiptilt_labels)
    axs[1][1].set_xticks(xticks_tiptilt, xticks_tiptilt_labels)

    axs[0][0].text(
        1.08, 1.1,
        'X-Axis',
        transform=axs[0][0].transAxes,
        color='black'
    )

    axs[1][0].text(
        1.08, 1.1,
        'Y-Axis',
        transform=axs[1][0].transAxes,
        color='black'
    )

    # axs[0][0].text(
    #     0.05, 0.9,
    #     '(c)',
    #     transform=axs[0][0].transAxes,
    #     color='black'
    # )
    #
    # axs[0][1].text(
    #     0.05, 0.9,
    #     '(d)',
    #     transform=axs[0][1].transAxes,
    #     color='black'
    # )
    #
    # axs[1][0].text(
    #     0.05, 0.9,
    #     '(e)',
    #     transform=axs[1][0].transAxes,
    #     color='black'
    # )
    #
    # axs[1][1].text(
    #     0.05, 0.9,
    #     '(f)',
    #     transform=axs[1][1].transAxes,
    #     color='black'
    # )

    plt.subplots_adjust(left=0.1, right=0.99, bottom=0.08, top=0.9, wspace=0.4, hspace=0.35)

    plt.savefig(write_path / 'Tiptilt_histogram.pdf', format='pdf', dpi=300)

def make_frequency_power_plot():

    # base_path = Path('/mnt/f/Harsh/CourseworkRepo/Tip Tilt Data/Closed Loop')

    # base_path = Path('/run/media/harsh/DE52135F52133BA9/TipTiltData/')

    base_path = Path('F:\\Harsh\\CourseworkRepo\\Tip Tilt Data\\Closed Loop')

    # write_path = Path('/home/harsh/CourseworkRepo/KTTAnalysis/figures/')

    write_path = Path('F:\\Harsh\\CourseworkRepo\\KTTAnalysis\\figures')

    data_tiptilt = np.loadtxt(base_path / '20230603_073530'/ 'Shifts_Uncorrected.csv', delimiter=',')[:, 2:4] * 0.22

    data_no_tiptilt = np.genfromtxt(base_path / '20230520_074727' / 'Shifts_Uncorrected.csv', delimiter=',', invalid_raise=False)[:, 2:4] * 0.22

    segments = 50

    segment_size = 1 # minute

    fps = 655 # per second

    seconds_in_one_minute = 60

    data_size = fps * seconds_in_one_minute * segment_size

    total_fft_no_tiptilt_x = np.zeros(data_size)
    total_fft_tiptilt_x = np.zeros(data_size)
    total_fft_no_tiptilt_y = np.zeros(data_size)
    total_fft_tiptilt_y = np.zeros(data_size)

    hamming_window = A0 + (A0-1)*np.cos(2*np.pi*np.arange(data_size)/(data_size-1))

    hamming_percentage = 0.2

    HAMMINGWINDOW_CUT = int(hamming_percentage * data_size / 2)

    for i in np.arange(HAMMINGWINDOW_CUT, data_size - HAMMINGWINDOW_CUT, 1):
        hamming_window[i] = hamming_window[HAMMINGWINDOW_CUT - 1]

    hamming_window /= hamming_window.max()

    for i in range(segments):
        segment_no_tiptilt_x = data_no_tiptilt[i * data_size :i * data_size + data_size, 0] * hamming_window
        segment_no_tiptilt_fft_x = np.fft.fft(segment_no_tiptilt_x)
        segment_tiptilt_x = data_tiptilt[i * data_size:i * data_size + data_size, 0] * hamming_window
        segment_tiptilt_fft_x = np.fft.fft(segment_tiptilt_x)
        total_fft_no_tiptilt_x += np.abs(segment_no_tiptilt_fft_x) * (1/655)# * (1/np.sqrt(segment_no_tiptilt_x.shape[0]))
        total_fft_tiptilt_x += np.abs(segment_tiptilt_fft_x) * (1/655)# * (1/np.sqrt(segment_tiptilt_fft_x.shape[0]))

        segment_no_tiptilt_y = data_no_tiptilt[i * data_size:i * data_size + data_size, 1] * hamming_window
        segment_no_tiptilt_fft_y = np.fft.fft(segment_no_tiptilt_y)
        segment_tiptilt_y = data_tiptilt[i * data_size:i * data_size + data_size, 1] * hamming_window
        segment_tiptilt_fft_y = np.fft.fft(segment_tiptilt_y)
        total_fft_no_tiptilt_y += np.abs(segment_no_tiptilt_fft_y) * (1/655)# * (1 / np.sqrt(segment_no_tiptilt_fft_y.shape[0]))
        total_fft_tiptilt_y += np.abs(segment_tiptilt_fft_y) * (1/655) #* (1 / np.sqrt(segment_tiptilt_fft_y.shape[0]))

    total_fft_no_tiptilt_x /= segments
    total_fft_tiptilt_x /= segments
    total_fft_no_tiptilt_y /= segments
    total_fft_tiptilt_y /= segments

    fftfreq = np.fft.fftfreq(data_size, d=1/655)

    ind_non_zero = np.where(fftfreq > 0)[0]

    font = {'size': 8}

    matplotlib.rc('font', **font)

    plt.close('all')
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.5))

    axs[0].plot(fftfreq[ind_non_zero], total_fft_no_tiptilt_x[ind_non_zero], color='blue')
    axs[0].plot(fftfreq[ind_non_zero], total_fft_tiptilt_x[ind_non_zero], color='black')
    axs[1].plot(fftfreq[ind_non_zero], total_fft_no_tiptilt_y[ind_non_zero], color='blue')
    axs[1].plot(fftfreq[ind_non_zero], total_fft_tiptilt_y[ind_non_zero], color='black')

    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')

    locmaj = matplotlib.ticker.LogLocator(base=10)
    axs[0].xaxis.set_major_locator(locmaj)
    axs[1].xaxis.set_major_locator(locmaj)
    axs[1].set_yticklabels([])

    axs[0].set_xlabel('Frequency [Hz]')
    axs[1].set_xlabel('Frequency [Hz]')
    axs[0].set_ylabel(r'Power [$\mathrm{arcsec^{2}}$ $\mathrm{Hz^{-1}}$]')


    axs[0].text(
        0.45, 0.93,
        'X-Axis',
        transform=axs[0].transAxes,
        color='brown'
    )

    axs[1].text(
        0.45, 0.93,
        'Y-Axis',
        transform=axs[1].transAxes,
        color='brown'
    )

    axs[1].text(
        0.7, 0.93,
        r'$-$ uncorrected',
        transform=axs[1].transAxes,
        color='blue'
    )

    axs[1].text(
        0.7, 0.88,
        r'$-$ corrected',
        transform=axs[1].transAxes,
        color='black'
    )

    plt.subplots_adjust(left=0.1, right=0.99, bottom=0.2, top=0.99, wspace=0.1, hspace=0.1)

    plt.savefig(write_path / 'TipTiltPerformenace.pdf', format='pdf', dpi=300)


def make_tiptilt_polarimeter_performance_plots():

    # base_path = Path('/mnt/f/Harsh/CourseworkRepo/InstrumentalUncorrectedStokes')

    base_path = Path('F:\\Harsh\\CourseworkRepo\\InstrumentalUncorrectedStokes')

    no_tiptilt_directory = base_path / '20230520'

    tiptilt_directory = base_path / '20230603'

    no_tiptilt_halpha, _ = sunpy.io.read_file(no_tiptilt_directory / 'Level-3' / 'residuals_075906_DETECTOR_1.fits')[0]
    no_tiptilt_ca, _ = sunpy.io.read_file(base_path / '20230519' / 'Level-3' / 'residuals_101327_DETECTOR_2.fits')[0]

    tiptilt_halpha, _ = sunpy.io.read_file(tiptilt_directory / 'Level-3' / 'residuals_073616_DETECTOR_1.fits')[0]
    tiptilt_ca, _ = sunpy.io.read_file(tiptilt_directory / 'Level-3' / 'residuals_073616_DETECTOR_3.fits')[0]

    plt.close('all')

    font = {'size': 8}

    matplotlib.rc('font', **font)

    sns.set_style("ticks")

    fig, axs = plt.subplots(2, 2, figsize=(7, 7))

    extent = [0, 37 * 0.6, 0, 29 * 0.6]

    halpha_indices = np.array(
        [
            685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697,
            698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710,
            711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723,
            724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736,
            737, 738, 739
        ]
    )

    ca_indices = np.array([390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402,
       403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415,
       416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428,
       429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441,
       442, 443, 444])

    data00 = np.sqrt(np.mean(np.square(no_tiptilt_halpha[3, :, 21:50, halpha_indices].T / 2), 2))
    data01 = np.sqrt(np.mean(np.square(tiptilt_halpha[3, 18:, 35:, halpha_indices].T / 2), 2))
    data10 = np.sqrt(np.mean(np.square(no_tiptilt_ca[3, 0:37, 10:39, ca_indices].T / 2), 2))
    data11 = np.sqrt(np.mean(np.square(tiptilt_ca[3, :-18, 10:-10, ca_indices].T / 2), 2))

    rmsdata00 = np.sqrt(np.mean(np.square(no_tiptilt_halpha[3, :, 21:50, halpha_indices].T / 2)))
    rmsdata01 = np.sqrt(np.mean(np.square(tiptilt_halpha[3, 18:, 35:, halpha_indices].T / 2)))
    rmsdata10 = np.sqrt(np.mean(np.square(no_tiptilt_ca[3, 0:37, 10:39, ca_indices].T / 2)))
    rmsdata11 = np.sqrt(np.mean(np.square(tiptilt_ca[3, :-18, 10:-10, ca_indices].T / 2)))

    # axs[0][0].imshow(data00, cmap='magma', origin='lower', vmin=0, vmax=0.09, extent=extent)
    # im01 = axs[0][1].imshow(data01, cmap='magma', origin='lower', vmin=0, vmax=0.09, extent=extent)
    # axs[1][0].imshow(data10, cmap='magma', origin='lower', vmin=0, vmax=0.03, extent=extent)
    # im11 = axs[1][1].imshow(data11, cmap='magma', origin='lower', vmin=0, vmax=0.03, extent=extent)

    bins = np.arange(0, 0.08, 0.001)

    element = 'bars'

    kde = False

    sns.histplot(
        x=data00.flatten(),
        bins=bins,kde=kde,
        element=element,
        ax=axs[0][0],
        weights=np.ones_like(
            data00.flatten()) / len(data00.flatten())
    )
    sns.histplot(
        x=data01.flatten(),
        bins=bins,kde=kde,
        element=element,
        ax=axs[0][1],
        weights=np.ones_like(
            data01.flatten()) / len(data01.flatten())
    )
    sns.histplot(
        x=data10.flatten(),
        bins=bins,kde=kde,
        element=element,
        ax=axs[1][0],
        weights=np.ones_like(
            data10.flatten()) / len(data10.flatten())
    )
    sns.histplot(
        x=data11.flatten(),
        bins=bins,kde=kde,
        element=element,
        ax=axs[1][1],
        weights=np.ones_like(
            data11.flatten()) / len(data11.flatten())
    )

    axs[0][0].text(
        0.35, 0.9,
        r'rms = {}'.format(np.round(rmsdata00, 3)),
        transform=axs[0][0].transAxes
    )

    axs[0][1].text(
        0.35, 0.9,
        r'rms = {}'.format(np.round(rmsdata01, 3)),
        transform=axs[0][1].transAxes
    )

    axs[1][0].text(
        0.35, 0.9,
        r'rms = {}'.format(np.round(rmsdata10, 3)),
        transform=axs[1][0].transAxes
    )

    axs[1][1].text(
        0.35, 0.9,
        r'rms = {}'.format(np.round(rmsdata11, 3)),
        transform=axs[1][1].transAxes
    )

    axs[0][0].set_ylim(0, 0.2)
    axs[0][1].set_ylim(0, 0.2)
    axs[1][0].set_ylim(0, 0.2)
    axs[1][1].set_ylim(0, 0.2)

    axs[0][0].set_xlim(0, 0.08)
    axs[0][1].set_xlim(0, 0.08)
    axs[1][0].set_xlim(0, 0.08)
    axs[1][1].set_xlim(0, 0.08)

    axs[0][0].yaxis.set_minor_locator(MultipleLocator(0.01))
    axs[0][1].yaxis.set_minor_locator(MultipleLocator(0.01))
    axs[1][0].yaxis.set_minor_locator(MultipleLocator(0.01))
    axs[1][1].yaxis.set_minor_locator(MultipleLocator(0.01))

    axs[0][0].xaxis.set_minor_locator(MultipleLocator(0.001))
    axs[0][1].xaxis.set_minor_locator(MultipleLocator(0.001))
    axs[1][0].xaxis.set_minor_locator(MultipleLocator(0.001))
    axs[1][1].xaxis.set_minor_locator(MultipleLocator(0.001))

    yticks = np.arange(0, 0.3, 0.05)
    xticks = np.arange(0, 0.1, 0.005)

    ytick_labels = [str(np.int64(ytick * 100)) if np.int64(ytick * 100) % 5 == 0 else '' for ytick in yticks]
    xticks_labels = [str(np.int64(xtick*100)) if  (xtick*100).is_integer() else '' for xtick in xticks] #np.int64(xtick * 100) % 2 == 0 and

    axs[0][0].set_yticks(yticks, ytick_labels)
    axs[0][1].set_yticks(yticks, ytick_labels)
    axs[1][0].set_yticks(yticks, ytick_labels)
    axs[1][1].set_yticks(yticks, ytick_labels)

    # axs[0][0].set_xticks(xticks, xticks_labels)
    # axs[1][0].set_xticks(xticks, xticks_labels)
    #
    # axs[0][1].set_xticks(xticks, xticks_labels)
    # axs[1][1].set_xticks(xticks, xticks_labels)

    axs[0][0].set_xlabel('rms noise')
    axs[0][0].set_ylabel('Counts [%]')

    axs[0][1].set_xlabel('rms noise')
    axs[0][1].set_ylabel('Counts [%]')

    axs[1][0].set_xlabel('rms noise')
    axs[1][0].set_ylabel('Counts [%]')

    axs[1][1].set_xlabel('rms noise')
    axs[1][1].set_ylabel('Counts [%]')

    axs[0][0].text(
        0.25, 1.2,
        'Without Tip tilt',
        transform=axs[0][0].transAxes
    )

    axs[0][1].text(
        0.25, 1.2,
        'With Tip tilt',
        transform=axs[0][1].transAxes
    )

    axs[0][0].text(
        1, 1.1,
        r'H$\alpha$ residual',
        transform=axs[0][0].transAxes
    )

    axs[1][0].text(
        0.95, 1.1,
        r'Ca II 8662 $\mathrm{\AA}$ residual',
        transform=axs[1][0].transAxes
    )
    plt.subplots_adjust(left=0.1, right=0.98, bottom=0.06, top=0.9, wspace=0.3, hspace=0.3)

    write_path = Path('F:\\Harsh\\CourseworkRepo\\KTTAnalysis\\figures')

    plt.savefig(write_path / 'Polarization_noise_comparison.pdf', format='pdf', dpi=300)


def make_context_image():
    base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes/20230527/Level-5-alt-alt')

    ca_ha_file = base_path / 'aligned_Ca_Ha_stic_profiles_20230527_074428.nc_straylight_secondpass.nc'

    ca_ha_residual_file = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes/20230527/Level-4-alt-alt') / 'aligned_residual_Ca_Ha_20230527_074428.fits'

    fcaha = h5py.File(ca_ha_file, 'r')

    residual, _ = sunpy.io.read_file(ca_ha_residual_file)[0]

    ind = np.where(fcaha['profiles'][0, 0, 0, :, 0] != 0)[0]

    ha_ind = ind[0:800]

    ca_ind = ind[800:]

    wave = fcaha['wav'][()]

    halpha_indices = np.array(
        [
            685, 739
        ]
    )

    ca_indices = np.array([390, 444])

    ha_wave = wave[ha_ind][halpha_indices]

    ca_wave = wave[ca_ind][ca_indices]

    plt.close('all')

    font = {'size': 8}

    matplotlib.rc('font', **font)

    fig, axs = plt.subplots(3, 2, figsize=(7, 7.5))

    hX, hY = np.meshgrid(wave[ha_ind[::-1]], np.arange(fcaha['profiles'].shape[2]) * 0.6)

    cX, cY = np.meshgrid(wave[ca_ind], np.arange(fcaha['profiles'].shape[2]) * 0.6)

    print('{} == {}'.format(np.min(fcaha['profiles'][0, 30][:, ha_ind[::-1], 0]), np.max(fcaha['profiles'][0, 30][:, ha_ind[::-1], 0])))
    print('{} == {}'.format(np.min(fcaha['profiles'][0, 30][:, ca_ind, 0]), np.max(fcaha['profiles'][0, 30][:, ca_ind, 0])))
    print('{} == {}'.format(np.min(fcaha['profiles'][0, 30][:, ha_ind[::-1], 3] / fcaha['profiles'][0, 30][:, ha_ind[::-1], 0]), np.max(fcaha['profiles'][0, 30][:, ha_ind[::-1], 3] / fcaha['profiles'][0, 30][:, ha_ind[::-1], 0])))
    print('{} == {}'.format(np.min(fcaha['profiles'][0, 30][:, ca_ind, 3] / fcaha['profiles'][0, 30][:, ca_ind, 0]), np.max(fcaha['profiles'][0, 30][:, ca_ind, 3] / fcaha['profiles'][0, 30][:, ca_ind, 0])))
    print('{} == {}'.format(np.min(residual[3, 30, :, :800][:, ::-1]), np.max(residual[3, 30, :, :800][:, ::-1])))
    print('{} == {}'.format(np.min(residual[3, 30, :, 800:]), np.max(residual[3, 30, :, 800:])))

    im00 = axs[0][0].pcolormesh(hX, hY, fcaha['profiles'][0, 30][:, ha_ind[::-1], 0], cmap='gray', shading='gouraud', vmin=0.24, vmax=1)

    im01 = axs[0][1].pcolormesh(cX, cY, fcaha['profiles'][0, 30][:, ca_ind, 0], cmap='gray', shading='gouraud', vmin=0.12, vmax=0.6)

    hasv = fcaha['profiles'][0, 30][:, ha_ind, 3] / fcaha['profiles'][0, 30][:, ha_ind, 0]

    mm = np.mean(hasv[:, 740:], 1)

    hasv = hasv - mm[:, np.newaxis]

    hasv = hasv[:, ::-1]

    casv = fcaha['profiles'][0, 30][:, ca_ind, 3] / fcaha['profiles'][0, 30][:, ca_ind, 0]

    mm = np.mean(casv[:, 500:], 1)

    casv = casv - mm[:, np.newaxis]

    im10 = axs[1][0].pcolormesh(hX, hY, hasv, cmap='gray', shading='gouraud', vmin=-0.06, vmax=0.06)

    im11 = axs[1][1].pcolormesh(cX, cY, casv, cmap='gray', shading='gouraud', vmin=-0.13, vmax=0.13)

    im20 = axs[2][0].pcolormesh(hX, hY, residual[3, 30, :, :800][:, ::-1], cmap='gray', shading='gouraud', vmin=-0.06, vmax=0.06)

    im21 = axs[2][1].pcolormesh(cX, cY, residual[3, 30, :, 800:], cmap='gray', shading='gouraud', vmin=-0.05, vmax=0.05)

    cbaxes = inset_axes(
        axs[0][0],
        width="80%",
        height="3%",
        loc='upper left',
        borderpad=0,
        bbox_to_anchor=(0.17, 0.0, 0.2, 0.85),
        bbox_transform=axs[0][0].transAxes
    )
    cbar = fig.colorbar(
        im00,
        cax=cbaxes,
        ticks=[0.5, 1],
        orientation='horizontal'
    )
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=8, colors='black')

    cbaxes = inset_axes(
        axs[1][0],
        width="80%",
        height="3%",
        loc='upper left',
        borderpad=0,
        bbox_to_anchor=(0.17, 0.0, 0.2, 0.85),
        bbox_transform=axs[1][0].transAxes
    )
    cbar = fig.colorbar(
        im10,
        cax=cbaxes,
        # ticks=[-0.06, 0, 0.06],
        # labels=[-6, 0, 6],
        orientation='horizontal'
    )
    cbar.set_ticks([-0.06, 0, 0.06],  labels=['-6', '0', '6'])
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=8, colors='black')

    cbaxes = inset_axes(
        axs[2][0],
        width="80%",
        height="3%",
        loc='upper left',
        borderpad=0,
        bbox_to_anchor=(0.17, 0.0, 0.2, 0.85),
        bbox_transform=axs[2][0].transAxes
    )
    cbar = fig.colorbar(
        im20,
        cax=cbaxes,
        orientation='horizontal'
    )
    cbar.set_ticks([-0.06, 0, 0.06], labels=['-0.06', '0', '0.06'])
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=8, colors='black')

    cbaxes = inset_axes(
        axs[0][1],
        width="80%",
        height="3%",
        loc='upper left',
        borderpad=0,
        bbox_to_anchor=(0.81, 0.0, 0.2, 0.85),
        bbox_transform=axs[0][1].transAxes
    )
    cbar = fig.colorbar(
        im01,
        cax=cbaxes,
        orientation='horizontal'
    )
    cbar.set_ticks([0.5], labels=['0.5'])
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=8, colors='black')

    cbaxes = inset_axes(
        axs[1][1],
        width="80%",
        height="3%",
        loc='upper left',
        borderpad=0,
        bbox_to_anchor=(0.81, 0.0, 0.2, 0.85),
        bbox_transform=axs[1][1].transAxes
    )
    cbar = fig.colorbar(
        im11,
        cax=cbaxes,
        orientation='horizontal'
    )
    cbar.set_ticks([-0.13, 0, 0.13], labels=['-13', '0', '13'])
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=8, colors='black')

    cbaxes = inset_axes(
        axs[2][1],
        width="80%",
        height="3%",
        loc='upper left',
        borderpad=0,
        bbox_to_anchor=(0.81, 0.0, 0.2, 0.85),
        bbox_transform=axs[2][1].transAxes
    )
    cbar = fig.colorbar(
        im21,
        cax=cbaxes,
        orientation='horizontal'
    )
    cbar.set_ticks([-0.05, 0, 0.05], labels=['-0.05', '0', '0.05'])
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=8, colors='black')

    axs[0][1].set_yticklabels([])

    axs[1][1].set_yticklabels([])

    axs[2][1].set_yticklabels([])

    axs[0][0].set_xticklabels([])

    axs[0][1].set_xticklabels([])

    axs[1][0].set_xticklabels([])

    axs[1][1].set_xticklabels([])

    # axs[0].set_ylabel('Slit direction [arcsec]')

    axs[0][0].yaxis.set_minor_locator(MultipleLocator(1))

    axs[0][1].yaxis.set_minor_locator(MultipleLocator(1))

    axs[0][0].xaxis.set_minor_locator(MultipleLocator(0.1))

    axs[0][1].xaxis.set_minor_locator(MultipleLocator(0.1))

    axs[1][0].yaxis.set_minor_locator(MultipleLocator(1))

    axs[1][1].yaxis.set_minor_locator(MultipleLocator(1))

    axs[1][0].xaxis.set_minor_locator(MultipleLocator(0.1))

    axs[1][1].xaxis.set_minor_locator(MultipleLocator(0.1))

    axs[2][0].yaxis.set_minor_locator(MultipleLocator(1))

    axs[2][1].yaxis.set_minor_locator(MultipleLocator(1))

    axs[2][0].xaxis.set_minor_locator(MultipleLocator(0.1))

    axs[2][1].xaxis.set_minor_locator(MultipleLocator(0.1))

    axs[0][0].set_xticks([6561, 6562, 6563, 6564], [])

    axs[0][1].set_xticks([8662, 8663], [])

    axs[1][0].set_xticks([6561, 6562, 6563, 6564], [])

    axs[1][1].set_xticks([8662, 8663], [])

    axs[2][0].set_xticks([6561, 6562, 6563, 6564], [6561, 6562, 6563, 6564])

    axs[2][1].set_xticks([8662, 8663], [8662, 8663])

    for w in ha_wave:
        axs[0][0].axvline(x=w, color='black', linewidth=0.5)
        axs[1][0].axvline(x=w, color='black', linewidth=0.5)
        axs[2][0].axvline(x=w, color='black', linewidth=0.5)

    for w in ca_wave:
        axs[0][1].axvline(x=w, color='black', linewidth=0.5)
        axs[1][1].axvline(x=w, color='black', linewidth=0.5)
        axs[2][1].axvline(x=w, color='black', linewidth=0.5)


    axs[2][0].text(
        0.9, -0.18,
        r'Wavelength [$\mathrm{\AA}$]',
        transform=axs[2][0].transAxes,
        color='black'
    )

    axs[1][0].text(
        -0.14, 0.1,
        'Slit direction [arcsec]',
        transform=axs[1][0].transAxes,
        color='black',
        rotation=90
    )

    plt.subplots_adjust(left=0.07, right=0.99, bottom=0.1, top=0.99, wspace=0.08, hspace=0.08)

    write_path = Path('/home/harsh/CourseworkRepo/KTTAnalysis/figures')

    plt.savefig(write_path / 'context.pdf', format='pdf', dpi=300)


if __name__ == '__main__':
    # make_tiptilt_polarimeter_performance_plots()

    # make_histogram_tiptilt_plots()

    # make_frequency_power_plot()

    make_context_image()
