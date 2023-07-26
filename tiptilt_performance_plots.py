import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
import sunpy.io
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


A0 = 0.54348


def make_frequency_power_plot():

    base_path = Path('/mnt/f/Harsh/CourseworkRepo/Tip Tilt Data/Closed Loop')

    base_path = Path('/run/media/harsh/DE52135F52133BA9/TipTiltData/')

    write_path = Path('/home/harsh/CourseworkRepo/KTTAnalysis/figures/')

    data_tiptilt = np.loadtxt(base_path / '20230603_073530'/ 'Shifts_Uncorrected.csv', delimiter=',')

    data_no_tiptilt = np.genfromtxt(base_path / '20230520_074727' / 'Shifts_Uncorrected.csv', delimiter=',', invalid_raise=False)

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
        segment_no_tiptilt_x = data_no_tiptilt[i * data_size :i * data_size + data_size, 2] * hamming_window
        segment_no_tiptilt_fft_x = np.fft.fft(segment_no_tiptilt_x)
        segment_tiptilt_x = data_tiptilt[i * data_size:i * data_size + data_size, 2] * hamming_window
        segment_tiptilt_fft_x = np.fft.fft(segment_tiptilt_x)
        total_fft_no_tiptilt_x += np.abs(segment_no_tiptilt_fft_x)
        total_fft_tiptilt_x += np.abs(segment_tiptilt_fft_x)

        segment_no_tiptilt_y = data_no_tiptilt[i * data_size:i * data_size + data_size, 3] * hamming_window
        segment_no_tiptilt_fft_y = np.fft.fft(segment_no_tiptilt_y)
        segment_tiptilt_y = data_tiptilt[i * data_size:i * data_size + data_size, 3] * hamming_window
        segment_tiptilt_fft_y = np.fft.fft(segment_tiptilt_y)
        total_fft_no_tiptilt_y += np.abs(segment_no_tiptilt_fft_y)
        total_fft_tiptilt_y += np.abs(segment_tiptilt_fft_y)

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
    axs[0].set_ylabel(r'Power [$\mathrm{pixel^{2}}$ $\mathrm{Hz^{-1}}$]')


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

    plt.savefig(write_path / 'TipTiltPerformenace_2.pdf', format='pdf', dpi=300)


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

    font = {'size': 10}

    matplotlib.rc('font', **font)

    fig, axs = plt.subplots(2, 2, figsize=(7, 6))

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

    axs[0][0].imshow(data00, cmap='magma', origin='lower', vmin=0, vmax=0.09, extent=extent)
    im01 = axs[0][1].imshow(data01, cmap='magma', origin='lower', vmin=0, vmax=0.09, extent=extent)
    axs[1][0].imshow(data10, cmap='magma', origin='lower', vmin=0, vmax=0.03, extent=extent)
    im11 = axs[1][1].imshow(data11, cmap='magma', origin='lower', vmin=0, vmax=0.03, extent=extent)

    axs[0][0].text(
        0.15, 1.08,
        r'rms = {}'.format(np.round(rmsdata00, 3)),
        transform=axs[0][0].transAxes
    )

    axs[0][1].text(
        0.35, 1.08,
        r'rms = {}'.format(np.round(rmsdata01, 3)),
        transform=axs[0][1].transAxes
    )

    axs[1][0].text(
        0.15, 1.08,
        r'rms = {}'.format(np.round(rmsdata10, 3)),
        transform=axs[1][0].transAxes
    )

    axs[1][1].text(
        0.35, 1.08,
        r'rms = {}'.format(np.round(rmsdata11, 3)),
        transform=axs[1][1].transAxes
    )

    axins01 = inset_axes(
        axs[0][1],
        width="40%",
        height="100%",
        loc="lower right",
        bbox_to_anchor=(1.0, 0, 0.1, 1),
        bbox_transform=axs[0][1].transAxes,
        borderpad=0,
    )
    fig.colorbar(im01, cax=axins01, ticks=[0, 0.03, 0.06, 0.09])

    axins11 = inset_axes(
        axs[1][1],
        width="40%",
        height="100%",
        loc="lower right",
        bbox_to_anchor=(1.0, 0, 0.1, 1),
        bbox_transform=axs[1][1].transAxes,
        borderpad=0,
    )
    fig.colorbar(im11, cax=axins11, ticks=[0, 0.01, 0.02, 0.03])

    axs[0][0].set_ylabel('Slit direction [arcsec]')
    axs[1][0].set_ylabel('Slit direction [arcsec]')

    axs[1][0].set_xlabel('Scan direction [arcsec]')
    axs[1][1].set_xlabel('Scan direction [arcsec]')

    xticks = [0, 5, 10, 15, 20]
    yticks = [0, 5, 10, 15]

    axs[0][0].set_xticks(xticks, [])
    axs[0][1].set_xticks(xticks, [])
    axs[1][0].set_xticks(xticks, xticks)
    axs[1][1].set_xticks(xticks, xticks)

    axs[0][0].set_yticks(yticks, yticks)
    axs[0][1].set_yticks(yticks, [])
    axs[1][0].set_yticks(yticks, yticks)
    axs[1][1].set_yticks(yticks, [])


    axs[0][0].text(
        0.25, 1.25,
        'Without Tip tilt',
        transform=axs[0][0].transAxes
    )

    axs[0][1].text(
        0.25, 1.25,
        'With Tip tilt',
        transform=axs[0][1].transAxes
    )

    axs[0][0].text(
        0.9, 1.1,
        r'H$\alpha$ residual',
        transform=axs[0][0].transAxes
    )

    axs[1][0].text(
        0.8, 1.1,
        r'Ca II 8662 $\mathrm{\AA}$ residual',
        transform=axs[1][0].transAxes
    )
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.08, top=0.9, wspace=0.1, hspace=0.15)

    write_path = Path('F:\\Harsh\\CourseworkRepo\\KTTAnalysis\\figures')

    plt.savefig(write_path / 'Polarization_noise_comparison.pdf', format='pdf', dpi=300)


if __name__ == '__main__':
    make_tiptilt_polarimeter_performance_plots()
