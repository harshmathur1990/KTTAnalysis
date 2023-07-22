import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib

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


if __name__ == '__main__':
    make_frequency_power_plot()
