import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


A0 = 0.54348


def make_frequency_power_plot():

    base_path = Path('/mnt/f/Harsh/CourseworkRepo/Tip Tilt Data/Closed Loop')

    data_tiptilt = np.loadtxt(base_path / '20230603_073530'/ 'Shifts_Uncorrected.csv', delimiter=',')

    data_no_tiptilt = np.genfromtxt(base_path / '20230519_100926' / 'Shifts_Uncorrected.csv', delimiter=',', invalid_raise=False)

    segments = 6

    segment_size = 5 # minute

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

    axs[0].set_xticks([1e-1, 1, 1e1, 1e2])
    axs[0].set_xticklabels([1e-1, 1, 1e1, 1e2])
    axs[1].set_xticks([1e-1, 1, 1e1, 1e2])
    axs[1].set_xticklabels([1e-1, 1, 1e1, 1e2])
    plt.show()


if __name__ == '__main__':
    make_frequency_power_plot()
