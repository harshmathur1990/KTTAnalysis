import sys
# sys.path.insert(1, '/home/harsh/CourseworkRepo/stic/example')
# sys.path.insert(2, '/home/harsh/CourseworkRepo/WFAComparison')
# sys.path.insert(1, '/mnt/f/Harsh/CourseworkRepo/stic/example')
sys.path.insert(1, 'F:\\Harsh\\CourseworkRepo\\stic\\example')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py
from pathlib import Path
import scipy.ndimage
from aiapy.calibrate import register
import sunpy.map
from astropy import units as u
import astropy.coordinates
import scipy.ndimage
from sunkit_image.coalignment import mapsequence_coalign_by_match_template as mc_coalign
from astropy.coordinates import SkyCoord
from sunpy.coordinates import HeliographicStonyhurst
import scipy.ndimage
import sunpy.io
from prepare_data import *


'''
To Call this method:

Run ipython:
import sunpy.io.fits
from flicker import flicker
data, header = sunpy.io.fits.read('sunspot.fits')[1]
flicker(data[0], data[1])
'''


def flicker(image1, image2, fill_value_1=0, fill_value_2=0, rate=1, animation_path=None, limits=None):

    if limits is None:
        limits = [[None, None], [None, None]]

    plt.close('all')

    image1 = image1.copy().astype(np.float64)

    image2 = image2.copy().astype(np.float64)

    final_image_1 = np.ones(
        shape=(
            max(
                image1.shape[0],
                image2.shape[0]
            ),
            max(
                image1.shape[1],
                image2.shape[1]
            )
        )
    ) * fill_value_1


    final_image_2 = np.ones(
        shape=(
            max(
                image1.shape[0],
                image2.shape[0]
            ),
            max(
                image1.shape[1],
                image2.shape[1]
            )
        )
    ) * fill_value_2

    final_image_1[0: image1.shape[0], 0: image1.shape[1]] = image1

    final_image_2[0: image2.shape[0], 0: image2.shape[1]] = image2

    final_image_1 = final_image_1 / np.nanmax(final_image_1)

    final_image_2 = final_image_2 / np.nanmax(final_image_2)

    imagelist = [final_image_1, final_image_2]

    rate = rate * 1000

    fig = plt.figure()  # make figure

    im = plt.imshow(
        imagelist[0],
        origin='lower',
        cmap='gray',
        interpolation='none'
    )

    # im.set_clim(0, 1)
    # function to update figure
    def updatefig(j):
        # set the data in the axesimage object
        im.set_array(imagelist[j])
        # im.set_clim(0, 1)
        # return the artists set
        return [im]
    # kick off the animation
    ani = animation.FuncAnimation(
        fig,
        updatefig,
        frames=range(2),
        interval=rate, blit=True
    )

    if animation_path:
        Writer = animation.writers['ffmpeg']

        writer = Writer(
            fps=1,
            metadata=dict(artist='Me'),
            bitrate=1800
        )

        ani.save(animation_path, writer=writer, dpi=1200)

    else:
        plt.show()


def rotate(image, angle, fill_value):
    return scipy.ndimage.rotate(image, angle, cval=fill_value, reshape=True)

def make_correct_shape(image1, image2, fill_value_1, fill_value_2):
    final_image_1 = np.ones(
        shape=(
            max(
                image1.shape[3],
                image2.shape[3]
            ),
            max(
                image1.shape[4],
                image2.shape[4]
            )
        )
    ) * fill_value_1[:, :, :, np.newaxis, np.newaxis]

    final_image_2 = np.ones(
        shape=(
            max(
                image1.shape[3],
                image2.shape[3]
            ),
            max(
                image1.shape[4],
                image2.shape[4]
            )
        )
    ) * fill_value_2[:, :, :, np.newaxis, np.newaxis]

    final_image_1[:, :, :, 0: image1.shape[3], 0: image1.shape[4]] = image1

    final_image_2[:, :, :, 0: image2.shape[3], 0: image2.shape[4]] = image2

    return final_image_1, final_image_2

def merge_ca_ha_data():
    # base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes')

    # base_path = Path('/mnt/f/Harsh/CourseworkRepo/InstrumentalUncorrectedStokes')

    base_path = Path('F:\\Harsh\\CourseworkRepo\\InstrumentalUncorrectedStokes')

    datestring = '20230603'

    level3path = base_path / datestring / 'Level-3'

    level4path = base_path / datestring / 'Level-4'

    hmi_image, hmi_header = sunpy.io.read_file(level4path / 'hmi.Ic_720s.20230603_043600_TAI.3.continuum.fits')[1]

    hmi_mag, hmi_mag_header = sunpy.io.read_file(level4path / 'hmi.M_720s.20230603_043600_TAI.3.magnetogram.fits')[1]

    hmi_map = sunpy.map.Map(hmi_image, hmi_header)

    hmi_mag_map = sunpy.map.Map(hmi_mag, hmi_mag_header)

    aia_map = register(hmi_map)

    hmi_mag_aia_map = register(hmi_mag_map)

    timestring = '073616'

    '''
    For the Date Time 2023-06-03_07:36:16 AM IST
    approximate center coordinates for HMI submap
    init_x, init_y = (-397, -244)
    rotation_angle_wrf_hmi = -15
    Numpy slice parameters in order of rotated Halpha and Ca II 8662 data and submap data
    a, b, c, d, e, f, g, h, i, j, k, l = 14, 84, 2, -12, 0, -14, 0, -14, 0, -18, 14, -9
    '''

    spread = 50

    init_x, init_y = -397, -244

    init = (init_x - spread / 2, init_y - spread / 2)

    final = (init_x + spread / 2, init_y + spread / 2)

    y0 = init[1] * u.arcsec

    x0 = init[0] * u.arcsec

    xf = final[0] * u.arcsec

    yf = final[1] * u.arcsec

    bottom_left1 = astropy.coordinates.SkyCoord(
        x0, y0, frame=aia_map.coordinate_frame
    )

    top_right1 = astropy.coordinates.SkyCoord(
        xf, yf, frame=aia_map.coordinate_frame
    )

    submap = aia_map.submap(bottom_left=bottom_left1, top_right=top_right1)

    mag_submap = hmi_mag_aia_map.submap(bottom_left=bottom_left1, top_right=top_right1)

    rotation_angle_wrf_hmi = -15

    a, b, c, d, e, f, g, h, i, j, k, l = 14, 84, 2, -12, 0, -14, 0, -14, 0, -18, 14, -9

    file1 = h5py.File(level3path / 'Halpha_{}_{}_stic_profiles.nc'.format(datestring, timestring), 'r')
    file2 = h5py.File(level3path / 'CaII8662_{}_{}_stic_profiles.nc'.format(datestring, timestring), 'r')

    vec_rotate = np.vectorize(rotate, signature='(m,n),(),()->(p,q)')

    fill_values_1 = np.median(file1['profiles'][()], axis=(1, 2))
    fill_values_2 = np.median(file2['profiles'][()], axis=(1, 2))
    file_1_profiles_rotated_data = np.transpose(
        vec_rotate(
            np.transpose(
                file1['profiles'][()],
                axes=(0, 3, 4, 2, 1)
            ),
            rotation_angle_wrf_hmi,
            fill_values_1
        ),
        axes=(0, 4, 3, 1, 2)
    )[:, c:d, a:b]

    file_2_profiles_rotated_data = np.transpose(
        vec_rotate(
            np.transpose(
                file2['profiles'][()],
                axes=(0, 3, 4, 2, 1)
            ),
            rotation_angle_wrf_hmi,
            fill_values_2
        ),
        axes=(0, 4, 3, 1, 2)
    )[:, g:h, e:f]

    aa, bb = make_correct_shape(
        np.transpose(
            file_1_profiles_rotated_data,
            axes=(0, 3, 4, 2, 1)
        ),
        np.transpose(
            file_2_profiles_rotated_data,
            axes=(0, 3, 4, 2, 1)
        ),
        fill_values_1,
        fill_values_2
    )

    sc_file_1_profiles_rotated_data = np.transpose(
        aa,
        axes=(0, 3, 4, 1, 2)
    )

    sc_file_2_profiles_rotated_data = np.transpose(
        bb,
        axes=(0, 3, 4, 1, 2)
    )

    # i=0, j=-18, k=14, l=-9
    bl = submap.pixel_to_world(k * u.pixel, i * u.pixel)

    tr = submap.pixel_to_world((submap.data.shape[1] - np.abs(l) - 1) * u.pixel, (submap.data.shape[1] - np.abs(j) - 1) * u.pixel)

    sm = submap.submap(bottom_left=bl, top_right=tr)

    mag_sm = mag_submap.submap(bottom_left=bl, top_right=tr)

    sunpy.io.write_file(
        str(
                level4path / 'HMI_Reference_Image_{}_{}.fits'.format(
                datestring, timestring
            ),
        ),
        sm.data,
        sm.meta,
        overwrite=True
    )

    sunpy.io.write_file(
        str(
            level4path / 'HMI_Reference_Magnetogram_{}_{}.fits'.format(
                datestring, timestring
            ),
        ),
        mag_sm.data,
        mag_sm.meta,
        overwrite=True
    )

    flicker(sc_file_1_profiles_rotated_data[0, :, :, 32, 0], sc_file_2_profiles_rotated_data[0, :, :, 32, 0])
    flicker(sc_file_1_profiles_rotated_data[0, :, :, 32, 0], sm.data)

    ha = sp.profile(nx=sc_file_1_profiles_rotated_data.shape[2], ny=sc_file_1_profiles_rotated_data.shape[1], ns=4, nw=file1['wav'].shape[0])
    ca = sp.profile(nx=sc_file_2_profiles_rotated_data.shape[2], ny=sc_file_2_profiles_rotated_data.shape[1], ns=4, nw=file2['wav'].shape[0])

    ha.wav[:] = file1['wav'][()]

    ha.dat[0, :, :, :, :] = sc_file_1_profiles_rotated_data

    ha.weights = file1['weights'][()]

    ca.wav[:] = file2['wav'][()]

    ca.dat[0, :, :, :, :] = sc_file_2_profiles_rotated_data

    ca.weights = file2['weights'][()]

    all_profiles = ca + ha

    all_profiles.write(
        level4path / 'aligned_Ca_Ha_stic_profiles_{}_{}.nc'.format(datestring, timestring)
    )

    rho_p = np.sqrt(init_x ** 2 + init_y ** 2)

    mu = np.sqrt(1 - np.square(rho_p / hmi_map.rsun_obs.value))

def run_flicker():
    # base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes')

    base_path = Path('/mnt/f/Harsh/CourseworkRepo/InstrumentalUncorrectedStokes')

    datestring = '20230603'

    level3path = base_path / datestring / 'Level-3'
    level4path = base_path / datestring / 'Level-4'

    file1 = h5py.File(level3path / 'Halpha_20230603_073616_stic_profiles.nc', 'r')
    file2 = h5py.File(level3path / 'CaII8662_20230603_073616_stic_profiles.nc', 'r')
    data = np.loadtxt(level4path / 'submap.txt')

    fill_value_1 = np.nanmedian(file1['profiles'][0, :, :, 32, 0])
    fill_value_2 = np.nanmedian(file2['profiles'][0, :, :, 32, 0])
    fill_value_3 = np.nanmedian(data)
    rotated_data_1 = scipy.ndimage.rotate(file1['profiles'][0, :, :, 32, 0].T, -15, cval=fill_value_1, reshape=True)
    rotated_data_2 = scipy.ndimage.rotate(file2['profiles'][0, :, :, 32, 0].T, -15, cval=fill_value_2, reshape=True)
    rotated_data_1[np.where(np.isnan(rotated_data_1))] = fill_value_1
    rotated_data_2[np.where(np.isnan(rotated_data_2))] = fill_value_2

    flicker(rotated_data_1[14:, 2:-12], rotated_data_2[:-14, :-14], fill_value_1, fill_value_2)
    # flicker(rotated_data_1[14:, 2:-12], data[:-18, 14:-9], 0, fill_value_3)
    # flicker(rotated_data_1[14:, :-12], rotated_data_2[:, :])
    # print(file1['profiles'][0, :, 10:, 32, 0].T.shape)
    # print(data[0:-27, 12:-12].shape)
    # flicker(file1['profiles'][0, :, 10:, 32, 0].T, data[0:-27, 12:-12])


if __name__ == '__main__':
    merge_ca_ha_data()
    # run_flicker()
