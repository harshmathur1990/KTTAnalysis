import sys
sys.path.insert(1, '/home/harsh/CourseworkRepo/stic/example')
# sys.path.insert(2, '/home/harsh/CourseworkRepo/WFAComparison')
# sys.path.insert(1, '/mnt/f/Harsh/CourseworkRepo/stic/example')
# sys.path.insert(1, 'F:\\Harsh\\CourseworkRepo\\stic\\example')
# sys.path.insert(1, '/Volumes/HarshHDD-Data/Documents/CourseworkRepo/stic/example')
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


def flicker(
        image1, image2, fill_value_1=0, fill_value_2=0,
        rate=1, animation_path=None, limits=None,
        offset_1_y=0, offset_1_x=0, offset_2_y=0, offset_2_x=0
):

    if limits is None:
        limits = [[None, None], [None, None]]

    plt.close('all')

    image1 = image1.copy().astype(np.float64)

    image2 = image2.copy().astype(np.float64)

    final_image_1 = np.ones(
        shape=(
            max(
                image1.shape[0] + offset_1_y,
                image2.shape[0] + offset_2_y
            ),
            max(
                image1.shape[1] + offset_1_x,
                image2.shape[1] + offset_2_x
            )
        )
    ) * fill_value_1


    final_image_2 = np.ones(
        shape=(
            max(
                image1.shape[0] + offset_1_y,
                image2.shape[0] + offset_2_y
            ),
            max(
                image1.shape[1] + offset_1_x,
                image2.shape[1] + offset_2_x
            )
        )
    ) * fill_value_2

    final_image_1[0 + offset_1_y: image1.shape[0] + offset_1_y, 0 + offset_1_x: image1.shape[1] + offset_1_x] = image1

    final_image_2[0 + offset_2_y: image2.shape[0] + offset_2_y, 0 + offset_2_x: image2.shape[1] + offset_2_x] = image2

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
        mn, mx = imagelist[j].min() * 0.9, imagelist[j].max() * 1.1
        im.set_clim(mn, mx)
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


def align_and_merge_ca_ha_data(
        datestring, timestring,
        offset_1_y=0, offset_1_x=0, offset_2_y=0, offset_2_x=0
):
    base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes')

    # base_path = Path('/mnt/f/Harsh/CourseworkRepo/InstrumentalUncorrectedStokes')

    # base_path = Path('C:\\Work Things\\InstrumentalUncorrectedStokes')

    level7path = base_path / datestring / 'Level-3-alt'
    level8path = base_path / datestring / 'Level-4-alt-alt'

    file1 = h5py.File(level7path / 'Halpha_{}_{}_stic_profiles.nc'.format(datestring, timestring), 'r')

    file2 = h5py.File(level7path / 'CaII8662_{}_{}_stic_profiles.nc'.format(datestring, timestring), 'r')

    nx = max(
        file1['profiles'].shape[1] + offset_1_y,
        file2['profiles'].shape[1] + offset_2_y
    )

    ny = max(
        file1['profiles'].shape[2] + offset_1_x,
        file2['profiles'].shape[2] + offset_2_x
    )

    ha = sp.profile(
        nx=nx, ny=ny, ns=4,
        nw=file1['wav'].shape[0]
    )
    ca = sp.profile(
        nx=nx, ny=ny, ns=4,
        nw=file2['wav'].shape[0]
    )

    ha.wav[:] = file1['wav'][()]

    ha.weights = file1['weights'][()]

    ca.wav[:] = file2['wav'][()]

    ca.weights = file2['weights'][()]

    d1 = file1['profiles'][0]

    d2 = file2['profiles'][0]

    fill_value_1 = np.median(d1, axis=(0, 1))

    fill_value_2 = np.median(d2, axis=(0, 1))

    ha.dat[0] = fill_value_1[np.newaxis, np.newaxis, :, :]

    ca.dat[0] = fill_value_2[np.newaxis, np.newaxis, :, :]

    ha.dat[0, 0 + offset_1_x: d1.shape[1] + offset_1_x, 0 + offset_1_y: d1.shape[0] + offset_1_y] = np.transpose(
        d1,
        axes=(1, 0, 2, 3)
    )

    ca.dat[0, 0 + offset_2_x: d2.shape[1] + offset_2_x, 0 + offset_2_y: d2.shape[0] + offset_2_y] = np.transpose(
        d2,
        axes=(1, 0, 2, 3)
    )

    all_profiles = ha + ca

    all_profiles.write(
        level8path / 'aligned_Ca_Ha_stic_profiles_{}_{}.nc'.format(datestring, timestring)
    )

    residual_1, header = sunpy.io.read_file(level7path / 'residuals_{}_DETECTOR_1.fits'.format(timestring))[0]

    residual_2_file = level7path / 'residuals_{}_DETECTOR_3.fits'.format(timestring)

    if not residual_2_file.exists():
        residual_2_file = level7path / 'residuals_{}_DETECTOR_2.fits'.format(timestring)

    residual_2, _ = sunpy.io.read_file(residual_2_file)[0]

    new_residual = np.zeros(
        (
            4,
            ny,
            nx,
            residual_1.shape[3] + residual_2.shape[3]
        ),
        dtype=np.float64
    )

    new_residual[:, 0 + offset_1_x: d1.shape[1] + offset_1_x, 0 + offset_1_y: d1.shape[0] + offset_1_y, 0:residual_1.shape[3]] = np.transpose(
        residual_1,
        axes=(0, 2, 1, 3)
    )

    new_residual[:, 0 + offset_2_x: d2.shape[1] + offset_2_x, 0 + offset_2_y: d2.shape[0] + offset_2_y, residual_1.shape[3]: residual_1.shape[3] + residual_2.shape[3]] = np.transpose(
        residual_2,
        axes=(0, 2, 1, 3)
    )

    sunpy.io.write_file(level8path / 'aligned_residual_Ca_Ha_{}_{}.fits'.format(datestring, timestring), new_residual, header, overwrite=True)


def run_flicker(datestring, timestring, offset_1_y=0, offset_1_x=0, offset_2_y=0, offset_2_x=0, create_files=False):
    base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes')

    # base_path = Path('/mnt/f/Harsh/CourseworkRepo/InstrumentalUncorrectedStokes')

    # base_path = Path('F:\\Harsh\\CourseworkRepo\\InstrumentalUncorrectedStokes')

    # base_path = Path('C:\\Work Things\\InstrumentalUncorrectedStokes')

    level7path = base_path / datestring / 'Level-3-alt'
    level8path = base_path / datestring / 'Level-4-alt-alt'

    level8path.mkdir(parents=True, exist_ok=True)


    file1 = h5py.File(level7path / 'Halpha_{}_{}_stic_profiles.nc'.format(datestring, timestring), 'r')
    file2 = h5py.File(level7path / 'CaII8662_{}_{}_stic_profiles.nc'.format(datestring, timestring), 'r')

    fill_value_1 = np.nanmedian(file1['profiles'][0, :, :, 32, 0])
    fill_value_2 = np.nanmedian(file2['profiles'][0, :, :, 32, 0])

    if not create_files:
        flicker(
            file1['profiles'][0, :, :, 32, 0], file2['profiles'][0, :, :, 32, 0], fill_value_1, fill_value_2,
            offset_1_y=offset_1_y, offset_1_x=offset_1_x, offset_2_y=offset_2_y, offset_2_x=offset_2_x
        )

    else:
        align_and_merge_ca_ha_data(
            datestring=datestring, timestring=timestring,
            offset_1_y=offset_1_y, offset_1_x=offset_1_x, offset_2_y=offset_2_y, offset_2_x=offset_2_x
        )


def run_hmi_flicker(datestring, timestring, hmi_cont_file, hmi_mag_file, angle=-13, offset_x=0, offset_y=0, init_x=307, init_y=-314, create_files=False):
    base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes')

    # base_path = Path('/mnt/f/Harsh/CourseworkRepo/InstrumentalUncorrectedStokes')

    # base_path = Path('F:\\Harsh\\CourseworkRepo\\InstrumentalUncorrectedStokes')

    # base_path = Path('C:\\Work Things\\InstrumentalUncorrectedStokes')

    level4path = base_path / datestring / 'Level-4-alt-alt'

    file1 = h5py.File(level4path / 'aligned_Ca_Ha_stic_profiles_{}_{}.nc'.format(datestring, timestring), 'r')
    hmi_image, hmi_header = sunpy.io.read_file(level4path / hmi_cont_file)[1]
    hmi_map = sunpy.map.Map(hmi_image, hmi_header)
    aia_map = register(hmi_map)

    spread = 100

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

    rotated_submap = submap.rotate(
        angle=angle * u.deg
    )

    rotated_data = rotated_submap.data

    plt.imshow(rotated_data, cmap='gray', origin='lower')

    plt.show()

    flicker(
        file1['profiles'][0, :, :, 32, 0], rotated_data[offset_y:file1['profiles'].shape[1] + offset_y, offset_x:file1['profiles'].shape[2] + offset_x]
    )

    hmi_image_mag, hmi_header_mag = sunpy.io.read_file(level4path / hmi_mag_file)[1]
    hmi_map_mag = sunpy.map.Map(hmi_image_mag, hmi_header_mag)
    aia_map_mag = register(hmi_map_mag)

    submap_mag = aia_map_mag.submap(bottom_left=bottom_left1, top_right=top_right1)

    rotated_submap_mag = submap_mag.rotate(
        angle=angle * u.deg
    )

    rotated_data_mag = rotated_submap_mag.data

    # flicker(
    #     file1['profiles'][0, :, :, 32, 0],
    #     rotated_data_mag[offset_y:file1['profiles'].shape[1] + offset_y, offset_x:file1['profiles'].shape[2] + offset_x]
    # )

    header = dict()

    header['CDELT2'] = 0.6

    header['CDELT3'] = 0.6

    header['CUNIT2'] = 'arcsec'

    header['CUNIT3'] = 'arcsec'

    header['CTYPE2'] = 'HPLT-TAN'

    header['CTYPE3'] = 'HPLN-TAN'

    header['CNAME2'] = 'HPC lat'

    header['CNAME3'] = 'HPC lon'

    sunpy.io.write_file(level4path / 'HMI_reference_image_{}_{}.fits'.format(datestring, timestring), rotated_data[offset_y:file1['profiles'].shape[1] + offset_y, offset_x:file1['profiles'].shape[2] + offset_x], header, overwrite=True)

    sunpy.io.write_file(level4path / 'HMI_reference_magnetogram_{}_{}.fits'.format(datestring, timestring),
                            rotated_data_mag[offset_y:file1['profiles'].shape[1] + offset_y,
                            offset_x:file1['profiles'].shape[2] + offset_x], header, overwrite=True)


def calculate_mu(
        datestring,
        hmi_cont_file,
        init_x, init_y
):
    base_path = Path('/Users/harshmathur/CourseworkRepo/InstrumentalUncorrectedStokes/')

    level4path = base_path / datestring / 'Level-4'
    hmi_image, hmi_header = sunpy.io.read_file(level4path / hmi_cont_file)[1]
    hmi_map = sunpy.map.Map(hmi_image, hmi_header)

    rho_p = np.sqrt(init_x ** 2 + init_y ** 2)

    mu = np.sqrt(1 - np.square(rho_p / hmi_map.rsun_obs.value))

    print(mu)


def get_aia_reference_image(
    datestring='20230527', timestring='074428',
    aia_file='hmi.Ic_720s.20230527_044800_TAI.3.magnetogram.fits',
    angle=-25,
    offset_x=97, offset_y=65,
    init_x=0,
    init_y=-260
):
    base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes')

    # base_path = Path('/mnt/f/Harsh/CourseworkRepo/InstrumentalUncorrectedStokes')

    # base_path = Path('F:\\Harsh\\CourseworkRepo\\InstrumentalUncorrectedStokes')

    # base_path = Path('C:\\Work Things\\InstrumentalUncorrectedStokes')

    level4path = base_path / datestring / 'Level-4-alt-alt'

    file1 = h5py.File(level4path / 'aligned_Ca_Ha_stic_profiles_{}_{}.nc'.format(datestring, timestring), 'r')

    aia_image, aia_header = sunpy.io.read_file(level4path / aia_file)[1]

    aia_image[np.where(np.isnan(aia_image))] = 0

    aia_map = sunpy.map.Map(aia_image, aia_header)

    registered_aia_map = register(aia_map)

    spread = 100

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

    submap_registered_aia_map = registered_aia_map.submap(bottom_left=bottom_left1, top_right=top_right1)

    rotated_submap_registered_aia_map = submap_registered_aia_map.rotate(angle=angle * u.deg, missing=0)

    rotated_data = rotated_submap_registered_aia_map.data

    header = dict()

    header['CDELT2'] = 0.6

    header['CDELT3'] = 0.6

    header['CUNIT2'] = 'arcsec'

    header['CUNIT3'] = 'arcsec'

    header['CTYPE2'] = 'HPLT-TAN'

    header['CTYPE3'] = 'HPLN-TAN'

    header['CNAME2'] = 'HPC lat'

    header['CNAME3'] = 'HPC lon'

    sunpy.io.write_file(
        level4path / '{}_{}_{}.fits'.format(Path(aia_file).name, datestring, timestring),
        rotated_data[offset_y:file1['profiles'].shape[1] + offset_y,
        offset_x:file1['profiles'].shape[2] + offset_x], header, overwrite=True)


def merge_hmi_stokes_files():

    base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes/20230527/Level-4-alt-alt/')

    si = [
        'hmi.s_720s.20230527_044800_TAI.3.I0.fits_20230527_074428.fits',
        'hmi.s_720s.20230527_044800_TAI.3.I1.fits_20230527_074428.fits',
        'hmi.s_720s.20230527_044800_TAI.3.I2.fits_20230527_074428.fits',
        'hmi.s_720s.20230527_044800_TAI.3.I3.fits_20230527_074428.fits',
        'hmi.s_720s.20230527_044800_TAI.3.I4.fits_20230527_074428.fits',
        'hmi.s_720s.20230527_044800_TAI.3.I5.fits_20230527_074428.fits'
    ]
    sq = [
        'hmi.s_720s.20230527_044800_TAI.3.Q0.fits_20230527_074428.fits',
        'hmi.s_720s.20230527_044800_TAI.3.Q1.fits_20230527_074428.fits',
        'hmi.s_720s.20230527_044800_TAI.3.Q2.fits_20230527_074428.fits',
        'hmi.s_720s.20230527_044800_TAI.3.Q3.fits_20230527_074428.fits',
        'hmi.s_720s.20230527_044800_TAI.3.Q4.fits_20230527_074428.fits',
        'hmi.s_720s.20230527_044800_TAI.3.Q5.fits_20230527_074428.fits'
    ]
    su = [
        'hmi.s_720s.20230527_044800_TAI.3.U0.fits_20230527_074428.fits',
        'hmi.s_720s.20230527_044800_TAI.3.U1.fits_20230527_074428.fits',
        'hmi.s_720s.20230527_044800_TAI.3.U2.fits_20230527_074428.fits',
        'hmi.s_720s.20230527_044800_TAI.3.U3.fits_20230527_074428.fits',
        'hmi.s_720s.20230527_044800_TAI.3.U4.fits_20230527_074428.fits',
        'hmi.s_720s.20230527_044800_TAI.3.U5.fits_20230527_074428.fits'
    ]
    sv = [
        'hmi.s_720s.20230527_044800_TAI.3.V0.fits_20230527_074428.fits',
        'hmi.s_720s.20230527_044800_TAI.3.V1.fits_20230527_074428.fits',
        'hmi.s_720s.20230527_044800_TAI.3.V2.fits_20230527_074428.fits',
        'hmi.s_720s.20230527_044800_TAI.3.V3.fits_20230527_074428.fits',
        'hmi.s_720s.20230527_044800_TAI.3.V4.fits_20230527_074428.fits',
        'hmi.s_720s.20230527_044800_TAI.3.V5.fits_20230527_074428.fits'
    ]

    combined_stokes = np.zeros((1, 65, 56, 6, 4), dtype=np.float64)

    for index, (i, q, u,v) in enumerate(zip(si, sq, su, sv)):
        di, hi = sunpy.io.read_file(base_path / i)[0]
        dq, hq = sunpy.io.read_file(base_path / q)[0]
        du, hu = sunpy.io.read_file(base_path / u)[0]
        dv, hv = sunpy.io.read_file(base_path / v)[0]

        combined_stokes[0, :, :, index, 0] = di
        combined_stokes[0, :, :, index, 1] = dq
        combined_stokes[0, :, :, index, 2] = du
        combined_stokes[0, :, :, index, 3] = dv

    sunpy.io.write_file(base_path / 'combined_hmi_stokes.fits', combined_stokes, hi, overwrite=True)


def merge_hmi_stokes_files_full_fov():

    base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes/20230527/Level-4-alt-alt/')

    si = [
        'hmi.s_720s.20230527_044800_TAI.3.I0.fits',
        'hmi.s_720s.20230527_044800_TAI.3.I1.fits',
        'hmi.s_720s.20230527_044800_TAI.3.I2.fits',
        'hmi.s_720s.20230527_044800_TAI.3.I3.fits',
        'hmi.s_720s.20230527_044800_TAI.3.I4.fits',
        'hmi.s_720s.20230527_044800_TAI.3.I5.fits'
    ]
    sq = [
        'hmi.s_720s.20230527_044800_TAI.3.Q0.fits',
        'hmi.s_720s.20230527_044800_TAI.3.Q1.fits',
        'hmi.s_720s.20230527_044800_TAI.3.Q2.fits',
        'hmi.s_720s.20230527_044800_TAI.3.Q3.fits',
        'hmi.s_720s.20230527_044800_TAI.3.Q4.fits',
        'hmi.s_720s.20230527_044800_TAI.3.Q5.fits'
    ]
    su = [
        'hmi.s_720s.20230527_044800_TAI.3.U0.fits',
        'hmi.s_720s.20230527_044800_TAI.3.U1.fits',
        'hmi.s_720s.20230527_044800_TAI.3.U2.fits',
        'hmi.s_720s.20230527_044800_TAI.3.U3.fits',
        'hmi.s_720s.20230527_044800_TAI.3.U4.fits',
        'hmi.s_720s.20230527_044800_TAI.3.U5.fits'
    ]
    sv = [
        'hmi.s_720s.20230527_044800_TAI.3.V0.fits',
        'hmi.s_720s.20230527_044800_TAI.3.V1.fits',
        'hmi.s_720s.20230527_044800_TAI.3.V2.fits',
        'hmi.s_720s.20230527_044800_TAI.3.V3.fits',
        'hmi.s_720s.20230527_044800_TAI.3.V4.fits',
        'hmi.s_720s.20230527_044800_TAI.3.V5.fits'
    ]

    d, h = sunpy.io.read_file(base_path / si[0])[1]

    m = sunpy.map.Map(d, h)

    am = register(m)

    combined_stokes = np.zeros((1, am.data.shape[0], am.data.shape[1], 6, 4), dtype=np.float64)

    for index, (i, q, u,v) in enumerate(zip(si, sq, su, sv)):
        di, hi = sunpy.io.read_file(base_path / i)[1]
        dq, hq = sunpy.io.read_file(base_path / q)[1]
        du, hu = sunpy.io.read_file(base_path / u)[1]
        dv, hv = sunpy.io.read_file(base_path / v)[1]

        mi = sunpy.map.Map(di, hi)
        ami = register(mi)

        mq = sunpy.map.Map(dq, hq)
        amq = register(mq)

        mu = sunpy.map.Map(du, hu)
        amu = register(mu)

        mv = sunpy.map.Map(dv, hv)
        amv = register(mv)

        combined_stokes[0, :, :, index, 0] = ami.data
        combined_stokes[0, :, :, index, 1] = amq.data
        combined_stokes[0, :, :, index, 2] = amu.data
        combined_stokes[0, :, :, index, 3] = amv.data

    sunpy.io.write_file(base_path / 'combined_hmi_stokes_full_fov.fits', combined_stokes, hi, overwrite=True)


if __name__ == '__main__':
    # run_flicker(
    #     datestring='20230603', timestring='073616',
    #     offset_1_y=0, offset_1_x=0, offset_2_y=2, offset_2_x=15,
    #     create_files=True
    # )
    # run_flicker(
    #     datestring='20230603', timestring='092458',
    #     offset_1_y=0, offset_1_x=0, offset_2_y=2, offset_2_x=15,
    #     create_files=True
    # )
    # run_flicker(
    #     datestring='20230601', timestring='081014',
    #     offset_1_y=0, offset_1_x=0, offset_2_y=2, offset_2_x=14,
    #     create_files=True
    # )
    # run_flicker(
    #     datestring='20230527', timestring='074428',
    #     offset_1_y=0, offset_1_x=0, offset_2_y=1, offset_2_x=16,
    #     create_files=True
    # )
    # run_hmi_flicker(
    #     datestring='20230603', timestring='073616',
    #     hmi_cont_file='hmi.Ic_720s.20230603_043600_TAI.3.continuum.fits',
    #     hmi_mag_file='hmi.M_720s.20230603_043600_TAI.3.magnetogram.fits',
    #     angle=-16.5,
    #     offset_x=80, offset_y=58,
    #     init_x=-397,
    #     init_y=-244
    # )

    # run_hmi_flicker(
    #     datestring='20230603', timestring='092458',
    #     hmi_cont_file='hmi.Ic_720s.20230603_063600_TAI.3.continuum.fits',
    #     hmi_mag_file='hmi.M_720s.20230603_063600_TAI.3.magnetogram.fits',
    #     angle=-16,
    #     offset_x=76, offset_y=66,
    #     init_x=-380,
    #     init_y=-244
    # )

    # run_hmi_flicker(
    #     datestring='20230601', timestring='081014',
    #     hmi_cont_file='hmi.Ic_720s.20230601_050000_TAI.3.continuum.fits',
    #     hmi_mag_file='hmi.M_720s.20230601_050000_TAI.3.magnetogram.fits',
    #     angle=-20,
    #     offset_x=81, offset_y=66,
    #     init_x=-717,
    #     init_y=-250
    # )

    # run_hmi_flicker(
    #     datestring='20230527', timestring='074428',
    #     hmi_cont_file='hmi.Ic_720s.20230527_044800_TAI.3.continuum.fits',
    #     hmi_mag_file='hmi.Ic_720s.20230527_044800_TAI.3.magnetogram.fits',
    #     angle=-25,
    #     offset_x=97, offset_y=65,
    #     init_x=0,
    #     init_y=-260
    # )

    # calculate_mu(
    #     datestring='20230603',
    #     hmi_cont_file='hmi.Ic_720s.20230603_043600_TAI.3.continuum.fits',
    #     init_x=-397,
    #     init_y=-244
    # )

    # calculate_mu(
    #     datestring='20230603',
    #     hmi_cont_file='hmi.Ic_720s.20230603_063600_TAI.3.continuum.fits',
    #     init_x=-380,
    #     init_y=-244
    # )

    # calculate_mu(
    #     datestring='20230601',
    #     hmi_cont_file='hmi.Ic_720s.20230601_050000_TAI.3.continuum.fits',
    #     init_x=-717,
    #     init_y=-250
    # )
    #
    # calculate_mu(
    #     datestring='20230527',
    #     hmi_cont_file='hmi.Ic_720s.20230527_044800_TAI.3.continuum.fits',
    #     init_x=0,
    #     init_y=-260
    # )

    # get_aia_reference_image(
    #     datestring='20230527', timestring='074428',
    #     aia_file='aia.lev1_euv_12s.2023-05-27T044810Z.171.image_lev1.fits',
    #     angle=-25,
    #     offset_x=97, offset_y=65,
    #     init_x=0,
    #     init_y=-260
    # )

    # get_aia_reference_image(
    #     datestring='20230527', timestring='074428',
    #     aia_file='aia.lev1_uv_24s.2023-05-27T044752Z.1600.image_lev1.fits',
    #     angle=-25,
    #     offset_x=97, offset_y=65,
    #     init_x=0,
    #     init_y=-260
    # )

    # get_aia_reference_image(
    #     datestring='20230527', timestring='074428',
    #     aia_file='aia.lev1_euv_12s.2023-05-27T044802Z.335.image_lev1.fits',
    #     angle=-25,
    #     offset_x=97, offset_y=65,
    #     init_x=0,
    #     init_y=-260
    # )

    # get_aia_reference_image(
    #     datestring='20230527', timestring='074428',
    #     aia_file='hmi.s_720s.20230527_044800_TAI.3.I0.fits',
    #     angle=-25,
    #     offset_x=97, offset_y=65,
    #     init_x=0,
    #     init_y=-260
    # )
    #
    # get_aia_reference_image(
    #     datestring='20230527', timestring='074428',
    #     aia_file='hmi.s_720s.20230527_044800_TAI.3.Q0.fits',
    #     angle=-25,
    #     offset_x=97, offset_y=65,
    #     init_x=0,
    #     init_y=-260
    # )
    #
    # get_aia_reference_image(
    #     datestring='20230527', timestring='074428',
    #     aia_file='hmi.s_720s.20230527_044800_TAI.3.U0.fits',
    #     angle=-25,
    #     offset_x=97, offset_y=65,
    #     init_x=0,
    #     init_y=-260
    # )
    #
    # get_aia_reference_image(
    #     datestring='20230527', timestring='074428',
    #     aia_file='hmi.s_720s.20230527_044800_TAI.3.V0.fits',
    #     angle=-25,
    #     offset_x=97, offset_y=65,
    #     init_x=0,
    #     init_y=-260
    # )
    #
    # get_aia_reference_image(
    #     datestring='20230527', timestring='074428',
    #     aia_file='hmi.s_720s.20230527_044800_TAI.3.I1.fits',
    #     angle=-25,
    #     offset_x=97, offset_y=65,
    #     init_x=0,
    #     init_y=-260
    # )
    #
    # get_aia_reference_image(
    #     datestring='20230527', timestring='074428',
    #     aia_file='hmi.s_720s.20230527_044800_TAI.3.Q1.fits',
    #     angle=-25,
    #     offset_x=97, offset_y=65,
    #     init_x=0,
    #     init_y=-260
    # )
    #
    # get_aia_reference_image(
    #     datestring='20230527', timestring='074428',
    #     aia_file='hmi.s_720s.20230527_044800_TAI.3.U1.fits',
    #     angle=-25,
    #     offset_x=97, offset_y=65,
    #     init_x=0,
    #     init_y=-260
    # )
    #
    # get_aia_reference_image(
    #     datestring='20230527', timestring='074428',
    #     aia_file='hmi.s_720s.20230527_044800_TAI.3.V1.fits',
    #     angle=-25,
    #     offset_x=97, offset_y=65,
    #     init_x=0,
    #     init_y=-260
    # )
    #
    # get_aia_reference_image(
    #     datestring='20230527', timestring='074428',
    #     aia_file='hmi.s_720s.20230527_044800_TAI.3.I2.fits',
    #     angle=-25,
    #     offset_x=97, offset_y=65,
    #     init_x=0,
    #     init_y=-260
    # )
    #
    # get_aia_reference_image(
    #     datestring='20230527', timestring='074428',
    #     aia_file='hmi.s_720s.20230527_044800_TAI.3.Q2.fits',
    #     angle=-25,
    #     offset_x=97, offset_y=65,
    #     init_x=0,
    #     init_y=-260
    # )
    #
    # get_aia_reference_image(
    #     datestring='20230527', timestring='074428',
    #     aia_file='hmi.s_720s.20230527_044800_TAI.3.U2.fits',
    #     angle=-25,
    #     offset_x=97, offset_y=65,
    #     init_x=0,
    #     init_y=-260
    # )
    #
    # get_aia_reference_image(
    #     datestring='20230527', timestring='074428',
    #     aia_file='hmi.s_720s.20230527_044800_TAI.3.V2.fits',
    #     angle=-25,
    #     offset_x=97, offset_y=65,
    #     init_x=0,
    #     init_y=-260
    # )
    #
    # get_aia_reference_image(
    #     datestring='20230527', timestring='074428',
    #     aia_file='hmi.s_720s.20230527_044800_TAI.3.I3.fits',
    #     angle=-25,
    #     offset_x=97, offset_y=65,
    #     init_x=0,
    #     init_y=-260
    # )
    #
    # get_aia_reference_image(
    #     datestring='20230527', timestring='074428',
    #     aia_file='hmi.s_720s.20230527_044800_TAI.3.Q3.fits',
    #     angle=-25,
    #     offset_x=97, offset_y=65,
    #     init_x=0,
    #     init_y=-260
    # )
    #
    # get_aia_reference_image(
    #     datestring='20230527', timestring='074428',
    #     aia_file='hmi.s_720s.20230527_044800_TAI.3.U3.fits',
    #     angle=-25,
    #     offset_x=97, offset_y=65,
    #     init_x=0,
    #     init_y=-260
    # )
    #
    # get_aia_reference_image(
    #     datestring='20230527', timestring='074428',
    #     aia_file='hmi.s_720s.20230527_044800_TAI.3.V3.fits',
    #     angle=-25,
    #     offset_x=97, offset_y=65,
    #     init_x=0,
    #     init_y=-260
    # )
    #
    # get_aia_reference_image(
    #     datestring='20230527', timestring='074428',
    #     aia_file='hmi.s_720s.20230527_044800_TAI.3.I4.fits',
    #     angle=-25,
    #     offset_x=97, offset_y=65,
    #     init_x=0,
    #     init_y=-260
    # )
    #
    # get_aia_reference_image(
    #     datestring='20230527', timestring='074428',
    #     aia_file='hmi.s_720s.20230527_044800_TAI.3.Q4.fits',
    #     angle=-25,
    #     offset_x=97, offset_y=65,
    #     init_x=0,
    #     init_y=-260
    # )
    #
    # get_aia_reference_image(
    #     datestring='20230527', timestring='074428',
    #     aia_file='hmi.s_720s.20230527_044800_TAI.3.U4.fits',
    #     angle=-25,
    #     offset_x=97, offset_y=65,
    #     init_x=0,
    #     init_y=-260
    # )
    #
    # get_aia_reference_image(
    #     datestring='20230527', timestring='074428',
    #     aia_file='hmi.s_720s.20230527_044800_TAI.3.V4.fits',
    #     angle=-25,
    #     offset_x=97, offset_y=65,
    #     init_x=0,
    #     init_y=-260
    # )
    #
    # get_aia_reference_image(
    #     datestring='20230527', timestring='074428',
    #     aia_file='hmi.s_720s.20230527_044800_TAI.3.I5.fits',
    #     angle=-25,
    #     offset_x=97, offset_y=65,
    #     init_x=0,
    #     init_y=-260
    # )
    #
    # get_aia_reference_image(
    #     datestring='20230527', timestring='074428',
    #     aia_file='hmi.s_720s.20230527_044800_TAI.3.Q5.fits',
    #     angle=-25,
    #     offset_x=97, offset_y=65,
    #     init_x=0,
    #     init_y=-260
    # )
    #
    # get_aia_reference_image(
    #     datestring='20230527', timestring='074428',
    #     aia_file='hmi.s_720s.20230527_044800_TAI.3.U5.fits',
    #     angle=-25,
    #     offset_x=97, offset_y=65,
    #     init_x=0,
    #     init_y=-260
    # )
    #
    # get_aia_reference_image(
    #     datestring='20230527', timestring='074428',
    #     aia_file='hmi.s_720s.20230527_044800_TAI.3.V5.fits',
    #     angle=-25,
    #     offset_x=97, offset_y=65,
    #     init_x=0,
    #     init_y=-260
    # )

    # merge_hmi_stokes_files()

    # merge_hmi_stokes_files_full_fov()

    get_aia_reference_image(
        datestring='20230527', timestring='074428',
        aia_file='hmi.V_720s.20230527_044800_TAI.3.Dopplergram.fits',
        angle=-25,
        offset_x=97, offset_y=65,
        init_x=0,
        init_y=-260
    )
