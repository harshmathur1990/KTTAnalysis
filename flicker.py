import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py
from pathlib import Path

'''
To Call this method:

Run ipython:
import sunpy.io.fits
from flicker import flicker
data, header = sunpy.io.fits.read('sunspot.fits')[1]
flicker(data[0], data[1])
'''


def flicker(image1, image2, rate=1, animation_path=None, limits=None):

    if limits is None:
        limits = [[None, None], [None, None]]

    plt.close('all')

    image1 = image1.copy().astype(np.float64)

    image2 = image2.copy().astype(np.float64)

    image1 = image1 / np.nanmax(image1)

    image2 = image2 / np.nanmax(image2)

    final_image_1 = np.zeros(
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
    )

    final_image_2 = np.zeros(
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
    )

    final_image_1[0: image1.shape[0], 0: image1.shape[1]] = image1

    final_image_2[0: image2.shape[0], 0: image2.shape[1]] = image2

    imagelist = [final_image_1, final_image_2]

    rate = rate * 1000

    fig = plt.figure()  # make figure

    im = plt.imshow(
        imagelist[0],
        origin='lower',
        cmap='gray',
        interpolation='none'
    )

    im.set_clim(limits[0][0], limits[0][1])
    # function to update figure
    def updatefig(j):
        # set the data in the axesimage object
        im.set_array(imagelist[j])
        im.set_clim(limits[j][0], limits[j][1])
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


if __name__ == '__main__':
    base_path = Path('/home/harsh/CourseworkRepo/InstrumentalUncorrectedStokes')

    datestring = '20230603'

    level3path = base_path / datestring / 'Level-3'

    file1 = h5py.File(level3path / 'Halpha_20230603_092458_stic_profiles.nc', 'r')
    file2 = h5py.File(level3path / 'CaII8662_20230603_092458_stic_profiles.nc', 'r')

    flicker(file1['profiles'][0, :, 13:, 100, 0], file2['profiles'][0, :, :-3, 100, 0])

