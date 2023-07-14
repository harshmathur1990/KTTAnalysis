from pathlib import Path
import sunpy.io
import numpy as np

def adjust_headers(datepath):

    level2path = datepath / 'Level-2'

    all_files = level2path.glob('**/*')

    all_fits_files = [file for file in all_files if file.name.startswith('v2u') and file.name.endswith('.fits')]

    for fits_file in all_fits_files:
        data, header = sunpy.io.read_file(fits_file)[0]

        wave_type = fits_file.name.split('_')[-2]

        timestring = fits_file.name.split('_')[-1].split('.')[0]

        if wave_type == 'halpha':
            source = datepath / 'total_stokes_{}_DETECTOR_1.fits'.format(timestring)
        else:
            source = datepath / 'total_stokes_{}_DETECTOR_3.fits'.format(timestring)

            if not source.exists():
                source = datepath / 'total_stokes_{}_DETECTOR_2.fits'.format(timestring)

        _, source_header = sunpy.io.read_file(source)[0]

        header['CDELT2'] = source_header['CDELTX']

        header['CDELT3'] = source_header['CDELTY']

        header['CUNIT2'] = 'arcsec'

        header['CUNIT3'] = 'arcsec'

        header['CTYPE2'] = 'HPLT-TAN'

        header['CTYPE3'] = 'HPLN-TAN'

        header['CNAME2'] = 'HPC lat'

        header['CNAME3'] = 'HPC lon'

        sunpy.io.write_file(fits_file, data, header, overwrite=True)


if __name__ == '__main__':
    base_path = Path('/mnt/f/harsh/CourseworkRepo/InstrumentalUncorrectedStokes')

    datestring = '20230519'

    datepath = base_path / datestring

    adjust_headers(datepath)
