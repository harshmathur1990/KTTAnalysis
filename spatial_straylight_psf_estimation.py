import sys
sys.path.insert(1, '/home/harsh/CourseworkRepo/stic/example')
# sys.path.insert(2, '/home/harsh/CourseworkRepo/WFAComparison')
# sys.path.insert(1, '/mnt/f/Harsh/CourseworkRepo/stic/example')
# sys.path.insert(1, 'F:\\Harsh\\CourseworkRepo\\stic\\example')
import numpy as np
from prepare_data import *


h = 6.62606957e-34
c = 2.99792458e8
kb = 1.380649e-23


def getContSi(lam):
    s = satlas.satlas()
    x,y,c = s.getatlas(lam-0.1,lam+0.1, si=True)
    return np.median(c)


def prepare_planck_function(wavelength_in_angstrom):
    f = c / wavelength_in_angstrom

    def planck_function(temperature_in_kelvin):
        return (2 * h * f**3 / c**2) * (1 / (np.exp((h * f) / (kb * temperature_in_kelvin)) - 1))

    return planck_function


def prepare_temperature_from_intensity(wavelength_in_angstrom):

    f = c / wavelength_in_angstrom

    def get_temperature_from_intensity(intensity_in_si):

        I = intensity_in_si
        return (h * f) / (np.log((2 * h * f**3 / (c**2 * I)) + 1) * kb)

    return get_temperature_from_intensity