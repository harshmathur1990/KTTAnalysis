import sys
sys.path.insert(1, 'F:\\Harsh\\CourseworkRepo\\RH\\python')
sys.path.insert(2, '/home/harsh/CourseworkRepo/rh/RH-uitenbroek/python')
import rhanalyze
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def compare_with_atlas():

    wd = Path(
        'F:\\Harsh\\CourseworkRepo\\RH\\rhf1d\\run'
    )

    os.chdir(wd)

    out = rhanalyze.rhout()

    waves = out.spectrum.waves

    waves *= 10

    ind = np.where((waves >= 8660) & (waves <= 8664))[0]

    catalog = np.loadtxt('F:\\Harsh\\CourseworkRepo\\KTTAnalysis\\catalog_8662.txt')

    ind_c = np.where((catalog[:, 0] >= 8660) & (catalog[:, 0] <= 8664))[0]

    plt.plot(waves[ind], out.rays[0].I[ind] / out.rays[0].I[ind][0], color='navy')

    plt.plot(catalog[ind_c, 0], catalog[ind_c, 1] / catalog[ind_c, 1][0], color='gray', linestyle='--')

    plt.show()


def create_falc_for_inversions():
    base_path = Path('/home/harsh/CourseworkRepo/rh/RH-uitenbroek/rhf1d/run')

    cwd = os.getcwd()

    os.chdir(base_path)

    out = rhanalyze.rhout()

    os.chdir(cwd)

    waves = out.spectrum.waves

    waves *= 10

    ind = list()



if __name__ == '__main__':
    compare_with_atlas()
