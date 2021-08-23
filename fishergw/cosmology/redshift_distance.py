import numpy as np
from scipy.integrate import simps
from scipy.interpolate import interp1d
from os.path import realpath, dirname

full_path = realpath(__file__)
dir_path = dirname(full_path)

from ..constants import speed_of_light, omega_matter, omega_lamda, H0
cc = speed_of_light*1e-3 # Km/s

def luminosity_distance(z):
    integrand = lambda x: 1/np.sqrt(omega_matter*(1+x)**3+omega_lamda)
    dz = 1e-3
    X = np.arange(0,z+dz,dz)
    Y = integrand(X)
    dL = cc/H0*(1+z)*simps(Y,X)
    return dL

## calibrated from redshift 0 to redshift 20
filename = dir_path+'/redshift_distance_table.dat'

try:
    data = np.genfromtxt(filename).T
except:
    print('File %s does not exist!'%filename)
    print('Generating %s...'%filename)
    redshifts = np.linspace(0,20,1000)
    distances = np.array([luminosity_distance(z) for z in redshifts])
    X = np.vstack((redshifts,distances)).T
    np.savetxt(filename,X)
    print('%s generated!'%filename)
    data = np.genfromtxt(filename).T

redshift_from_distance = interp1d(data[1],data[0])
distance_from_redshift = interp1d(data[0],data[1])

