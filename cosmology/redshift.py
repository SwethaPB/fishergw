import numpy as np
from scipy.integrate import simps
from scipy.interpolate import interp1d
from os.path import realpath, dirname

full_path = realpath(__file__)
dir_path = dirname(full_path)

def Ldist_(z):
    ## comsological parameteres from
    ## Planck 2018 results VI
    ## https://arxiv.org/abs/1807.06209
    OM = 0.31
    OLamda = 0.69
    H0 = 68.
    integrand = lambda x: 1/np.sqrt(OM*(1+x)**3+OLamda)
    dz = 1e-3
    X = np.arange(0,z+dz,dz)
    Y = integrand(X)
    cc = 299792.458
    out = cc/H0*(1+z)*simps(Y,X)
    return out

## calibrated from redshift 0 to redshift 20
data = np.genfromtxt(dir_path+'/redshift_distance_table.dat').T
redshift_from_distance = interp1d(data[1],data[0])
distance_from_redshift = interp1d(data[0],data[1])

