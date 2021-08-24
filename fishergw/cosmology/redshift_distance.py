import numpy as np
from scipy.integrate import simps
from scipy.optimize import root_scalar

from ..constants import speed_of_light, omega_matter, omega_lamda, H0
cc = speed_of_light*1e-3 # Km/s

def distance_from_redshift(z):
    """
    Returns the luminosity distance d_L from the redshift z.
   
    Uses the Planck 2018 cosmological parameters from Tab. 1 in https://arxiv.org/abs/1807.06209.

    Parameters:
        z : float
            Redshift.

    Returns:
        d_L : float
            Luminosity distance (in units of Mpc).
    """
    integrand = lambda x: 1/np.sqrt(omega_matter*(1+x)**3+omega_lamda)
    dz = 1e-3
    X = np.arange(0,z+dz,dz)
    Y = integrand(X)
    d_L = cc/H0*(1+z)*simps(Y,X)
    return d_L

def redshift_from_distance(d_L):
    """
    Returns the redshift z from the luminosity distance d_L.

    Uses the Planck 2018 cosmological parameters from Tab. 1 in https://arxiv.org/abs/1807.06209.

    Parameters:
        d_L : float
            Luminosity distance (in units of Mpc). Must be less than 231518.

    Returns:
        z : float
            Redshift.

    Notes:
        A solution for z is searched in the interval [0,20]. This restricts the allowed values of d_L in [0,231518] Mpc.
    """
    if d_L>231518:
        raise ValueError('d_L must be less than 231518 Mpc!')
    f = lambda x: d_L - distance_from_redshift(x)
    z = root_scalar(f,method='bisect',bracket=[0,20])
    return z.root
