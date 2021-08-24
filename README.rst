About
-----
A Python package to compute Fisher matrices for gravtational wave models

Installation
------------
Install from folder
    
   $ pip install .

Install from pip

   $ pip install fishergw

Usage of taylorf2
-----------------
    >>> from fishergw.taylorf2 import CompactObject, TaylorF2
    >>> from fishergw.taylorf2 import Fisher
    >>>
    >>> m1, m2 = 1.46, 1.27
    >>> DL = 40
    >>> s1, s2 = 0., 0.
    >>> obj1 = CompactObject(m1,s1,Lamda=199)
    >>> obj2 = CompactObject(m2,s2,Lamda=474)
    >>> signal = TaylorF2(obj1,obj2,DL=DL,redshift=True)
    >>>
    >>> keys=['t_c','phi_c','M_c','eta','Lamda','Lamda_2']
    >>> fisher = Fisher(signal,detector='ET',keys=keys)
    >>> fmin = 5
    >>> fmax = signal.isco(mode='static')
    >>>
    >>> snr = fisher.snr(fmin,fmax,nbins=int(1e4))
    >>> fm = fisher.fisher_matrix(fmin,fmax,nbins=int(1e4))
    >>> cov, sigma = fisher.covariance_matrix(fm)

Usage of cosmology
------------------

    >>> from fishergw.cosmology import redshift_from_distance, distance_from_redshift
    >>> z = redshift_from_distance(100)
    >>> d = distance_from_redshift(z)
