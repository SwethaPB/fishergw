import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps
from os.path import realpath, dirname

full_path = realpath(__file__)
dir_path = dirname(full_path)

class Fisher():
    """
    An object to load the power spectral density (PSD) and compute the signal-to-noise ratio (SNR) and the Fisher matrix elements of a gravitational wave signal. The SNR and the Fisher matrix are averaged over orientation and inclination angles.
    
    Attributes:
        signal : Waveform object.

        integration_method : An integration method from the scipy.integrate module.

        psd : Interpolant of the PSD.

        fmin : If ``psd_name`` is provided, ``fmin`` is the corresponding minimum frequency. Otherwise, ``fmin`` = ``None``.

        fmax : If ``psd_name`` is provided, ``fmax`` is the corresponding maximum frequency. Otherwise, ``fmax`` = ``None``.

        Qavg : Angle-averaging factor. When the ``self.load_psd`` is called, the PSD is divided by ``Qavg`` **2 to ensure that the SNR and the Fisher matrix elements are angle-averaged.
    
        keys : List of the independent variables w.r.t. which the Fisher matrix is evaluated.

        _detectors_ : Dictionary mapping default detector names to their ``psd_name`` and ``Qavg`` factor. Default detectors are Advanced Ligo ('aLigo'), Cosmic Explorer ('CE'), Einstein Telescope ('ET') and LISA ('lisa').
        The following conventions hold for Qavg:
        
            'aLigo' and 'CE' are mapped to the factor ``Qavg`` =2/5 for a two-armed 90-degrees detector (see, e.g., Eq. (7.177) in [1]);
        
            'ET' is mapped to ``Qavg`` =2/5*sqrt(3/2), the additional factor sqrt(3/2) coming from the fact that ET is a three-armed 60-degrees detector with two channels (see Eq. (4) in https://arxiv.org/abs/1012.0908);
        
            'lisa' is mapped to ``Qavg`` =2/sqrt(5). This only accounts for averaging over the inclination angle, because the LISA sensitivity curve is already averaged over orientation and detector channels. (see Eq.s (2,8-9) in https://arxiv.org/abs/1803.01944).

        [1] Maggiore, Michele. Gravitational waves: Volume 1: Theory and experiments. Vol. 1. Oxford university press, 2008.
    """

    _detectors_ = {'aLigo':(dir_path+'/../detector/aligo_psd.dat',2/5),\
                 'CE':(dir_path+'/../detector/ce_psd.dat',2/5),\
                 'ET':(dir_path+'/../detector/etd_psd.dat',2/5*np.sqrt(3/2)),\
                 'lisa':(dir_path+'/../detector/lisa_psd.dat',2/5*np.sqrt(5))}
    
    def __init__(self,signal,integration_method=simps,\
            psd_name=None,detector=None,keys=None):
        """
        :param signal: The waveform object.
        :type signal: TaylorF2

        :param integration_method: The integration method to compute the SNR and the Gisher matrix elements.
        :type integration_method: ``scipy.integrate`` object, default= ``scipy.integrate.simps``

        :param psd_name: The path to a text file with the tabulated PSD. If ``None``, psd defaults to 1.0. If ``detector`` is not ``None``, ``psd_name`` is read from the ``_detectors_`` dictionary
        :type psd_name: filepath or None, optional
            
        :param detector: If not ``None``, must be one of ['aLigo', 'CE', 'ET', 'lisa'].
        :type detector: str or None, optional

        :param keys: Independent variables w.r.t. which the Fisher matrix is evaluated. If ``None``, defaults to ``self.signal.keys``.
        :type keys: list of str or None
        """
        self.signal = signal
        if not keys:
            self.keys = self.signal.keys
        else:
            self.keys = keys
        self.integration_method = simps
        if detector:
            psd_name, Qavg = self._detectors_[detector]
            self.load_psd(psd_name,Qavg)
        elif psd:
            self.Qavg = 1.0
            self.psd, self.fmin, self.fmax = self.load_psd(psd_name)
        else:
            self.Qavg = 1.0
            self.psd = lambda x: 1.0
            self.fmin, self.fmax = None, None
        
    def snr(self,fmin=None,fmax=None,nbins=int(1e5)):
        """
        Returns the SNR of the signal.

        :param fmin: Minimum frequency. If ``None``, defaults to the minimum frequency set by the provided PSD file.
        :type fmax: float or None, optional

        :param fmax: Maximum frequency. If ``None``, defaults to the maximum frequency set by the provided PSD file.
        :type fmax: float or None, optional

        :param nbins: Binning of the integration domain.
        :type nbins: int, default=1e5

        :rtype: float
        """
        if not fmin:
            fmin = self.fmin
        if not fmax:
            fmax = self.fmax
        x = np.linspace(fmin,fmax,int(nbins))
        y = 4*np.abs(self.signal(x))**2/self.psd(x)
        snr = self.integration_method(y,x)
        snr = np.sqrt(snr)
        return snr
    
    def load_psd(self,psd_name,Qavg=1.0):
        """
        Loads the PSD from a text file. Initializes the attributes ``psd``, ``fmin``, ``fmax`` and ``Qavg``.

        :param psd_name: Path to the PSD text file.
        :type psd_name: filepath

        :param Qavg: Value of the angle-averaging factor. See the description in the Fisher attributes.
        :type Qavg: float, default=1.0

        :returns: ``self``
        :rtype: Fisher
        """
        s = np.genfromtxt(psd_name).T
        self.Qavg = Qavg
        self.fmin = s[0].min()
        self.fmax = s[0].max()
        self.psd = interp1d(s[0],s[1]/Qavg**2)
        return self
    
    def fisher_matrix(self,fmin=None,fmax=None,nbins=int(1e5)):
        """
        Returns the Fisher matrix.
        
        :param fmin: Minimum frequency. If ``None``, defaults to the minimum frequency set by the provided PSD file.
        :type fmax: float or None, optional

        :param fmax: Maximum frequency. If ``None``, defaults to the maximum frequency set by the provided PSD file.
        :type fmax: float or None, optional

        :param nbins: Binning of the integration domain.
        :type nbins: int, default=1e5

        :rtype: ``numpy.array``, shape [len(``keys``),len(``keys``)]
        """
        Nabla = self.signal._evaluate_Nabla_(keys=self.keys)
        dim = len(self.keys)
        if not fmin:
            fmin = self.fmin
        if not fmax:
            fmax = self.fmax
        fm = np.zeros((dim,dim))
        derivatives = list(Nabla.values())
        f = np.linspace(fmin,fmax,int(nbins))
        for i in range(dim):
            for j in range(i,dim):
                y = 4*np.real(derivatives[i](f)*np.conj(derivatives[j](f)))/self.psd(f)
                fm[i,j] = self.integration_method(y,f)
                fm[j,i] = fm[i,j]
        return fm
    
    def covariance_matrix(self,fm):
        """
        Returns the covariance matrix.

        :param fm: The \Fisher matrix.
        :type fm: ``numpy.array``, shape [len(``keys``),len(``keys``)]

        :rtype: ``numpy.array``, shape [len(``keys``),len(``keys``)]
        """
        inverse_fm = np.matrix(fm).I
        cov = np.zeros_like(inverse_fm)
        dim = len(fm)
        for i in range(dim):
            for j in range(i,dim):
                cov[i,j] = inverse_fm[i,j]/np.sqrt(inverse_fm[i,i]*inverse_fm[j,j])
                cov[j,i] = cov[i,j]
        sigma = {self.keys[i]:np.sqrt(inverse_fm[i,i]) for i in range(dim)}
        return cov
    
    def sigma1d(self,fm):
        """
        Returns the standard deviations of the 1D marginalized posteriors.

        :param fm: The \Fisher matrix.
        :type fm: ``numpy.array``, shape [len(``keys``),len(``keys``)]

        :returns: Dictionary mapping ``keys`` to the corresponding standard deviations.
        :rtype: dict
        """
        inverse_fm = np.matrix(fm).I
        dim = len(fm)
        sigma = {self.keys[i]:np.sqrt(inverse_fm[i,i]) for i in range(dim)}
        return sigma

