import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps
from os.path import realpath, dirname

full_path = realpath(__file__)
dir_path = dirname(full_path)

detectors = {'aLigo':(dir_path+'/../detector/aligo_psd.dat',2/5),\
             'CE':(dir_path+'/../detector/ce_psd.dat',2/5),\
             'ET':(dir_path+'/../detector/etd_psd.dat',2/5*np.sqrt(3/2)),\
             'lisa':(dir_path+'/../detector/lisa_psd.dat',2/5*np.sqrt(5))}

class Fisher():
    def __init__(self,signal,integration_method=simps,\
            psd=None,detector=None):
        self.signal = signal
        self.integration_method = simps
        if detector:
            psd, norm = detectors[detector]
            self.load_psd(psd,norm)
        elif psd:
            self.load_psd(psd)
        else:
            self.psd = lambda x: 1
        
    def SNR(self,fmin=None,fmax=None,nbins=1e5):
        if not fmin:
            fmin = self.fmin
        if not fmax:
            fmax = self.fmax
        x = np.linspace(fmin,fmax,int(nbins))
        y = 4*np.abs(self.signal(x))**2/self.psd(x)
        out = self.integration_method(y,x)
        return np.sqrt(out)
    
    def load_psd(self,psd_name,norm=1):
        s = np.genfromtxt(psd_name).T
        self.fmin = s[0].min()
        self.fmax = s[0].max()
        self.psd = interp1d(s[0],s[1]/norm**2)
        return None
    
    def FisherMatrix(self,fmin=None,fmax=None,nbins=1e5,keys=None):
        if not keys:
            keys = self.signal.keys
        self.keys = keys
        self.signal.evaluate_Nabla(keys=keys)
        dim = len(keys)
        if not fmin:
            fmin = self.fmin
        if not fmax:
            fmax = self.fmax
        self.snr = self.SNR(fmin,fmax,nbins=nbins)
        self.fm = np.zeros((dim,dim))
        derivatives = list(self.signal.Nabla.values())
        f = np.linspace(fmin,fmax,int(nbins))
        for i in range(dim):
            for j in range(i,dim):
                y = 4*np.real(derivatives[i](f)*np.conj(derivatives[j](f)))/self.psd(f)
                self.fm[i,j] = self.integration_method(y,f)
                self.fm[j,i] = self.fm[i,j]
        return None
    
    def CovarianceMatrix(self):
        ifm = np.matrix(self.fm).I
        self.cov = np.zeros_like(ifm)
        dim = len(self.cov)
        for i in range(dim):
            for j in range(i,dim):
                self.cov[i,j] = ifm[i,j]/np.sqrt(ifm[i,i]*ifm[j,j])
                self.cov[j,i] = self.cov[i,j]
            self.cov[i,i] = np.sqrt(ifm[i,i])
        self.sigma = {self.keys[i]:self.cov[i,i] for i in range(dim)}
        #self.sigma_rel = {k:v/self.signal.__dict__[k]*100 for (k,v) in self.sigma.items()}
        return None
