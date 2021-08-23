import numpy as np
import sympy as sp
from sympy import Rational
from copy import deepcopy

from ..cosmology.redshift_distance import redshift_from_distance
from ..constants import speed_of_light, solar_mass, G, Mpc

cc = speed_of_light
msun = solar_mass
## Schwarzschild radius of the Sun
rsun = msun*G/cc**2 ## m

## define sympy symbols
eta, delta, M, c = sp.symbols('eta, delta, M, c')
chi_s, chi_a, kappa_s, kappa_a, Lamda_T, delta_Lamda =\
 sp.symbols('chi_s, chi_a, kappa_s, kappa_a, Lamda_T, delta_Lamda')
v, f = sp.symbols('v, f')
tc, phic, DL = sp.symbols('t_c, phi_c, d_L')

class CompactObject():
    """
    A class to define an isolated compact object.

    Attributes
    ----------
    mass : float
        Mass (in units of salr masses).
    spin : float
        Dimensionless spin.
    Lamda : float
        Tidal deformability.
    kappa : float
        Dimensionless quadrupole deformability.

    Notes
    -----
    This class is under development. Lamda can only be specified by hand and kappa is fixed to the black hole value.
    To do list:
    -- support equations of state for neutron stars.
    """
    def __init__(self, mass, spin, Lamda=0.0):
        """
        Parameters
        ----------
        mass : float
            Mass (in units of salr masses).
        spin : float
            Dimensionless spin.
        Lamda : float, default=0.0
            Tidal deformability.
        """
        self.mass = mass
        self.spin = spin
        self.kappa = 1.
        self.Lamda = Lamda


def amplitude_coefficients():
    """
    Returns the amplitude coefficients of the TaylorF2 template.
    Coefficients are expanded in powers of v=(pi*M*f/c)**(1/3) and normalized to 1 for v=0.
    Includes terms up to 3PN order (see Eq.s(B14-B20) in https://arxiv.org/abs/1508.07253).

    Returns
    -------
    out : list
        list of sympy symbolic expressions of the PN coefficients.
    """
    ## numerical factors
    pi = sp.pi
    ## spin corrections
    sigma = (8*eta - Rational(81,32))*chi_a**2 + (17*eta/8 - Rational(81,32))*chi_s**2\
            - Rational(81,16)*delta*chi_s*chi_a
    
    gamma = (Rational(285197,16128) - 1579*eta/4032)*delta*chi_a +\
            (Rational(285197,16128) - 15317*eta/672 - 2227*eta**2/1008)*chi_s
    
    xi = (Rational(1614569,64512) - 1873643*eta/16128 + 2167*eta**2/42)*chi_a**2 + \
         (31*pi/12 - 7*pi*eta/3)*chi_s +\
         (Rational(1614569,64512) - 61392*eta/1344 + 57451*eta**2/4032)*chi_s**2 +\
         (31*pi/12 + (Rational(1614569,32256) - 165961*eta/2688)*chi_s)*delta*chi_a
    
    ## output
    out = []
    out.append(1)
    out.append(0)
    out.append(-Rational(323,224) + 451*eta/168)
    out.append((27*delta*chi_a/8 + (Rational(27,8) - 11*eta/6)*chi_s))
    out.append(-Rational(27312085,8128512) - 1975055*eta/338688 + 105271*eta**2/24192 + sigma)
    out.append(-85*pi/64 + 85*pi*eta/16 + gamma)
    out.append(-Rational(177520268561,8583708672) + (Rational(545384828789,5007163392)\
             - 205*pi**2/48)*eta - 3248849057*eta**2/178827264 + 34473079*eta**3/6386688 + xi)
    return out


def phase_coefficients():
    """
    Returns the phase coefficients of the TaylorF2 template.
    Coefficients are expanded in powers of v=(pi*M*f/c)**(1/3) and normalized to 1 for v=0.
    Includes point-particle and spin-induced terms up to 3.5PN order (see Eq.s(B6-B13) in https://arxiv.org/abs/1508.07253 and Eq.s(0.5a-c) in https://arxiv.org/abs/1701.06318).

    Returns
    -------
    out : list
        list of sympy symbolic expressions of the PN coefficients.

    Notes
    -----
    The 3.5PN spin-induced term neglects the quadrupole and octupole moment corrections from Eq.(0.5c) in https://arxiv.org/pdf/1701.06318.pdf. Quadrupole corrections are instead included in the 2PN and 3PN terms, as per Eq.s(0.5a-b). 
    """
    ## numerical factors
    pi = sp.pi
    gE = sp.EulerGamma
    ## spin corrections
    beta = Rational(113,3)*delta*chi_a + (113-76*eta)/3*chi_s

    sigma = - 5*chi_s**2/8*(1 + 156*eta + 80*delta*kappa_a + 80*(1-2*eta)*kappa_s)\
        - 5*chi_a**2/8*((1-160*eta) + 80*delta*kappa_a + 80*(1-2*eta)*kappa_s)\
        - 5*chi_s*chi_a/4*(delta + 80*(1-2*eta)*kappa_a + 80*delta*kappa_s)

    gamma = chi_a*delta*(-Rational(732985,2268) - 140*eta/9)\
            + chi_s*(-Rational(732985,2268) + 24260*eta/81 + 340*eta**2/9)

    xi = sp.pi*( 2270*chi_a*delta/3 + chi_s*(Rational(2270,3) - 520*eta) )\
         + chi_s**2*( -Rational(1344475,2016) + 829705*eta/504\
        + 3415*eta**2/9\
        + delta*kappa_a*(Rational(26015,28) - 1495*eta/6)\
        + kappa_s*(Rational(26015,28) - 44255*eta/21 - 240*eta**2) )\
        + chi_a**2*( -Rational(1344475,2016) + 267815*eta/252 - 240*eta**2\
        + delta*kappa_a*(Rational(26015,28) - 1495*eta/6)\
        + kappa_s*(Rational(26015,28) - 44255*eta/21 - 240*eta**2) )\
        + chi_s*chi_a*( kappa_a*(Rational(26015,14) - 88510*eta/21 - 480*eta**2)\
        + delta*(-Rational(1344475,1008) + 745*eta/18\
        + kappa_s*(Rational(26015,14) - 1495*eta/3)) )

    zeta = delta*chi_a*( -Rational(25150083775,3048192) + 26804935*eta/6048\
        - 1985*eta**2/48 ) + chi_s*( -Rational(25150083775,3048192)\
        + 10566655595*eta/762048 - 1042165*eta**2/3024 + 5345*eta**3/36 )\
        + chi_s**3*( Rational(265,24) + 4035*eta/2 - 20*eta**2/3\
        + (Rational(3110,3) - 10250*eta/3 + 40*eta**2)\
        - 440*(1-3*eta) )\
        + chi_a**3*( delta*(Rational(265,24) - 2070*eta\
        + (Rational(3110,3) - 750*eta) - 440*(1-eta)) )\
        + chi_s**2*chi_a*( delta*(Rational(265,8) + 12055*eta/6\
        + (3110 - 10310*eta/3) - 1320*(1-eta)) )\
        + chi_s*chi_a**2*( Rational(265,8) - 6500*eta/3 + 40*eta**2\
        + (3110 - 27190*eta/3 + 40*eta**2)\
        - 1320*(1-3*eta) )
    
    ## output
    out = []
    out.append(1.)
    out.append(0.)
    out.append(Rational(3715,756) + 55*eta/9)
    ## leading SO
    out.append(-16*pi + beta)
    ## leading SS
    out.append(Rational(15293365,508032) + 27145*eta/504 + 3085*eta**2/72 + sigma)
    out.append((38645*pi/756 - 65*pi*eta/9 + gamma)*(1+3*sp.log(v)))
    out.append(Rational(11583231236531,4694215680) - 6848*gE/21 -\
        640*pi**2/3 + (2255*pi**2/12 - Rational(15737765635,3048192))*eta +\
        76055*eta**2/1728 - 127825*eta**3/1296 + xi - Rational(6848,63)*sp.log(64*v**3))
    out.append( 77096675*pi/254016 + 378515*pi/1512*eta\
        -74045*pi/756*eta**2 + zeta )
    return out


class TaylorF2():
    """
    TaylorF2 frequency-domain template for the inspiral waveform from a binary coalescence.

    Attributes
    ----------
    d_L : float, default=100.0
        Luminosity distance in units of Mpc.

    tc : float, default=0.0
        Time of coalescence.

    phic : float, default=0.0
        Phase of coalescence.

    M : float
        Total mass in the detector frame, in units of m.
    
    kappa_s : float
        Symmetrized quadrupole moment.
    
    kappa_a : float
        Antisymmetrized quadrupole moment.
    
    eta : float
        Symmetric mass ratio.
    
    M_c : float
        Chirp mass in the detector frame, in units of m.
    
    q : float
        Mass ratio obj1.mass/obj2.mass.
    
    chi_s : float
        Symmetrized dimensionless spin.
    
    chi_a : float
        Antisymmetrized dimensionless spin.
    
    Lamda_T : float
        Tidal deformability of the binary, according to Eq.(14) in https://arxiv.org/abs/1410.8866.

    delta_Lambda : float
        Auxiliary tidal parameter, according to Eq.(15) in https://arxiv.org/abs/1410.8866.
    
    keys : list of str, default=['t_c','phi_c','M_c','eta','chi_s','chi_a']
        Independent variables w.r.t. which the Fisher matrix is evaluated. If Lambda_T is not zero, ['Lamda_T','delta_Lamda'] are added to keys.

    Notes
    -----
    Because the TaylorF2 phase is linear in t_c and phi_c, the actual values of t_c and phi_c are irrelevant to the computation of the Fisher matrix and can be left to their default.

    TaylorF2 assumes that the normalization of Eq.(7.177) in [1], without the angular factor Q.
    
    [1] Maggiore, Michele. Gravitational waves: Volume 1: Theory and experiments. Vol. 1. Oxford university press, 2008.
    """

    def __init__(self,obj1,obj2,d_L=100.0,t_c=0.0,phi_c=0.0,redshift=False):
        """
        Parameters
        ----------
        obj1 : CompactObject instance
            Primary compact object in the binary.

        obj2 : CompactObject instance
            Secondary compact object in the binary.

        d_L : float, default=100.0
            Luminosity distance in units of Mpc.

        tc : float, default=0.0
            Time of coalescence.

        phic : float, default=0.0
            Phase of coalescence.

        redshift : bool, default=False
            If ``True``, the masses are redshifted, otherwise the redshift is neglected.
        """
        if redshift:
            self.redshift = redshift_from_distance(d_L)
        else:
            self.redshift = 0.0
        self.keys = ['t_c','phi_c','M_c','eta','chi_s','chi_a']
        self.__dict__['t_c'] = t_c
        self.__dict__['phi_c'] = phi_c
        self.__dict__['d_L'] = d_L*Mpc
        self.__dict__['M'] = (obj1.mass + obj2.mass)*rsun*(1+self.redshift)
        self.__dict__['kappa_s'] = 0.5*(obj1.kappa + obj2.kappa)
        self.__dict__['kappa_a'] = 0.5*(obj1.kappa - obj2.kappa)
        self.__dict__['eta'] = obj1.mass*obj2.mass/(obj1.mass + obj2.mass)**2
        #self.__dict__['mu'] = self.__dict__['eta']*self.__dict__['M']
        self.__dict__['M_c'] = self.__dict__['M']*self.__dict__['eta']**0.6*(1+self.redshift)
        #self.__dict__['delta'] = (obj1.mass - obj2.mass)/(obj1.mass + obj2.mass)
        self.__dict__['q'] = obj1.mass / obj2.mass
        self.__dict__['chi_s'] = 0.5*(obj1.spin +  obj2.spin)
        self.__dict__['chi_a'] = 0.5*(obj1.spin - obj2.spin)
        self.__dict__['Lamda_T'] = 16/13*((1+12/self.__dict__['q'])*(obj1.mass**5)*obj1.Lamda +\
                (1+12*self.__dict__['q'])*(obj2.mass**5)*obj2.Lamda)/(obj1.mass+obj2.mass)**5
        self.__dict__['delta_Lamda'] = 16/4361*( (-919+3179*(1+self.__dict__['q'])/self.__dict__['q']-\
                2286*self.__dict__['q']/(1+self.__dict__['q'])+\
                260*(self.__dict__['q']/(1+self.__dict__['q']))**2)*(obj1.mass**5)*obj1.Lamda +\
                (-919+3179*(1+self.__dict__['q'])-2286/(1+self.__dict__['q'])+\
                260/(1+self.__dict__['q'])**2)*(obj2.mass**5)*obj2.Lamda )/(obj1.mass+obj2.mass)**5
        if self.__dict__['Lamda_T']:
            self._tidal_ = True
            self.keys += ['Lamda_T','delta_Lamda']
        else:
            self._tidal_ = False
        self._eval_ = False

    def isco(self,mode='static'):
        """
        Compute the ISCO frequency of the system. It is the recommended maximum frequency when computing the SNR and the Fisher matrix.

        Parameters
        ----------
        mode : str, default='static'
            If ``static``, neglects the contribution of the indivudal spins.

        Returns
        -------
        fmax : float
            The ISCO frequency.

        Notes
        -----
        This method is under development. The only supported option so far is the ISCO frequency of a lack hole binary.
        To do list:
        -- add a ``spinning`` mode to compute the ISCO frequency without neglecting the spins;
        -- implement the contact frequency in case one or both compact objects have a tidally deformable surface.
        """
        if mode == 'static':
            fmax = cc/(self.M*6**1.5*np.pi)
        return fmax

    def start_frequency_from_obs_time(self,obs_time=1.0):
        """
        Returns the starting frequency given the osbervational time, as per Eq.(2.15) in https://arxiv.org/abs/gr-qc/0411129v2.
        
        Parameters
        ----------
        obs_time : float, default=1
            The observational time (in units of yr).

        Returns
        -------
        fmin : float
            The starting frequency.
        """
        fmin = 4.149e-5*(obs_time)**(-3/8)*(self.M_c*1e-6)**(-5/8)
        return fmin

    def __call__(self,f):
        """
        Value of the strain at a given frequency.

        Parameters
        ----------
        f : float
            Frequency (in units of Hz).

        Returns
        -------
        strain : complex
            Value of the strain at f.
        """
        if not self._eval_:
            ## update eval
            self._eval_ = True
            params = {k:self.__dict__[k] for k in self.keys}
            ## phase
            self.phase_eval = self._phase_().subs(params)
            self.phase_eval = sp.lambdify('f',self.phase_eval,modules='numpy')
            ## amplitude
            self.amplitude_eval = self._amplitude_().subs(params)
            self.amplitude_eval = sp.lambdify('f',self.amplitude_eval,modules='numpy')
        amplitude = self.amplitude_eval(f)
        phase = self.phase_eval(f)
        return amplitude*np.exp(1j*phase)

    def _evaluate_Nabla_(self,keys=None):
        """
        Computes the derivatives w.r.t. the arguments in keys.

        Parameters
        ----------
        keys : list of str or None
            Independent variables w.r.t. which the Fisher matrix is evaluated. If ``None``, the default self.keys is assumed.
        
        Returns
        -------
        Nabla : dict
            Argument names mapped to their derivative estimators.
        
        """
        Nabla = {}
        if not keys:
            keys = self.keys
        for argument in keys:
            Nabla[argument] = self._diff_(argument)
        return Nabla

    def _diff_(self,argument):
        """
        Derivative of the strain w.r.t. to the argument.

        Parameters
        ----------
        argument : str
            Variable w.r.t. which the differential is evaluated.

        Returns
        -------
        out : lambda function
            Estimator the differential.

        Notes
        -----
        Because the amplitude is treated as an independent variable, only the phase is differentiated.
        """
        keys = deepcopy(self.keys)
        keys.remove(argument)
        params = {k:self.__dict__[k] for k in keys}
        ## phase
        ph_diff = self._phase_().subs(params)
        ph_diff = sp.diff(ph_diff,argument)
        ph_diff = ph_diff.subs({argument:self.__dict__[argument]})
        ph_diff = sp.lambdify('f',ph_diff,modules='numpy')
        ##
        out = lambda f: self(f)*ph_diff(f)
        return out

    def _amplitude_(self,PN=0):
        """
        Returns a sympy expression for the amplitude in terms of the independent varialbles in self.keys.
        The amplitude is truncated at the specified PN order. 
        """
        cfs = amplitude_coefficients()
        ## restrict to PN order
        out = 0
        for i in range(int(2*PN)+1):
            out += cfs[i]*v**i
        ## add normalization
        out *= sp.sqrt(sp.pi*eta)*v**Rational(-7,2)
        out *= sp.sqrt(Rational(5,24))
        out *= M**2/c/DL
        out = out.subs('v','(pi*M*f/c)**Rational(1,3)')
        ## change variables
        out = out.subs([('M','M_c/(eta)**Rational(3,5)'),('delta','sqrt(1-4*eta)')])
        ## replace numerical physical quantities
        out = out.subs([('c',cc),('d_L',self.__dict__['d_L']),\
                       ('kappa_a',self.__dict__['kappa_a']),\
                       ('kappa_s',self.__dict__['kappa_s'])])
        return out

    def _phase_(self,PN=3.5):
        """
        Returns a sympy expression for the phase in term of the independent variables in self.keys.
        The phase is truncated at the specified PN order.
        If self.tidal=True, tidal terms at 5PN and 6PN are also added.
        """
        cfs = phase_coefficients()
        ## restrict to PN order
        out = 0
        for i in range(int(2*PN)+1):
            out += cfs[i]*v**i
        ## add tidal terms
        if self._tidal_ == True:
            out += -Rational(39,2)*Lamda_T*v**10 -Rational(3115,64)*Lamda_T*v**12 + Rational(6595,364)*delta*delta_Lamda*v**12
        ## add normalization
        out *= 3/(128*eta*v**5)
        ## add constant and linear terms in the phase
        out += -sp.pi/4 + 2*c*tc/M*v**3 - phic
        out = out.subs('v', '(pi*M*f/c)**Rational(1,3)')
        ## change variables
        out = out.subs([('M','M_c/(eta)**Rational(3,5)'),('delta','sqrt(1-4*eta)')])
        ## replace numerical physical quantities
        out = out.subs([('c',cc),('d_L',self.__dict__['d_L']),\
                       ('kappa_a',self.__dict__['kappa_a']),\
                       ('kappa_s',self.__dict__['kappa_s'])])
        return out
