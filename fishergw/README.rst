fishergw
---
Usage of taylorf2::
>>>from fishergw.taylorf2.waveform import CompactObject, TaylorF2
>>>from fishergw.taylorf2.fisher import Fisher
>>>m1, m2 = 1.46, 1.27
>>>DL = 40
>>>s1, s2 = 0., 0.
>>obj1 = CompactObject(m1,s1,Lamda=199)
>>obj2 = CompactObject(m2,s2,Lamda=474)
>>signal = TaylorF2(obj1,obj2,DL=DL)
>>>fisher = Fisher(signal,detector='ET')
>>>fmin = 5
>>>fmax = signal.ISCO(mode='static')
>>>snr = fisher.SNR(fmin,fmax,nbins=int(1e4))
>>>keys=['t_c','phi_c','M_c','eta','Lamda','Lamda_2']
>>>fisher.FisherMatrix(fmin,fmax,nbins=int(1e4),keys=keys)
>>>fisher.CovarianceMatrix()
>>>sigma = fisher.sigma
