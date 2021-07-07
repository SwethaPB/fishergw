TaylorF2 bibliography
---------------------

The ``TaylorF2`` template in this module restricts to the (l=2,m=2) harmonic. 

The expressions for the amplitude and phase of the waveform are extracted from the following papers [*]_

Point particle terms up to 3.5PN in the phase
  Eq.s (B6-13) in `1508.07253 <https://arxiv.org/abs/1508.07253>`_
Spin-induced terms up to 3.5PN in the phase [*]_
  Eq.s (0.5a-c) in `1701.06318 <https://arxiv.org/abs/1701.06318>`_
Terms up to 3.5PN in the amplitude
  Eq.s (B14-20) in  `1508.07253 <https://arxiv.org/abs/1508.07253>`_
Tidal terms at 5PN and 6PN in the phase
  Eq.s (14-16) in `1410.8866 <https://arxiv.org/abs/1410.8866>`_

.. [*] These papers were not necessarily the first to present the corresponding expressions. We just refer to them for simplicity.
.. [*] We included quadrupole corrections at 3PN, but we neglected quadrupole and octupole corrections at 3.5PN.

Normalization conventions
-------------------------
The ``taylorf2.waveform.TaylorF2`` assumes that the normalization of Eq. (7.177) in [1]_, without the angular factor Q.

When ``taylorf2.fisher.Fisher`` calls a given PSD, it divides it by a factor of Q^2, so as to account for angle averaging.

However, the conventions about Q change with the PSD, depending on: the design experiment (e.g., the number of arms in the detector and their relative orientation); whether the PSD has been preprocessed with a transfer function; and so on.

Here below we motivate our choices of Q.

aligo_psd
  The standard factor Q=2/5 for a 2-arms 90-degrees detectors is assumed. See Eq. (7.180) in [1]_.
etd_psd
  The factor Q=2/5 is multiplied by an additional sqrt(3/2) to account for the fact that ET is a 3-arms 60-degrees detector with two main channels. See Eq. (4) in [2]_.
lisa_psd
  The factor Q=2/sqrt(5) only accounts for the inclination angle. The PSD has been already preprocessed with a transfer function, thus accounting for orientation averaging. See Eq.s (2), (8-9) in [3]_.
ce_psd
  same as for *aligo_psd*.

.. [1] Maggiore, Michele. Gravitational waves: Volume 1: Theory and experiments. Vol. 1. Oxford university press, 2008.
.. [2] `1012.0908 <https://arxiv.org/abs/1012.0908>`_
.. [3] `1803.01944 <https://arxiv.org/pdf/1803.01944.pdf>`_

