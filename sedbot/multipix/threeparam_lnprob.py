#!/usr/bin/env python
# encoding: utf-8
"""
Three-parameter star formation history for the multi-pixel model.

Instantiate one posterior function per compute core.
"""

import numpy as np

import fsps

from sedbot.photconv import abs_ab_mag_to_mjy
from sedbot.zinterp import bracket_logz, interp_logz


class ThreeParamLnProb(object):
    """
    FIXME add addition of background.

    Parameters in theta (pixel-local) space are:

    0. logmass
    1. logZZsol
    2. logtau
    3. const
    4. sf_start
    5. dust1
    6. dust2

    Parameters in phi (global) are:

    0. d
    """
    def __init__(self, fsps_params, obs_bands,
                 obs_sed=None, obs_err=None, priors=None,
                 fsps_compute_bands=None):
        super(ThreeParamLnProb, self).__init__()
        self._sp = fsps.StellarPopulation(**fsps_params)
        self._bands = obs_bands
        self._obs_sed = obs_sed
        self._obs_err = obs_err
        self._priors = priors
        self._param_names = ['logmass',
                             'logZZsol',
                             'logtau',
                             'const',
                             'sf_start',
                             'dust1',
                             'dust2']
        # pre-allocate memory
        self._prior_vals = np.empty(self.ndim)
        # Placeholders for metadata at high and low metallicity brackets
        self._meta1 = np.empty(5)
        self._meta2 = np.empty(5)

        # Possible to compute SED in more bands than necessary
        if fsps_compute_bands is None:
            self._compute_bands = self._bands
            self._band_indices = np.ones(len(self._bands))
        else:
            self._compute_bands = fsps_compute_bands
            self._band_indices = np.array([self._compute_bands.index(b)
                                           for b in self._bands])

    @property
    def ndim(self):
        """Number of dimensions in theta (pixel-level) parameter space."""
        return len(self._param_names)

    @property
    def nbands(self):
        """Number of *observed* bandpasses in the likelihood."""
        return len(self._bands)

    @property
    def ndim_phi(self, arg1):
        """Number of global parameters"""
        return 1.

    def reset_pixel(self, obs_sed, obs_err, priors):
        """Change observation and priors for a new pixel."""
        self._obs_sed = obs_sed
        self._obs_err = obs_err
        self._priors = priors

    def __call__(self, theta, B, phi):
        """Compute the ln posterior probability.

        Parameters
        ----------
        theta : ndarray
            The theta parameters, shape ``(n_theta_params,)``.
        B : ndarray
            The background parameters, shape ``(n_bands,)``.
        phi : ndarray
            The phi parameter space, here, just a 1D ndarray with d (parsecs).

        Returns
        -------
        ln_post : float
            The ln-posterior probablity.
        blob : list
            The blob metadata.
        """
        # Physically we'd expect dust1 > dust2 if young stars are more embedded
        if theta[5] < theta[6]:
            return -np.inf, np.nan

        # Evaluate priors
        lnprior = 0.
        for i, name in enumerate(self._param_names):
            lnprior += self._priors[name](theta[i])
        if not np.isfinite(lnprior):
            return -np.inf, np.nan

        # Evaluate the ln-likelihood function by interpolating between
        # metallicty bracket
        logm = theta[0]  # logmass
        logZZsol = theta[1]
        self._sp.params['tau'] = 10. ** theta[2]
        self._sp.params['const'] = theta[3]
        self._sp.params['sf_start'] = theta[4]
        self._sp.params['dust1'] = theta[5]
        self._sp.params['dust2'] = theta[6]
        zmet1, zmet2 = bracket_logz(logZZsol)
        # Compute fluxes with low metallicity
        self._sp.params['zmet'] = zmet1
        f1 = abs_ab_mag_to_mjy(
            self._sp.get_mags(tage=13.8, bands=self._compute_bands),
            phi[0])
        self._meta1[0] = self._sp.stellar_mass
        self._meta1[1] = self._sp.dust_mass
        self._meta1[2] = self._sp.log_lbol
        self._meta1[3] = self._sp.sfr
        self._meta1[4] = self._sp.log_age
        # Compute fluxes with high metallicity
        self._sp.params['zmet'] = zmet2
        f2 = abs_ab_mag_to_mjy(
            self._sp.get_mags(tage=13.8, bands=self._compute_bands),
            phi[0])
        self._meta2[0] = self._sp.stellar_mass
        self._meta2[1] = self._sp.dust_mass
        self._meta2[2] = self._sp.log_lbol
        self._meta2[3] = self._sp.sfr
        self._meta2[4] = self._sp.log_age

        # Interpolate and scale the SED by mass
        model_mjy = 10. ** logm * interp_logz(zmet1, zmet2, logZZsol, f1, f2)

        # Interpolate metadata between the two metallicities
        meta = interp_logz(zmet1, zmet2, logZZsol, self._meta1, self._meta2)

        # Compute likelihood
        L = -0.5 * np.sum(
            np.power((model_mjy[self._band_indices] + B
                     - self._obs_sed) / self._obs_err, 2.))

        # Compute ln posterior probability
        lnpost = lnprior + L
        if not np.isfinite(lnpost):
            lnpost = -np.inf

        # Scale statistics by the total mass
        log_m_star = theta[0] + np.log10(meta[0])  # log solar masses
        log_m_dust = theta[0] + np.log10(meta[1])  # log solar masses
        log_lbol = theta[0] * meta[2]  # log solar luminosities
        log_sfr = theta[0] * meta[3]  # star formation rate, M_sun / yr
        log_age = meta[4]  # log(age / yr)
        blob = (lnpost, log_m_star, log_m_dust, log_lbol, log_sfr, log_age,
                model_mjy)
        return lnpost, blob

    def init_blob_chain(self, n):
        """Creates a blob chain record array that the calling function can
        store.

        Parameters
        ----------
        n : int
            Length of the blob chain. This should be the total length of
            the flatchain.

        Returns
        -------
        blob_chain : ndarray
            An empty blob chain.
        """
        dt = [('lnpost', np.float),
              ('logMstar', np.float),
              ('logMdust', np.float),
              ('logSFR', np.float),
              ('logLbol', np.float),
              ('logAge', np.float),
              ('model_mjy', np.float, self.nbands)]
        blob_chain = np.nan * np.empty(n, dtype=np.dtype(dt))
        return blob_chain

    def append_blobs(self, i, blob_chain, blobs):
        """Append the ``blobs`` list to the ``blob_chain`` created
        by :meth:`init_blob_chain`.

        Parameters
        ----------
        i : int
            First index along ``blob_chain`` to write into (i.e., current
            step x number of walkers).
        blob_chain : ndarray
            An array created by :meth:`init_blob_chain`.
        blobs : list
            The blobs list returned by the posterior function.
        """
        for step_blobs in blobs:
            for b in step_blobs:
                blob_chain['lnpost'][i] = b[0]
                i += 1

    def _estimate_backgrounds(self, obs_sed, blob):
        """Estimate the background in the observed SED given the model SED
        in the blob.
        """
        model_sed = blob[6][self._band_indices]
        return obs_sed - model_sed


class GlobalThreeParamLnProb(ThreeParamLnProb):
    """Global-level version of the :class:`ThreeParamLnProb` to sample in
    the phi parameter space.

    Parameters
    ----------
    fsps_params : dict
        Parameters to initialize the python-fsps `StellarPopulation` with.
    obs_bands : list
        FSPS bandpass names that matches the `obs_seds` array.
    obs_seds : ndarray
        A ``(n_pix, n_bands)`` array of all SEDs.
    obs_errs : ndarray
        A ``(n_pix, n_bands)`` array of all SED uncertainties.
    priors : list
        A list of prior functions. For this model is is simply a prior on
        distance.
    """
    def __init__(self, fsps_params, obs_bands, obs_seds, obs_errs, priors):
        super(GlobalThreeParamLnProb, self).__init__()
        self._sp = fsps.StellarPopulation(**fsps_params)
        self._bands = obs_bands
        self._obs_seds = obs_seds
        self._obs_errs = obs_errs
        self._priors = priors
        self._param_names = ['d']

    @property
    def ndim(self):
        return 1

    def __call__(self, phi, thetas, B):
        """Compute the ln posterior probability.

        Parameters
        ----------
        phi : ndarray
            The phi parameter space, here, just a 1D ndarray with d (parsecs).
        thetas : ndarray
            The theta (pixel) parameters, shape ``(n_pix, n_theta_params)``.
        B : ndarray
            The background parameters, shape ``(n_bands,)``.

        Returns
        -------
        ln_post : float
            The ln-posterior probablity.
        """
        lnprior = 0.
        for i, name in enumerate(self._param_names):
            lnprior += self._priors[name](phi[i])
        if not np.isfinite(lnprior):
            return -np.inf
        joint_ln_likelihood = 0.
        for i in self._obs_seds.shape[0]:
            joint_ln_likelihood += self._single_pixel_ln_likelihood(
                i, thetas[i, :], B, phi)
        return lnprior + joint_ln_likelihood

    def _single_pixel_ln_likelihood(self, i, theta, B, phi):
        """Compute likelihood against a single pixel ``i``."""
        # Evaluate the ln-likelihood function by interpolating between
        # metallicty bracket
        logm = theta[0]  # logmass
        logZZsol = theta[1]
        self._sp.params['tau'] = 10. ** theta[2]
        self._sp.params['const'] = theta[3]
        self._sp.params['sf_start'] = theta[4]
        self._sp.params['dust1'] = theta[5]
        self._sp.params['dust2'] = theta[6]
        zmet1, zmet2 = bracket_logz(logZZsol)
        # Compute fluxes with low metallicity
        self._sp.params['zmet'] = zmet1
        f1 = abs_ab_mag_to_mjy(
            self._sp.get_mags(tage=13.8, bands=self._obs_bands),
            phi[0])
        # Compute fluxes with high metallicity
        self._sp.params['zmet'] = zmet2
        f2 = abs_ab_mag_to_mjy(
            self._sp.get_mags(tage=13.8, bands=self._obs_bands),
            phi[0])

        # Interpolate and scale the SED by mass
        model_mjy = 10. ** logm * interp_logz(zmet1, zmet2, logZZsol, f1, f2)

        # Compute likelihood
        L = -0.5 * np.sum(np.power((model_mjy + B
                                    - self._obs_sed) / self._obs_err, 2.))
        return L
