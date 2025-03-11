############################################
# Electron Cyclotron Emission functionlity #
############################################

import numpy as np

from astropy.table import QTable, Column

import astropy.units as u
import astropy.constants as c

from scipy.interpolate import griddata

from .utils import normalize_frequency


def ecm_allowed(n_e, B):
    """
    Function to determine if a cell can produce ECM emission.

    Llama et al 2018 eq 3
    """
    
    if isinstance(n_e, u.Quantity):
        n_e = n_e.to(u.m**-3).value
        
    if isinstance(B, u.Quantity):
        B = B.to(u.G).value
        
    return (n_e/10**14) < (28 * B / 900)**2


def gyrofrequency(B, s=1):
    """
    Given a magnetic field strength and harmonic number calculate 
    the gyroresonance emission frequency.
    
    Parameters
    ----------
    B : Astropy Quantity or float
        If float, will be assumed to be in Gauss.
    s : int
        Harmonic number
        
    Returns
    -------
    response : Astropy Quantity
        Gyroresonance emission frequency in GHz
    """
    
    if isinstance(B, u.Quantity):
        B = B.to(u.G).value
        
    return 2.8 * 10**-3 * s * B * u.GHz


def ecmfrac_calc(model, s=1):
    """
    Calculate the ECM fraction along every field line.
    
    Note, currently doing this field lines with prominences only.
    """
    
    model["gyro_freq"] = gyrofrequency(model["Bmag"], s)
    model["ECM valid"] = ecm_allowed(model["ndens"]/2, model["Bmag"])
        
    if not "ecm_frac" in model.colnames:
        model.add_column(Column(name="ecm_frac", length=len(model)))
    else: # Clear out anything that might be in there
        model["ecm_frac"] = 0

    if not model["ECM valid"].any():
        raise ValueError("No ECM possible.")
        
    for ln_num in np.unique(model["line_num"][model.prom]):

        line_inds = (model["line_num"] == ln_num)

        ds_gf =  model["gyro_freq"][line_inds] * model["ds"][line_inds]

        model["ecm_frac"][line_inds] = (ds_gf/ds_gf.sum()) *  model["ECM valid"][line_inds]



def ecm_flux(model, field_lines, tau, epsilon=0.1, sigma=1*u.deg, verbose=False):
    """
    Calculate ECM intensity at a given phase.

    Note: should put in a check for prominences
    """

    # Clearing any old data
    #model["delta_theta"] = np.nan*u.rad  # Could take this out in future when I am sure it's working correctly
    model["ECM"] = 0*u.mJy*u.GHz
    
    for ln_num in field_lines:

        line_inds = np.where(model["line_num"] == ln_num)[0]
        if len(line_inds) < 4:
            if verbose:
                print(f"Line {ln_num} is too short, skipping.")
            continue

        # Calculate the dx values
        xval = model['x'][line_inds]
        dx = np.abs(np.diff(xval))

        # We remove the first point bc both ends are on the surface so we are considering the ds's going backwards
        # TODO: this might need to be updated for Magex 
        line_inds = line_inds[1:]

        # Pulling out the line sub-array for convenience
        line_arr = model[line_inds]

        # Calculate total available enery and power
        m_prom = line_arr["Mprom"].sum() 
        r_prom = line_arr["radius"][line_arr["Mprom"] > 0*u.kg].mean()
        m_star = model.meta.get("Mass", c.M_sun) 

        Etot = (c.G * m_star * m_prom / r_prom).cgs
        power_tot = (epsilon * Etot / tau).cgs

        sinval = dx/line_arr['ds']

        # Our dx is approx so this bit ensures no nans
        sinval[sinval<-1] = -1
        sinval[sinval>1] = 1

        delta_theta = np.arcsin(sinval)
        visible_fraction = np.exp(-delta_theta**2/(2*sigma**2))
        
        if np.isclose(visible_fraction, 0).all():
            # No visibility
            if verbose:
                print(f"No visibility on line {ln_num}")
            continue


        radsigma = sigma.to(u.rad).value
        #modr = model.meta["Distance"] + line_arr["ds"]/radsigma# pulling out the distance for convenience
        
        ecm = (visible_fraction * (power_tot/(2*np.pi*radsigma*model.meta["Distance"]**2)) * line_arr["ecm_frac"])

        model["ECM"][line_inds] = ecm
        


        
def ecm_by_freq(model, freq_bin_edges):
    """
    Given a model with ECM already calculated and a frequency list,
    return the total ECM intensity for each frequency.
    """

    intensities = np.zeros(len(freq_bin_edges)-1)*model["ECM"].unit
    
    ecm_arr = model["gyro_freq", "ECM"][model["ECM"]!=0]
    
    inds = np.digitize(ecm_arr["gyro_freq"], freq_bin_edges)
    
    # We want the right bin to be closed on both sides
    inds[ecm_arr["gyro_freq"] == freq_bin_edges[-1]] = len(freq_bin_edges) - 1
    
    for i in np.unique(inds):

        if i in (len(freq_bin_edges), 0): # Value is outside range of frequencies
            continue

        intensities[i-1] = ecm_arr["ECM"][np.where(inds==i)].sum()

    delta_nus = np.diff(freq_bin_edges) # how wide each bin is
        
    return intensities/delta_nus


def dynamic_spectrum(model, freqs, phases, field_lines, tau, epsilon=0.1, sigma=1*u.deg):
    """
    TODO: WRITE THIS

    TODO: not sure about the return params, seems a little out of hand
    """

    # Frequency preprocessing
    if isinstance(freqs, int):
        num_freqs = freqs + 1 # plus one bc we are calculating frequency bin edges

        min_freq = model["gyro_freq"][model["ECM valid"] & np.isin(model["line_num"], field_lines)].min()
        max_freq = model["gyro_freq"][model["ECM valid"] & np.isin(model["line_num"], field_lines)].max()

        freq_edges = np.linspace(min_freq, max_freq, num_freqs)
    else:
        freq_edges = freqs

    # Phase preprocessing
    if isinstance(phases, int):
        num_phases = phases
        phases = np.linspace(0, 360, num_phases, endpoint=False)*u.deg
        
    freqs = (freq_edges[:-1] + freq_edges[1:])/2 # get frequency midpoints
    
    diagram_arr = np.zeros((len(freqs),len(phases)))*u.mJy
   
    for i,phs in enumerate(phases):

        model.phase = phs
        
        # Calculate the ECM intensity at this viewing angle/phase            
        ecm_flux(model, field_lines, tau=tau, epsilon=epsilon, sigma=sigma)

        # Calculate the ECM flux accross the frequencies and fill the associated column
        diagram_arr[:,i] = ecm_by_freq(model, freq_edges)

    return diagram_arr, freqs, phases, freq_edges
