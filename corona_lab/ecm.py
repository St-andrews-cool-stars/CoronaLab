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
    Determine if a plasma cell meets the conditions for Electron Cyclotron Maser (ECM) emission.

    Based on Llama et al. (2018), Equation 3.

    Parameters
    ----------
    n_e : float or `~astropy.units.Quantity`
        Electron number density. Can be unitless (assumed in m⁻³) or a Quantity with units.
    B : float or astropy.units.Quantity
        Magnetic field strength. Can be unitless (assumed in Gauss) or a Quantity with units.

    Returns
    -------
    bool
        True if ECM emission is allowed under the input conditions, False otherwise.

    Notes
    -----
    The ECM condition is:

        n_e / 1e14 < (28 * B / 900)^2

    where `n_e` is in m⁻³ and `B` in Gauss.
    """
    
    if isinstance(n_e, u.Quantity):
        n_e = n_e.to(u.m**-3).value
        
    if isinstance(B, u.Quantity):
        B = B.to(u.G).value
        
    return (n_e/10**14) < (28 * B / 900)**2


def gyrofrequency(B, s=1):
    """
    Calculate the gyroresonance emission frequency for a given magnetic
    field strength and harmonic number.

    Parameters
    ----------
    B : float or `~astropy.units.Quantity`
        Magnetic field strength. If unitless, assumed to be in Gauss.
    s : int, optional
        Harmonic number (default is 1).

    Returns
    -------
    ~astropy.units.Quantity
        Gyroresonance emission frequency in GHz.
    """
    
    if isinstance(B, u.Quantity):
        B = B.to(u.G).value
        
    return 2.8 * 10**-3 * s * B * u.GHz


def ecmfrac_calc(model, s=1):
    """
    Calculate the fraction of total electron cyclotron maser (ECM) energy emitted by
    each cell along every magnetic field line in a corona model.

    The ECM fraction is weighted by the local gyrofrequency and path length (`ds`)
    and only includes regions where ECM emission is allowed. The result is stored
    in the `ecm_frac` column of the model table.

    Parameters
    ----------
    model : `~corona_lab.ModelCorona` or `~astropy.table.QTable`
        Table-like object containing the magnetic model of the system.
        Required columns:
        - "Bmag" : ~astropy.units.Quantity, magnetic field strength
        - "ndens" : ~astropy.units.Quantity, number density
        - "ds" : ~astropy.units.Quantity, cell path length along the field line
        - "line_num" : int, field line identifier
        - "prom" : bool, mask for prominence regions
    s : int, optional
        Harmonic number used in the gyrofrequency calculation (default is 1).

    Raises
    ------
    ValueError
        If no regions in the model allow ECM emission.

    Notes
    -----
    The ECM fraction is calculated only for field lines with prominences.
    """
    
    model["gyro_freq"] = gyrofrequency(model["Bmag"], s)
    model["ECM valid"] = ecm_allowed(model["ndens"]/2, model["Bmag"])
        
    if not "ecm_frac" in model.colnames:
        model.add_column(Column(name="ecm_frac", length=len(model)))
    else: # Clear out anything that might be in there
        model["ecm_frac"] = 0

    if not model["ECM valid"].any():
        raise ValueError("No ECM possible.")
        
    for ln_num in np.unique(model["line_num"][model["proms"]]):

        line_inds = (model["line_num"] == ln_num)

        ds_gf =  model["gyro_freq"][line_inds] * model["ds"][line_inds]

        model["ecm_frac"][line_inds] = (ds_gf/ds_gf.sum()) *  model["ECM valid"][line_inds]



def ecm_flux(model, field_lines, tau, epsilon=0.1, sigma=1*u.deg, verbose=False):
    """
    Calculate the ECM (Electron Cyclotron Maser) intensity at a given phase.

    This function loops over specified prominence-bearing field lines, computes the total available 
    ECM power from prominence ejections, distributes it along the field-line and computes the visible intensity
    for each field-line cell. The input model should have calculated cartesian coordinates such that the observer
    is looking down the x-axis for the desired observing angle.

    Parameters
    ----------
    model : `~corona_lab.ModelCorona` or `~astropy.table.QTable`
        Table-like object containing the magnetic model of the system.
        Required columns:
            - 'x' : ~astropy.units.Quantity
            - 'ds' : ~astropy.units.Quantity
            - 'Mprom' : ~astropy.units.Quantity
            - 'radius' : ~astropy.units.Quantity
            - 'ecm_frac' : float (unitless)
            - 'line_num' : int
        Metadata must include:
            - 'Distance' : ~astropy.units.Quantity (distance to observer)
            - 'Mass' (optional): ~astropy.units.Quantity (stellar mass; defaults to 1 M_sun)
    field_lines : list[int]
        List of field line numbers to include in the flux calculation.
    tau : `~astropy.units.Quantity`
        Timescale over which energy is radiated.
    epsilon : float, optional
        Efficiency factor for ECM emission (default: 0.1).
    sigma : `~astropy.units.Quantity`, optional
        Angular width of the emission cone (default: 1 deg) (emission cone has a fixed
        opening angle of 90 deg).
    verbose : bool, optional
        Whether to print debug messages.

    Returns
    -------
    None
        The function modifies the `model` table in-place by populating the 'ECM' column
        with intensities in units of mJy GHz.

    Notes
    -----
    - Assumes ECM emission is emitted isotropically in a Gaussian angular profile.
    - Skips field lines with <4 points or no visible regions.
    - Currently assumes prominence mass (`Mprom`) is already set on the model.
    """

    # TODO: fix the no prominences case, right now you get nans in the ECM flux
    
    # Clearing any old data
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
        
        ecm = (visible_fraction * (power_tot/(2*np.pi*radsigma*model.meta["Distance"]**2)) * line_arr["ecm_frac"])

        model["ECM"][line_inds] = ecm

        
def ecm_by_freq(model, freq_bin_edges):
    """
    Bin total ECM emission by frequency and return spectral flux density.

    Given a model that contains precomputed ECM intensities (for the current observing angle of the
    model) and corresponding gyroresonant frequencies, this function bins the total ECM power into
    specified frequency bins and returns the resulting spectral flux density (mJy).

    Parameters
    ----------
    model : `~corona_lab.ModelCorona` or `~astropy.table.QTable`
        Table-like object containing the magnetic model of the system.
        Required columns:
        - "ECM": ECM emission (with units of flux × frequency)
        - "gyro_freq" is the emission frequency (as ~astropy.units.Quantity).

    freq_bin_edges : `~astropy.units.Quantity`
        1D array of frequency bin edges (must have units of frequency, e.g., GHz).
        The output will have one less element than the length of this array.

    Returns
    -------
    intensities : `~astropy.units.Quantity`
        Array of total ECM flux per frequency bin divided by bin width, resulting
        in spectral flux density. Units are equivalent to flux density (mJy).
    
    Notes
    -----
    - Values of `gyro_freq` outside the provided bin range are ignored.
    - The upper edge of the last bin is inclusive.
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
    Generate a dynamic spectrum of Electron Cyclotron Maser (ECM) emission over a range of 
    frequencies and rotational phases.

    Parameters
    ----------
    model : `~model_corona.ModelCorona`
        Model corona object. Must be pre-setup with ecm_frac already calculated.
    freqs : int or array-like
        If int, the number of frequency bins (edges will be computed automatically from 
        valid gyrofrequencies in `model`). If array-like, the frequency bin edges as 
        `~astropy.units.Quantity` or array of floats (assumed to be in Hz or compatible).
    phases : int or `~astropy.units.Quantity`
        If int, the number of evenly spaced rotational phases from 0° to 360°. 
        If a `Quantity`, the exact phases (in degrees) to evaluate.
    field_lines : list[int]
        List of field line numbers to include in the flux calculation.
    tau : `~astropy.units.Quantity`
        Timescale over which energy is radiated.
    epsilon : float, optional
        Efficiency factor for ECM emission (default: 0.1).
    sigma : `~astropy.units.Quantity`, optional
        Angular width of the emission cone (default: 1 deg) (emission cone has a fixed
        opening angle of 90 deg).

    Returns
    -------
    diagram_arr : `~astropy.units.Quantity`
        2D array of flux densities with shape (n_freqs, n_phases), in mJy.
    freqs : `~astropy.units.Quantity`
        Frequency midpoints corresponding to each row in `diagram_arr`.
    phases : `~astropy.units.Quantity`
        Rotational phases (in degrees) corresponding to each column in `diagram_arr`.
    freq_edges : `~astropy.units.Quantity`
        Frequency bin edges used to compute the midpoints.
    """

    # TODO: Move this to the object itself?
    
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
