###################################
# Free-free emission functionlity #
###################################

import numpy as np

from astropy.table import QTable, Column
from astropy.modeling.physical_models import BlackBody

import astropy.units as u
import astropy.constants as c

from scipy.interpolate import griddata

from .utils import normalize_frequency


def kappa_ff(teff, freq, ni, Z=1.128):
    """
    Calculate the free-free absorption coefficient (κ_ff).

    This function computes the free-free absorption coefficient for a plasma 
    based on the given effective temperature (teff), observation frequency, 
    and ion number density.

    Note: The calculation assumes the presence of 
    hydrogen, so ni = 2n_e = 2n_p.

    Parameters
    ----------
    teff : float or quantity
        Effective temperature. If a float is provided, it is assumed to be 
        in Kelvin.
    freq : float or quantity
        Observation frequency. If a float is provided, it is assumed to be 
        in Hz.
    ni : float or quantity
        Number density. If a float is provided, it is assumed 
        to be in m^-3.
    Z : float, optional
        Mean ion charge (default is 1.128, from Simon's paper TODO proper citation). 

    Returns
    -------
    result : float
        Free-free absorption coefficient (κ_ff) in units of cm^-1.
    """
    
    # Deal with the quantities if any

    try:
        teff = teff.to(u.K).value
    except AttributeError:
        pass # assume if we can't do this it came in as a float or float array
        
    try:
        freq = freq.to(u.Hz).value
    except AttributeError:
        pass # assume if we can't do this it came in as a float
        
    try:
        ni = ni.to(u.cm**-3).value
    except AttributeError:
        ni = (ni*u.m**-3).to(u.cm**-3).value

    gff = 9.77 + 1.27*np.log10(np.power(teff,1.5)/(freq*Z))
    
    return (0.0178 * Z**2 * gff / (np.power(teff,1.5)*freq**2) * (ni/2)**2) / u.cm


def freefree_image(field_table, sidelen_pix, sidelen_rad=None, distance=None, kff_col="kappa_ff"):
    """
    Generate a free-free emission intensity image along the x-direction line of sight.

    This function takes a magnetic field data table (as a QTable) and constructs
    a square 2D image by integrating thermal free-free radiation along the x-axis.

    Parameters
    ----------
    field_table : astropy.table.QTable
        A QTable containing 3D positions and physical parameters.
        Required columns: 'x', 'y', 'z', 'blackbody', and `kff_col`.
        Required metadata: "Source Surface Radius" and "Radius".
        TODO: explain these columns better (esp how they depend on observing frequency)
    sidelen_pix : int
        The number of pixels per side in the output square image.
    sidelen_rad : float or None, optional
        Side length of the image in radial units. If None, defaults to 2×Source Surface Radius.
    distance : astropy.units.Quantity or None, optional
        Distance to the object. If provided, output will be flux (in mJy), otherwise an intensity image will be returned.
    kff_col : str, optional
        Column name for the free-free opacity coefficient. Default is "kappa_ff".

    Returns
    -------
    image : astropy.units.Quantity
        2D flux (or intensity) image.

    Raises
    ------
    TypeError
        If input is not a QTable, or required columns are missing.
    """
    if not isinstance(field_table, QTable):
        raise TypeError("Input table must be a QTable!")

    if not all(x in field_table.colnames for x in ['x', 'y', 'z']):
        raise TypeError("X,Y,Z columns must be present to make  image.")


    rss = field_table.meta["Source Surface Radius"]/field_table.meta["Radius"]
    if sidelen_rad is None:
        edge = rss
    else:
        edge = sidelen_rad/2
    px_sz = 2*edge/sidelen_pix*field_table.meta["Radius"]
        
    # Setting up our grids
    grid_z, grid_x, grid_y = np.meshgrid(np.linspace(-edge,edge,sidelen_pix), 
                                         np.linspace(edge,-edge,sidelen_pix), 
                                         np.linspace(-edge,edge,sidelen_pix))
   
    # The all important interpolation step
    grid_blackbody = griddata(list(zip(field_table["x"].data, field_table["y"].data, field_table["z"].data)), 
                              field_table["blackbody"], (grid_x, grid_y, grid_z), 
                              method='nearest', fill_value=0)*field_table["blackbody"].unit

    grid_kappa = griddata(list(zip(field_table["x"].data, field_table["y"].data, field_table["z"].data)), 
                          field_table[kff_col], (grid_x, grid_y, grid_z), 
                          method='nearest', fill_value=0)*field_table[kff_col].unit
    
    # Making everything beyond the source surface 0 (i.e. removing everything outside our model volume)
    beyond_scope = ((grid_x**2 + grid_y**2 + grid_z**2) > rss**2)
    grid_blackbody[beyond_scope] = 0
    grid_kappa[beyond_scope] = 0
    
    # Removing points hidden by the star
    behind_in_star = (((grid_y**2 + grid_z**2) < 1) & (grid_x < 0) ) | ((grid_x**2 + grid_y**2 + grid_z**2) < 1)
    grid_blackbody[behind_in_star] = 0
    grid_kappa[behind_in_star] = 0

    dx = (grid_x[0,0,0]-grid_x[1,0,0])*field_table.meta["Radius"]
    
    # Calculating the optical depth
    tau_grid = (grid_kappa*dx).to("")
    tau_grid = np.cumsum(tau_grid, axis=0)
    tau_grid = np.append([np.zeros(tau_grid[0].shape)], tau_grid[:-1], axis=0)
    
    # Calculating the intensity
    dI_grid = grid_blackbody*np.exp(-tau_grid)*(1 - np.exp(-dx*grid_kappa))
    
    image = np.sum(dI_grid, axis=0)

    if distance is not None:
        image = (image * (px_sz/distance)**2 * np.pi*u.sr).to(u.mJy)

    return image
