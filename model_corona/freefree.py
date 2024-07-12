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
    Calculating kappa_ff. Free-free absorption coefficient
    TODO: add references


    Parameters
    ----------
    teff : float or quantity
        Effective temperature (if float assumed K)
    freq : float or quantity
        Observation frequency (if float assumed Hz)
    ni : float or quantity
        Number density, assuming hydrogen, so ni = 2n_e = 2n_p (if float assumed m^-3)
    Z = mean ion charge 1.128 (from simon's paper)
        
    
    Returns
    -------
    result : float
        Free-free absorption coefficient for given parameters.
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


def freefree_image(field_table, obs_freq, sidelen_pix, sidelen_rad=None, distance=None, kff_col="kappa_ff"):
    """
    Given a magnetic field table with necessary columns (TODO: specify)
    and a pixel size, create an intensity image along the x-direction
    line of sight. (Only does square image)

    TODO: note required columns etc
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
