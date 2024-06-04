######################################
# Radio regime specific functionlity #
######################################

import numpy as np

from itertools import combinations

from astropy.table import QTable

import astropy.units as u
import astropy.constants as c

from scipy.interpolate import griddata
from scipy import ndimage

from skimage.feature import peak_local_max

from .utils import parsed_angle, xy2polar


def kappa_ff(teff, freq, ni, Z=1.128):
    """
    Calculating $\kappa_\nu$. Free-free absorption coefficient
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


def make_radio_image(field_table, sidelen_pix, sidelen_rad=None, distance=None):
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
                          field_table["kappa_ff"], (grid_x, grid_y, grid_z), 
                          method='nearest', fill_value=0)*field_table["kappa_ff"].unit
    
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


def get_greatest_sep(props):
    """
    Get the largest distance between two or more points.
    """
    
    bestij = [0,0]
    bestdist = 0
    for i,j in combinations(range(len(props)),2):

        dist = np.sqrt((props["x"][j] - props["x"][i])**2 + (props["y"][j] - props["y"][i])**2)
        
        if dist > bestdist:
            bestdist = dist
            bestij = [i,j]
            
    return bestdist, bestij

def smooth_img(img, px_sz, beam_size):
    """
    px_sz and beam_size (beam diameter) in R*
    """
    
    # Get the beam radius in px
    r = int((beam_size/px_sz)//2)

    if np.sum(np.isclose(img, img.max()))/img.size > 0.4:
        trunc = 4
    elif np.sum(np.isclose(img, img.max()))/img.size > 0.1:
        trunc = 3
    else:
        trunc = 2
    
    return ndimage.gaussian_filter(img/img.mean(), sigma=r-trunc, truncate=trunc, mode='constant')


def get_image_lobes(image_array, px_sz, beam_size):
    """
    TODO

    px_sz and beam_size (beam diameter) in R* (or really just the same units)
    """


    # Get smoothed image
    im = smooth_img(image_array, px_sz, beam_size)

    # Get the peaks
    coordinates = peak_local_max(im, min_distance=1, exclude_border=False)

    props = QTable(names=["y","x"], rows=coordinates)
    xpix,ypix = im.shape
    props['r'], props['theta'] = xy2polar(props['x'], props['y'], (xpix/2, ypix/2))
    
    props.meta["Pixel size"] = px_sz
    
    if len(props) < 2: # one one peak, so we'll call this the one blob situation
        props.meta["Seperation"] = 0*px_sz 
        props.meta["Angular separation"] = 0*u.deg
        return props    
    
    bestdist, bestij = get_greatest_sep(props)

    props.meta["Seperation"] = bestdist*px_sz 
    
    dist = np.abs(props['theta'][bestij[0]] - props['theta'][bestij[1]])
    if dist > 180*u.deg:
        dist = 360*u.deg - dist
    
    props.meta["Angular separation"] = dist
    
    return props
