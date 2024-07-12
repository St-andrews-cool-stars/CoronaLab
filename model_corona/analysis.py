##########################################
# Functionlity related to analysis tasks #
##########################################

import numpy as np

from itertools import combinations

from astropy.table import QTable
import astropy.units as u

from scipy import ndimage

from skimage.feature import peak_local_max

from .utils import xy2polar


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
        props.meta["Separation"] = 0*px_sz 
        props.meta["Angular separation"] = 0*u.deg
        return props    
    
    bestdist, bestij = get_greatest_sep(props)

    props.meta["Separation"] = bestdist*px_sz 
    
    dist = np.abs(props['theta'][bestij[0]] - props['theta'][bestij[1]])
    if dist > 180*u.deg:
        dist = 360*u.deg - dist
    
    props.meta["Angular separation"] = dist
    
    return props
