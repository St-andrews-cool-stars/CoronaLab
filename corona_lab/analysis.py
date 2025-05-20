##########################################
# Functionlity related to analysis tasks #
##########################################

import numpy as np

from itertools import combinations

from astropy.table import QTable
import astropy.units as u

from scipy import ndimage
from scipy.signal import find_peaks

from skimage.feature import peak_local_max

from .utils import xy2polar


def get_greatest_sep(props):
    """
    From two or more points determine which two are the furthest from each other, and return that distance.

    Parameters
    ----------
    props : `~astropy.table.Table` or dict
        A table with two columns or dictionary with two keys:
        - 'x': The x-coordinates of the points.
        - 'y': The y-coordinates of the points.

    Returns
    -------
    bestdist : float
        The largest Euclidean distance between any two points.
    bestij : list of int
        Indices [i, j] of the pair of points that are furthest apart.

    Raises
    ------
    ValueError
        If the input does not contain at least two points.
    """

    # Check for at least two points
    if len(props["x"]) < 2:
        raise ValueError("At least two points must be supplied.")
        
    
    bestij = [0,0]
    bestdist = -1
    for i,j in combinations(range(len(props)),2):

        dist = np.sqrt((props["x"][j] - props["x"][i])**2 + (props["y"][j] - props["y"][i])**2)
        
        if dist > bestdist:
            bestdist = dist
            bestij = [i,j]
            
    return bestdist, bestij


def smooth_img(img, px_sz, beam_size):
    """
    Smooth an image using a Gaussian filter based on telescope "beam size".

    The standard deviation (sigma) of the Gaussian filter is derived from the
    ratio of the beam size to pixel size. The filter's truncation is dynamically
    chosen based on the proportion of maximum-intensity pixels in the image.

    Parameters
    ----------
    img : ndarray
        2D image array to be smoothed.
    px_sz : float
        Pixel size in R* units.
    beam_size : float
        Beam diameter in R* units.

    Returns
    -------
    smoothed_img : ndarray
        The smoothed image.

    Notes
    -----
    - The function normalizes the input image by its mean before applying the filter.
    - The `truncate` parameter is chosen based on the proportion of pixels close to
      the maximum value:
        - >40% -> truncate = 4
        - >10% -> truncate = 3
        - otherwise -> truncate = 2
    - Uses 'constant' mode for filtering boundaries.
    """

    # Check for constant value image
    if (img == img[0]).all():
        return np.ones(img.shape) # normalized array of constant value will always be 1s
    
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
    Identify peak regions in 2D a image.

    The image is first smoothed using a Gaussian filter determined by the
    beam size and pixel size. Peaks in the smoothed image are then identified,
    and their polar coordinates relative to the image center are computed.
    If multiple peaks are found, the function computes the greatest euclidian separation
    between them and the corresponding angular distance.

    Parameters
    ----------
    image_array : ndarray
        2D image array representing intensity values.
    px_sz : float
        Pixel size in consistent units (e.g., R* or km).
    beam_size : float
        Beam diameter in the same units as `px_sz`.

    Returns
    -------
    props : `~astropy.table.QTable`
        Table of detected peak coordinates and derived polar coordinates.
        The metadata includes:
        - "Pixel size": the input pixel size.
        - "Separation": the distance (in input units) between the two most distant peaks.
        - "Angular separation": angular difference between the two most distant peaks.

    Notes
    -----
    - If fewer than 2 peaks are detected, separation and angular separation are set to zero.
    - Peaks are detected using `skimage.feature.peak_local_max`.
    - Polar coordinates are computed with respect to the image center.
    """

    # TODO: add beam size to metadata


    # Get smoothed image
    im = smooth_img(image_array, px_sz, beam_size)

    # Get the peaks
    coordinates = peak_local_max(im, min_distance=1, exclude_border=False)

    # Deal with a constant image (no peaks)
    if not coordinates.size:
        coordinates = None
    

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




    
    
