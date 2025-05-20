#########################
# General use functions #
#########################

import numpy as np

import astropy.units as u

def parsed_angle(angle):
    """
    Convert an angle or array of angles to radians.
    
    Parameters
    ----------
    angle : float, array-like, or astropy.units.Quantity
        The input angle(s). If a float or array of floats is provided,
        it is assumed to be in degrees.

    Returns
    -------
    astropy.units.Quantity
        Angle(s) converted to radians.
    """
    
    if not isinstance(angle, u.Quantity):
        angle = angle*u.deg

    return angle.to(u.rad)


def normalize_frequency(frequency):
    """
    Normalize input frequency to gigahertz (GHz).

    Parameters
    ----------
    frequency : float or astropy.units.Quantity
        The input frequency. If a float is provided, it is assumed to be in GHz.

    Returns
    -------
    astropy.units.Quantity
        Frequency converted to GHz.
    """
    
    if isinstance(frequency, float):
        frequency = frequency * u.GHz # assume GHz by default
    frequency = frequency.to(u.GHz) # normalize to GHz

    return frequency


def xy2polar(x, y, center=(0,0)):
    """
    Convert Cartesian coordinate(s) to polar coordinates.

    Parameters
    ----------
    x : float or array-like
        X-coordinate(s) of the point(s).
    y : float or array-like
        Y-coordinate(s) of the point(s).
    center : tuple of floats, optional
        The (x, y) coordinates of the center point. Defaults to (0, 0).

    Returns
    -------
    tuple
        A tuple containing:
        - r : float or ndarray
            Radial distance(s) from the center.
        - theta : astropy.units.Quantity
            Polar angle(s) in degrees (0-360).
    """
   
    z = (x-center[0]) + 1j*(y-center[0])
    return ( np.abs(z), (np.angle(z, deg=True)%360)*u.deg )


def make_serializable(thing):
    """
    Recursively convert a data structure into a JSON-serializable form.
    Specifically designed to serialize unit information with the data.

    Parameters
    ----------
    thing : any
        The input object to be converted. Can be a nested combination of dicts,
        lists, NumPy arrays, astropy Quantities, and standard types.

    Returns
    -------
    any
        A JSON-serializable representation of the input object. 
        - `astropy.units.Quantity` is converted into a tuple: (value, unit as string)
        - numpy scalars are cast to Python native types
        - numpy arrays and lists are converted element-wise
        - Dictionaries are processed recursively
    """
    
    if isinstance(thing, dict):
        return {x: make_serializable(y) for x, y in thing.items()}
    elif isinstance(thing, u.Quantity):
        return (make_serializable(thing.value), thing.unit.to_string())
    elif isinstance(thing, (list, np.ndarray)):
        return [make_serializable(x) for x in thing]

    # Dealing with stupid numpy data types
    elif isinstance(thing, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                            np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(thing)
    elif isinstance(thing, (np.float16, np.float32, np.float64)):
        return float(thing)
    elif isinstance(thing, (np.bool_)):
        return bool(thing)

    # Everything else
    else:
        return thing

    
def read_serialized(thing):
    """
    Reconstruct Python objects from a JSON-serializable form,
    speficially as output by `make_serializable`.

    Note
    ----
    This is not a fully lossless round-trip from `make_serializable`, but it
    should return objects that are functionally equivalent (e.g. tuples get
    round-tripped into lists). 

    Parameters
    ----------
    thing : any
        The serialized object. Typically a dict, list, or tuple containing
        primitive types or (value, unit) pairs.

    Returns
    -------
    any
        Reconstructed object, with astropy Quantities restored when detected.
    """
    
    if isinstance(thing, dict):
        return {x: read_serialized(y) for x, y in thing.items()}
    elif isinstance(thing, (list,tuple)):
        if (len(thing) == 2) and isinstance(thing[1], str) and not isinstance(thing[0], str):
            try:
                return thing[0]*u.Unit(thing[1])
            except ValueError:
                # Not a valid unit or multiplication failed
                return [read_serialized(x) for x in thing]
        else:
            return [read_serialized(x) for x in thing]
    else:
        return thing

