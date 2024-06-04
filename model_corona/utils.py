#########################
# General use functions #
#########################

import numpy as np

import astropy.units as u

def parsed_angle(angle):
    """
    Takes in an angle or array of angles as either Quantities or floats.
    If floats assume degrees.
    Return angle in radians as Quantity.
    """
    if not isinstance(angle, u.Quantity):
        angle = angle*u.deg

    return angle.to(u.rad)

def xy2polar(x, y, center=(0,0)):
    """
    Trans form cartesian coordinates into polar coordinates,
    with respect to the given center point.
    """
    z = (x-center[0]) + 1j*(y-center[0])
    return ( np.abs(z), (np.angle(z, deg=True)%360)*u.deg )

def make_serializable(thing):
    """
    TODO
    """
    
    if isinstance(thing, dict):
        return {x: make_serializable(y) for x, y in thing.items()}
    elif isinstance(thing, u.Quantity):
        return (make_serializable(thing.value), thing.unit.to_string())
    elif isinstance(thing, (list, np.ndarray)):
        return [make_serializable(x) for x in thing]
    else:
        return thing

    
def read_serialized(thing):
    """
    TODO
    """
    
    if isinstance(thing, dict):
        return {x: read_serialized(y) for x, y in thing.items()}
    elif isinstance(thing, (list, tuple)):
        if len(thing) == 2:
            try:
                return thing[0]*u.Unit(thing[1])
            except ValueError:
                return [read_serialized(x) for x in thing]
        else:
            return [read_serialized(x) for x in thing]
    else:
        return thing

