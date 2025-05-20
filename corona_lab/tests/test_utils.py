import pytest

import numpy as np
import astropy.units as u

from  .. import utils


def test_parsed_angle():

    res_ang = np.pi/2*u.rad

    deg_ang = np.degrees(np.pi/2)

    assert utils.parsed_angle(res_ang) == res_ang
    assert utils.parsed_angle(deg_ang) == res_ang
    assert utils.parsed_angle(deg_ang*u.deg) == res_ang


def test_normalize_frequency():
    res_freq = 9.3*u.GHz

    assert utils.normalize_frequency(res_freq) == res_freq
    assert utils.normalize_frequency(9.3) == res_freq
    assert utils.normalize_frequency(9.3e9*u.Hz) == res_freq


def test_xy2polar():
    xy = np.array([[1,-1],[1,1]])

    r,phi = utils.xy2polar(xy[0], xy[1])
    assert (r == np.sqrt(2)).all()
    assert (phi == [45,135]*u.deg).all()

    r,phi = utils.xy2polar(xy[0], xy[1], center=(1,1))
    assert (r == np.array([0,2])).all()
    assert (phi == [0,180]*u.deg).all()
    


def test_serialization():
    num1 = 5
    num2 = 86.3
    unit1 = u.deg
    unit2 = u.m/u.s
    
    canon = {"Thing 1": "Test",
             "Thing 2": num1,
             "Thing 3": num1*unit1,
             "Thing 4": [num1*unit1, num2*unit2],
             "Thing 5": np.array((num1,num2)),
             "Thing 6": [num2,num1]*unit2,
             "Thing 7": {"a": num1, "b": num2}}

    serialed = utils.make_serializable(canon)

    assert isinstance(serialed, dict)
    assert serialed["Thing 1"] == canon["Thing 1"]
    assert serialed["Thing 2"] == num1
    assert serialed["Thing 3"] == (num1, unit1.to_string())
    assert serialed["Thing 4"] == [(num1, unit1.to_string()), (num2, unit2.to_string())]
    assert serialed["Thing 5"] == [num1,num2]
    assert serialed["Thing 6"] == ([num2,num1], unit2.to_string())
    assert serialed["Thing 7"] == canon["Thing 7"]

    unserialed = utils.read_serialized(serialed)
    assert unserialed["Thing 1"] == canon["Thing 1"]
    assert unserialed["Thing 2"] == canon["Thing 2"]
    assert unserialed["Thing 3"] == canon["Thing 3"]
    assert unserialed["Thing 4"] == canon["Thing 4"]
    assert unserialed["Thing 5"] == list(canon["Thing 5"])
    assert (unserialed["Thing 6"] == canon["Thing 6"]).all()
    assert unserialed["Thing 7"] == canon["Thing 7"]
    
    
