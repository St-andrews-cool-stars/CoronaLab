import pytest

import numpy as np
import astropy.units as u

from astropy.table import QTable

from .utils_for_test import data_path
from  .. import freefree




def test_kappa_ff():

    # With floats
    result = freefree.kappa_ff(1e6, 5e9, 1e15)
    assert isinstance(result, u.Quantity)
    assert result.unit == u.cm**-1 
    assert np.isclose(result, 2e-12/u.cm)

    # With quantities
    result = freefree.kappa_ff(1e6*u.K,  5*u.GHz, 1e15*u.m**-3)
    assert isinstance(result, u.Quantity)
    assert result.unit == u.cm**-1
    assert np.isclose(result, 2e-12/u.cm)



def test_freefree_image():


    canon_intensity = [[0, 0, 0, 0],
                       [0, 2.6e-13, 2.8e-12, 0],
                       [0, 2.8e-12, 3.6e-11, 0],
                       [0, 0, 0, 0]]*u.Unit('erg / (Hz s sr cm2)')

    canon_flux = [[0, 0, 0, 0],
                  [0, 0.00404718, 0.0426036 , 0],
                  [0, 0.04319599, 0.5488469 , 0],
                  [0, 0, 0, 0]]*u.mJy
    
    field_table = QTable.read(data_path("example_model.ecsv"))
    field_table["kappa_ff"] = field_table["8.4 GHz Kappa_ff"]


    # No distance
    img = freefree.freefree_image(field_table, 4)
    assert np.isclose(img, canon_intensity).all()

    # With distance
    img = freefree.freefree_image(field_table, 4, distance=14*u.pc)
    assert np.isclose(img, canon_flux).all()

    # Checking the specific columns
    img = freefree.freefree_image(field_table, 10, sidelen_rad=5, distance=14*u.pc)
    img_2 = freefree.freefree_image(field_table, 10, sidelen_rad=5, distance=14*u.pc,
                                    kff_col="8.4 GHz Kappa_ff")
    assert (img == img_2).all()
    
