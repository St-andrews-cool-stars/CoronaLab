import pytest

import numpy as np
import astropy.units as u
from astropy.table import QTable
import astropy.constants as c


from corona_lab import build_corona
from .utils_for_test import data_path


def my_gen(tab, isopen=True):
    """
    Generator function to get the lists of open and closed field lines for the processing class
    """

    tab = tab.copy()
    tab = tab["radius", "theta", "phi", "ds", "Bmag", "Brad", "Btheta", "Bphi", "line_num"] # only use the columns we need
    tab.meta = {} # removing the metadata
    
    if isopen:
        lines = np.unique(tab[tab["line_num"] < 0]["line_num"])
    else:
        lines = np.unique(tab[tab["line_num"] >= 0]["line_num"])
        
    for ln in lines:
        yield tab[tab["line_num"] == ln]


def test_fieldine_processor():

    input_lines = QTable.read(data_path("process_fieldlines_ex.ecsv"))

    rad = input_lines.meta["Radius"]
    rss = input_lines.meta["Source Surface Radius"]
    r_exp = input_lines.meta["Ratio Exponent"]
    T_cor = input_lines.meta["Corona Temperature"]
    T_prom = input_lines.meta["Prominence Temperature"]

    dtheta = np.pi/64
    dphi = np.pi/64


    my_processor = build_corona.FieldlineProcessor(radius=rad, mass=c.M_sun, period=0.53*u.day, verbose=False)

    
    closed_fieldlines = my_gen(input_lines, False)
    open_fieldlines = my_gen(input_lines, True)

    
    pymodc = my_processor.build_model_corona(closed_fieldlines, open_fieldlines, rss,
                                             r_exp, T_cor, T_prom, dtheta, dphi, distance=15*u.pc)

    for col in pymodc.colnames:
        assert np.isclose(pymodc[col], input_lines[col]).all(), f"{col} should match between input table and result"


    


    
