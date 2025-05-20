import pytest

import numpy as np
import astropy.units as u

from astropy.table import QTable

from .utils_for_test import data_path
from corona_lab import corona, ecm

def test_ecm_allowed():

    # True
    n_e = 1e13 * u.m**-3
    B = 100 * u.G
    assert ecm.ecm_allowed(n_e, B) == True

    # False
    n_e = 5e14 * u.m**-3
    B = 20 * u.G
    assert ecm.ecm_allowed(n_e, B) == False

    # no units True
    n_e = 2e13  # assumed to be in m^-3
    B = 80      # assumed to be in Gauss
    assert ecm.ecm_allowed(n_e, B) == True

    # no units false
    n_e = 1e15
    B = 10
    assert ecm.ecm_allowed(n_e, B) == False

    # Edge case (get exactly the condition)
    B = 50  # G
    rhs = (28 * B / 900)**2
    n_e = rhs * 1e14
    assert ecm.ecm_allowed(n_e, B) == False  # ECM condition is strictly less than

    # Arrays
    n_e = [1e7, 5e8] * u.cm**-3
    B =  [100, 20] * u.G
    assert (ecm.ecm_allowed(n_e, B) == (True, False)).all()

    # Table
    prop_tab = QTable(names=("ndens", "Bmag"), data=[n_e, B])
    assert (ecm.ecm_allowed(prop_tab["ndens"], prop_tab["Bmag"]) == (True, False)).all()


def test_gyrofrequency():

    # basic
    B = 100 * u.G
    freq = ecm.gyrofrequency(B)
    expected = 2.8e-3 * 100 * u.GHz
    assert freq.unit == u.GHz
    assert freq == expected

    # higher harmonic
    B = 200 * u.G
    s = 3
    freq = ecm.gyrofrequency(B, s=s)
    expected = 2.8e-3 * s * 200 * u.GHz
    assert freq.unit == u.GHz
    assert freq == expected

    # no units
    B = 150  # assumed Gauss
    freq = ecm.gyrofrequency(B)
    expected = 2.8e-3 * 150 * u.GHz
    assert freq.unit == u.GHz
    assert freq == expected


    # Array
    B = [100, 200] * u.G
    freq = ecm.gyrofrequency(B)
    expected = 2.8e-3 * ([100, 200] * u.GHz)
    assert freq.unit == u.GHz
    assert (freq == expected).all()
    
    
    # zero field
    B = 0 * u.G
    freq = ecm.gyrofrequency(B)
    assert freq == 0 * u.GHz

    # bad input
    B = 100 * u.G
    with pytest.raises(TypeError):
        ecm.gyrofrequency(B, s="second")  # invalid type for s



def test_ecmfrac():

    model = corona.ModelCorona.read(data_path("example_model.ecsv"))
    
    ecm.ecmfrac_calc(model)

    assert "gyro_freq" in model.colnames
    assert "ecm_frac" in model.colnames
    assert len(model["ecm_frac"]) == len(model)
    
    assert np.any(model["ecm_frac"] > 0)
    assert np.all(model["ecm_frac"] >= 0)

    model["ndens"] = np.ones(len(model)) * 1e20 * u.m**-3  # Too dense for ECM
    with pytest.raises(ValueError, match="No ECM possible"):
        ecm.ecmfrac_calc(model)

    # TODO: expand tests


def test_ecm_flux():
    model = corona.ModelCorona.read(data_path("example_model.ecsv"))
    ecm.ecmfrac_calc(model)
    
    tau = 1 * u.hr
    lines = np.unique(model["line_num"][model["proms"]])
    
    ecm.ecm_flux(model, lines, tau)
    
    assert "ECM" in model.colnames
    assert model["ECM"].unit == u.mJy * u.GHz
    assert ~np.isnan(model["ECM"]).any()
    
    assert np.any(model["ECM"].to_value() > 0), "Expected non-zero ECM emission"


    # test short line
    model["ECM"] = 0 * u.mJy * u.GHz  # reset
    model_short = model.copy()
    
    # artificially truncate one field line to < 4 points
    short_line_num = 882

    inds = np.where(model_short['line_num'] == short_line_num)[0]
    model_short.remove_rows(np.append(inds[1:21], inds[23:]))
    
    lines = [short_line_num]
    ecm.ecm_flux(model_short, lines, tau)
    
    assert model_short["ECM"][model_short["line_num"] == short_line_num].sum() == 0 * u.mJy * u.GHz


    # no proms
    # TODO this fails because get a nan
    #model_noprom = model.copy()
    #model_noprom["Mprom"] = np.zeros(len(model_noprom)) * u.kg

    #ecm.ecm_flux(model_noprom, lines, tau, verbose=True)

    #assert np.all(model_noprom["ECM"] == 0 * u.mJy * u.GHz), "ECM should be zero if no prominence mass"
    
    # changing sigma

    model["ECM"] = 0 * u.mJy * u.GHz  # reset
    ecm.ecm_flux(model, lines, tau, sigma=0.5*u.deg)
    flux1 = model["ECM"].sum()

    model["ECM"] = 0 * u.mJy * u.GHz  # reset
    ecm.ecm_flux(model, lines, tau, sigma=5*u.deg)
    flux2 = model["ECM"].sum()

    assert flux2 > flux1, "Wider sigma should result in more total flux"


    # Changing time scale
    tau_short = 0.5 * u.hr
    tau_long = 5 * u.hr

    model["ECM"] = 0 * u.mJy * u.GHz  # reset
    ecm.ecm_flux(model, lines, tau_short)
    flux_short = model["ECM"].sum()

    model["ECM"] = 0 * u.mJy * u.GHz  # reset
    ecm.ecm_flux(model, lines, tau_long)
    flux_long = model["ECM"].sum()

    assert flux_short > flux_long, "Shorter tau should give more intense emission"

    # test no distance
    model["ECM"] = 0 * u.mJy * u.GHz  # reset
    del model.meta["Distance"]

    with pytest.raises(KeyError):
        ecm.ecm_flux(model, lines, tau)

    

def test_ecm_by_flux():
    model = corona.ModelCorona.read(data_path("example_model.ecsv"))
    ecm.ecmfrac_calc(model)
    
    tau = .25 * u.day
    lines = np.unique(model["line_num"][model["proms"]])
    ecm.ecm_flux(model, lines, tau)

    bins = np.linspace(0.05, 1.25, 11) * u.GHz  # 10 bins
    output = ecm.ecm_by_freq(model, bins)

    assert len(output) == len(bins) - 1
    assert output.unit.is_equivalent(u.mJy), "Output should be flux density (mJy)"

    total_flux = (output * np.diff(bins)).sum()
    original_flux = model["ECM"].sum()

    assert np.isclose(total_flux.sum(), original_flux, rtol=1e-3), "Total flux should be conserved"

    # Completely outside freq range
    bins = np.linspace(100, 200, 5) * u.GHz  
    output = ecm.ecm_by_freq(model, bins)
    assert np.all(output == 0*output.unit), "Should return zero flux for out-of-range bins"


    # Last bin goes to exactly upper edge
    max_freq = model["gyro_freq"].max()
    bins = np.linspace(0, max_freq.to_value(u.GHz), 5) * u.GHz

    output = ecm.ecm_by_freq(model, bins)

    total_flux = (output * np.diff(bins)).sum()
  

    assert np.isclose(total_flux.sum(), original_flux, rtol=1e-3), "Total flux should be conserved"

    # One bin
    bins = [0.05, 1.25] * u.GHz
    output = ecm.ecm_by_freq(model, bins)
    
    assert len(output) == 1
    assert np.isclose((output*bins.diff())[0], original_flux, rtol=1e-3)

    
    # ECM is all 0
    model["ECM"] = 0 * u.mJy * u.GHz

    bins = np.linspace(0.05, 1.25, 11) * u.GHz  # 10 bins
    output = ecm.ecm_by_freq(model, bins)

    assert np.all(output == 0 * output.unit)



def test_dynamic_spectrum():

    model = corona.ModelCorona.read(data_path("example_model.ecsv"))
    ecm.ecmfrac_calc(model)
    
    tau = .25 * u.day
    field_lines = np.unique(model["line_num"][model["proms"]])

    # giving numbers of freqs and phases
    freqs = 10
    phases = 8

    diagram, f_mids, phs, f_edges = ecm.dynamic_spectrum(model, freqs, phases, field_lines, tau)

    assert diagram.shape == (freqs, phases)
    assert diagram.unit.is_equivalent(u.mJy)
    assert f_mids.unit.is_equivalent(u.Hz)
    assert f_edges.unit.is_equivalent(u.Hz)
    assert phs.unit.is_equivalent(u.deg)
    assert len(f_mids) == freqs
    assert len(phs) == phases
    assert len(f_edges) == freqs + 1


    # Giving specific freqs and phases
    freq_edges = np.linspace(0.05, 1.25, 11) * u.GHz 
    phases = np.linspace(0, 360, 12, endpoint=False) * u.deg

    diagram, f_mids, phs, f_edges = ecm.dynamic_spectrum(model, freq_edges, phases, field_lines, tau)

    assert diagram.shape == (len(freq_edges) - 1, len(phases))
    assert diagram.unit.is_equivalent(u.mJy)
    assert np.all(phs == phases)
    assert np.allclose(f_mids.value, ((freq_edges[:-1] + freq_edges[1:]) / 2).value)


    
    
