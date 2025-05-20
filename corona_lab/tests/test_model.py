import pytest

import numpy as np
from os import path

from astropy.table import QTable

import astropy.units as u
import astropy.constants as c

from .utils_for_test import data_path
from  .. import corona


def test_from_field_lines():

    init_field = QTable.read(data_path("example_table.ecsv"))

    model = corona.ModelCorona.from_field_lines(init_field, distance=15*u.pc)

    # Check that the meta data from the input table made it into the model object
    for key in init_field.meta:
        assert init_field.meta[key] == model.meta[key]

    # Check the colnames (should be the same with the addition of 'wind' in model)
    assert model.colnames[:-1] == init_field.colnames

    # Check column data
    for col in init_field.colnames:
        assert (init_field[col] == model[col]).all()

    assert (model["wind"] == (model["line_num"] < 0)).all()
    assert (model["wind"] == model.wind).all()

    #  Check new metadata items
    assert model.meta["Total Prominence Mass"] == model['Mprom'].sum()
    assert model.meta["Corona Temperature"] == model['temperature'][~model.wind &
                                                                    ~model["proms"]].mean()
    assert model.meta["Prominence Temperature"] == model['temperature'][model["proms"]].mean()

    assert isinstance(model.meta["Source Surface Radius"], u.Quantity)
    
    # TODO: This needs better testing
    

def test_obs_freqs():

    model = corona.ModelCorona.read(data_path("example_model.ecsv"))

    model.clear_observation_freqs()
    assert np.array_equal(model.observation_freqs, []*u.GHz)

    obs_freq = 8.4*u.GHz
    model.add_observation_freq(obs_freq)

    assert len(model.meta["Observation frequencies"]) == 1
    assert model.meta["Observation frequencies"][0] == obs_freq

    assert f"{obs_freq} Kappa_ff" in model.colnames

    obs_freq = 9.1*u.MHz
    model.add_observation_freq(obs_freq)

    assert len(model.meta["Observation frequencies"]) == 2
    assert model.meta["Observation frequencies"][1] == obs_freq
    assert f"{obs_freq.to(u.GHz)} Kappa_ff" in model.colnames

    assert (model.observation_freqs == model.meta["Observation frequencies"]).all()
    
    model.clear_observation_freqs()
    assert np.array_equal(model.observation_freqs, []*u.GHz)


def test_cartesian_coords():

    model = corona.ModelCorona.read(data_path("coord_test.ecsv"))

    canon_x = model["x"].copy()
    canon_y = model["y"].copy()
    canon_z = model["z"].copy()

    # Clear canon values
    model.remove_columns(("x","y","z"))

    
    model.add_cartesian_coords((2,30)*u.deg, 0)
    print(model.colnames)
        
    assert (model["x"] == canon_x).all()
    assert (model["y"] == canon_y).all()
    assert (model["z"] == canon_z).all()

    model.remove_columns(("x","y","z"))

    model.observation_angle = (2,30)
    assert (model["x"] == canon_x).all()
    assert (model["y"] == canon_y).all()
    assert (model["z"] == canon_z).all()

    model.observation_angle = (0,0)*u.deg
    assert (model["x"] != canon_x).all()
    assert (model["y"] != canon_y).all()
    assert (model["z"] != canon_z).all()

    # TODO: test phase setting on its own
    


def test_ff_img():

    true_img = corona.ModelImage.read(data_path("test.img"))

    model = corona.ModelCorona.read(data_path("example_model.ecsv"))

    img = model.freefree_image(8.4*u.GHz, 10)

    assert (img == true_img).all()
    assert img.meta.keys() == true_img.meta.keys()

    for k in img.meta.keys():
        if k in ('Observation angle', 'Image size'):
            print((img.meta[k] == true_img.meta[k]).all())
        else:
            print(img.meta[k] == true_img.meta[k])


def test_radio_cube():

    true_cube = corona.PhaseCube.read(data_path("example_cube.ecsv"))

    model = corona.ModelCorona.read(data_path("example_model.ecsv"))

    cube = model.radio_phase_cube(8.4*u.GHz, 4, 10, 5)

    assert cube.colnames == true_cube.colnames
    for col in true_cube.colnames:
        assert (cube[col] == true_cube[col]).all()
        
    assert cube.meta.keys() == true_cube.meta.keys()
    for k in cube.meta.keys():
        if k in ('Observation angle', 'Image size'):
            print((cube.meta[k] == true_cube.meta[k]).all())
        else:
            print(cube.meta[k] == true_cube.meta[k])

def test_dynamic_spectrum():

    model = corona.ModelCorona.read(data_path("example_model.ecsv"))

    # Testing with one frequency bin
    freqs = [0.1, 0.2]*u.GHz
    phases = 4
    field_lines = [882]
    tau = 0.25*u.day
    dyn_spec = model.dynamic_spectrum(freqs, phases, field_lines, tau=0.25*u.day)

    assert (dyn_spec.meta["Observation angle"] == model.observation_angle).all()
    assert dyn_spec.meta["Distance"] == model.distance
    assert dyn_spec.meta["Parent UID"] == model.uid
    
    assert len(dyn_spec.meta["Phases"]) == 4
    assert (dyn_spec.meta["Phases"] == [0, 90, 180, 270]*u.deg).all() 
    assert np.isclose(dyn_spec.meta["Frequencies"], 0.15*u.GHz).all()
    assert (dyn_spec.meta["Frequency bin edges"] == freqs).all()
    assert dyn_spec.meta["Tau"] == tau
    assert dyn_spec.meta["Sigma"] == 1*u.deg
    assert dyn_spec.meta["Epsilon"] == 1e-5
    assert dyn_spec.meta["Ejected Line IDs"] == field_lines

    assert np.isclose(dyn_spec[0], [2.64895e-01, 3.6904e-01, 4.6538e-06, 0]*u.mJy).all()

    
    # TODO: This needs better testing
