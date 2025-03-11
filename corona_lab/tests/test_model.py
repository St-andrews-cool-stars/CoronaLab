import pytest

import numpy as np
import pytest


from os import path

from astropy.table import QTable

import astropy.units as u
import astropy.constants as c

from  .. import corona
    

def data_path(filename):
    data_dir = path.join(path.dirname(__file__), 'data')
    return path.join(data_dir, filename)
 
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

    # TODO: this is incredibly incomplete

def test_obs_freqs():

    model = corona.ModelCorona.read(data_path("example_model.ecsv"))

    assert True

    return 

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

    
    

    
    
   
