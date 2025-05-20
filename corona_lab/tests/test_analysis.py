import pytest

import numpy as np
import astropy.units as u

from astropy.table import QTable

from corona_lab import analysis


def test_get_greatest_sep():

    # Test with dict
    props = {
        "x": np.array([0, 3]),
        "y": np.array([0, 4])
    }
    dist, indices = analysis.get_greatest_sep(props)
    assert np.isclose(dist, 5.0)
    assert sorted(indices) == [0, 1]

    # Test with table
    props = QTable({"x": [1, 4, 0],
                    "y": [1, 5, 0]})
    dist, indices = analysis.get_greatest_sep(props)
    assert np.isclose(dist, np.sqrt(4**2 + 5**2))  # distance between index 1 and 2
    assert sorted(indices) == sorted([1, 2])

    # The same point twice
    props = QTable({"x": [2, 2],
                    "y": [3, 3]})
    dist, indices = analysis.get_greatest_sep(props)
    assert dist == 0.0
    assert sorted(indices) == [0, 1]

    # Invalid input
    props = {
        "x": np.array([1]),
        "y": np.array([2])
    }
    with pytest.raises(ValueError):
        analysis.get_greatest_sep(props)



def test_smooth_img():
    img = np.random.rand(50, 50)*u.mJy
    smoothed = analysis.smooth_img(img, px_sz=1.0, beam_size=10.0)
    assert smoothed.shape == img.shape
    assert isinstance(smoothed, np.ndarray)
    
    img = np.full((20, 20), 5.0)*u.Unit('erg / (Hz s sr cm2)')
    smoothed = analysis.smooth_img(img, px_sz=1.0, beam_size=5.0)
    # After normalization, all values should be 1, and smoothing should do nothing
    assert np.allclose(smoothed, smoothed[0, 0])
    assert isinstance(smoothed, np.ndarray)

    img = np.ones((30, 30))
    img[15, 15] = 10.0  # small spike
    smoothed = analysis.smooth_img(img, px_sz=1.0, beam_size=6.0)
    assert smoothed.shape == img.shape
    assert np.where(smoothed == smoothed.max()) == ([15],[15])

    
    img = np.ones((30, 30)) * 5 * u.mJy
    img[0:10, 0:10] = 103.5*u.mJy  # region with higher values
    smoothed = analysis.smooth_img(img, px_sz=1.5, beam_size=6.0)
    assert smoothed.shape == img.shape

    xes, yes = np.where(smoothed == smoothed.max())
    assert (np.unique(xes) == np.arange(0,10)).all()
    assert (np.unique(yes) == np.arange(0,10)).all()


    # check all zeros is OK
    img = np.zeros((10, 10))*u.mJy
    smoothed = analysis.smooth_img(img, px_sz=1.0, beam_size=10.0)
    assert (smoothed == 1).all()



def test_get_image_lobes():

    # One peak
    img = np.zeros((20, 20))*u.mJy
    img[10, 10] = 1*u.mJy  # single peak
    px_sz = 1.0
    beam_size = 5.0

    props = analysis.get_image_lobes(img, px_sz, beam_size)

    assert isinstance(props, QTable)
    assert len(props) == 1
    assert props.meta["Separation"] == 0.0
    assert props.meta["Angular separation"] == 0 * u.deg


    # Two peaks
    img = np.zeros((20, 20))*u.mJy
    img[5, 5] = 10*u.mJy
    img[15, 15] = 15*u.mJy
    px_sz = 2.0
    beam_size = 4.0

    props = analysis.get_image_lobes(img, px_sz, beam_size)

    assert isinstance(props, QTable)
    assert len(props) == 2

    # Check separation and angle are positive
    assert props.meta["Separation"] > 0
    assert 0 * u.deg <= props.meta["Angular separation"] <= 180 * u.deg

    # Check px size preserved
    assert props.meta["Pixel size"] == px_sz

    # more than two peaks
    img = np.ones((30, 30))*u.Unit('erg / (Hz s sr cm2)')
    coords = [(5, 5), (15, 25), (25, 5)]
    for y, x in coords:
        img[y, x] = 10*u.Unit('erg / (Hz s sr cm2)')

    px_sz = 1.0
    beam_size = 4.0

    props = analysis.get_image_lobes(img, px_sz, beam_size)

    assert len(props) == 3
    assert "Separation" in props.meta
    assert "Angular separation" in props.meta
    assert props.meta["Separation"] > 0
    assert props.meta["Angular separation"].unit == u.deg

    # Constant image (no peaks)
    img = np.ones((20, 20))
    px_sz = 1.0
    beam_size = 3.0

    props = analysis.get_image_lobes(img, px_sz, beam_size)

    assert len(props) == 0 
    assert "Separation" in props.meta
    assert "Angular separation" in props.meta

    
