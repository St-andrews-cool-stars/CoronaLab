import warnings
import json
import io
import numpy as np

from pathlib import Path
from hashlib import md5
from re import search

from astropy.table import Table, QTable, Column
from astropy.coordinates import SkyCoord
from astropy.utils.metadata import MetaData
from astropy.modeling.physical_models import BlackBody

import astropy.constants as c
import astropy.units as u

from .radio import kappa_ff, make_radio_image, get_image_lobes
from .utils import parsed_angle, make_serializable, read_serialized


class RadioImage(u.Quantity):
    meta = MetaData()
        
    @property
    def lobes(self):

        if not hasattr(self, '_lobe_array'):
            self._lobe_array = get_image_lobes(self, self.meta.get("Pixel size", 1*u.pix))
            self.meta["Separation"] = self._lobe_array.meta["Separation"]

        return self._lobe_array

    @property
    def lobe_separation(self):
        sep = self.meta.get("Separation")

        if not sep:
            sep = self.lobes.meta["Separation"]

        return sep

    def write(self, filename):

        # TODO: add checking for existing file (and overwrite arg)

        content_dict = {}
        content_dict["meta"] = make_serializable(self.meta)
        content_dict["unit"] = self.unit.to_string()

        memfile = io.BytesIO()
        np.save(memfile, self.value)
        content_dict["array"] = memfile.getvalue().decode('latin-1')

        with open(Path(self.meta["Directory"]).joinpath(self.meta["Filename"]), "w") as FLE:
            json.dump(content_dict, FLE)


class RadioCube(QTable):

    @property
    def observation_freq(self):
        return self.meta.get('Observation Frequency')
    

class ModelCorona(QTable):

    _special_props = ["distance"]

    @classmethod
    def from_field_lines(cls, input_table, **model_parms):
        """
        Build a ModelCorona object from an field lines table and optional
        additional metadata.

        Parameters
        ----------
        input_table : Any valid input to `astropy.table.QTable()`
            The table object containing a set of model corona field lines
        **model_parms
            Additional model parameters to go in the table meta data.
            There is no requirement to supply any of these things, however
            some functions require various meta data.

        Returns
        -------
        response : `ModelCorona`
            ModelCorona object with the given field lines and metadata.
        """

        instance = cls(input_table)

        for parm in model_parms:
            instance.meta[parm] = model_parms[parm]

        instance._regularize()
        
        return instance

    def _regularize(self):
        """
        Function to make sure everything conforms to expectations by the functions.
        """

        # Making sure we have all the columns we need/want
        # TODO: allow for if the input table didn't come out of my translation file
        self["wind"] = self["line_num"] < 0

        # Making sure the meta data matches the table
        self.meta["Total Prominence Mass"] = self['Mprom'].sum()
        self.meta["Corona Temperature"] = self['temperature'][~self["wind"] & ~self["proms"]].mean()
        self.meta["Total Prominence Mass"] = self['temperature'][self["proms"]].mean()


        if not isinstance(self.meta['Radius'], u.Quantity):
            self.meta['Radius'] *= c.R_sun  # Assume in solar radii
       
        if not isinstance(self.meta['Source Surface Radius'], u.Quantity):
            self.meta['Source Surface Radius'] *= self.meta['Radius']  # Assume in stellar radii

        # Regularizing the meta dara attributes we care about
        for prop in self.meta.keys():
            if prop in self._special_props:
                setattr(self, prop, self.meta[prop])

    @property
    def distance(self):
        return self.meta.get("distance")

    @distance.setter
    def distance(self, value):
        if not isinstance(value, u.Quantity):
            value *= u.pc  # Assume parsecs
        self.meta["distance"] = value.to(u.pc)

    @property
    def observation_freq(self):
        return self.meta.get('Observation Frequency')

    @observation_freq.setter
    def observation_freq(self, obs_freq):
        """
        Doing all setup that requires the observation frequency.

        Added columns:
        - blackbody : blackbody emission at the observing frequency (erg / (cm2 Hz s sr))
        - kappa_ff : free-free absorption coefficient at the observing frequency (cm^-1)
        """
        
        if isinstance(obs_freq, float):
            obs_freq = obs_freq * u.GHz # assume GHz by default

        # Doing blackbody calculations
        bb_corona = BlackBody(temperature=self.corona_temp )
        bb_prominence = BlackBody(temperature=self.prom_temp)
    
        self["blackbody"] = bb_corona(obs_freq)
        self["blackbody"][self["proms"]] = bb_prominence(obs_freq)
        self["blackbody"][self["wind"]] = 0
    
        # Calculating the free-free absorption coefficient
        with warnings.catch_warnings(): # suppressing divide by zero warning on wind points
            warnings.simplefilter("ignore")
            self["kappa_ff"] = kappa_ff(self["temperature"], obs_freq, self["ndens"])
        self["kappa_ff"][self["wind"]] = 0

        # Recording the observation frequency
        self.meta["Observation Frequency"] = obs_freq

    def add_cartesian_coords(self, obs_angle, phase=0):
        """
        Given a viewing angle and optional phase add columns with the cartesian coordinats for each row.
        Assumes field_table has columns radius, theta, phi

        Note, this goes into a right handed cartesian coordinate system.
        """

        obs_angle = parsed_angle(obs_angle)
        phi0, theta0 = obs_angle
        phase = parsed_angle(phase)
    
        phi = self["phi"]+phase
        theta = self["theta"]
        r = self["radius"]
    
        self["x"] = r * (np.cos(theta0)*np.cos(theta) + np.sin(theta0)*np.sin(theta)*np.cos(phi-phi0))
        self["y"] = r * np.sin(theta)*np.sin(phi-phi0)
        self["z"] = r * (np.sin(theta0)*np.cos(theta) - np.cos(theta0)*np.sin(theta)*np.cos(phi-phi0))

        self.meta["Observation angle"] = obs_angle.to(u.deg)
        self.meta["Phase"] = phase.to(u.deg)
        
    @property
    def observation_angle(self):
        return self.meta.get('Observation angle')

    @observation_angle.setter
    def observation_angle(self, value):
        self.add_cartesian_coords(value, self.meta.get('Phase',0))

    @property
    def phase(self):
        return self.meta.get('Phase')

    @phase.setter
    def phase(self, value):
        if not self.observation_angle:
            raise AttributeError("You cannot set a phase with no Observation Angle in place.")
        self.add_cartesian_coords(self.observation_angle, value)
 
    @property
    def corona_temp(self):
        return self.meta.get("Corona Temperature")

    @property
    def prom_temp(self):
        return self.meta.get("Prominence Temperature")

    @property
    def radius(self):
        return self.meta.get('Radius')

    @property
    def rss(self):
        return self.meta.get('Source Surface Radius')

    def make_radio_image(self, sidelen_pix, *, sidelen_rad=None, obs_angle=None, phase=None):
        """
        Make a (square) radio image given the current object parameters.

        Parameters
        ----------

        Returns
        -------
        
        """

        if self.distance is None:
            warnings.warn("No distance found, the returned image will be in intensity rather than flux.")

        if (obs_angle is None) & (self.observation_angle is None):
            raise AttributeError("Observation angle neither supplied nor already set.")
        elif obs_angle:
            phase = 0 if phase is none else phase
            self.add_cartesian_coords(obs_angle, phase)

        if sidelen_rad is None:
            rss = self.meta["Source Surface Radius"]/self.meta["Radius"]
            px_sz = 2*rss/sidelen_pix
        else:
            px_sz = sidelen_rad/sidelen_pix
        
        image = make_radio_image(self, sidelen_pix, sidelen_rad, self.distance)   
        image_meta = {"Observing frequency": self.observation_freq,
                      "Observation Angle": self.observation_angle,
                      "Stellar Phase":  self.phase,
                      "Image size": (sidelen_pix, sidelen_pix)*u.pix,
                      "Pixel size": px_sz * self.radius}

        if self.distance is not None:
            image_meta["distance"] = self.distance
            image_meta["Total Flux"] = np.sum(image)

        image = RadioImage(image)
        image.meta.update(image_meta)

        return image

    
    def make_radio_phase_cube(self, num_steps, sidelen_pix, *, sidelen_rad=None,
                              obs_angle=None, min_phi=0*u.deg, max_phi=360*u.deg):
        """
        Make a number (square) radio image given the current object parameters, evenly
        spaced between the min and max phases.

        Parameters
        ----------

        Returns
        -------
        """

        if self.distance is None:
            warnings.warn("No distance found, the returned image will be in intensity rather than flux.")

        if (obs_angle is None) & (self.observation_angle is None):
            raise AttributeError("Observation angle neither supplied nor already set.")
        elif obs_angle:
            obs_angle = parsed_angle(obs_angle)
        else:
            obs_angle = self.observation_angle

        # Regularising the angles
        min_phi = parsed_angle(min_phi)
        max_phi = parsed_angle(max_phi)
        obs_angle = parsed_angle(obs_angle)
    
        # Get the phases we want
        phase_list = np.linspace(min_phi, max_phi, num_steps)

        # Go ahead and to this calculation here
        if sidelen_rad is None:
            sidelen_rad = 2*self.meta["Source Surface Radius"]/self.meta["Radius"]
        px_sz = sidelen_rad/sidelen_pix
        
        cube_dict = {"phi":[], "flux":[], "separation":[], "image":[]}

        for phase in phase_list:

            self.add_cartesian_coords(obs_angle, phase)
            image = make_radio_image(self, sidelen_pix, sidelen_rad, self.distance)
            lobes = get_image_lobes(image, px_sz)
        
            cube_dict["phi"].append(phase.to('deg'))
            cube_dict["flux"].append(np.sum(image))
            cube_dict["separation"].append(lobes.meta["Separation"])
            cube_dict["image"].append(image)
        
        cube_table = RadioCube(cube_dict)

        cube_meta = {"Observing frequency": self.meta["Observation Frequency"],
                     "Observation Angle": self.meta["Observation angle"],
                     "Image size": (sidelen_pix, sidelen_pix)*u.pix,
                     "Pixel size": px_sz * self.meta["Radius"],
                     "Average Flux": cube_table["flux"].mean(),
                     "Average Separation": cube_table["separation"].mean()}
        cube_table.meta.update(cube_meta)

        return cube_table


    
    

    
        


