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

    @classmethod
    def read(cls, filename):
        # TODO: WRITE THIS
        pass

    @property
    def distance(self):
        return self.meta.get("Distance")

    @property
    def observation_freq(self):
        return self.meta.get('Observation frequency')

    @property
    def observation_angle(self):
        return self.meta.get('Observation angle')

    @property
    def phase(self):
        return self.meta.get('Phase')

    @property
    def pix_size(self):
        return self.meta.get('Pixel size')

    @property
    def size_angular(self):
        # TODO: make safe
        self.meta.get('Pixel size') * self.meta.get('"Image size"')
        
    @property
    def flux(self):
        self.meta.get('Total Flux')

    @property
    def uid(self):
        uid = self.meta.get('UID')
        if uid is None:
            uid = md5(self).hexdigest()
        return uid

    @property
    def parent_uid(self):
        return self.meta.get("Parent UID")
    
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

    


class RadioCube(QTable):     

    @property
    def observation_freq(self):
        return self.meta.get('Observation frequency')

    @property
    def observation_angle(self):
        return self.meta.get('Observation angle')


    @property
    def pix_size(self):
        return self.meta.get('Pixel size')

    @property
    def size_angular(self):
        # TODO: make safe
        self.meta.get('Pixel size') * self.meta.get('"Image size"')
        
    @property
    def ave_flux(self):
        self.meta.get('Average Flux')

    @property
    def ave_separation(self):
        return self.meta.get("Average Separation")

    @property
    def uid(self):
        uid = self.meta.get('UID')
        if uid is None:
            uid =  md5(self.as_array()).hexdigest()
        return uid

    @property
    def parent_uid(self):
        return self.meta.get("Parent UID")


class ModelCorona(QTable):

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
            Optional. Additional model parameters to go in the table meta data.
            If these parameters also appear in the input_table meta object,
            the argument values will be perfered (i.e. these inputs override
            anything already in the table metadata)
            The valid names for parameters used throughout this class are:
            *distance, radius, rss, obs_freq, obs_angle, phase*

        Returns
        -------
        response : `ModelCorona`
            ModelCorona object with the given field lines and metadata.
        """

        instance = cls(input_table)

        instance["wind"] = instance["line_num"] < 0

        # TODO: check column names/units
        
        instance.meta["Total Prominence Mass"] = instance['Mprom'].sum()
        instance.meta["Corona Temperature"] = instance['temperature'][~instance.wind & ~instance["proms"]].mean()
        instance.meta["Total Prominence Mass"] = instance['temperature'][instance["proms"]].mean()


        # Dealing with the required metadata

        # Radius
        radius = model_parms.pop('radius', None) if 'radius' in model_parms.keys() else instance.meta.get("Radius")

        if radius is None:
            warnings.warn("No stellar radius found, assuming solar radius")
            radius = c.R_sun # Assume solar radius

        if not isinstance(radius, u.Quantity):
            radius *= c.R_sun  # Assume in solar radii

        instance.meta["Radius"] = radius

        # Source surface radius
        rss = model_parms.pop('rss', None) if 'rss' in model_parms.keys() else instance.meta.get("Source Surface Radius")

        if rss is None:
            raise AttributeError("No source surface found, this is a required parameter.")

        if not isinstance(rss, u.Quantity):
            rss *= radius  # Assume in stellar radii

        instance.meta["Source Surface Radius"] = rss

        # Dealing with optional meta data that we nonetheless want to conform 

        # Distance 
        distance = model_parms.pop('distance', None) if 'distance' in model_parms.keys() else instance.meta.get("Distance")

        if distance is not None:
            instance.distance = distance

        # observation frequency
        obs_freq = model_parms.pop('obs_freq', None) if 'obs_freq' in model_parms.keys() \
            else instance.meta.get("Observation frequency")

        if obs_freq is not None:
            instance.observation_freq = obs_freq

        # observation angle and phase
        obs_angle = model_parms.pop('obs_angle', None) if 'obs_angle' in model_parms.keys() \
            else instance.meta.get("Observation angle")
        phase = model_parms.pop('phase', None) if 'phase' in model_parms.keys() else instance.meta.get("Phase")

        if obs_angle is not None:
            phase = 0 if phase is None else phase
            instance.add_cartesian_coords(obs_angle, phase)

        # Adding the rest of the given meta data (could be anything, we don't care)
        for parm in model_parms:
            instance.meta[parm] = model_parms[parm]

        # Adding a unique id (hash)
        instance.meta["UID"] = instance.uid
        
        return instance

    @property
    def distance(self):
        return self.meta.get("Distance")

    @distance.setter
    def distance(self, value):
        if not isinstance(value, u.Quantity):
            value *= u.pc  # Assume parsecs
        self.meta["Distance"] = value.to(u.pc)

    @property
    def observation_freq(self):
        return self.meta.get('Observation frequency')

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
        self["blackbody"][self.wind] = 0
    
        # Calculating the free-free absorption coefficient
        with warnings.catch_warnings(): # suppressing divide by zero warning on wind points
            warnings.simplefilter("ignore")
            self["kappa_ff"] = kappa_ff(self["temperature"], obs_freq, self["ndens"])
        self["kappa_ff"][self.wind] = 0

        # Recording the observation frequency
        self.meta["Observation frequency"] = obs_freq

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
            raise AttributeError("You cannot set a phase with no Observation angle in place.")
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

    @property
    def uid(self):
        uid = self.meta.get('UID')
        if uid is None:
            uid = md5(self['radius', 'theta', 'phi', 'Bmag', 'proms'].as_array()).hexdigest()
        return uid

    @property
    def wind(self):
        return self["wind"]

    @property
    def prom(self):
        return self["proms"]
        
    def print_meta(self):
        pass


    def add_plasma_beta(self):
        """
        Calculate the plasma beta (ratio of plasma pressure to magnetic pressure) for each cell:

        $\beta = \frac{nkT}{B^2/2\mu_0}$

        Note: The wind had Bmag set to 0 in it, so we must exclue those points
        """

        self.add_column(Column(name="plasma_beta", length=len(self)))

        p_plasma = self[~self.wind]["ndens"]*c.k_B*self[~self.wind]["temperature"]
        p_mag = self[~self.wind]["Bmag"]**2/(2*c.mu0)

        self["plasma_beta"][~self.wind] = (p_plasma/p_mag).to("")
        
    
    def make_radio_image(self, sidelen_pix, *, sidelen_rad=None, obs_angle=None, phase=None):
        """
        Make a (square) radio image given the current object parameters.

        Parameters
        ----------
        sidelen_pix : int
            Image side length in pixels (image will be sidelen_pix x sidelen_pix pixels)
        sidelen_rad : float
            Optional. Image side length in stellar radii. If not given the source surface radius
            will be used.
        obs_angle : 2 lonrg array of float or `astropy.units.Quantity`
            Optional. Format is (ra, dec). If not given the current observation angle stored
            in meta will be used.
        phase : float or `astropy.units.Quantity`
            Optional. The stellar rotation phase/latitude. If not given the current phase stored
            in meta will be used.
            
        
        Returns
        -------
        response : `RadioImage`
            The calculated radio image as a RadioImage object, which is a `astropy.units.Quantity`
            array with metadata.
        """

        if self.distance is None:
            warnings.warn("No distance found, the returned image will be in intensity rather than flux.")

        if (obs_angle is None) & (self.observation_angle is None):
            raise AttributeError("Observation angle neither supplied nor already set.")
        elif obs_angle:
            phase = 0 if phase is None else phase
            self.add_cartesian_coords(obs_angle, phase)

        if sidelen_rad is None:
            rss = self.meta["Source Surface Radius"]/self.meta["Radius"]
            px_sz = 2*rss/sidelen_pix
        else:
            px_sz = sidelen_rad/sidelen_pix
        
        image = make_radio_image(self, sidelen_pix, sidelen_rad, self.distance)   
        image_meta = {"Observation frequency": self.observation_freq,
                      "Observation angle": self.observation_angle,
                      "Stellar Phase":  self.phase,
                      "Image size": (sidelen_pix, sidelen_pix)*u.pix,
                      "Pixel size": px_sz * self.radius}

        if self.distance is not None:
            image_meta["Distance"] = self.distance
            image_meta["Total Flux"] = np.sum(image)

        image = RadioImage(image)
        image.meta.update(image_meta)

        # Adding UID info
        image.meta["UID"] = image.uid
        image.meta["Parent UID"] = self.uid
        
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

        cube_meta = {"Observation frequency": self.meta["Observation frequency"],
                     "Observation angle": self.meta["Observation angle"],
                     "Image size": (sidelen_pix, sidelen_pix)*u.pix,
                     "Pixel size": px_sz * self.meta["Radius"],
                     "Average Flux": cube_table["flux"].mean(),
                     "Average Separation": cube_table["separation"].mean()}
        cube_table.meta.update(cube_meta)

        # Adding UID info
        cube_table.meta["UID"] = cube_table.uid
        cube_table.meta["Parent UID"] = self.uid

        return cube_table


    
    

    
        


