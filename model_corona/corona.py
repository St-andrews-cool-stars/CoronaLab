import warnings
import json
import io
import numpy as np

from pathlib import Path
from hashlib import md5
from re import search

from astropy.table import Table, QTable, Column
from astropy.coordinates import SkyCoord, distances
from astropy.utils.metadata import MetaData
from astropy.modeling.physical_models import BlackBody

import astropy.constants as c
import astropy.units as u

from .freefree import kappa_ff, freefree_image
from .ecm import ecmfrac_calc, dynamic_spectrum
from .utils import parsed_angle, make_serializable, read_serialized, normalize_frequency
from .analysis import get_image_lobes


class ModelArray(u.Quantity):
    meta = MetaData()

    def write(self, filename):

        # TODO: add checking for existing file (and overwrite arg)

        content_dict = {}
        content_dict["meta"] = make_serializable(self.meta)
        content_dict["unit"] = self.unit.to_string()

        memfile = io.BytesIO()
        np.save(memfile, self.value)
        content_dict["array"] = memfile.getvalue().decode('latin-1')

        with open(filename, "w") as FLE:
            json.dump(content_dict, FLE)

    @classmethod
    def read(cls, filename):

        with open(filename, "r") as FLE:
            img_dict = json.load(FLE)

        img_str = img_dict.pop("array")
        fp = io.BytesIO(img_str.encode('latin-1'))
        arr_np = np.load(fp, encoding='latin1')

        instance = cls(arr_np*u.Unit(img_dict.pop("unit")))

        instance.meta.update(read_serialized(img_dict.pop("meta")))

        return instance

    @property
    def distance(self):
        return self.meta.get("Distance")

    @property
    def observation_angle(self):
        return self.meta.get('Observation angle')

    @property
    def uid(self):
        uid = self.meta.get('UID')
        if uid is None:
            uid = md5(self).hexdigest()
        return uid

    @property
    def parent_uid(self):
        return self.meta.get("Parent UID")
    

class ModelImage(ModelArray):

    @property
    def observation_freq(self):
        return self.meta.get('Observation frequency')

    @property
    def phase(self):
        return self.meta.get('Phase')

    @property
    def stellar_radius(self):
        return self.meta.get('Stellar Radius')

    @property
    def pix_size(self):
        return self.meta.get('Pixel size')

    @property
    def size_angular(self):
        # TODO: make safe
        return self.meta.get('Pixel size') * self.meta.get('"Image size"')
        
    @property
    def flux(self):
        return self.meta.get('Total Flux')


class ModelDynamicSpectrum(ModelArray):

    @property
    def phases(self):
        return self.meta.get("Phases")

    @property
    def freqs(self):
        return self.meta.get("Frequencies")

    @property
    def ejected_mass(self):
        return self.meta.get("Ejected Mass")

    @property
    def ejected_lines(self):
        return self.meta.get("Ejected Line IDs")

    @property
    def tau(self):
        return self.meta.get("Tau")

    @property
    def epsilon(self):
        return self.meta.get("Epsilon")

    @property
    def sigma(self):
        return self.meta.get("Sigma")

    @property
    def light_curve(self):
        return np.mean(self, axis=0)

    @property
    def sed(self):
        return np.mean(self, axis=1)

    
class PhaseCube(QTable):     

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
            self.meta['UID'] = uid
        return uid

    @property
    def parent_uid(self):
        return self.meta.get("Parent UID")


class FrequencyCube(QTable):

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
    def uid(self):
        uid = self.meta.get('UID')
        if uid is None:
            uid =  md5(self.as_array()).hexdigest()
            self.meta['UID'] = uid
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
        instance.meta["Prominence Temperature"] = instance['temperature'][instance["proms"]].mean()

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
        if 'rss' in model_parms.keys():
            rss = model_parms.pop('rss', None)
        elif 'Rss' in model_parms.keys():
            rss = model_parms.pop('Rss', None)
        else:
            rss = instance.meta.get("Source Surface Radius")

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
        obs_freq = model_parms.pop('obs_freq', None)
        if obs_freq is not None:
            instance.add_observation_frequency(obs_freq)
        
        obs_freq = instance.meta.pop("Observation frequency", None)
        if obs_freq is not None:
            instance.add_observation_frequency(obs_freq)
            instance.meta

        # observation angle and phase
        obs_angle = model_parms.pop('obs_angle', None) if 'obs_angle' in model_parms.keys() \
            else instance.meta.get("Observation angle")
        phase = model_parms.pop('phase', None) if 'phase' in model_parms.keys() else instance.meta.get("Phase")

        if obs_angle is not None:
            phase = 0 if phase is None else phase
            instance.add_cartesian_coords(obs_angle, phase)

        # Adding the rest of the given meta data (could be anything, we don't care)
        # Note: currently we just ditch anything with an already used key value 
        for parm in model_parms:
            if not parm in instance.meta.keys():
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
    def observation_freqs(self):
        return self.meta.get('Observation frequencies', []*u.GHz)

    def add_observation_freq(self, obs_freq, cache=True):
        """
        Add freq specific column. "<obs_freq> Kappa_ff"
        """

        obs_freq = normalize_frequency(obs_freq)
            
        if (cache == True) and (obs_freq in self.observation_freqs):
            return # No need to recalculate

        # Calculating the free-free absorption coefficient
        with warnings.catch_warnings(): # suppressing divide by zero warning on wind points
            warnings.simplefilter("ignore")
            self[f"{obs_freq} Kappa_ff"] = kappa_ff(self["temperature"], obs_freq, self["ndens"])
        self[f"{obs_freq} Kappa_ff"][self.wind] = 0

        # Recording the observation frequency in the metadata
        self.meta["Observation frequencies"] = np.concatenate((self.meta.get("Observation frequencies", []*u.GHz),
                                                               [obs_freq]))

    def clear_observation_freqs(self, obs_freqs="all"):

        if obs_freqs == "all":
            obs_freqs = self.observation_freqs
        elif (not isinstance(obs_freqs, u.Quantity) and  np.isscalar(obs_freqs)) or \
             (isinstance(obs_freqs, u.Quantity) and  np.isscalar(obs_freqs.value)): 
            obs_freqs = [obs_freqs]

        for freq in obs_freqs:
            freq = normalize_frequency(freq)

            if not freq in self.observation_freqs: # Freq is not actuall in our table
                continue

            self.remove_column(f"{freq} Kappa_ff")
            
            self.meta['Observation frequencies'] = np.delete(self.meta['Observation frequencies'],
                                                             np.where(self.meta['Observation frequencies'] == freq))

        
    def add_cartesian_coords(self, obs_angle, phase=0, recalculate=False):
        """
        Given a viewing angle and optional phase add columns with the cartesian coordinats for each row.
        Assumes field_table has columns radius, theta, phi

        Note, this goes into a right handed cartesian coordinate system.
        """
        
        obs_angle = parsed_angle(obs_angle)
        phase = parsed_angle(phase)

        # Check if we actually need to redo the calculation or not
        if ((not recalculate) and
            (self.meta.get("Phase") == phase) and
            (np.isclose(self.meta.get("Observation angle", [np.nan]*2*u.deg), obs_angle).all())):
            return
        
        phi0, theta0 = obs_angle
        phi = self["phi"]+phase
        theta = self["theta"]

        # Sometime radius is a distance object, and passing that on to x,y,z leads to problems when reading/writing
        # because distances are not allowed to be negative numbers
        if isinstance(self["radius"], distances.Distance):
            r = u.Quantity(self["radius"])
        else:
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
        if not isinstance(self.observation_angle, u.Quantity):
            raise AttributeError("You cannot set a phase with no Observation angle in place.")
        self.add_cartesian_coords(self.observation_angle, value)
 
    @property
    def corona_temp(self):
        return self.meta.get("Corona Temperature")

    @property
    def prom_temp(self):
        return self.meta.get("Prominence Temperature")

    @property
    def bb_corona(self):
        if getattr(self, "_bb_corona", None) is None:
            self._bb_corona = BlackBody(temperature=self.corona_temp)
        return self._bb_corona

    @property
    def bb_prominence(self):
        if getattr(self, "_bb_prominence", None) is None:
            self._bb_prominence = BlackBody(temperature=self.prom_temp)
        return self._bb_prominence

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

    @property
    def cor_only(self):
        return ~self["wind"] & ~self["proms"]
        
    def print_meta(self):
        for key, val in self.meta.items():
            if key == "Radius":
                print(f"{key}: {val/c.R_sun:.1f} Rsun")
            elif key == "Source Surface Radius":
                rad = self.meta.get("Radius", c.R_sun)
                print(f"{key}: {val/rad:.1f} R*")
            elif key == "Corona Temperature":
                print(f"{key}: {val:.1e}")
            elif key == "Total Prominence Mass":
                print(f"{key}: {val:.1e}")
            elif key in ("nrad", "UID"):
                continue
            else:
                print(f"{key}: {val}")

    def add_plasma_beta(self):
        r"""
        Calculate the plasma beta (ratio of plasma pressure to magnetic pressure) for each cell:

        .. math::
            \beta = \frac{nkT}{B^2/2\mu_0}

        Note: The wind had Bmag set to 0 in it, so we must exclue those points
        """

        self.add_column(Column(name="plasma_beta", length=len(self)))

        p_plasma = self[~self.wind]["ndens"]*c.k_B*self[~self.wind]["temperature"]
        p_mag = self[~self.wind]["Bmag"]**2/(2*c.mu0)

        self["plasma_beta"][~self.wind] = (p_plasma/p_mag).to("")

    def _add_bb_col(self, obs_freq):
        """
        The blackbody column is not labelled by frequency so in general
        this function should not be called by the user.
        """

        if not hasattr(self, 'bb_corona'):
            self.bb_corona = BlackBody(temperature=self.corona_temp)
            
        if not hasattr(self, 'bb_prominence'):
            self.bb_prominence = BlackBody(temperature=self.prom_temp)
            
        self["blackbody"] = self.bb_corona(obs_freq)
        self["blackbody"][self["proms"]] = self.bb_prominence(obs_freq)
        self["blackbody"][self.wind] = 0
        
    
    def freefree_image(self, obs_freq, sidelen_pix, *, sidelen_rad=None, obs_angle=None, phase=None):
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
        response : `ModelImage`
            The calculated radio image as a RadioImage object, which is a `astropy.units.Quantity`
            array with metadata.
        """

        if self.distance is None:
            warnings.warn("No distance found, the returned image will be in intensity rather than flux.")

        # Getting the observation frequency set up
        obs_freq = normalize_frequency(obs_freq)
        self.add_observation_freq(obs_freq, cache=True)
        self._add_bb_col(obs_freq)

        # Handling the observation angle/phase
        if obs_angle is None:
            if self.observation_angle is None:
                raise AttributeError("Observation angle neither supplied nor already set.")
            else:
                obs_angle = self.observation_angle

        if phase is None:
            phase = self.meta.get("Phase", 0)

        self.add_cartesian_coords(obs_angle, phase)

        if sidelen_rad is None:
            rss = self.meta["Source Surface Radius"]/self.meta["Radius"]
            px_sz = 2*rss/sidelen_pix
        else:
            px_sz = sidelen_rad/sidelen_pix

        image = freefree_image(self, obs_freq, sidelen_pix, sidelen_rad, self.distance, kff_col=f"{obs_freq} Kappa_ff")   
        image_meta = {"Observation frequency": obs_freq,
                      "Observation angle": self.observation_angle,
                      "Stellar Phase":  self.phase,
                      "Image size": (sidelen_pix, sidelen_pix)*u.pix,
                      "Pixel size": (px_sz * self.radius).to(self.radius),
                      "Stellar Radius": self.radius}

        if self.distance is not None:
            image_meta["Distance"] = self.distance
            image_meta["Total Flux"] = np.sum(image)

        image = ModelImage(image)
        image.meta.update(image_meta)

        # Adding UID info
        image.meta["UID"] = image.uid
        image.meta["Parent UID"] = self.uid
        
        return image

    
    def radio_phase_cube(self, obs_freq, num_steps, sidelen_pix, beam_size, *, sidelen_rad=None,
                        obs_angle=None, min_phi=0*u.deg, max_phi=360*u.deg):
        """
        Make a number (square) radio image given the current object parameters, evenly
        spaced between the min and max phases.

        TODO: this should allow ecm imagery to be added but later
        TODO: this does not need to be so specific

        Parameters
        ----------

        Returns
        -------
        """

        if self.distance is None:
            warnings.warn("No distance found, the returned image will be in intensity rather than flux.")

        if (obs_angle is None) & (self.observation_angle is None):
            raise AttributeError("Observation angle neither supplied nor already set.")
        elif obs_angle is not None:
            obs_angle = parsed_angle(obs_angle)
        else:
            obs_angle = self.observation_angle

        # Getting the observation frequency set up
        obs_freq = normalize_frequency(obs_freq)
        self.add_observation_freq(obs_freq, cache=True)
        self._add_bb_col(obs_freq)

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
        
        cube_dict = {"phi":[], "flux":[], "separation":[], "num_peaks":[], "ang_sep":[], "image":[]}
        for phase in phase_list:

            self.add_cartesian_coords(obs_angle, phase)
            image = freefree_image(self, obs_freq, sidelen_pix, sidelen_rad, self.distance, kff_col=f"{obs_freq} Kappa_ff")
            lobes = get_image_lobes(image, px_sz, beam_size)
        
            cube_dict["phi"].append(phase.to('deg'))
            cube_dict["flux"].append(np.sum(image))
            cube_dict["separation"].append(lobes.meta["Separation"])
            cube_dict["num_peaks"].append(len(lobes))
            cube_dict["ang_sep"].append(lobes.meta["Angular separation"])

            
            cube_dict["image"].append(image)
        
        cube_table = PhaseCube(cube_dict)

        
        cube_meta = {"Observation frequency": obs_freq,
                     "Observation angle": self.meta["Observation angle"],
                     "Image size": (sidelen_pix, sidelen_pix)*u.pix,
                     "Stellar Radius": self.meta["Radius"],
                     "Pixel size": px_sz,
                     "Beam size": beam_size,
                     "Average Flux": cube_table["flux"].mean(),
                     "Percent 2 Peaks": sum(cube_table["num_peaks"]==2)/len(cube_table)}
        if cube_meta["Percent 2 Peaks"] > 0:
             cube_meta["Average Separation"] = np.mean(cube_table["separation"][cube_table["num_peaks"]>1]),
             cube_meta["Average Angular Separation"] = np.mean(cube_table["ang_sep"][cube_table["num_peaks"]>1])
            
        cube_table.meta.update(cube_meta)

        # Adding UID info
        cube_table.meta["UID"] = cube_table.uid
        cube_table.meta["Parent UID"] = self.uid

        return cube_table

    
    def dynamic_spectrum(self, freqs, phases, field_lines, tau, epsilon=0.1, sigma=1*u.deg,
                         harmonic=1, distance=None, obs_angle=None):
        """
        Return ECM dynamic spectrum

        
        """

        # Setting up the distance
        if (distance is None) & (self.distance is None):
            raise AttributeError("Distance neither supplied nor already set.")
        elif distance is not None:
            self.distance = distance

        # Turning the field_lines into an array of line numbers
        if isinstance(field_lines, str) and (field_lines == "prom"):
            field_lines = np.unique(model["line_num"][model.prom])
        elif isinstance(field_lines, int):
            field_lines = [field_lines]
        # TODO: obv there are still a lot of ways this could error out
        
        # Setting up the observation angle
        if (obs_angle is None) & (self.observation_angle is None):
            raise AttributeError("Observation angle neither supplied nor already set.")
        elif obs_angle is not None:
            self.observation_angle = obs_angle

            
        # Making sure the one-time work is done
        # this needs better error handling
        if (not "ECM valid" in self.colnames) or (self.meta.get("ECM Harmonic") != harmonic):
            ecmfrac_calc(self, harmonic)
            self.meta["ECM Harmonic"] = harmonic

        #if (self["ecm_frac"] == 0).all(): 
        #    raise ValueError("No ECM possible.")

        dyn_spec, freqs, phases, bin_edges = dynamic_spectrum(self, freqs, phases, field_lines, tau, epsilon, sigma)

        spec_meta = {"Observation angle": self.observation_angle,
                     "Distance": self.distance,
                     "Phases": phases,
                     "Frequencies": freqs,
                     "Ejected Mass": self["Mprom"][np.isin(self["line_num"], field_lines)].sum(),
                     "Ejected Line IDs": np.array(field_lines),
                     "Harmonic": harmonic,
                     "Tau": tau,
                     "Epsilon": epsilon,
                     "Sigma": sigma,
                     "Frequency bin edges": bin_edges}
        
        dyn_spec = ModelDynamicSpectrum(dyn_spec)
        dyn_spec.meta.update(spec_meta)

        # Adding UID info
        dyn_spec.meta["UID"] = dyn_spec.uid
        dyn_spec.meta["Parent UID"] = self.uid
        
        return dyn_spec


    



   
