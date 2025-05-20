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
    """
    Quantitative array with associated metadata, supporting serialization to and from JSON files.
    Base class for array-based synthetic observation classes.

    Parameters
    ----------
    value : array-like
        Numeric data array.
    unit : ~astropy.units.Quantity or str or ~astropy.units.Unit
        Unit for the array values. Can be an `astropy.units.Unit` object, a
        `~astropy.units.Quantity` (unit part is used), or a string parsable
        by `u.Unit`.
    meta : dict, optional
        Initial metadata dictionary.
        Defaults to an empty metadata container.

    Attributes
    ----------
    meta : MetaData
        Metadata container for storing arbitrary key–value pairs.
    unit : ~astropy.units.Unit
        Physical unit of the array values.
    value : numpy.ndarray
        Underlying numeric data array.

    Notes
    -----
    Inherits from `~astropy.units.Quantity`, so all Quantity operations
    apply to `ModelArray` instances.
    """

    meta = MetaData()

    def write(self, filename):
        """
        Serialize this ModelArray to a JSON file.

        The output file will contain:
        - `"meta"`: metadata dictionary, JSON-serializable.
        - `"unit"`: unit string, as from `~astropy.units.Unit.to_string()`.
        - `"array"`: raw array data encoded in Latin-1 after being saved
          with `numpy.save`.

        Parameters
        ----------
        filename : str
            Path to the JSON file to write. Overwrites existing files.

        Raises
        ------
        IOError
            If the file cannot be written.
        """

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
        """
        Deserialize a ModelArray from a JSON file.

        Reads the JSON file written by `write()`, reconstructs the array,
        unit, and metadata.

        Parameters
        ----------
        filename : str
            Path to the JSON file to read.

        Returns
        -------
        ModelArray
            New instance created from the file contents.

        Raises
        ------
        IOError
            If the file cannot be read.
        """

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
    """
    Class for holding synthetic image data.
    Quantitative array with associated metadata, supporting serialization to and from JSON files.
    
    Parameters
    ----------
    value : array-like
        Numeric data array.
    unit : ~astropy.units.Quantity or str or ~astropy.units.Unit
        Unit for the array values. Can be an `astropy.units.Unit` object, a
        `~astropy.units.Quantity` (unit part is used), or a string parsable
        by `u.Unit`.
    meta : dict, optional
        Initial metadata dictionary.
        Defaults to an empty metadata container.

    Attributes
    ----------
    meta : MetaData
        Metadata container for storing arbitrary key–value pairs.
    unit : ~astropy.units.Unit
        Physical unit of the array values.
    value : numpy.ndarray
        Underlying numeric data array.

    Notes
    -----
    Inherits from `~astropy.units.Quantity`, so all Quantity operations
    apply to `ModelImage` instances.
    """

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
    """
    Class for holding a synthetic dynamic spectrum.
    Quantitative array with associated metadata, supporting serialization to and from JSON files.
    
    Parameters
    ----------
    value : array-like
        Numeric data array.
    unit : ~astropy.units.Quantity or str or ~astropy.units.Unit
        Unit for the array values. Can be an `astropy.units.Unit` object, a
        `~astropy.units.Quantity` (unit part is used), or a string parsable
        by `u.Unit`.
    meta : dict, optional
        Initial metadata dictionary.
        Defaults to an empty metadata container.

    Attributes
    ----------
    meta : MetaData
        Metadata container for storing arbitrary key–value pairs.
    unit : ~astropy.units.Unit
        Physical unit of the array values.
    value : numpy.ndarray
        Underlying numeric data array.

    Notes
    -----
    Inherits from `~astropy.units.Quantity`, so all Quantity operations
    apply to `ModelDynamicSpectrum` instances.
    """

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
    """
    Table of synthetic observations accross rotation phases with associated
    metadata for frequency, angle, and pixel properties.

    Inherits from `~astropy.table.QTable`, adding convenience properties for
    accessing metadata fields related to the associated corona model

    Parameters
    ----------
    data : array-like or dict or `~astropy.table.Table`
        Table data to initialize the PhaseCube.
    meta : dict, optional
        Initial metadata dictionary. Keys should match expected properties
        (e.g., 'Observation frequency', 'Pixel size'). Defaults to empty.

    Attributes
    ----------
    meta : dict
        Metadata container storing observational parameters.
    columns : `~astropy.table.Column` instances
        Columns inherited from `QTable`.
    """

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
    """
    Table of synthetic observations accross observing frequencies with
    associated metadata for frequency, angle, and pixel properties.

    Inherits from `~astropy.table.QTable`, adding convenience properties for
    accessing metadata fields related to the associated corona model.

    Parameters
    ----------
    data : array-like or dict or `~astropy.table.Table`
        Table data to initialize the PhaseCube.
    meta : dict, optional
        Initial metadata dictionary. Keys should match expected properties
        (e.g., 'Observation frequency', 'Pixel size'). Defaults to empty.

    Attributes
    ----------
    meta : dict
        Metadata container storing observational parameters.
    columns : `~astropy.table.Column` instances
        Columns inherited from `QTable`.
    """

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
    """
    Table of model coronal field–line data with computed properties and metadata.

    Wraps an `~astropy.table.QTable` containing stellar corona field‐line columns
    (e.g., `radius`, `theta`, `phi`, `Bmag`, `ndens`, `temperature`, `line_num`)
    and adds methods and metadata for modeling coronal emission, absorption,
    geometry, and dynamics.

    Notes
    -----
    - Many properties and methods assume the presence of specific metadata
      entries; missing keys typically return `None` or raise an error if required.
    - All quantities stored in `meta` should be `~astropy.units.Quantity` where
      appropriate, and methods will interpret plain numerics in standard units
      (parsecs, solar radii, GHz, degrees) when setting metadata.
    - The class uses MD5 hashing of core data arrays to generate stable UIDs.
    """

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
        `ModelCorona`
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
            # TODO: add check for other ways it could be invalid
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
        """
        Returns
        -------
        `~astropy.units.Quantity`
            Distance to the star in parsecs.

        When setting this property, if the input is not a `~astropy.units.Quantity`,
        it is assumed to be in parsecs and converted accordingly. 
        """
        return self.meta.get("Distance")

    @distance.setter
    def distance(self, value):
        if not isinstance(value, u.Quantity):
            value *= u.pc  # Assume parsecs
        self.meta["Distance"] = value.to(u.pc)

    @property
    def observation_freqs(self):
        """
        Returns
        -------
        `~astropy.units.Quantity`
            Observation frequencies for which the free-free
            absoption coefficient has been calculated.
        """
        return self.meta.get('Observation frequencies', []*u.GHz)

    def add_observation_freq(self, obs_freq, cache=True):
        """
        Add a frequency-specific column for free-free absorption coefficient.

        Computes the free-free absorption coefficient (`kappa_ff`) at a given
        observation frequency and stores the result in a column named
        "<obs_freq> Kappa_ff". If the frequency has already been computed and
        `cache` is True, the method skips recomputation.

        Parameters
        ----------
        obs_freq : float or astropy.units.Quantity
            The observation frequency to compute `kappa_ff` for. If not a Quantity,
        the value is assumed to be in GHz.
    
        cache : bool, optional
            Whether to skip recomputation if the frequency already exists in
        the metadata. Default is True.
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
        """
        Remove one or more observation frequencies and their associated data.

        Deletes the `<freq> Kappa_ff` columns and removes the corresponding
        frequencies from the metadata. By default, all frequencies are cleared.

        Parameters
        ----------
        obs_freqs : str, scalar, Quantity, or list of Quantity, optional
            The observation frequency or list of frequencies to remove. Can be:
            - "all" (default): removes all stored observation frequencies.
            - Single frequency (e.g., 5.0 or 5.0 * u.GHz).
            - List or array of frequencies.
        """
        if obs_freqs == "all":
            obs_freqs = self.observation_freqs
        elif (not isinstance(obs_freqs, u.Quantity) and  np.isscalar(obs_freqs)) or \
             (isinstance(obs_freqs, u.Quantity) and  np.isscalar(obs_freqs.value)): 
            obs_freqs = [obs_freqs]

        for freq in obs_freqs:
            freq = normalize_frequency(freq)

            if not freq in self.observation_freqs: # Freq is not actually in our table
                continue

            self.remove_column(f"{freq} Kappa_ff")
            
            self.meta['Observation frequencies'] = np.delete(self.meta['Observation frequencies'],
                                                             np.where(self.meta['Observation frequencies'] == freq))

        
    def add_cartesian_coords(self, obs_angle, phase=0, recalculate=False):
        """
        Compute and add Cartesian coordinates (x, y, z) from spherical coordinates.

        Given a viewing angle (`obs_angle`) and an optional rotation (`phase`),
        this function computes the Cartesian coordinates assuming a right-handed
        coordinate system and stores them in columns ('x', 'y', 'z') in the table.

        The transformation assumes the existence of `radius`, `theta`, and `phi`
        columns. If the coordinates have already been computed for the given angle
        and phase and `recalculate` is False, the computation is skipped.

        Parameters
        ----------
        obs_angle : float, tuple, or Quantity
            Observation angle as a tuple (phi, theta) or parsable input.
            Units should be convertible to degrees.

        phase : float or Quantity, optional
            Rotation to apply around the z-axis (in degrees). Default is 0.

        recalculate : bool, optional
            If False, avoids recomputation when current coordinates already match
            the provided angle and phase. Default is False.
        """

        obs_angle = parsed_angle(obs_angle)
        phase = parsed_angle(phase)

        # Check if we actually need to redo the calculation or not
        if ((not recalculate) and
            (self.meta.get("Phase") == phase) and
            (np.isclose(self.meta.get("Observation angle", [np.nan]*2*u.deg), obs_angle).all()) and
            ("x" in self.colnames)):
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
        """
        Returns
        -------
        `~astropy.units.Quantity` or None
            The observation angle (phi, theta) in degrees, used to compute the
            Cartesian coordinate (x, y, z) columns.
            None if not set.

        When setting, if no units are given degrees is assumed.
        """
        return self.meta.get('Observation angle')

    @observation_angle.setter
    def observation_angle(self, value):
        self.add_cartesian_coords(value, self.meta.get('Phase',0))

    @property
    def phase(self):
        """
        Returns
        -------
        `~astropy.units.Quantity` or None
           Value in degrees giving the phase offset applied to the
           coordinate transformation for the Cartesian coordinate (x, y, z) columns.
            None if not set.
        """
        return self.meta.get('Phase')

    @phase.setter
    def phase(self, value):
        if not isinstance(self.observation_angle, u.Quantity):
            raise AttributeError("You cannot set a phase with no Observation angle in place.")
        self.add_cartesian_coords(self.observation_angle, value)
 
    @property
    def corona_temp(self):
        """
        Returns
        -------
        `~astropy.units.Quantity`
            The temperature of the corona.
            Calculated from the table, and therefore not settable.
        """
        return self.meta.get("Corona Temperature")

    @property
    def prom_temp(self):
        """
        Returns
        -------
        `~astropy.units.Quantity`
            The temperature of the stellar prominences.
            Calculated form the table, and therefore not settable.
        """
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
        """
        Returns
        -------
        `~astropy.units.Quantity`
            The stellar radius.
            Intrinsic to the model and therefore not settable.
        """
        return self.meta.get('Radius')

    @property
    def rss(self):
        """
        Returns
        -------
        `~astropy.units.Quantity`
            The model extent, i.e. source surface radius.
            Intrinsic to the model and therefore not settable.
        """
        return self.meta.get('Source Surface Radius')

    @property
    def uid(self):
        """
        Returns
        -------
        str
            Unique identifier string for this model.
        """
        uid = self.meta.get('UID')
        if uid is None:
            uid = md5(self['radius', 'theta', 'phi', 'Bmag', 'proms'].as_array()).hexdigest()
        return uid

    @property
    def wind(self):
        """
        Returns
        -------
        `numpy.ndarray`
             Boolean mask for wind cells in the corona model.
        """
        return self["wind"]

    @property
    def prom(self):
        """
        Returns
        -------
        `numpy.ndarray`
             Boolean mask for prominence cells in the corona model.
        """
        return self["proms"]

    @property
    def cor_only(self):
        """
        Returns
        -------
        `numpy.ndarray`
            Boolean mask for model cells that are in the closed corona (not the wind)
            but do not contain prominences.
        """
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
        Calculate and add the plasma β (ratio of plasma pressure to magnetic pressure) for each cell:
        
        .. math::
            \beta = \frac{n k_B T}{B^2 / 2 \mu_0}

        where:
            - :math:`n` is the number density,
            - :math:`k_B` is the Boltzmann constant,
            - :math:`T` is the temperature,
            - :math:`B` is the magnetic field magnitude,
            - :math:`\mu_0` is the permeability of free space.

        The value is only computed for non-wind cells, since wind regions have
        `Bmag = 0` and would result in a division by zero.

        Adds a new column `"plasma_beta"` to the table.
        """
        
        self.add_column(Column(name="plasma_beta", length=len(self))) # TODO: check for existing column

        p_plasma = self[~self.wind]["ndens"]*c.k_B*self[~self.wind]["temperature"]
        p_mag = self[~self.wind]["Bmag"]**2/(2*c.mu0)

        self["plasma_beta"][~self.wind] = (p_plasma/p_mag).to("")

    def _add_bb_col(self, obs_freq):
        """
        Add a blackbody column for a given observing frequency/
        
        The blackbody column is not labeled by frequency so this function will
        be frequency overwritten, and in general should not be called by the user. 
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
        `ModelImage`
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

        image = freefree_image(self, sidelen_pix, sidelen_rad, self.distance, kff_col=f"{obs_freq} Kappa_ff")   
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
        Generate a cube of radio images across a range of rotational phases.

        This method computes free-free emission images at a given observation frequency
        and viewing angle across `num_steps` rotational phases. The resulting data includes
        flux, lobe separations, peak counts, and angular separations for each phase.

        Parameters
        ----------
        obs_freq : float or `~astropy.units.Quantity`
            The observation frequency. Will be converted to GHz if not already.

        num_steps : int
            Number of evenly spaced phases to generate between `min_phi` and `max_phi`.

        sidelen_pix : int
            Number of pixels per side of the square image.

        beam_size : float or `~astropy.units.Quantity`
            Size of the synthetic observational beam (for peak/lobe identification).

        sidelen_rad : float or `~astropy.units.Quantity`, optional
            Physical size of the image in angular units (radians). If not provided,
            computed from the source surface radius and stellar radius.

        obs_angle : float, tuple, or `~astropy.units.Quantity`, optional
            Observation angle as (phi, theta) in degrees. If None, will use previously set value.

        min_phi : `~astropy.units.Quantity`, optional
            Starting rotational phase angle in degrees. Default is 0°.

        max_phi : `~astropy.units.Quantity`, optional
            Ending rotational phase angle in degrees. Default is 360°.

        Returns
        -------
        `PhaseCube`
            A PhaseCube object containing:
                - phi: phase angle for each image
                - flux: total flux from each image
                - separation: maximum peak separation 
                - num_peaks: number of emission peaks
                - ang_sep: maximum angular separation
                - image: 2D flux image (or intensity if distance is not set)
        
        Metadata includes observation frequency, angles, image specs, and summary stats.

        Raises
        ------
        AttributeError
            If no observation angle is provided or previously set.

        Warnings
        --------
        UserWarning
            If the stellar distance is missing, intensity is returned instead of flux.
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
            image = freefree_image(self, sidelen_pix, sidelen_rad, self.distance, kff_col=f"{obs_freq} Kappa_ff")
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

    
    def dynamic_spectrum(self, freqs, phases, field_lines, tau, epsilon=1e-5, sigma=1*u.deg,
                         harmonic=1, distance=None, obs_angle=None):
        """
        Generate the electron cyclotron maser (ECM) dynamic spectrum.

        This method calculates the ECM emission resulting from the election of stellar prominences
        for a given set of frequencies and rotational phases. 

        Parameters
        ----------
        freqs : int or `astropy.units.Quantity`
            Frequencies for the dynamic spectra. Can be:
            - int: number of frequency bins spanning the range of ECM emission
            - Quantity: Array of frequency bin edges
        
        phases : int or `astropy.units.Quantity`
            Phases for the dynamic spectra. Can be:
            - int: number of phase bins (0-360 deg)
            - Quantity: Array of phases
        
        field_lines : str, int, or array-like
            Magnetic field lines to include in the calculation. Can be:
            - "prom": selects all lines marked as prominences in the model,
            - int: a single field line number,
            - list of ints: specific field lines.
        
        tau : `astropy.units.Quantity`
            Disipation timescale
        
        epsilon : float, optional
            ECM efficiency factor, default 1e-5.
        
        sigma : Quantity or float, optional
            A Width of the ECM emission cone, by default 1 deg, per TODO: citation.
        
        harmonic : int, optional
            Harmonic number to use for ECM emission, by default 1 (fundamental).
        
        distance : Quantity, optional
            Distance to the object. If None, uses `self.distance`.
        
        obs_angle : Quantity or float, optional
            Observation angle. If None, uses `self.observation_angle`.

        Returns
        -------
        `ModelDynamicSpectrum`
            The computed dynamic spectrum, with metadata including observation
            angle, frequencies, field lines, tau, and harmonic.

        Raises
        ------
        AttributeError
            If distance or observation angle is not provided and not set on the object.
        ValueError
            If ECM fraction is zero for all field lines.
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

        if (self["ecm_frac"] == 0).all(): 
            raise ValueError("No ECM possible.")

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


    



   
