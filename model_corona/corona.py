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
    def _loaded(self):
        if not hasattr(self, "_data_loaded"):
            # Should only happen when object was initialised with data in it
            # This is really me getting around not having access to __init__
            # There is probably a better way but here we are
            self._data_loaded = True
        return  self._data_loaded

    @_loaded.setter
    def _loaded(self, value):
        self._data_loaded = value
    
    @classmethod
    def from_meta(cls, metadata):
        """
        Initialize (empty) image object from existing metadata
        """
        size = metadata.get("Image size", [0,0]*u.pix).value.astype(int)
        instance = cls(np.full(size, np.nan))
        instance._loaded = False
        instance.meta.update(metadata)
        return instance

    @classmethod
    def load(cls, filename): 
        pass

    def load(self):
        
        with open(Path(self.meta["Directory"]).joinpath(self.meta["Filename"]), "r") as FLE:
            sdict = json.load(FLE)

        memfile = io.BytesIO()
        memfile.write(sdict.pop("array").encode('latin-1'))
        memfile.seek(0)
        self[:] = np.load(memfile)
        self._set_unit(u.Unit(sdict.pop("unit")))

        # TODO: checking
        self.meta.update(_read_serialized(sdict))
        
        self._loaded = True
        
        
    def to_value(self, unit=None, equivalencies=[]):
        if self._loaded == False:
            self.load()
        
        return super().to_value(unit, equivalencies)
        
    value = property(
        to_value,
        doc="""The numerical value of this instance.

    See also
    --------
    to_value : Get the numerical value in a given unit.
    """,
    )
        
    
    def write(self, *, base_dir=None, filename=None):

        # TODO: add checking for existing file
        # add better default filename
        
        self.meta["Directory"] = base_dir if base_dir is not None else self.meta.get("Directory", ".")
        self.meta["Filename"] = filename if filename is not None else self.meta.get("Filename", "image.json")

        content_dict = {}
        content_dict["meta"] = make_serializable(self.meta)
        content_dict["unit"] = self.unit.to_string()

        memfile = io.BytesIO()
        np.save(memfile, self.value)
        content_dict["array"] = memfile.getvalue().decode('latin-1')

        with open(Path(self.meta["Directory"]).joinpath(self.meta["Filename"]), "w") as FLE:
            json.dump(content_dict, FLE)
        
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
    def _loaded(self):
        if not hasattr(self, "_data_loaded"):
            # Should only happen when object was initialised with data in it
            # This is really me getting around not having access to __init__
            # There is probably a better way but here we are
            self._data_loaded = True
        return  self._data_loaded

    @_loaded.setter
    def _loaded(self, value):
        self._data_loaded = value

    #def __getattribute__(self,name):  # Load data as needed (hopefully)
    #    if (name not in ('_loaded','_meta', 'from_meta', '_data_loaded')) and (self._data_loaded is False):
    #        self.load()
    #    return QTable.__getattribute__(self, name)
            
    @classmethod
    def from_meta(cls, metadata):
        """
        Initialize (empty) image object from existing metadata
        """
        instance = cls()
        instance._loaded = False
        instance.meta.update(metadata)
        return instance

    def load(self):

        temp = QTable.read(Path(self.meta["Directory"]).joinpath(self.meta["Filename"]))

        #TODO: checking
        self.update(temp)
        self._loaded = True


class ModelCorona:

    meta = MetaData(copy=False)
    
    def __init__(self, metadata=None):

        self.field_lines = QTable()
        self.images = list()
        self.cubes = list()
        self._path = None
        
        if metadata:
            self.meta.update(metadata)
        else:
            self._initialize_metadata()

    @classmethod
    def from_field_lines(cls, input_table, distance=None, obs_freq=None):
        """
        TODO: write docstring
        """

        instance = cls()

        # Ingesting the table
        instance.field_lines = input_table.copy()  # Copying for safety

        # Additional processing
        instance.field_lines.meta["Total Prominence Mass"] = input_table['Mprom'].sum()
        
        if distance is not None:
            instance.set_distance(distance)

        if obs_freq is not None:
            instance.set_obs_freq(obs_freq)

        instance.meta["model_params"] = instance.field_lines.meta

        return instance
        
    def _initialize_metadata(self):

        self.meta.update({"model_params": None,
                          "image_params": list(),
                          "cube_params": list()})

    @classmethod    
    def load(cls, metafile):
        """
        TODO
        """
        
        # Read in and load metadata
        with open(metafile, 'r') as FLE:
            meta = json.load(FLE)
            
        instance = cls(read_serialized(meta))
        instance._path = Path(metafile).parent

        # Reading in the field lines
        instance.field_lines = QTable.read(instance._path.joinpath(instance.meta["model_params"]["Filename"]))

        # ADD DATA CHECK
        instance.meta["model_params"] = instance.field_lines.meta  # reconnecting the metadata
        
        # Setting up the Image objects
        instance.images = list([None]*len(instance.meta["image_params"]))
        for i, image_meta in enumerate(instance.meta["image_params"]):
            instance.images[i] = RadioImage.from_meta(image_meta)
            instance.images[i]._base_dir = instance._path
            instance.meta["image_params"][i] = instance.images[i].meta  # reconnecting the metadata


        # Setting up the Cube objects
        instance.cubes = list([None]*len(instance.meta["cube_params"]))
        for i, cube_meta in enumerate(instance.meta["cube_params"]):
            instance.cubes[i] = RadioCube.from_meta(cube_meta)
            instance.cubes[i]._base_dir = instance._path
            instance.meta["cube_params"][i] = instance.images[i].meta  # reconnecting the metadata


        return instance

    
    def write(self, metafile, overwrite=True):
        """
        TODO
        """

        meta_path = Path(metafile)
        base_path = meta_path.parent

        if not overwrite and  meta_path.exists():
            raise OSError(f'{meta_path} already exists. '
                          'If you mean to replace it then use the argument "overwrite=True".')

        data_dir = Path(meta_path.stem+"_data")
        base_path.joinpath(data_dir).mkdir(exist_ok=True)

        # Write field line table
        fieldline_file = data_dir.joinpath("field_lines.ecsv")
        self.field_lines.meta["Filename"] = str(fieldline_file)
        self.field_lines.write(base_path.joinpath(fieldline_file), overwrite=True)
        
        # Write all images
        for image in self.images:
            image_file = data_dir.joinpath(f'img_{image.meta["Hash"]}.json')
            image.meta["Filename"] = str(image_file)
            image.meta["Directory"] = str(base_path)
            image.write()
            
        # Write all cubes
        for cube in self.cubes:
            cube_file = data_dir.joinpath(f'cube_{cube.meta["Hash"]}.ecsv')
            cube.meta["Filename"] = str(cube_file)
            cube.meta["Directory"] = str(base_path)
            cube.write(Path(cube.meta["Directory"]).joinpath(cube.meta["Filename"]), format="ascii.ecsv")

        # Write metadata
        content_dict = make_serializable(self.meta)
        with open(meta_path, "w") as FLE:
            json.dump(content_dict, FLE)
        
        

    def set_obs_freq(self, obs_freq):
        """
        Doing all setup that requires the observation frequency.

        Added columns:
        - blackbody : blackbody emission at the observing frequency (erg / (cm2 Hz s sr))
        - kappa_ff : free-free absorption coefficient at the observing frequency (cm^-1)
        """

        field_table = self.field_lines
        
        if isinstance(obs_freq, float):
            obs_freq = obs_freq * u.GHz # assume GHz by default

        # Doing blackbody calculations
        bb_corona = BlackBody(temperature=field_table.meta["Corona Temperature"] )
        bb_prominence = BlackBody(temperature=field_table.meta["Prominence Temperature"] )

        wind = field_table["line_num"] < 0
    
        field_table["blackbody"] = bb_corona(obs_freq)
        field_table["blackbody"][field_table["proms"]] = bb_prominence(obs_freq)
        field_table["blackbody"][wind] = 0
    
        # Calculating the free-free absorption coefficient
        with warnings.catch_warnings(): # suppressing divide by zero warning on wind points
            warnings.simplefilter("ignore")
            field_table["kappa_ff"] = kappa_ff(field_table["temperature"], obs_freq, field_table["ndens"])
        field_table["kappa_ff"][wind] = 0

        # Recording the observation frequency
        field_table.meta["Observation Frequency"] = obs_freq

    def set_distance(self, distance):
        """
        Setting the distance to the star.
        """

        if not isinstance(distance, u.Quantity):
            distance *= u.pc  # Assume parsecs
        self.field_lines.meta["Distance"] = distance.to(u.pc)
            

    def add_cartesian_coords(self, obs_angle=(0,0), phase=0):
        """
        Given a viewing angle and optional phase add columns with the cartesian coordinats for each row.
        Assumes field_table has columns radius, theta, phi

        Note, this goes into a right handed cartesian coordinate system.
        """

        field_table = self.field_lines

        obs_angle = parsed_angle(obs_angle)
        phi0, theta0 = obs_angle
        phase = parsed_angle(phase)
    
        phi = field_table["phi"]+phase
        theta = field_table["theta"]
        r = field_table["radius"]
    
        field_table["x"] = r * (np.cos(theta0)*np.cos(theta) + np.sin(theta0)*np.sin(theta)*np.cos(phi-phi0))
        field_table["y"] = r * np.sin(theta)*np.sin(phi-phi0)
        field_table["z"] = r * (np.sin(theta0)*np.cos(theta) - np.cos(theta0)*np.sin(theta)*np.cos(phi-phi0))

        field_table.meta["Observation angle"] = obs_angle.to(u.deg)
        field_table.meta["Phase"] = phase.to(u.deg)


    def make_radio_image(self, sidelen_pix, *, sidelen_rad=None, obs_angle=(0,60), phase=0):
        """
        Given a magnetic field table with necessary columns (TODO: specify)
        and a pixel size, create an intensity image along the x-direction
        line of sight. (Only does square image)
        """

        distance = self.meta["model_params"].get("Distance")
        if distance is None:
            warnings.warn("No distance found, the returned image will be in intensity rather than flux.\n"
                          "Set distance with `ModelCorona.set_distance`.")

        obs_angle = parsed_angle(obs_angle)
        phase = parsed_angle(phase)
        if not (((obs_angle.to(u.deg) != self.field_lines.meta.get("Observation angle")).any()) and
                (phase.to(u.deg) != self.field_lines.meta.get("Phase"))):
            self.add_cartesian_coords(obs_angle, phase)

        if not sidelen_rad:
            rss = self.field_lines.meta["Source Surface Radius"]/self.field_lines.meta["Radius"]
            px_sz = 2*rss/sidelen_pix
        else:
            px_sz = sidelen_rad/sidelen_pix
        
        image = make_radio_image(self.field_lines, sidelen_pix, sidelen_rad, distance)   
        image_meta = {"Observing frequency": self.field_lines.meta["Observation Frequency"],
                      "Observation Angle": self.field_lines.meta["Observation angle"],
                      "Stellar Phase":  self.field_lines.meta["Phase"],
                      "Image size": (sidelen_pix, sidelen_pix)*u.pix,
                      "Pixel size": px_sz * self.field_lines.meta["Radius"]}

        if distance is not None:
            image_meta["Distance"] = distance
            image_meta["Total Flux"] = np.sum(image)

        image = RadioImage(image)
        image.meta.update(image_meta)
        image.meta["Hash"] = md5(image).hexdigest()

        self.meta["image_params"].append(image.meta)
        self.images.append(image)

        return image

    
    def make_radio_phase_cube(self, num_steps, sidelen_pix, *, sidelen_rad=None,
                              min_phi=0*u.deg, max_phi=360*u.deg, obs_angle=(0,60)*u.deg):
        """
        TODO: FIX
        Do the prep work for animating the images, and making flux vs phase plots.
        """

        # Checking on the distance
        distance = self.meta["model_params"].get("Distance")
        if distance is None:
            warnings.warn("No distance found, the returned image will be in intensity rather than flux.\n"
                          "Set distance with `ModelCorona.set_distance`.")
        
        # Regularising the angles
        min_phi = parsed_angle(min_phi)
        max_phi = parsed_angle(max_phi)
        obs_angle = parsed_angle(obs_angle)
    
        # Get the phases we want
        phase_list = np.linspace(min_phi, max_phi, num_steps)

        # Go ahead and to this calculation here
        if not sidelen_rad:
            sidelen_rad = 2*self.field_lines.meta["Source Surface Radius"]/self.field_lines.meta["Radius"]
        px_sz = sidelen_rad/sidelen_pix
        
        cube_dict = {"phi":[], "flux":[], "separation":[], "image":[]}
        #pix_sz = 2*(field_table.meta["Source Surface Radius"]/field_table.meta["Radius"])/width_pix
    
        for phase in phase_list:

            self.add_cartesian_coords(obs_angle, phase)
            image = make_radio_image(self.field_lines, sidelen_pix, sidelen_rad, distance)
            lobes = get_image_lobes(image, px_sz)
        
            cube_dict["phi"].append(phase.to('deg'))
            cube_dict["flux"].append(np.sum(image))
            cube_dict["separation"].append(lobes.meta["Separation"])
            cube_dict["image"].append(image)
        
        cube_table = RadioCube(cube_dict)

        cube_meta = {"Observing frequency": self.field_lines.meta["Observation Frequency"],
                     "Observation Angle": self.field_lines.meta["Observation angle"],
                     "Image size": (sidelen_pix, sidelen_pix)*u.pix,
                     "Pixel size": px_sz * self.field_lines.meta["Radius"],
                     "Average Flux": cube_table["flux"].mean(),
                     "Average Separation": cube_table["separation"].mean()}
        cube_table.meta.update(cube_meta)
        cube_table.meta["Hash"] = md5(cube_table.as_array()).hexdigest()


        self.meta["cube_params"].append(cube_table.meta)
        self.cubes.append(cube_table)

        return cube_table
