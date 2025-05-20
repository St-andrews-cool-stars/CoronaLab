import sys
import io
import warnings

import numpy as np
from pathlib import Path

from scipy.interpolate import griddata

from astropy.table import Table, QTable, Column, vstack
from astropy.coordinates import SkyCoord, cartesian_to_spherical
from astropy.time import Time

import astropy.constants as c
import astropy.units as u

import sunpy.map
from sunpy.coordinates import HeliographicCarrington
from sunpy.util import SunpyMetadataWarning

from sunkit_magex import pfss as pfsspy
from sunkit_magex.pfss import coords, tracing
from sunkit_magex.pfss.grid import Grid

from corona_lab import corona





class FieldlineProcessor():
    """
    Class processing closed and open magnetic field lines, finding stable points and prominences,
    for corona modeling.

    Parameters
    ----------
    radius : `~astropy.units.Quantity`
        Stellar radius.
    mass : `~astropy.units.Quantity`
        Stellar mass.
    period : `~astropy.units.Quantity`
        Stellar rotation period.
    mean_ptc_mass : `~astropy.units.Quantity`, optional
        Mean particle mass, default is 0.5 * (proton mass + electron mass).
    verbose : bool, optional
        If True, prints verbose output. Default is False.
    """

    def __init__(self, radius, mass, period, mean_ptc_mass=0.5*(c.m_p + c.m_e), verbose=False):

        # Add units checking for these
        self.radius = radius
        self.mass = mass
        self.period = period
        
        self.verbose = verbose
        
        self.mean_ptc_mass = mean_ptc_mass  # default is fully ionized hydrogen

        self.kappa_p = None
        self.T_cor = None
        self.T_prom = None
        self.c_sound = None
        self.sonic = None
        self.wind_vel = None
        self.ang_vel = None

        self.prom_count = 0
        self.nlambda = 4 # number of pressure scale heights for prominence extent

        self.dtheta = None
        self.dphi = None


    @staticmethod
    def _find_stable_pts(radius, pressure, min_rad=1.1):
        """
        Identify stable points along a magnetic field line based on pressure profile.

        Parameters
        ----------
        radius : array_like
            Radius values in units of stellar radii.
        pressure : array_like
            Pressure values corresponding to radius.
        min_rad : float, optional
            Minimum radius to consider as stable, by default 1.1.

        Returns
        -------
        ndarray
            Indices of stable points in the radius array.
        """

        cond1 = (pressure[2:-1] > pressure[1:-2]) & (pressure[2:-1] > pressure[3:])
        cond2 = (pressure[1:-2] == pressure[2:-1])
        cond3 = (pressure[:-3] < pressure[2:-1]) & (pressure[2:-1] == pressure[1:-2])

        stable_inds = np.where((radius[2:-1] > min_rad) & (cond1 | cond2 | cond3))
        stable_inds = stable_inds[0] + 2  # 1D so will be first result, add 2 bc the "i" we are measuring from is the pressure[2:-1]

        return stable_inds
        
            
    def find_prominences(self, fieldline):
        """
        Identify stable points and determine prominence mass along a closed magnetic field line.

        Parameters
        ----------
        fieldline : `~astropy.table.QTable`
            Table representing the closed field line.
            Required columns: "radius", "theta", "phi", "ds", "Bmag", "Brad", "Btheta", "Bphi".
            Optional columns: "s_pos" (path position)

        Returns
        -------
        fieldline : `~astropy.table.QTable`
            Modified field line table with prominence information added, including:
            "proms", "Rhoprom", "Mprom", and cross-sectional/volumetric columns.
        """
        
        if not "s_pos" in fieldline.colnames:
            s_pos = np.cumsum(fieldline["ds"])
            fieldline["s_pos"] = np.concatenate(([0*u.m], s_pos[:-1])).to(self.radius)
        
        summit_ind = np.argmax(fieldline["radius"])

        # this indicates some more needed inputs
        summit_area = self.dtheta * self.dphi * fieldline["radius"][summit_ind]**2
       
    
        # Adding cross sectional area column
        fieldline.add_column(Column(name="dA_c", length=len(fieldline), unit=self.radius**2))

        # Adjusting the cross sectional area such that the loop fits into a cell at the summit
        # See line 1566 in allsp3
        fieldline["dA_c"] = summit_area*fieldline["Bmag"][summit_ind]/fieldline["Bmag"]

        # Getting the cell volume
        fieldline["dV"] = fieldline["dA_c"] * fieldline["ds"]

        # calculate mass flow rate along the flux tube, because this is the same everywhere we
        # can just calculate at the the foot point (because that's easiest)
        # the proper units are wrapped up in kappa_p
        p0 = (self.kappa_p * 0.5*((fieldline["Bmag"][0].to(u.G).value)**2 +
                                 (fieldline["Bmag"][-1].to(u.G).value)**2) * u.Pa)  
        mdot = ((p0/self.c_sound**2) * self.wind_vel * fieldline["dA_c"][0]).si # TODO: is this used?

        bsign = (int(fieldline["Brad"][0].value > 0) * 2) - 1 # 1 or -1 for positive/negative B_rad respectively
 

        # Calculating the r and theta gravity components (there is no phi componant)
        fieldline["gr"] = ((-c.G*self.mass/fieldline["radius"]**2) +
                            (self.ang_vel**2 * fieldline["radius"] * np.sin(fieldline["theta"])**2))
        fieldline["gt"] = (self.ang_vel**2 * fieldline["radius"] *
                            np.sin(fieldline["theta"]) * np.cos(fieldline["theta"]))

        # Effective gravity i.e. the component along the field line (in the ds direction)
        fieldline["g_eff"] = ((fieldline["Brad"]*fieldline["gr"] +
                                fieldline["Btheta"]*fieldline["gt"])/fieldline["Bmag"]).si

        # Calculating the pressure
        integrated_g = np.cumsum((bsign * fieldline["ds"] * (self.mean_ptc_mass/(c.k_B*self.T_cor)) *
                                  fieldline["g_eff"]).si)
        integrated_g -= integrated_g[0]  # Setting the first value to 0 so we get p0 for the foot point I think TODO
        fieldline["pressure"] = p0*np.exp(integrated_g)

        stable_pt_list = self._find_stable_pts(fieldline["radius"].value, fieldline["pressure"], min_rad=1.1)

        if not len(stable_pt_list):
            # Adding the prom columns for table consistency
            fieldline["proms"] = False
            fieldline["Rhoprom"] = 0*u.kg*u.m**-3
            fieldline["Mprom"] = 0*u.kg

            # Removing columns we only needed for internal calculations
            fieldline.remove_columns(("gr", "gt", "g_eff"))
        
            return fieldline

        stable_pt = stable_pt_list[0]  # TODO: Sort out what to do if there are multiple stable points

        # Use Heron's formula to get the radius of curvature
        # (https://artofproblemsolving.com/wiki/index.php/Circumradius)
        a_heron = fieldline["ds"][stable_pt]
        b_heron = fieldline["ds"][stable_pt+1]

        #c_heron = fieldline["coords"][stable_pt-1].separation_3d(line_table["coords"][stable_pt+1])

        r1 = fieldline["radius"][stable_pt-1]
        t1 = fieldline["theta"][stable_pt-1]
        p1 = fieldline["phi"][stable_pt-1]
        
        r2 = fieldline["radius"][stable_pt+1]
        t2 = fieldline["theta"][stable_pt+1]
        p2 = fieldline["phi"][stable_pt+1]
        
        c_heron = np.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * (np.cos(t1)*np.cos(t2)*np.cos(p1-p2) + np.sin(t1)*np.sin(t2)))

        s_heron = 0.5 * (a_heron + b_heron + c_heron)
        area_heron = np.sqrt(np.abs(s_heron * (s_heron-a_heron) * (s_heron-b_heron) * (s_heron-c_heron)))

        R_c = ((a_heron * b_heron * c_heron) / (4 * area_heron)).to(self.radius)

        # find normal component of g at the stable point
        gmag_sq = fieldline["gr"][stable_pt]**2 + fieldline["gt"][stable_pt]**2
        g_norm = np.sqrt(np.abs(gmag_sq - fieldline["g_eff"][stable_pt]**2)).si

        # Calculate the density at the stable point using TODO 
        dens_stable_point = (fieldline["Bmag"][stable_pt]**2 / (c.mu0*g_norm*R_c)).si

        # Check if our prominence is dense enough (needs to be denser than the surronding gas)
        if not (dens_stable_point > (fieldline["pressure"][stable_pt] * self.mean_ptc_mass / (self.T_cor * c.k_B))):
            # TODO this is a repeat of above so idk figure that out
            # Adding the prom columns for table consistency
            fieldline["proms"] = False
            fieldline["Rhoprom"] = 0*u.kg*u.m**-3
            fieldline["Mprom"] = 0*u.kg

            # Removing columns we only needed for internal calculations
            fieldline.remove_columns(("gr", "gt", "g_eff"))
        
            return fieldline

        # This is all the indices within the scale height of the stable point, but not the stable point
        m = ((fieldline["pressure"] < fieldline["pressure"][stable_pt]) & 
             (fieldline["pressure"] > fieldline["pressure"][stable_pt]*np.exp(-self.nlambda*self.T_prom/self.T_cor)))
        
        # Making sure the pressure gradiant also passes (TODO, see paper(s))
        p_diffs = np.concatenate(([0], np.diff(fieldline["pressure"])))

        left_prom = m & (p_diffs > 0)
        left_prom[stable_pt:] = False

        right_prom = m & (p_diffs < 0)
        right_prom[:stable_pt] = False

        if not (left_prom + right_prom).any():
            print("remove stable pt only")
            # TODO this is a repeat of above so idk figure that out
            # Adding the prom columns for table consistency
            fieldline["proms"] = False
            fieldline["Rhoprom"] = 0*u.kg*u.m**-3
            fieldline["Mprom"] = 0*u.kg

            # Removing columns we only needed for internal calculations
            fieldline.remove_columns(("gr", "gt", "g_eff"))
        
            return fieldline
            

        # You only get here if there is a stable point and it can hold a prominence
        self.prom_count += 1
        
        # Getting the final set of prominence points
        fieldline["proms"] = left_prom + right_prom
        fieldline["proms"][stable_pt] = True # TODO: I added this bc I feel like it should be there but idk (ask Moira)
        
        # Adding prominence density and mass
        fieldline.add_column(Column(name="Rhoprom", length=len(fieldline), unit="kg m-3"))
        fieldline["Rhoprom"][fieldline["proms"]] = (dens_stable_point *
                                                      (fieldline["pressure"][fieldline["proms"]]/
                                                       fieldline["pressure"][stable_pt]))

        fieldline["Mprom"] = fieldline["Rhoprom"]*fieldline["dV"]

        # Removing columns we only needed for internal calculations
        fieldline.remove_columns(("gr", "gt", "g_eff"))

        return fieldline


    def process_wind_fieldline(self, fieldline):
        """
        Process an open (stellar wind) magnetic field line.
        Note: This function contains minimal processing, and so should be overwritten
        for modelling that includes real treatment of the wind,

        Parameters
        ----------
        fieldline : `~astropy.table.QTable`
           Table representing the open field line.
           Required columns: "radius", "theta", "phi", "ds", "Bmag", "Brad", "Btheta", "Bphi".
           Optional columns: "s_pos" (path position)
        Returns
        -------
        fieldline : `~astropy.table.QTable`
           Table with wind-related quantities added and placeholder prominence fields.
        """
         
        if not "s_pos" in fieldline.colnames:
            s_pos = np.cumsum(fieldline["ds"])
            fieldline["s_pos"] = np.concatenate(([0*u.m], s_pos)).to(self.radius)
        

        rss_area = self.dtheta * self.dphi * fieldline["radius"][-1]**2

        # Adding cross sectional area column
        fieldline.add_column(Column(name="dA_c", length=len(fieldline), unit=c.R_sun**2))

        # Adjusting the cross sectional area such that the flux tube fits into a cell at the source surface
        # See line 1566 in allsp3 TODO
        # this is a clunky solution to the wind processing problem
        if not (fieldline["Bmag"] == 0*u.G).all():
            fieldline["dA_c"] = rss_area*fieldline["Bmag"][-1]/fieldline["Bmag"]
        else:
             fieldline["dA_c"] = rss_area*0

        # Getting the cell volume
        fieldline["dV"] = fieldline["dA_c"] * fieldline["ds"]

        # Setting pressure to 0 everywhere
        fieldline["pressure"] = 0*u.Pa

        # Adding the prominence columns for table consistency  
        fieldline["proms"] = False
        fieldline["Rhoprom"] = 0*u.kg*u.m**-3
        fieldline["Mprom"] = 0*u.kg

        return fieldline


    def _set_model_constants(self, kappa_power, T_cor, T_prom):
        """
        Set up all model constants needed for finding prominences.

        Parameters
        ----------
        kappa_power : float
            Logarithmic scaling for pressure constant, i.e. kappa_p = 10 ** (-kappa_power).
        T_cor : `~astropy.units.Quantity`
            Corona temperature.
        T_prom : `~astropy.units.Quantity`
            Prominence temperature.
        """

        # Check units for these
        self.kappa_p = 10**-kappa_power
        self.T_cor = T_cor
        self.T_prom = T_prom

        # Calculated quantities
        self.c_sound = np.sqrt(c.k_B * self.T_cor/self.mean_ptc_mass).si # sound speed
        self.sonic = c.G * self.mass / (2. * self.radius * self.c_sound**2) # sonic point in units of Rstar
        self.wind_vel = self.c_sound * self.sonic**2 * np.exp(-2*self.sonic + 1.5) # wind vel at surface

        self.ang_vel = 2*np.pi/self.period # angular velocity (implied units of radians on top)

        
        
    
    def build_model_corona(self, closed_fieldlines, open_fieldlines, rss, kappa_power, T_cor, T_prom,
                           dtheta, dphi, distance=None):
        """
        Construct a model corona from closed and open field lines.
        
        Parameters
        ----------
        closed_fieldlines : list of `~astropy.table.QTable`
            List of closed magnetic field lines.
        open_fieldlines : list of `~astropy.table.QTable`
            List of open magnetic field lines.
        rss : `~astropy.units.Quantity`
            Source surface radius.
        kappa_power : float
            Power-law index for pressure scaling (used in 10^-kappa_power).
        T_cor : `~astropy.units.Quantity`
            Coronal temperature.
        T_prom : `~astropy.units.Quantity`
            Prominence temperature.
        dtheta : `~astropy.units.Quantity`
            Angular resolution in theta.
        dphi : `~astropy.units.Quantity`
            Angular resolution in phi.
        distance : `~astropy.units.Quantity`, optional
            Distance to the star, used for converting to observables.

        Returns
        -------
        model_corona : `~corona.ModelCorona`
            Modeled corona containing processed field lines and associated metadata.
            Ready for synthetic data creation.
        """

        # reset prom count
        self.prom_count = 0

        # Set the grid size
        self.dtheta = dtheta
        self.dphi = dphi
          
        self._set_model_constants(kappa_power, T_cor, T_prom)
        fieldline_tables = []

        if self.verbose:
            print("Looking for prominences.")
            
        # Find prominences
        line_num = 1
        for closed_line in closed_fieldlines:
            #if len(closed_line) < 3:  # Skip the short ones # TODO: deal with the short ones
            #    continue

            line_table = self.find_prominences(closed_line)
            line_table["line_num"] = line_num
            line_num +=1
            
            fieldline_tables.append(line_table)

        if self.verbose:
            print(f"{self.prom_count} prominences found.")
            print("Processing open lines")
        
        # Build the wind lines
        line_num = -1
        for open_line in open_fieldlines:
            line_table = self.process_wind_fieldline(open_line)
            line_table["line_num"] = line_num
            line_num -=1
            
            fieldline_tables.append(line_table)
            
        if self.verbose:
            print("Building model corona")
            
        # build coronal model table
        corona_model = vstack(fieldline_tables)

        
        # make into actual ModelCorona object
        corona_model["temperature"] = self.T_cor
        corona_model["temperature"][corona_model["proms"]] = self.T_prom

        corona_model["ndens"] = ( (corona_model["pressure"]) / (c.k_B*self.T_cor) ).si
    
        # Use the Rhoprom (prominence density) to calculate the prominence number density
        corona_model["ndens"][corona_model["proms"]] = corona_model["Rhoprom"][corona_model["proms"]] / ((c.m_e+c.m_p)/2)

        corona_model = corona.ModelCorona.from_field_lines(corona_model , distance=distance,
                                                           radius=self.radius, rss=rss)

        return corona_model
    
