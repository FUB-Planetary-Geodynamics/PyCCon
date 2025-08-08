# -*- coding: utf-8 -*-
"""
Created in Dec 2022

@author: Lena Noack
"""
import math

####################################
# don't change anything below for  #
# your current simulation, just    #
# reset the appropriate parameters #
# in input.data.py!                #
####################################    

def input(self): 

    self.geom = 0 # 0 - box, 1 - cylinder, 2 - spherical annulus
    self.regional = 0.25 # only used for geom=1/2; 1 is full anulus, 0.25 quarter, etc.
    self.periodic = 0
    self.corrRot = 0
    self.nl = 40
    self.nr = 40
    self.rmin = 1
    self.rmax = 2
    self.lmin = 0
    self.lmax = 1

    self.figsize = (1,1) # if (1,1) then calculated based on number of figures and aspect ratio
    self.Plx = 0 # if 0 then set based on number of figures
    self.Ply = 0 # if 0 then set based on number of figures
    self.plot = 'Tvuw' # T-Temp,v-vel,u-lat vel,w-rad vel,c-compgrid,C-composition,V-visc,p-pressure
    self.plot_vel = 'std' # 'ang' for angular velocity (radians/time), 'std' is standard velocity (length/time)
    self.output = 'Default_Ra1e4_40x40' # don't add "txt" or similar 
    self.showplot = 0 # if 1, show figures in notebook (not advised for large simulations)
    self.saveplot = 1 # if 1, save figures in Google Drive

    self.free_slip_t = 1 #True
    self.free_slip_b = 1 #True
    self.inner_bound = False
    self.botv = 0

    self.Ttop = 0
    self.Tbot = 1
    self.T0 = 0.1
    self.Tamp = 0.05 # intial sin T field perturbation
    self.sph = 1 # mode of spherical harmonics (if 0, random perturbation is applied)
    self.Tprofile = 1 # 0->use Tini and TBL, 1-> linear, 2-> conductive profile
    self.Tini = 0.5
    self.TBLt = 0.2
    self.TBLb = 0.2

    self.Cdt = 1
    self.dt_ini = 1e-6
    self.dt_min = 1e-6
    self.dt_max = 1e-3
    self.nbiter = 20 # if set to zero, then use tmax
    self.figiter = 1 # every ... time steps create a new figure; set to 0 to use figtime
    self.figtime = 0 # every ... time create a new figure
    self.tmax = 0

    self.T_ref = 0.0
    self.z_ref = 0.0 # reference depth, defined via reference pressure
    self.eta_ref = 1.0
    self.gamma_T = 0.0 # FK Viscosity: gamma_T - set this or exp value below
    self.gamma_p = 0.0 # FK Viscosity: gamma_p
    self.e_gamma_T = 1.0#e1 # FK Viscosity: exp(gamma_T)
    self.e_gamma_p = 1.0 # FK Viscosity: exp(gamma_p)
    self.eta_min = 0.0 # ignored if 0
    self.eta_max = 0.0 # ignored if 0
    self.E = 0.0
    self.V = 0.0
    self.ys_0 = 0.0
    self.ys_z = 0.0
    self.ys_eta = 0.0 # ignored if 0

    self.Ra = 1e4
    self.H0 = 0
    self.Hlambda = 0

    self.Cref = 1
    self.Clinear = 0
    self.Cbot = 0
    self.Ctop = 1
    self.Clayers = 0
    self.Clayer_thick = 0.3 # attached at bottom, will get Cbot value, rest Ctop value
    self.Cmin = 0 # composition plot min
    self.Cmax = 1 # composition plot max
    self.Cmap = 'RdGy' # composition color map 

    self.B = 0
    self.Le = 1e6
    self.Cimage = ''
    self.Image_fliplr = False
    self.Image_flipud = False
    self.Image_transpose = False
    self.np = 0 # 3*self.nl*self.nr # number of particles, set to 0 to use chemical field approach
    self.rk = 4
    self.trace_part = 0 # which particle to trace (plot with 'R'), if 0, no particle is being traced at all

    # for subduction simulation, only employed if velr != 0
    self.weak_visc = 0.01 # visc pre-factor
    self.weak_angle = 45 # angle of weak zone
    self.weak_pos = 2.0 # surface position of weak zone
    self.weak_width = 0.1 # width of weak zone
    self.weak_depth = 0.2 # depth of weak zone
    self.crust_visc = 100  # viscosity pre-factor for overriding plate
    self.crust_depth = 0.2  # thickness of overriding plate
    self.subd_visc = 100  # viscosity pre-factr for subducting plate
    self.subd_depth = 0.2  # thickness of subducting plate
    self.velr = 0 # 100 # velocity of incoming plate from the right
    self.z_in = 0.2 # thickness of inflow region
    self.z_out = 0.5 # thickness of outflow region on the left
    self.z_right = False # outflow on left or right side?
    self.sedi_d = 0 # thickness of sediment layer
    self.sedi_visc = 0.01
    self.sedi_density = 0.8

    self.debug = 0
    #self.upwind = True
    self.max_iter = 50 # max iterations per time step until T/v+p need to converge
    self.solver = 0# 0-FDM, 1-FVM
    self.lin_solver = 0# 0-spsolve, 1-bicg, 2-bicgstab
    self.convT = 1e-6
    self.convv = 1e-4
    self.penalty = 1e-7
    self.compress = 0 # 0-no compressiblity, use BA or EBA; 1-compressible terms in stokes, strain rate etc. and TALA formulation in stokes
    self.Di = 0 # dissipation number, for Earth, Di ~ 0.5
    self.Gr = 1 # Gruneisen parameter, for Earth, Gr ~ 1.2

    self.Tsol0 = 0 # if>0 then melt will be calculated
    self.Tsolz = 0
    self.Tliq0 = 0
    self.Tliqz = 0
    self.CO2_ini = 0
    self.H2O_ini = 0
    self.DIW = 4               # log 10 Delta IW change of oxygen fugacity from IW buffer
    self.D_H2O = 1             # partition coefficient of water
    self.D_CO2 = 1             # partition coefficient of CO2/carbonates, scales with redox state
    self.Chi_extr = 0.2        # amount of extrusive melt compared to total generated magma (0.2 = 20% is extrusive)
    self.kappa = 1             # thermal diffusivity of the mantle

    # dimensional parameters needed for outgassing simulations to have useful outputs
    self.Rp = 2                # Planet radius in m (here top of domain), needed to rescale volatile outgassing values
    self.Rc = 1                # Core radius in m (here bottom of domain)
    self.T0 = 0.1              # surface temperature in K, to calculate melt temp at surface and speciation based on that temperature
    self.DeltaT = 1            # temperature contrast from surface to lower mantle in K, to calculate melt temp at surface
    self.rho_m = 1             # average mantle density in kg/m^3
    self.grav = 1              # surface gravitational acceleration in m/s^2, for calculating atmospheric pressure

    self.snap_iter = 100 # write a snapshot every 200 time steps
    self.read_snap = False # if True, then read last snapshot and start simulation from there

    return