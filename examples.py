def get_sim_param(self, SimCase):
    # self: adapt data self object depending on simulation case
    # SimCase: names related to different test simulations

    if SimCase == "Melting":
        ###############################################################################################
        # Atmospheric evolution due to volcanic outgassing, uses realistic planetary parameter values #
        ###############################################################################################
        self.geom = 0  # 0 - box, 1 - cylinder, 2 - spherical annulus
        self.nl = 40
        self.nr = 40
        self.np = 3 * self.nl * self.nr  # number of particles, needed for melt calculations
        self.Ra = 1e4

        self.Tsol0 = 0.5  # if self.Tsol0=0, then no melt is considered
        self.Tsolz = 0.55  # Tsol = Tsol0 + z*Tsolz
        self.Tliq0 = 0.8
        self.Tliqz = 0.55  # Tliq = Tliq0 + z*Tliqz
        self.CO2_ini = 2.0e-4  # initial amount of CO2 volatiles in mantle in weight fraction, used to set initial particle values
        self.H2O_ini = 2.0e-4  # initial amount of H2O volatiles in mantle in weight fraction, used to set initial particle values
        self.DIW = 4  # log 10 Delta IW change of oxygen fugacity from IW buffer
        self.D_H2O = 0.01  # partition coefficient of water
        self.D_CO2 = 10 ** (0.3 - self.DIW)  # partition coefficient of CO2/carbonates, scales with redox state
        self.Chi_extr = 0.2  # amount of extrusive melt compared to total generated magma (0.2 = 20% is extrusive)

        # parameters only needed here in main.ipynb for melt/outgassing routines, hence do not set in data object
        self.Rp = 6371000  # Planet radius in m (here top of domain), needed to rescale volatile outgassing values
        self.Rc = 3500000  # Core radius in m (here bottom of domain)
        self.T0 = 300  # surface temperature in K, to calculate melt temp at surface and speciation based on that temperature
        self.DeltaT = 2000  # temperature contrast from surface to lower mantle in K, to calculate melt temp at surface
        self.rho_m = 4500  # average mantle density in kg/m^3
        self.grav = 9.81  # surface gravitational acceleration in m/s^2, for calculating atmospheric pressure
        self.kappa = 1.0e-6  # thermal diffusivity of the mantle

        self.plot = 'TtsvFdgh'

        self.output = 'Melt/Bl_box_isovisc_Ra' + str(int(self.Ra)) + '_' + str(self.nl) + 'x' + str(
            self.nr) + '_Com_' + str(self.compress) + '_Di_' + str(self.Di) + '_Gr_' + str(self.Gr) + '_T0_' + str(
            self.T0) + '_DIW_' + str(self.DIW)

        self.Cdt = -1  # >0 Courant criterium (~1); <0 Delta criterium (~-1)
        self.dt_max = 1e-2
        self.figiter = 5
        self.figtime = 0  # every ... outputs create a new figure
        self.nbiter = 0  # if set to zero, then use tmax
        self.tmax = 0.3

        self.snap_iter = 25
        self.read_snap = False  # Set to True to restart simulation from last-written snapshot

    elif SimCase == "Plasticity":
        ##############
        # Plasticity #
        ##############
        self.Ra = 1e2
        # self.E = 23.03
        # self.T0 = 1#0.1
        self.e_gamma_T = 1.0e5  # FK Viscosity: exp(gamma_T)
        self.e_gamma_p = 1.0e1  # FK Viscosity: exp(gamma_p)
        self.T_ref = 0.0
        self.z_ref = 0.0
        self.eta_min = 1e-5
        self.eta_max = 1e5
        self.ys_0 = 1  # 3 #0
        self.ys_z = 0
        self.ys_eta = 1e-3

        # self.geom = 0 # 0 - box, 1 - cylinder, 2 - spherical annulus
        # self.nl = 40
        # self.nr = 40

        self.rmin = 1
        self.rmax = 2
        self.geom = 2  # 0 - box, 1 - cylinder, 2 - spherical annulus
        self.regional = 1  # only used for geom=1/2; 1 is full anulus, 0.25 quarter, etc.
        self.periodic = 1
        self.nr = 30  # 40
        self.nl = 6 * self.nr

        self.Tprofile = 0  # 0->use Tini and TBL, 1-> linear, 2-> conductive profile
        self.Tini = 0.4

        self.output = 'PT/FK_SphAn_reg_VCT' + str(int(self.e_gamma_T)) + '_VCp' + str(int(self.e_gamma_p)) \
                      + '_Ra' + str(int(self.Ra)) + '_' + str(self.nl) + 'x' + str(self.nr) \
                      + '_YS' + str(self.ys_0) + '_YSz' + str(self.ys_z) + '_Tini' + str(self.Tini)
        # self.output = 'PT/FK_Box_T15_VCT'+str(int(self.e_gamma_T))+'_VCp'+str(int(self.e_gamma_p))\
        #            +'_Ra'+str(int(self.Ra))+'_'+str(self.nl)+'x'+str(self.nr)\
        #            +'_YS'+str(self.ys_0)+'_YSz'+str(self.ys_z)

        self.plot = 'TvpsSV'
        self.sph = 1  # 10
        self.Tamp = 0.1

        self.tmax = 1
        self.Cdt = 1  # -0.001 # >0 Courant criterium (~1); <0 Delta criterium (~-0.1)
        self.nbiter = 0  # if set to zero, then use tmax
        self.figtime = 0  # 0.05
        self.figiter = 25  # 0 # every ... outputs create a new figure

        self.snap_iter = 100
        self.read_snap = False

        self.dt_min = 1e-6
        self.solver = 1  # 0-FDM, 1-FVM
        self.lin_solver = 0  # 0-spsolve, 1-bicg, 2-bicgstab, 3-pypardiso
        self.convT = 1e-7  # 6
        self.convv = 1e-4
        self.debug = 0  # 1

    elif SimCase == "SubductionZone":
        # ##########################################################
        # subduction zone
        # ##########################################################
        self.lmin = 0
        self.lmax = 4  # if lmax-lmin != 1 then aspect ratio of simulation changes

        # self.nl = 180#120
        # self.nr = 45#30
        self.nl = 120  # 180#120
        self.nr = 45  # 30
        self.np = 3 * self.nl * self.nr  # number of particles, set to 0 to use chemical field approach

        self.Tprofile = 0  # 0->use Tini and TBL, 1-> linear, 2-> conductive profile
        self.Tini = 0.7

        self.Ra = 10
        self.B = 27  # nondimensional value for 1/(alpha*DT) with alpha=2.5e-5 and DT=1500 (since only upper mantle)
        self.H0 = 1.3  # nondimensional value for H*D^2/(k*DT) = [W/m^3 * m^2 / (W/(m K)*K)] = [1], today H = 5e-12 W/kg * 3500 kg/m^3; D=670km, k=4W/(mK)
        self.E = 0  # 23.03
        self.T_ref = 1.0
        self.z_ref = 0.0
        self.T0 = 0.1
        self.eta_min = 1e-2
        self.eta_max = 1e5

        self.weak_angle = -35.0  # angle of weak zone
        self.weak_pos = 2.0  # surface position of weak zone
        self.weak_width = 0.1  # width of weak zone
        self.weak_depth = 0.3  # depth of weak zone
        self.weak_visc = 0.01  # visc pre-factor
        self.weak_density = 0.8  # density pre-factor

        self.crust_depth = 0.15  # 0.07 # thickness of overriding plate
        self.crust_visc = 100000  # viscosity pre-factor for overriding plate
        self.crust_density = 0.9  # density pre-factor for overriding plate, basaltic crust is lighter
        self.subd_depth = 0.25  # 0.1 # thickness of subducting plate
        self.subd_visc = 100000  # viscosity pre-factor for subducting plate
        self.subd_density = 1.1  # density pre-factor for subducting plate, formation of eclogite makes crust denser
        self.sedi_depth = 0.05  # thickness of sediment layer
        self.sedi_visc = 0.01
        self.sedi_density = 0.8

        self.velr = 1000  # velocity of incoming plate from the right # 5cm/yr -> v'=v*D/kappa=1000
        self.z_in = 0.3  # thickness of inflow region
        self.z_out = 0.5  # thickness of outflow region on the side boundary
        self.z_right = True  # True -> outflow also on the right side; False -> outflow on the left side

        self.output = 'Subd/Subd_sedi_' + str(self.sedi_depth) + '_' + str(self.nl) + 'x' + str(self.nr) + '_OPd' + str(
            self.crust_depth) + '_SPd' + str(self.subd_depth) + '_OPrho' + str(self.crust_density) + '_SPrho' + str(
            self.subd_density) + "_2"
        self.plot = 'TvuVC'
        self.figsize = (10, 13)
        self.Plx = 5
        self.Ply = 1
        self.Cmin = 0.8
        self.Cmax = 1.2
        self.Cmap = 'bwr'

        self.nbiter = 0  # if set to zero, then use tmax
        self.figiter = 10  # every ... outputs create a new figure
        self.Cdt = 0.1  # >0 Courant criterium (~1); <0 Delta criterium (~-1)
        self.dt_ini = 1e-10
        self.dt_max = 1e-2
        self.tmax = 0.001

        self.snap_iter = 50
        self.read_snap = False

    elif SimCase == "Compressibility":
        # ##########################################################
        # compressible convection
        # ##########################################################
        self.geom = 0  # 0 - box, 1 - cylinder, 2 - spherical annulus
        self.nl = 20
        self.nr = 20

        self.compress = 1  # 0-no compressiblity, use BA or EBA; 1-compressible terms in stokes, strain rate etc. and TALA formulation in stokes
        self.Di = 1  # dissipation number, for Earth, Di ~ 0.5
        self.Gr = 1  # Gruneisen parameter, for Earth, Gr ~ 1.2
        self.T0 = 0.1  # 0.091 #0.1

        self.Ra = 1e4
        self.solver = 1  # 0-FDM, 1-FVM

        if self.solver == 1:
            self.output = 'Compress/Bl_box_FVM_isovisc_Ra' + str(int(self.Ra)) + '_' + str(self.nl) + 'x' + str(
                self.nr) + '_Com_' + str(self.compress) + '_Di_' + str(self.Di) + '_Gr_' + str(self.Gr) + '_T0_' + str(
                self.T0)
        else:
            self.output = 'Compress/Bl_box_FDM_isovisc_Ra' + str(int(self.Ra)) + '_' + str(self.nl) + 'x' + str(
                self.nr) + '_Com_' + str(self.compress) + '_Di_' + str(self.Di) + '_Gr_' + str(self.Gr) + '_T0_' + str(
                self.T0)

        self.plot = 'Ttsvuw'

        self.Cdt = -1  # >0 Courant criterium (~1); <0 Delta criterium (~-1)
        self.dt_max = 1e-2
        self.figiter = 50  # 5
        self.figtime = 0  # 1e-4 #1e-2 # every ... outputs create a new figure
        self.nbiter = 0  # if set to zero, then use tmax
        self.tmax = 0.3

        self.snap_iter = 30
        self.read_snap = False


    elif SimCase == "Isoviscous":
        # ##########################################################
        # Blankenbach/Herndlund&Tackley isoviscous 2D Box/2D sphere
        # ##########################################################

        # box
        self.geom = 0  # 0 - box, 1 - cylinder, 2 - spherical annulus
        self.nl = 20
        self.nr = 20
        self.periodic = 0  # 1
        self.sph = 1

        # OR: spherical
        # self.geom = 2 # 0 - box, 1 - cylinder, 2 - spherical annulus
        # self.regional = 0.5 #0.25 # only used for geom=1/2; 1 is full anulus, 0.25 quarter, etc.
        # self.nl = 128 #64
        # self.nr = 32
        # self.periodic = 1
        # self.regional = 0.5
        # self.sph = 3#6

        self.Tamp = 0.1

        self.free_slip_b = 1
        self.free_slip_t = 1

        # Wrong -> HT08 used 1.22 and 2.22!
        self.rmin = 1
        self.rmax = 2

        self.Ra = 1e4
        self.output = 'Blank/Bl_box_per_FVM_isovisc_Ra' + str(int(self.Ra)) + '_' + str(self.nl) + 'x' + str(self.nr)
        # self.output = 'Blank/HT_sph_per_isovisc_Ra'+str(int(self.Ra))+'_'+str(self.nl)+'x'+str(self.nr)+'_sph'+str(self.sph)

        self.plot = 'Tpsvuw'
        self.solver = 1  # 0-FDM, 1-FVM

        self.Cdt = -1  # >0 Courant criterium (~1); <0 Delta criterium (~-1)
        self.dt_max = 1e-3
        self.figiter = 50  # 5
        self.figtime = 0  # 1e-4 #1e-2 # every ... outputs create a new figure
        self.nbiter = 1000  # 20 # if set to zero, then use tmax
        # self.tmax = 1

    elif SimCase == "Arrhenius":
        # ##########################################################
        #  Variable (Arrhenius-type) viscosity
        # ##########################################################

        self.Ra = 1e5
        self.E = 23.03
        self.T_ref = 1.0
        self.z_ref = 0.0
        self.T0 = 1  # 0.1
        self.eta_min = 1e-2
        self.eta_max = 1e5

        # OR: spherical
        self.geom = 2  # 0 - box, 1 - cylinder, 2 - spherical annulus
        self.regional = 1  # 0.25 # only used for geom=1/2; 1 is full anulus, 0.25 quarter, etc.
        self.periodic = 1
        self.nr = 16  # 32
        self.nl = 6 * self.nr
        self.output = 'Arrh/Neu_Arrh_sph_Full_per__E_' + str(int(self.E)) + '_Ra' + str(int(self.Ra)) + '_' + str(
            self.nl) + 'x' + str(self.nr)

        self.solver = 1  # 0-FDM, 1-FVM
        self.plot = 'TvuwpV'
        self.sph = 3  # 10#6
        self.Tamp = 0.1

        self.Cdt = -1  # >0 Courant criterium (~1); <0 Delta criterium (~-1)

        self.nbiter = 100  # if set to zero, then use tmax
        self.figiter = 20  # every ... outputs create a new figure

        self.snap_iter = 30
        self.read_snap = False


    elif SimCase == "FK":
        # ##########################################################
        # variable viscosity (Frank-Kamenetskii approximation) for benchmarks
        # ##########################################################
        self.lmin = 0
        self.lmax = 2  # if lmax-lmin != 1 then aspect ratio of simulation changes

        self.nl = 40
        self.nr = 20

        self.Ra = 10
        self.e_gamma_T = 1.0e5  # FK Viscosity: exp(gamma_T)

        self.T_ref = 0.0
        self.z_ref = 0.0
        self.T0 = 0.1

        self.output = 'FK/FK_VC_' + str(int(self.e_gamma_T)) + '_Ra' + str(int(self.Ra)) + '_' + str(
            self.nl) + 'x' + str(self.nr)
        self.plot = 'TvVuwp'

        self.nbiter = 0  # if set to zero, then use tmax
        self.figiter = 10  # every ... outputs create a new figure
        self.Cdt = -1  # >0 Courant criterium (~1); <0 Delta criterium (~-1)
        self.dt_max = 1e-2
        self.tmax = 1

    elif SimCase == "ThermoChemical":
        # ##########################################################
        # thermal and compositional convection
        # ##########################################################

        # box
        self.geom = 0  # 0 - box, 1 - cylinder, 2 - spherical annulus
        self.nl = 40
        self.nr = 40

        # OR: spherical
        # self.geom = 2 # 0 - box, 1 - cylinder, 2 - spherical annulus
        # self.regional = 0.25 # only used for geom=1/2; 1 is full anulus, 0.25 quarter, etc.
        # self.nl = 64
        # self.nr = 32

        self.Ra = 1e4
        self.B = 1  # buoyancy number, e.g. between 1 and 10

        # without particles
        # self.np = 0 # number of particles, set to 0 to use chemical field approach
        # self.plot = 'TvC'
        # self.Le = 1e6
        # self.output = 'Comp/ThermoChemicalConv_box_field_'+str(self.nl)+'x'+str(self.nr)+'_Le_'+str(self.Le)
        ##self.output = 'Comp/ThermoChemicalConv_sph_field_'+str(self.nl)+'x'+str(self.nr)+'_Le_'+str(self.Le)
        # with particles
        self.np = 3 * self.nl * self.nr  # number of particles, set to 0 to use chemical field approach
        self.plot = 'TvPC'
        self.output = 'Comp/ThermoChemicalConv_box_part_' + str(self.nl) + 'x' + str(self.nr)
        # self.output = 'Comp/ThermoChemicalConv_sph_part_'+str(self.nl)+'x'+str(self.nr)

        self.Cmap = 'magma'  # composition color map

        self.Tprofile = 1  # 0->use Tini and TBL, 1-> linear, 2-> conductive profile

        self.Clinear = 1
        self.Cmin = 0  # composition plot min
        self.Cmax = 1  # composition plot max

        self.nbiter = 0  # if set to zero, then use tmax
        self.tmax = 0.1
        self.figiter = 10  # every ... outputs create a new figure

    elif SimCase == "CompositionInput":
        # ##########################################################
        # pure compositional convection, read-in file
        # ##########################################################
        self.nl = 40
        self.nr = 40

        self.Ra = 1e4

        self.B = 2  # buoyancy number, e.g. between 1 and 10
        self.Cimage = 'Test-Circle.txt'
        self.Image_fliplr = False
        self.Image_flipud = False
        self.Image_transpose = False
        self.Cmap = 'magma'  # composition color map
        self.Cdt = 10

        # self.np = 3*self.nl*self.nr # number of particles, set to 0 to use chemical field approach
        # self.plot = 'vuwPcp'
        # self.output = 'Comp/Test-Circle_Part_'+str(self.nl)+'x'+str(self.nr)

        self.np = 0  # number of particles, set to 0 to use chemical field approach
        self.plot = 'vuwc'
        self.Le = 1
        self.output = 'Comp/Test-Circle_Field_' + str(self.nl) + 'x' + str(self.nr) + '_Le_' + str(self.Le)

        self.Ttop = 0
        self.Tbot = 0  # 1 # set to zero here to have no temp influence
        self.Tamp = 0  # no T field perturbation
        self.Tprofile = 0  # 0->use Tini and TBL, 1-> linear, 2-> conductive profile
        self.Tini = 0

        self.nbiter = 20  # if set to zero, then use tmax
        self.figiter = 1  # every ... outputs create a new figure


    #############################################################################################
    ##                                     Code test case                                      ##
    #############################################################################################
    elif SimCase == "TestRK":
        # ##########################################################
        # Trace particle depending on Runge-Kutta method
        # ##########################################################
        self.nl = 20
        self.nr = 20

        self.Ra = 1e4

        self.rk = 4
        self.output = 'RK_' + str(int(self.rk)) + '_' + str(self.nl) + 'x' + str(self.nr)
        self.plot = 'TvR'

        self.trace_part = 205  # which particle to trace (plot with 'R'), if 0, no particle is being traced at all
        self.B = 0  # buoyancy number, e.g. between 1 and 10
        self.np = 3 * self.nl * self.nr  # number of particles, set to 0 to use chemical field approach

        self.nbiter = 0  # if set to zero, then use tmax
        self.figiter = 25  # every ... outputs create a new figure
        self.nbiter = 500
        self.Cdt = -1  # >0 Courant criterium (~1); <0 Delta criterium (~-1)
        self.tmax = 1

    elif SimCase == "TestLateralMovement":
        # ##########################################################
        # TestCase lateral periodic movement
        # ##########################################################
        self.nl = 80  # 5 #20
        self.nr = 40  # 5 #20
        self.rmin = 1
        self.rmax = 2
        self.regional = 0.25
        self.periodic = 1
        self.geom = 1

        self.Ra = 1e4

        self.Tprofile = 0  # 0->use Tini and TBL, 1-> linear, 2-> conductive profile
        self.Tini = 0.0
        self.Tbot = 0.0
        self.Tamp = 0.0

        self.free_slip_b = 2
        self.botv = 100

        self.output = 'Test_NoT/New2_Test_NoT_Cyl_vstd_FVM_per_Ra' + str(int(self.Ra)) + '_' + str(self.nl) + 'x' + str(
            self.nr)
        self.plot = 'psSvuw'
        self.plot_vel = 'std'  # 'ang' for angular velocity (radians/time), 'std' is standard velocity (length/time)
        self.solver = 1  # 0-FDM, 1-FVM

        self.Cdt = -1  # >0 Courant criterium (~1); <0 Delta criterium (~-1)
        self.dt_max = 1e-3
        self.figiter = 1  # 5 # every ... outputs create a new figure
        self.nbiter = 1  # 20 # if set to zero, then use tmax
        # self.tmax = 1

    return