from supp_functions import *
from outgassing import calc_gas_spec
import solver
import datetime

def run(self):
    time_start = datetime.datetime.now(datetime.UTC)  # datetime.datetime.utcnow()

    self.initialize_fields()  # Initialize mesh and fields (T,p,u,w,eta)
    self.set_initial_conditions()  # Set initial conditions

    self.t = 0.0  # time
    self.dt = self.dt_ini  # initial timestep
    self.dt_old = self.dt
    self.it = 0  # index of timestep
    plot_time = 0.0

    if self.read_snap:
        self.read_snapshot()  # Overwrite T,p,u,w,C from last timestep
        # update plot_time:
        if (self.figtime > 0):
            while plot_time < self.t:
                plot_time = plot_time + self.figtime

    if self.np > 0:
        self.initialize_particles()  # Initialize particles

    if self.compress > 0:
        rho = np.exp(self.Di / self.Gr * (self.rmax - self.rmin))
        print('Compressible convection, rho increases from 1 to ', rho)

    finished = False  # set to 1 if max time or max timesteps reached
    self.Told1 = self.T.copy()  # needed so Told2 can be calculated at first timestep
    self.Cold1 = self.C.copy()  # needed so Cold2 can be calculated at first timestep
    self.Viold1 = self.Vi.copy()  # needed so Viold2 can be calculated at first timestep

    self.charval = [self.it, 0, self.t, self.dt, 0.0, 0.0, 0.0]
    self.write_output()

    self.create_plot()  # only to plot initial conditions before first time step

    time_TS = datetime.datetime.now(datetime.UTC)  # datetime.datetime.utcnow()

    # set initial values benchmark
    if self.velr != 0:
        self.update_bench_weak_zone(ini=1)  # written on particles -> need to update self.V/self.C field
        self.particles2field()  # update V field

    Tsol = self.Tsol0 + self.Tsolz * (self.rmax - self.rc)
    Tliq = self.Tliq0 + self.Tliqz * (self.rmax - self.rc)

    # Mass of atmosphere over time
    self.M_atm = [0]  # list of atmospheric masses, will be extended in each timestep
    self.M_CO2 = [0]
    self.M_CO = [0]
    self.M_H2O = [0]
    self.M_H2 = [0]
    self.time = [0]

    while finished == False:  # while end of simulation not reached
        self.it = self.it + 1
        step_OI = 0  # outer iteration between mm and e solver until convergence is reached
        finished_OI = False
        self.vel_old = 0  # needed to determine convergence of velocity field
        self.Temp_old = 0  # needed to determine convergence of temperature field

        # store fields from last time step for improved energy solver
        self.Told2 = self.Told1.copy()
        self.Told1 = self.T.copy()
        self.Cold2 = self.Cold1.copy()
        self.Cold1 = self.C.copy()
        self.Viold2 = self.Viold1.copy()  # Visc pre-factor
        self.Viold1 = self.Vi.copy()
        self.uold = self.u.copy()
        self.wold = self.w.copy()

        while finished_OI == False:  # while no convergence at current time step
            step_OI = step_OI + 1

            ############################
            # Mass and momentum solver #
            ############################

            solver.solver_mm(self)  # solve mass and momentum equation, update u,v,p fields in data
            self.get_rms_velocity()  # get root-mean-square velocity field (self.vel)
            self.update_strR_stress()

            #################
            # Energy solver #
            #################

            self.Titer = self.T.copy()
            solver.solver_e(self)  # solve energy equation, update T field in data

            self.update_viscosity()  # calculate eta with Arrhenius/FK and multiply with self.V field
            if self.ys_0 > 0 or self.ys_z > 0:
                self.plasticity()

            #########################
            # Check for convergence #
            #########################

            dif_T, dif_vel, av_T, av_v = self.conv_char_val()  # check for convergence of temperature field, get first characteristic values
            Nu_t, Nu_b = self.get_Nu()  # get Nusselt numbers

            if self.debug > 0:
                print("  OI: %3d, dif_v: %10.3E, dif_T: %10.3E, |v|: %7.4f, |T|: %7.4f"
                      % (step_OI, dif_vel, dif_T, av_v, av_T))

            # check if convergence was reached in both vel and T
            if (dif_vel < self.convv) & (dif_T < self.convT):
                finished_OI = True

            if (step_OI == self.max_iter):
                print('ERROR - max number of iterations reached in current time step')
                finished_OI = True

        ###################################
        # Calculate melt fraction, Ex 8.3 #
        ###################################
        if self.Tsol0 > 0:
            vol_melt = 0
            vol_depl = 0
            vol_domain = 0  # either add up over cells or calculate analytically with rmax etc.
            M_CO2_dt = 0
            M_CO_dt = 0
            M_H2O_dt = 0
            M_H2_dt = 0
            for k in range(1, self.nr + 1):
                for i in range(1, self.nl + 1):
                    if (self.geom == 0):
                        dV = self.dl * self.dr
                    else:
                        dV = self.dl * (self.rc[k] ** self.geom) * self.dr
                    vol_domain += dV

                    self.meltF[k, i] = 0.0
                    if self.T[k, i] > Tliq[k]:
                        self.meltF[k, i] = 1.0
                        vol_melt += dV * self.meltF[k, i]
                    elif self.T[k, i] > Tsol[k] + self.depl[k, i] * (Tliq[k] - Tsol[k]):  # include depletion
                        self.meltF[k, i] = (self.T[k, i] - Tsol[k] - self.depl[k, i] * (Tliq[k] - Tsol[k])) / (
                                    Tliq[k] - Tsol[k])
                        vol_melt += dV * self.meltF[k, i]
                    vol_depl += dV * self.depl[k, i]

                    if self.meltF[k, i] > 0:
                        Melt_CO2 = self.CO2[k, i] / self.meltF[k, i] * (1 - (1 - self.meltF[k, i]) ** (
                                    1 / self.D_CO2))  # concentration in melt to be outgassed
                        self.CO2[k, i] = Melt_CO2 * self.meltF[
                            k, i]  # extraction of volatiles from mantle values (X=C*F)
                        Melt_H2O = self.H2O[k, i] / self.meltF[k, i] * (1 - (1 - self.meltF[k, i]) ** (1 / self.D_H2O))

                        self.H2O[k, i] = Melt_H2O * self.meltF[k, i]
                        # CO2 and H2O fields here track depletion, below is subtracted from particle concentration values, at end particle values are given back to field

                        [XH2, XH2O, XCO, XCO2] = calc_gas_spec(self.T[k, i] * self.DeltaT + self.T0, 1, Melt_H2O,
                                                               Melt_CO2, self.DIW)

                        # XH2O and XCO2 were weight fractions, hence weight fraction of XH2 and XCO need to be corrected
                        XH2 = XH2 * 2 / 18  # 2 = molar mass of H2, 18 = molar mass of H2O
                        XCO = XCO * 28 / 44  # 28 = molar mass of CO, 44 = molar mass of CO2

                        # scale to planetary values at end of time step by dividing by vol_domain and multiplying with dimensional V_mantle
                        M_H2O_dt += XH2O * self.rho_m * dV * self.meltF[k, i] * self.Chi_extr
                        M_H2_dt += XH2 * self.rho_m * dV * self.meltF[k, i] * self.Chi_extr
                        M_CO2_dt += XCO2 * self.rho_m * dV * self.meltF[k, i] * self.Chi_extr
                        M_CO_dt += XCO * self.rho_m * dV * self.meltF[k, i] * self.Chi_extr
                    else:
                        self.CO2[k, i] = 0.0  # extraction of volatiles (will be subtracted from particle values)
                        self.H2O[k, i] = 0.0

            print('  Melt fraction = ', vol_melt / vol_domain, ', average depletion = ',
                  vol_depl / vol_domain)  # Exercise 8.4
            self.M_H2O.append(self.M_H2O[-1] + M_H2O_dt / vol_domain * 4 / 3 * np.pi * (self.Rp ** 3 - self.Rc ** 3))
            self.M_H2.append(self.M_H2[-1] + M_H2_dt / vol_domain * 4 / 3 * np.pi * (self.Rp ** 3 - self.Rc ** 3))
            self.M_CO2.append(self.M_CO2[-1] + M_CO2_dt / vol_domain * 4 / 3 * np.pi * (self.Rp ** 3 - self.Rc ** 3))
            self.M_CO.append(self.M_CO[-1] + M_CO_dt / vol_domain * 4 / 3 * np.pi * (self.Rp ** 3 - self.Rc ** 3))
            self.M_atm.append(
                self.M_atm[-1] + (M_H2O_dt + M_H2_dt + M_CO2_dt + M_CO_dt) / vol_domain * 4 / 3 * np.pi * (
                            self.Rp ** 3 - self.Rc ** 3))
            self.time.append(self.t + self.dt)

        # Update depletion particle values based on melt fraction field (for now zero field as long as self.meltF is not calculated)
        if self.Tsol0 > 0:
            if self.np > 0:  # only works if particle are used, if not then only melt fraction is calculated
                for m in range(0, self.np):
                    # get cell indices in which to find the particle
                    i, k = self.get_particle_cell_indices(self.p_pos[m, 0], self.p_pos[m, 1])
                    # update depletion value based on local melt fraction
                    self.p_D[m] += self.meltF[k, i]
                    self.p_CO2[m] = max(0, self.p_CO2[m] - self.CO2[
                        k, i])  # here CO2/H2O fields are extraction values, after TS become residual mantle values from particles
                    self.p_H2O[m] = max(0, self.p_H2O[m] - self.H2O[k, i])

        # Update compositional field
        if self.np == 0:
            solver.solver_c(self)  # solve composition equation, update C field in data
        else:
            self.move_particles()  # move particles
            self.particles2field()  # update C field

        self.t = self.t + self.dt

        # update dependent data
        self.get_rms_velocity()  # get root-mean-square velocity field (self.vel)
        self.update_strR_stress()
        if self.velr != 0:
            self.update_bench_weak_zone(ini=0)  # written on particles -> need to update self.V/self.C field
            self.particles2field()  # update V field
        self.update_viscosity()
        if self.ys_0 > 0 or self.ys_z > 0:
            self.plasticity()

        ViscContrast = np.max(self.eta) / np.min(self.eta)

        self.charval = [self.it, step_OI, self.t, self.dt, av_v, av_T, Nu_t, Nu_b, ViscContrast]
        self.write_output()

        time_TS2 = datetime.datetime.now(datetime.UTC)  # datetime.datetime.utcnow()

        if self.debug > 0:
            print(80 * '-')
        print('TS:%5d, OI:%3d, t:%10.3E, dt:%10.3E, |v|:%7.4f, |T|:%7.4f, |Nu_t/b|:%7.4f/%7.4f, VC:%5d -- %7.3fs'
              % (self.it, step_OI, self.t, self.dt, av_v, av_T, Nu_t, Nu_b, ViscContrast,
                 (time_TS2 - time_TS).total_seconds()))
        if self.debug > 0:
            print(80 * '-')

        time_TS = time_TS2

        self.dt_old = self.dt

        self.get_next_dt()  # get new timestep length depending on strength of convection

        if (self.nbiter > 0):
            # max number of iterations reached?
            if (self.it == self.nbiter):
                finished = True
        else:
            # max time reached?
            if (self.t >= self.tmax):
                finished = True

        '''
        ##########################
        # Plot convection fields #
        ##########################
        '''

        if (self.figiter > 0):
            if (self.it == 1) | (self.it % self.figiter == 0):
                self.create_plot()
        elif (self.figtime > 0):
            if plot_time < self.t:
                plot_time = plot_time + self.figtime
                self.create_plot()

        '''
        ##########################
        # Create snapshot of sim #
        ##########################
        '''

        if (self.snap_iter > 0):
            if (self.it % self.snap_iter == 0):
                print('Write snapshot')
                self.write_snapshot()

    # Create final plot
    self.create_plot()

    time_end = datetime.datetime.now(datetime.UTC)  # datetime.datetime.utcnow()

    print('Simulation ended after ' + str(time_end - time_start) + ' h:min:sec')

    return