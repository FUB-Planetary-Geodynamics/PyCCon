# -*- coding: utf-8 -*-
"""
Created in Dec 2022
Update Sep 2024 (sph. annulus)

@author: Lena Noack
"""
import numpy as np
import input_default
import math
import matplotlib.pyplot as plt
import csv
import os
from scipy.interpolate import RectBivariateSpline # for testing only
from scipy.interpolate import RegularGridInterpolator as RGI
from matplotlib.cm import ScalarMappable
import matplotlib.tri as tri
import random
#####################################################

class supp_functions():

    # Set planet parameters and initial conditions
    def __init__(self):
        #input_data.get_input(self) 
        input_default.input(self) 

    #####################################################
    def initialize_fields(self):
        """Initialize fields for convection simulation"""
    #####################################################
    
        ###################
        # initialize mesh #
        ###################
        '''
         lb   0   1   2     nl-1  nl
         lc 0   1   2          nl  nl+1   rc    rb
              |   |   |      |   |        nr+1
              -------------------              nr
              |   |   |      |   |        nr
              
              |   |   |      |   |        1
              -------------------              0
              |   |   |      |   |        0
              
              actual domain: 1-nl x 1-nr
              lc - lateral center 0:nl+1
              lb - lateral left boundary 0:nl
              rc - radius center 0:nr+1
              rb - radius bottom boundary 0:nr
        '''

        if (self.geom==0):    
          # Cartesian box, non-staggered grid
          self.lb = np.linspace(self.lmin,self.lmax,self.nl+1)
          self.lc = 0.5*(self.lb[0:-1]+self.lb[1:])
        else:
          # Cylinder/spherical annulus -> l represents angle fraction
          self.lmin = -math.pi*self.regional
          self.lmax =  math.pi*self.regional
          self.lb = np.linspace(self.lmin,self.lmax,self.nl+1)
          self.lc = 0.5*(self.lb[0:-1]+self.lb[1:])

        self.rb = np.linspace(self.rmin,self.rmax,self.nr+1)
        self.rc = 0.5*(self.rb[0:-1]+self.rb[1:])
    
        # uniform grid -> constant dl, dr
        # for geom=1/2 -> dl represents angle fraction
        self.dl = self.lc[1]-self.lc[0]
        self.dr = self.rc[1]-self.rc[0]
    
        # add boundary cells
        self.lc = np.concatenate([[self.lmin-0.5*self.dl],self.lc,[self.lmax+0.5*self.dl]])
        self.rc = np.concatenate([[self.rmin-0.5*self.dr],self.rc,[self.rmax+0.5*self.dr]])
    
        #####################
        # initialize fields #
        #####################
        '''
         u    0   1   2     nl-1 nl
         w  0   1   2          nl  nl+1
         p      0   1         nl-1 
         T  0   1   2          nl  nl+1   T     u    w  p
             ||   |   |      |   ||        nr+1  nr+1
             |--------------------|                  nr
             ||   |   |      |   ||        nr    nr      nr-1
             |                    |
             ||   |   |      |   ||        1     1       0
             |------------------- |                  0
             ||   |   |      |   ||        0     0
          
          actual domain: 1-nl x 1-nr
          lc - lateral center 0:nl+1
          lb - lateral left boundary 0:nl
          rc - radius center 0:nr+1
          rb - radius bottom boundary 0:nr
        '''
    
        self.T = np.zeros((self.nr+2,self.nl+2))
        self.p = np.zeros((self.nr,self.nl)) 
        if self.periodic==0:
          self.u = np.zeros((self.nr+2,self.nl+1))
        else:
          self.u = np.zeros((self.nr+2,self.nl+2)) # extra lateral vel on the right side needed for periodic bnd
        self.w = np.zeros((self.nr+1,self.nl+2))
        self.eta = np.ones((self.nr+2,self.nl+2)) # viscosity in center of cell
        self.etaA = np.ones((self.nr+2,self.nl+1)) # at left/right cell boundaries (like u)
        self.etaC = np.ones((self.nr+1,self.nl+2)) # at bottom/top cell boundaries (like w)
        self.etaN = np.ones((self.nr+1,self.nl+1)) # at corners of cell
        self.C = np.zeros((self.nr+2,self.nl+2)) # composition field
        self.Vi = np.ones((self.nr+2,self.nl+2)) # viscosity pre-factor field
        self.vel = np.zeros((self.nr+2,self.nl+2)) # rms velocity, only for dt and plots
        self.strR = np.zeros((self.nr,self.nl)) # strain rate field (2nd invariant)
        self.stress = np.zeros((self.nr,self.nl)) # stress field
        self.ViscD = np.zeros((self.nr,self.nl)) # viscous dissipation field
        self.meltF = np.zeros((self.nr+2,self.nl+2)) # melt fraction field
        self.depl = np.zeros((self.nr+2,self.nl+2)) # depletion field
        self.CO2 = np.zeros((self.nr+2,self.nl+2)) # carbon field (in weight fraction of carbon in form of CO2)
        self.H2O = np.zeros((self.nr+2,self.nl+2)) # hydrogen field (in weight fraction of hydrogen in form of H2O)

        return

    #####################################################
    def set_initial_conditions(self):
        """Set initial field values"""
    #####################################################
 
        # 1) temperature profile
        if self.Tprofile==0:
            # set interior temperature to Tini with thermal boundary layers
            
            for k in range(1,self.nr+1):
                if self.rmax-self.rc[k]<self.TBLt:
                    self.T[k,:] = self.Ttop+(self.Tini-self.Ttop) \
                        *math.sin(0.5*math.pi*(self.rmax-self.rc[k])/self.TBLt)
                elif self.rc[k]-self.rmin<self.TBLb:
                    self.T[k,:] = self.Tbot-(self.Tbot-self.Tini) \
                        *math.sin(0.5*math.pi*(self.rc[k]-self.rmin)/self.TBLb)
                else:
                    self.T[k,:] = self.Tini
        elif self.Tprofile==1:
            for k in range(1,self.nr+1):
                self.T[k,:] = self.Tbot - (self.Tbot-self.Ttop)*float(k)/float(self.nr+1)
        else: # 2-conductive
            for k in range(1,self.nr+1):
                Tmax = self.rmax**2 * (1-3*(self.rmin**2/self.rmax**2)+2*(self.rmin**3/self.rmax**3))/6
                self.T[k,:] = 1 - self.rc[k]**2 * (1-3*(self.rmin**2/self.rc[k]**2)+2*(self.rmin**3/self.rc[k]**3))/6/Tmax                

    
 
        # 2) temperature perturbation to initiate convection

        if self.sph==0:
            # random perturbation
            for k in range(1,self.nr+1):
                for i in range(1,self.nl+1):
                    self.T[k,i] = self.T[k,i] + self.Tamp*random.random()
        else:
            for k in range(1,self.nr+1):
                for i in range(1,self.nl+1):
                  if (self.geom==0):
                    sph_harm = math.cos(math.pi*self.lc[i]*self.sph)
                    sph_harm = sph_harm*math.sin(0.5*math.pi*(self.rc[-2]-self.rc[k])) 
                    self.T[k,i] = self.T[k,i] + self.Tamp*sph_harm
                  else:
                    r_scaled=(self.rc[k]-self.rc[1]) / (self.rc[-2]-self.rc[1])
                    sph_harm = math.cos(self.lc[i]*self.sph)
                    self.T[k,i] = self.T[k,i] + self.Tamp*sph_harm*math.sin(math.pi*r_scaled)


        # set boundary cells
        self.T[0,:] = self.Tbot
        self.T[-1,:] = self.Ttop
        self.T[:,0] = self.T[:,1]
        self.T[:,-1] = self.T[:,-2]


        # 3) compositional field: const, linear or layer or image.txt
        if (self.Cimage != ''):
            data = np.genfromtxt(self.Cimage, delimiter=' ', dtype=float)
            (Im_nl,Im_nr) = data.shape # dimension may be different

            if self.Image_fliplr:
                data_ = np.fliplr(data)
            else:
                data_ = data
                
            if self.Image_flipud:
                data_ = np.flipud(data_)
            else:
                data_ = data_
                
            xnew = np.linspace(self.lmin,self.lmax,self.nl+2)
            ynew = np.linspace(self.rmin,self.rmax,self.nr+2)
            if self.Image_transpose:
                data_ = data_.transpose() # shape Im_nr x Im_nl
#                interp = interp2d(np.linspace(self.lmin,self.lmax,Im_nl),np.linspace(self.rmin,self.rmax,Im_nr), data_)
#                data_interp = interp(np.linspace(self.lmin,self.lmax,self.nl+2),np.linspace(self.rmin,self.rmax,self.nr+2))
#                interp = RectBivariateSpline(np.linspace(self.lmin,self.lmax,Im_nl),np.linspace(self.rmin,self.rmax,Im_nr), data_.T)
#                data_interp = lambda xnew, ynew: interp(xnew, ynew).T
                interp = RGI((np.linspace(self.lmin,self.lmax,Im_nl), np.linspace(self.rmin,self.rmax,Im_nr)), data_.T, method='linear', bounds_error=False)
                xxnew, yynew = np.meshgrid(xnew, ynew)
                data_interp = interp((xxnew, yynew)).T
            else:
#                interp = interp2d(np.linspace(self.lmin,self.lmax,Im_nr),np.linspace(self.rmin,self.rmax,Im_nl), data_)
#                data_interp = interp(np.linspace(self.lmin,self.lmax,self.nl+2),np.linspace(self.rmin,self.rmax,self.nr+2))
#                interp = RectBivariateSpline(np.linspace(self.lmin,self.lmax,Im_nr),np.linspace(self.rmin,self.rmax,Im_nl), data_.T)
#                data_interp = lambda xnew, ynew: interp(xnew, ynew).T
                interp = RGI((np.linspace(self.lmin,self.lmax,Im_nr), np.linspace(self.rmin,self.rmax,Im_nl)), data_.T, method='linear', bounds_error=False)
                xxnew, yynew = np.meshgrid(xnew, ynew)
                data_interp = interp((xxnew, yynew)).T

            #print(data_interp)
            self.C[:,:] = data_interp             
            
            self.C[0,:] = self.C[1,:]
            self.C[self.nr+1,:] = self.C[self.nr,:]
            self.C[:,0] = self.C[:,1]
            self.C[:,self.nl+1] = self.C[:,self.nl]

        elif self.Clinear==1: # linear
            for k in range(0,self.nr+2):
                self.C[k,:] = self.Cbot - (self.Cbot-self.Ctop)*float(k)/float(self.nr+1)
        elif self.Clayers==1: # layers
            for k in range(0,self.nr+2):
                if (self.rc[k]-self.rmin)<self.Clayer_thick:
                    self.C[k,:] = self.Cbot
                else:
                    self.C[k,:] = self.Ctop
        else:
            # set composition to Cref
            self.C[:,:] = self.Cref

        self.CO2[:,:] = self.CO2_ini
        self.H2O[:,:] = self.H2O_ini

        return



    #####################################################
    def get_particle_cell_indices(self,pos_l,pos_r):
        """Find cell indices in which particle currently lies"""
    #####################################################

        pl=0            
        for i in range(0,self.nl):
            if ( (pos_l>self.lb[i]) & (pos_l<=self.lb[i+1]) ):
                pl=i+1
        pr=0
        for k in range(0,self.nr):
            if ( (pos_r>self.rb[k]) & (pos_r<=self.rb[k+1]) ):
                pr=k+1

        return pl,pr
    

    #####################################################
    def initialize_particles(self):
        """Initialize particles for tracing composition"""
    #####################################################
    
        self.p_pos = np.zeros((self.np,self.np))    # position of particles
        self.p_C = np.zeros(self.np)                # compositional values
        self.p_V = np.zeros(self.np)                # viscosity pre-factor values
        self.p_D = np.zeros(self.np)                # depletion values
        self.p_CO2 = np.zeros(self.np)              # CO2 values
        self.p_H2O = np.zeros(self.np)              # H2O values

        if self.np<(2*self.nl*self.nr):
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("Warning, increase particles to at least twice the number of cells")
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        # set one particle per cell in main domain (no particles in boundary cells)
        for k in range(0,self.nr):
            for i in range(0,self.nl):
                self.p_pos[i+k*self.nl,0] = self.lc[i+1]
                self.p_pos[i+k*self.nl,1] = self.rc[k+1]

        # find random position for rest of the particles
        for m in range(self.nl*self.nr,self.np):
            self.p_pos[m,0] = random.uniform(self.lmin,self.lmax)
            self.p_pos[m,1] = random.uniform(self.rmin,self.rmax)

        
        # Initialize particle values from initial compositional field
        for m in range(0,self.np):
            pl,pr = self.get_particle_cell_indices(self.p_pos[m,0],self.p_pos[m,1])            
            self.p_C[m] = self.C[pr,pl]
            self.p_V[m] = self.Vi[pr,pl]
            self.p_D[m] = self.depl[pr,pl]
            self.p_CO2[m] = self.CO2[pr,pl]
            self.p_H2O[m] = self.H2O[pr,pl]

        if self.trace_part>0:
            self.rk_part = [[self.p_pos[self.trace_part,0],self.p_pos[self.trace_part,1],0]]

    #####################################################
    def get_vel_part(self,pos_l,pos_r):
        """Get velocity of particle depending on vel field"""
    #####################################################

        pl,pr = self.get_particle_cell_indices(pos_l,pos_r) # in (1...nl,1...nr)

        #########################################################################
        # get lateral average velocity from vel surrounding cell i,k (here pl,pr):
        # u_i-1,k+1          u_i,k+1
        # u_i-1,k    (i,k)   u_i,k
        # u_i-1,k-1          u_i,k-1
        #########################################################################

        vel_u = 0.0
        sum_w = 0.0

        for i in range(pl-1,pl+1): 
            for k in range(pr-1,pr+2):
                # at particle: weight=1; at corner of neighbour cell: weight->0
                # weight is the same for geom=0 and geom=1/2
                weight = 1.0-math.sqrt(1/2*(((pos_r-self.rc[k])/(1.5*self.dr))**2
                                          + ((pos_l-self.lb[i])/(self.dl))**2))
                vel_u = vel_u + weight*self.u[k,i]
                sum_w = sum_w + weight

        vel_u = vel_u/sum_w

        #########################################################################
        # get radial average velocity from vel surrounding cell i,k (here pl,pr):
        #  w_i-1,k    w_i,k    w_i+1,k
        #             (i,k) 
        # w_i-1,k-1  w_i,k-1  w_i+1,k-1
        #########################################################################

        vel_w = 0.0
        sum_w = 0.0

        for i in range(pl-1,pl+2): 
            for k in range(pr-1,pr+1): 
                # at particle: weight=1; at corner of neighbour cell: weight->0
                weight = 1.0-math.sqrt(1/2*(((pos_r-self.rb[k])/(self.dr))**2
                                          + ((pos_l-self.lc[i])/(1.5*self.dl))**2))
                vel_w = vel_w + weight*self.w[k,i]
                sum_w = sum_w + weight

        vel_w = vel_w/sum_w

        return vel_u,vel_w 

    #####################################################
    def part_check_bound(self,pos_l,pos_r):
      """Check if particle is still in domain after moving"""
    #####################################################

      if self.periodic==0:
        # shoot particle back into inside of the domain
        if pos_l<self.lmin:
            pos_l = 2.0*self.lmin - pos_l
        if pos_l>self.lmax:
            pos_l = 2.0*self.lmax - pos_l
        if pos_r<self.rmin:
            pos_r = 2.0*self.rmin - pos_r
        if pos_r>self.rmax:
            pos_r = 2.0*self.rmax - pos_r
      else:
        # bring particle in on other side of the domain / reflect at top and bottom bnd
        if pos_l<self.lmin:
            pos_l = self.lmax - (self.lmin - pos_l)
        if pos_l>self.lmax:
            pos_l = self.lmin + (pos_l - self.lmax)
        if pos_r<self.rmin:
            pos_r = 2.0*self.rmin - pos_r
        if pos_r>self.rmax:
            pos_r = 2.0*self.rmax - pos_r

      # Only for one boundary treatment, otherwise error (vel too high):
      if (pos_l<self.lmin) | (pos_l>self.lmax) | (pos_r<self.rmin) | (pos_r>self.rmax):
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("Error: particles are shooting too far, vel too high, reduce time step")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

      return pos_l,pos_r            
    
    #####################################################
    def move_particles(self):
        """Move particles along convection stream"""
    #####################################################
        
        for m in range(0,self.np):
            # get average velocity
            vel_u,vel_w = self.get_vel_part(self.p_pos[m,0],self.p_pos[m,1])

            if self.rk==1:
                # Runge-Kutta method of 1st order: pos(n+1) = pos(n) + dt*v(pos(n))
                if (self.geom==0):
                  self.p_pos[m,0] = self.p_pos[m,0] + self.dt*vel_u
                else:
                  self.p_pos[m,0] = self.p_pos[m,0] + self.dt*vel_u/self.p_pos[m,1]
                self.p_pos[m,1] = self.p_pos[m,1] + self.dt*vel_w
                self.p_pos[m,0],self.p_pos[m,1] = self.part_check_bound(self.p_pos[m,0],self.p_pos[m,1])
                
            elif self.rk==2: 
                # Runge-Kutta method of order 2: step (n*) inbetween (n) and (n+1)
                # pos(n+1) = pos(n) + dt*v( pos(n)+dt/2 v(pos(n)) )
                # -> pos(n*)  = pos(n) + dt/2 v( pos(n) )
                #    pos(n+1) = pos(n) + dt   v( pos(n*) )
                
                if (self.geom==0):
                  pos_l = self.p_pos[m,0] + 0.5*self.dt*vel_u
                else:
                  pos_l = self.p_pos[m,0] + 0.5*self.dt*vel_u/self.p_pos[m,1]
                pos_r = self.p_pos[m,1] + 0.5*self.dt*vel_w                                
                pos_l,pos_r = self.part_check_bound(pos_l,pos_r)
                vel_u_1,vel_w_1 = self.get_vel_part(pos_l,pos_r)

                if (self.geom==0):
                  self.p_pos[m,0] = self.p_pos[m,0] + self.dt*vel_u_1
                else:
                  self.p_pos[m,0] = self.p_pos[m,0] + self.dt*vel_u_1/self.p_pos[m,1]
                self.p_pos[m,1] = self.p_pos[m,1] + self.dt*vel_w_1
                self.p_pos[m,0],self.p_pos[m,1] = self.part_check_bound(self.p_pos[m,0],self.p_pos[m,1])
                
            else: #(rk==4)
                # Runge-Kutta method of order 4: steps (n1), (n2) and (n3) inbetween (n) and (n+1)
                # pos(n1) = pos(n) + dt/2 v( pos(n) )
                # pos(n2) = pos(n) + dt/2 v( pos(n1) )
                # pos(n3) = pos(n) + dt v( pos(n2) )
                # pos(n+1) = pos(n) + dt/6*v(pos(n)) + dt/3*v(pos(n1)) + dt/3*v(pos(n2)) + dt/6*v(pos(n3))

                if (self.geom==0):
                  pos_l = self.p_pos[m,0] + 0.5*self.dt*vel_u
                else:
                  pos_l = self.p_pos[m,0] + 0.5*self.dt*vel_u/self.p_pos[m,1]
                pos_r = self.p_pos[m,1] + 0.5*self.dt*vel_w
                pos_l,pos_r = self.part_check_bound(pos_l,pos_r)
                vel_u_1,vel_w_1 = self.get_vel_part(pos_l,pos_r)
            
                if (self.geom==0):
                  pos_l = self.p_pos[m,0] + 0.5*self.dt*vel_u_1
                else:
                  pos_l = self.p_pos[m,0] + 0.5*self.dt*vel_u_1/self.p_pos[m,1]
                pos_r = self.p_pos[m,1] + 0.5*self.dt*vel_w_1
                pos_l,pos_r = self.part_check_bound(pos_l,pos_r)
                vel_u_2,vel_w_2 = self.get_vel_part(pos_l,pos_r)

                if (self.geom==0):
                  pos_l = self.p_pos[m,0] + self.dt*vel_u_2
                else:
                  pos_l = self.p_pos[m,0] + self.dt*vel_u_2/self.p_pos[m,1]
                pos_r = self.p_pos[m,1] + self.dt*vel_w_2
                pos_l,pos_r = self.part_check_bound(pos_l,pos_r)
                vel_u_3,vel_w_3 = self.get_vel_part(pos_l,pos_r)

                if (self.geom==0):
                  self.p_pos[m,0] = self.p_pos[m,0] + self.dt/6*vel_u + self.dt/3*vel_u_1 \
                                    + self.dt/3*vel_u_2 + self.dt/6*vel_u_3
                else:
                  self.p_pos[m,0] = self.p_pos[m,0] + (self.dt/6*vel_u + self.dt/3*vel_u_1 \
                                    + self.dt/3*vel_u_2 + self.dt/6*vel_u_3)/self.p_pos[m,1]
                self.p_pos[m,1] = self.p_pos[m,1] + self.dt/6*vel_w + self.dt/3*vel_w_1 \
                                  + self.dt/3*vel_w_2 + self.dt/6*vel_w_3
                self.p_pos[m,0],self.p_pos[m,1] = self.part_check_bound(self.p_pos[m,0],self.p_pos[m,1])

        if self.trace_part>0:
            (self.rk_part).append([self.p_pos[self.trace_part,0],self.p_pos[self.trace_part,1],self.it])                
    
    
    #####################################################
    def particles2field(self):
        """Update composition field from particles"""
    #####################################################

        # re-set compositional fields
        self.C[:,:] = 0
        self.Vi[:,:] = 0
        depl_save = self.depl.copy()
        self.depl[:,:] = 0
        CO2_save = self.CO2.copy()
        self.CO2[:,:] = 0
        H2O_save = self.H2O.copy()
        self.H2O[:,:] = 0
        vec_weight = np.zeros((self.nr+2,self.nl+2)) # boundary values will be zero, but this way easier for indices
        
        # add cell values with particle values times weight
        for m in range(0,self.np):
            pl,pr = self.get_particle_cell_indices(self.p_pos[m,0],self.p_pos[m,1])            
            
            # at cell center: weight=1; at corner of cell: weight=0; at mid-faces: weight=0.29
            # weight function works for geom=0 as well as geom=1/2
            weight = 1.0-math.sqrt(2*(((self.p_pos[m,1]-self.rc[pr])/self.dr)**2 
                               + ((self.p_pos[m,0]-self.lc[pl])/self.dl)**2))

            # arithmetic mean
            self.C[pr,pl] = self.C[pr,pl] + self.p_C[m]*weight
            self.Vi[pr,pl] = self.Vi[pr,pl] + self.p_V[m]*weight
            self.depl[pr,pl] = self.depl[pr,pl] + self.p_D[m]*weight
            self.CO2[pr,pl] = self.CO2[pr,pl] + self.p_CO2[m]*weight
            self.H2O[pr,pl] = self.H2O[pr,pl] + self.p_H2O[m]*weight
            vec_weight[pr,pl] = vec_weight[pr,pl] + weight

        # average value from all particles in cell
        for k in range(1,self.nr+1):
            for i in range(1,self.nl+1):
                if vec_weight[k,i]>0:
                    self.C[k,i] = self.C[k,i]/vec_weight[k,i]
                    self.Vi[k,i] = self.Vi[k,i]/vec_weight[k,i]
                    self.depl[k,i] = self.depl[k,i]/vec_weight[k,i]
                    self.CO2[k,i] = self.CO2[k,i]/vec_weight[k,i]
                    self.H2O[k,i] = self.H2O[k,i]/vec_weight[k,i]
                else:
                    self.C[k,i] = self.Cold1[k,i] # value from last time step if no particle in cell
                    self.Vi[k,i] = self.Viold1[k,i]
                    self.depl[k,i] = depl_save[k,i]
                    self.CO2[k,i] = CO2_save[k,i]
                    self.H2O[k,i] = H2O_save[k,i]

        # update boundary cells
        self.C[0,:] = self.C[1,:]
        self.C[self.nr+1,:] = self.C[self.nr,:]
        self.C[:,0] = self.C[:,1]
        self.C[:,self.nl+1] = self.C[:,self.nl]

        self.Vi[0,:] = self.Vi[1,:]
        self.Vi[self.nr+1,:] = self.Vi[self.nr,:]
        self.Vi[:,0] = self.Vi[:,1]
        self.Vi[:,self.nl+1] = self.Vi[:,self.nl]

        self.depl[0,:] = self.depl[1,:]
        self.depl[self.nr+1,:] = self.depl[self.nr,:]
        self.depl[:,0] = self.depl[:,1]
        self.depl[:,self.nl+1] = self.depl[:,self.nl]

        self.CO2[0,:] = self.CO2[1,:]
        self.CO2[self.nr+1,:] = self.CO2[self.nr,:]
        self.CO2[:,0] = self.CO2[:,1]
        self.CO2[:,self.nl+1] = self.CO2[:,self.nl]

        self.H2O[0,:] = self.H2O[1,:]
        self.H2O[self.nr+1,:] = self.H2O[self.nr,:]
        self.H2O[:,0] = self.H2O[:,1]
        self.H2O[:,self.nl+1] = self.H2O[:,self.nl]


    #####################################################
    def update_viscosity(self):
        """FK and Arrhenius Viscosity"""
    #####################################################

        #   |-----------|
        #   |     |     |
        #   |     |     |
        #   A----eta----|
        #   |     |     |
        #   |     |     |
        #   N-----C-----|
        
        # define viscosity in center of cell
        for k in range(self.nr+2):
            for i in range(self.nl+2):
                temperature = self.T[k,i]
                depth = self.rb[-1] - self.rc[k]
                if self.E > 0: # Arrhenius law
                  self.eta[k,i] = math.exp((self.E+depth*self.V)/(temperature+self.T0) - 
                  (self.E+self.z_ref*self.V)/(self.T_ref+self.T0))
                else: # FK viscosity
                  self.eta[k,i] = math.exp(self.gamma_T*(self.T_ref-temperature) 
                                          + self.gamma_p*(depth-self.z_ref) )
                if self.eta_min>0:
                  self.eta[k,i] = max(self.eta_min,self.eta[k,i])
                if self.eta_max>0:
                  self.eta[k,i] = min(self.eta_max,self.eta[k,i])
                self.eta[k,i] *= self.Vi[k,i] # visc prefactor, if not used then default value is 1

        # define viscosity at left/right cell boundaries (like u)
        for k in range(self.nr+2):
            for i in range(self.nl+1):
                depth = self.rb[-1] - self.rc[k]
                temperature = 0.5*(self.T[k,i]+self.T[k,i+1])
                if self.E > 0: # Arrhenius law
                  self.etaA[k,i] = math.exp((self.E+depth*self.V)/(temperature+self.T0) - 
                                            (self.E+self.z_ref*self.V)/(self.T_ref+self.T0))
                else: # FK viscosity
                  self.etaA[k,i] = math.exp(self.gamma_T*(self.T_ref-temperature) 
                                           + self.gamma_p*(depth-self.z_ref) )
                if self.eta_min>0:
                  self.etaA[k,i] = max(self.eta_min,self.etaA[k,i])
                if self.eta_max>0:
                  self.etaA[k,i] = min(self.eta_max,self.etaA[k,i])
                self.etaA[k,i] *= 0.5*(self.Vi[k,i]+self.Vi[k,i+1]) # visc prefactor, if not used then default value is 1

        # define viscosity at bottom/top cell boundaries (like w)
        for k in range(self.nr+1):
            for i in range(self.nl+2):
                depth = self.rb[-1] - self.rb[k]
                temperature = 0.5*(self.T[k,i]+self.T[k+1,i])
                if self.E > 0: # Arrhenius law
                  #print(k,i,(self.E+depth*self.V)/(temperature+self.T0) - 
                  #                          (self.E+self.z_ref*self.V)/(self.T_ref+self.T0))
                  self.etaC[k,i] = math.exp((self.E+depth*self.V)/(temperature+self.T0) - 
                                            (self.E+self.z_ref*self.V)/(self.T_ref+self.T0))
                else: # FK viscosity
                  self.etaC[k,i] = math.exp(self.gamma_T*(self.T_ref-temperature) 
                                           + self.gamma_p*(depth-self.z_ref) )
                if self.eta_min>0:
                  self.etaC[k,i] = max(self.eta_min,self.etaC[k,i])
                if self.eta_max>0:
                  self.etaC[k,i] = min(self.eta_max,self.etaC[k,i])
                self.etaC[k,i] *= 0.5*(self.Vi[k,i]+self.Vi[k+1,i]) # visc prefactor, if not used then default value is 1

        # define viscosity at corners of cell
        for k in range(self.nr+1):
            for i in range(self.nl+1):
                depth = self.rb[-1] - self.rb[k]
                temperature = 0.25*(self.T[k,i]+self.T[k+1,i]+self.T[k,i+1]+self.T[k+1,i+1])
                if self.E > 0: # Arrhenius law
                  self.etaN[k,i] = math.exp((self.E+depth*self.V)/(temperature+self.T0) - 
                                            (self.E+self.z_ref*self.V)/(self.T_ref+self.T0))
                else: # FK viscosity
                  self.etaN[k,i] = math.exp(self.gamma_T*(self.T_ref-temperature) 
                                           + self.gamma_p*(depth-self.z_ref) )
                if self.eta_min>0:
                  self.etaN[k,i] = max(self.eta_min,self.etaN[k,i])
                if self.eta_max>0:
                  self.etaN[k,i] = min(self.eta_max,self.etaN[k,i])
                self.etaN[k,i] *= 0.25*(self.Vi[k,i]+self.Vi[k+1,i]+self.Vi[k,i+1]+self.Vi[k+1,i+1])

        # To consider: take into account spherical geometry for temp at corner?

    #####################################################
    def update_bench_weak_zone(self, ini=0):
        """Reduce viscosity within weak zone, adapt density and visc fields"""
    #####################################################

        #   |-----------|
        #   |     |     |
        #   |     |     |
        #   A----eta----|
        #   |     |     |
        #   |     |     |
        #   N-----C-----|

        angle = self.weak_angle / 180.0 * np.pi
        weak_ws = self.weak_width / np.sin(abs(angle))
        for m in range(0,self.np):
            i,k = self.get_particle_cell_indices(self.p_pos[m,0],self.p_pos[m,1])
            pos = self.lc[i] - self.lmin
            depth = self.rb[-1] - self.rc[k]
            pos_w = self.weak_pos + depth * np.cos(angle) / np.sin(angle)

            if ini==1:
                if (pos < pos_w - 0.5 * weak_ws) & (depth < self.crust_depth):
                    self.p_V[m] = self.crust_visc
                    self.p_C[m] = self.crust_density
                if (pos < pos_w + 0.5 * weak_ws) & (pos > pos_w - 0.5 * weak_ws) & (depth < self.weak_depth):
                    self.p_V[m] = self.weak_visc
                    self.p_C[m] = self.weak_density
                if (pos > pos_w + 0.5 * weak_ws):
                    if (depth < self.sedi_depth):
                        self.p_V[m] = self.sedi_visc
                        self.p_C[m] = self.sedi_density
                    elif (depth < self.subd_depth+self.sedi_depth):
                        self.p_V[m] = self.subd_visc
                        self.p_C[m] = self.subd_density
            else: # update only right boundary for incoming plate conditions
                if (i>=self.nl):
                  if depth < self.sedi_depth:
                    self.p_V[m] = self.sedi_visc
                    self.p_C[m] = self.sedi_density
                  elif depth < self.subd_depth+self.sedi_depth:
                    self.p_V[m] = self.subd_visc
                    self.p_C[m] = self.subd_density

        '''
        angle = self.weak_angle / 180.0 * np.pi
        weak_ws = self.weak_width / np.sin(abs(angle))
        for k in range(self.nr+2):
            for i in range(self.nl+2):
                pos = self.lc[i]-self.lmin
                depth = self.rb[-1] - self.rc[k]
                pos_w = self.weak_pos + depth * np.cos(angle) / np.sin(angle)
                if (pos < pos_w - 0.5 * weak_ws) & (depth < self.crust_depth):
                    self.eta[k, i] *= self.crust_visc
                if (pos < pos_w + 0.5 * weak_ws) & (pos > pos_w - 0.5 * weak_ws) & (depth < self.weak_depth):
                    self.eta[k,i] *= self.weak_visc
                if (pos > pos_w + 0.5 * weak_ws) & (depth < self.subd_depth):
                    self.eta[k, i] *= self.subd_visc

        for k in range(self.nr+2):
            for i in range(self.nl+1):
                pos = self.lb[i-1]-self.lmin
                depth = self.rb[-1] - self.rc[k]
                pos_w = self.weak_pos + depth * np.cos(angle) / np.sin(angle)
                if (pos < pos_w - 0.5 * weak_ws) & (depth < self.crust_depth):
                    self.etaA[k, i] *= self.crust_visc
                if (pos < pos_w + 0.5 * weak_ws) & (pos > pos_w - 0.5 * weak_ws) & (depth < self.weak_depth):
                    self.etaA[k,i] *= self.weak_visc
                if (pos > pos_w + 0.5 * weak_ws) & (depth < self.subd_depth):
                    self.etaA[k, i] *= self.subd_visc

        for k in range(self.nr+1):
            for i in range(self.nl+2):
                pos = self.lc[i]-self.lmin
                depth = self.rb[-1] - self.rc[k]
                pos_w = self.weak_pos + depth * np.cos(angle) / np.sin(angle)
                if (pos < pos_w - 0.5 * weak_ws) & (depth < self.crust_depth):
                    self.etaC[k, i] *= self.crust_visc
                if (pos < pos_w + 0.5 * weak_ws) & (pos > pos_w - 0.5 * weak_ws) & (depth < self.weak_depth):
                    self.etaC[k,i] *= self.weak_visc
                if (pos > pos_w + 0.5 * weak_ws) & (depth < self.subd_depth):
                    self.etaC[k, i] *= self.subd_visc

        for k in range(self.nr+1):
            for i in range(self.nl+1):
                pos = self.lb[i-1]-self.lmin
                depth = self.rb[-1] - self.rc[k]
                pos_w = self.weak_pos + depth * np.cos(angle) / np.sin(angle)
                if (pos < pos_w - 0.5 * weak_ws) & (depth < self.crust_depth):
                    self.etaN[k, i] *= self.crust_visc
                if (pos < pos_w + 0.5 * weak_ws) & (pos > pos_w - 0.5 * weak_ws) & (depth < self.weak_depth):
                    self.etaN[k,i] *= self.weak_visc
                if (pos > pos_w + 0.5 * weak_ws) & (depth < self.subd_depth):
                    self.etaN[k, i] *= self.subd_visc
        '''


    #####################################################
    def update_strR_stress(self):
        """strain rate and stress updates, for now only FDM and not FVM"""
    #####################################################

    #    |---w(k,i)---|    rb(k)
    #    |            |
    # u(k,i-1)  *   u(k,i) rc(k)
    #    |  eta(k,i)  |
    #    |--w(k-1,i)--|    rc(k-1)
        dli = 1.0/self.dl
        dri = 1.0/self.dr

        compr = 0 # compressibility term, only used if self.compress > 0

        for k in range(1,self.nr+1):
          for i in range(1,self.nl+1):
            if self.geom==0:
              # box: StrRate  = sqrt ( (du/dx)^2 + (dw/dz)^2) + 0.5*( du/dz + dw/dx )^2 )
              self.strR[k-1,i-1] = math.sqrt( ((self.u[k,i]-self.u[k,i-1])*dli)**2 +
                                     ((self.w[k,i]-self.w[k-1,i])*dri)**2 + 
                                     (0.5*(self.u[k+1,i]+self.u[k+1,i-1]-self.u[k-1,i]-self.u[k-1,i-1])*0.5*dri +
                                      0.5*(self.w[k,i+1]+self.w[k-1,i+1]-self.w[k,i-1]-self.w[k-1,i-1])*0.5*dli)**2)
              # Test later: FVM, compare with CHIC version below
              # inv_dVi = 1.0_dp / mesh%dVi(i,j,k)
              # Ll = mesh%dr_vec(k)
              # Lr = mesh%dr_vec(k)
              # Lu = mesh%dl_vec(i)
              # Lt = mesh%dl_vec(i)

              # field%strain_rate(i,j,k) = sqrt( prefac * ( (inv_dVi*(Lr*field%u(i+1,j,k)-Ll*field%u(i,j,k))-compr)**2 &
              #                        & + (inv_dVi*(Lt*field%w(i,j,k+1)-Lu*field%w(i,j,k))-compr)**2 &
              #           + 0.5_dp*(inv_dVi*(0.25_dp*Lr*(field%w(i,j,k)+field%w(i+1,j,k)+field%w(i,j,k+1)+field%w(i+1,j,k+1)) &
              #                        &   - 0.25_dp*Ll*(field%w(i-1,j,k)+field%w(i,j,k)+field%w(i-1,j,k+1)+field%w(i,j,k+1)) &
              #                        &   + 0.25_dp*Lt*(field%u(i,j,k)+field%u(i+1,j,k)+field%u(i,j,k+1)+field%u(i+1,j,k+1)) & 
              #                        &   - 0.25_dp*Lu*(field%u(i,j,k-1)+field%u(i+1,j,k-1)+field%u(i,j,k)+field%u(i+1,j,k))))**2 ) )

              if self.compress>0:
                # compressibility term is applied to stress below, not to strain rate
                compr = (self.u[k,i]-self.u[k,i-1])*dli + (self.w[k,i]-self.w[k-1,i])*dri

            else:
              # sphere: follow formulation Hernlund and Tackley, 2008
              tau_rr = (self.w[k,i]-self.w[k-1,i])*dri
              tau_rr2 = (self.w[k,i]+self.w[k-1,i])/(2.0*self.rc[k])
              tau_pp = (self.u[k,i]-self.u[k,i-1])*dli/self.rc[k] \
                     + (self.w[k,i]+self.w[k-1,i])/(2.0*self.rc[k])
              tau_rp = 0.5*(self.w[k,i+1]+self.w[k-1,i+1]-self.w[k,i-1]-self.w[k-1,i-1])*0.5*dli/self.rc[k] \
                     + self.rc[k]*( 0.5*(self.u[k+1,i]+self.u[k+1,i-1])/self.rc[k+1] \
                                  - 0.5*(self.u[k-1,i]+self.u[k-1,i-1])/self.rc[k-1] )*0.5*dri
              self.strR[k-1,i-1] = math.sqrt( tau_rr**2 + (self.geom-1.0)*tau_rr2**2 + tau_pp**2 + 0.5*tau_rp**2 )

              if self.compress>0:
                # compressibility term is applied to stress below, not to strain rate
                compr = (self.u[k,i]-self.u[k,i-1])*dli/self.rc[k] +\
                        (self.w[k,i]*self.rb[k]**self.geom-self.w[k-1,i]*self.rb[k-1]**self.geom)*dri/self.rc[k]**self.geom

            # Stress is 2 eta * strR, here is 2nd invariant of stress tensor; ToDo: check compressible term here for PT with compressible convection
            self.stress[k-1,i-1] = 2 * self.eta[k,i]*(self.strR[k-1,i-1] - compr/3.0) # different indices since no strR/stress in boundary cells needed
            # Visc dissipation term is stress * (dv_i/dx_j) needed in energy equation
            self.ViscD[k-1,i-1] = 2 * self.eta[k,i]*(self.strR[k-1,i-1]**2 - compr**2/3.0) # Schubert p. 280ff

    #####################################################
    def plasticity(self):
        """Calculated effective viscosity"""
    #####################################################

        # effective viscosity in center of cell
        for k in range(1,self.nr+1):
            for i in range(1,self.nl+1):
                depth = self.rb[-1] - self.rc[k]
                strain_rate = self.strR[k-1,i-1]
                self.eta[k,i] = 2.0/( 1.0/self.eta[k,i] + 
                  1.0/( self.ys_eta + (self.ys_0 + self.ys_z*depth)/strain_rate ) )
        # bnd conditions
        self.eta[:,0] = self.eta[:,1]
        self.eta[:,self.nl+1] = self.eta[:,self.nl]
        self.eta[0,:] = self.eta[1,:]
        self.eta[self.nr+1,:] = self.eta[self.nr,:]

        # define viscosity at left/right cell boundaries (like u)
        for k in range(1,self.nr+1):
            for i in range(1,self.nl):
                depth = self.rb[-1] - self.rc[k]
                strain_rate = 0.5*(self.strR[k-1,i-1]+self.strR[k-1,i])
                self.etaA[k,i] = 2.0/( 1.0/self.etaA[k,i] + 
                  1.0/( self.ys_eta + (self.ys_0 + self.ys_z*depth)/strain_rate ) )
        # bnd conditions
        self.etaA[:,0] = self.etaA[:,1]
        self.etaA[:,self.nl] = self.etaA[:,self.nl-1]
        self.etaA[0,:] = self.etaA[1,:]
        self.etaA[self.nr+1,:] = self.etaA[self.nr,:]

        # define viscosity at bottom/top cell boundaries (like w)
        for k in range(1,self.nr):
            for i in range(1,self.nl+1):
                depth = self.rb[-1] - self.rb[k]
                strain_rate = 0.5*(self.strR[k-1,i-1]+self.strR[k,i-1])
                self.etaC[k,i] = 2.0/( 1.0/self.etaC[k,i] + 
                  1.0/( self.ys_eta + (self.ys_0 + self.ys_z*depth)/strain_rate ) )
        # bnd conditions
        self.etaC[:,0] = self.etaC[:,1]
        self.etaC[:,self.nl+1] = self.etaC[:,self.nl]
        self.etaC[0,:] = self.etaC[1,:]
        self.etaC[self.nr,:] = self.etaC[self.nr-1,:]

        # define viscosity at corners of cell
        for k in range(1,self.nr):
            for i in range(1,self.nl):
                depth = self.rb[-1] - self.rb[k]
                strain_rate = 0.25*(self.strR[k-1,i-1]+self.strR[k,i-1]+self.strR[k-1,i]+self.strR[k,i])
                self.etaN[k,i] = 2.0/( 1.0/self.etaN[k,i] + 
                  1.0/( self.ys_eta + (self.ys_0 + self.ys_z*depth)/strain_rate ) )
        # bnd conditions
        self.etaN[:,0] = self.etaN[:,1]
        self.etaN[:,self.nl] = self.etaN[:,self.nl-1]
        self.etaN[0,:] = self.etaN[1,:]
        self.etaN[self.nr,:] = self.etaN[self.nr-1,:]


    #####################################################
    def get_rms_velocity(self):
        """Get root-mean-square velocity"""
    #####################################################

        # version includign boundaries
        #k = 0
        #self.vel[k,0] = math.sqrt(self.u[k,0]**2+self.w[k,0]**2)
        #for i in range(1,self.nl+1):
        #    self.vel[k,i] = math.sqrt((0.5*(self.u[k,i-1]+self.u[k,i]))**2 + (self.w[k,i])**2)
        #self.vel[k,self.nl+1] = math.sqrt(self.u[k,self.nl]**2+self.w[k,self.nl+1]**2)
        #
        #for k in range(1,self.nr+1):
        #    i = 0
        #    self.vel[k,i] = math.sqrt(self.u[k,i]**2 + (0.5*(self.w[k-1,i]+self.w[k,i]))**2)
        #    for i in range(1,self.nl+1):
        #        self.vel[k,i] = math.sqrt((0.5*(self.u[k,i-1]+self.u[k,i]))**2 
        #                                + (0.5*(self.w[k-1,i]+self.w[k,i]))**2)
        #    i = self.nl+1
        #    self.vel[k,i] = math.sqrt(self.u[k,i-1]**2 + (0.5*(self.w[k-1,i]+self.w[k,i]))**2)
        #
        #k = self.nr+1
        #self.vel[k,0] = math.sqrt(self.u[k,0]**2+self.w[k-1,0]**2)
        #for i in range(1,self.nl+1):
        #    self.vel[k,i] = math.sqrt((0.5*(self.u[k,i-1]+self.u[k,i]))**2 + (self.w[k-1,i])**2)
        #self.vel[k,self.nl+1] = math.sqrt(self.u[k,self.nl]**2+self.w[k-1,self.nl+1]**2)

        for k in range(0,self.nr+2):
          for i in range(0,self.nl+2):
            if i==0:
              self.vel[k,i] = self.u[k,i]**2
            elif i==self.nl+1:
              self.vel[k,i] = self.u[k,i-1]**2
            else: 
              self.vel[k,i] = (0.5*(self.u[k,i-1]+self.u[k,i]))**2
            if ((self.geom>0) and (self.plot_vel=='ang')):
              # lateral velocity is only angular velocity
              self.vel[k,i] /= (self.rc[k])**2

            if k==0:
              self.vel[k,i] += self.w[k,i]**2
            elif k==self.nr+1:
              self.vel[k,i] += self.w[k-1,i]**2
            else:
              self.vel[k,i] += (0.5*(self.w[k-1,i]+self.w[k,i]))**2

            self.vel[k,i] = math.sqrt(self.vel[k,i])

        # online inside convecting box -> bnd zero, not correct
        #for k in range(1,self.nr+1):
        #  for i in range(1,self.nl+1):
        #    if ((self.geom>0) and (self.plot_vel=='std')):
        #      # lateral velocity in length/time instead of angular velocity
        #      self.vel[k,i] = math.sqrt((0.5*(self.u[k,i]+self.u[k,i+1])*self.rc[k]**self.geom)**2 
        #                      +(0.5*(self.w[k,i]+self.w[k+1,i]))**2)
        #    else:  
        #      self.vel[k,i] = math.sqrt((0.5*(self.u[k,i]+self.u[k,i+1]))**2 
        #                      +(0.5*(self.w[k,i]+self.w[k+1,i]))**2)


        return



    #####################################################
    def conv_char_val(self):
        """Get convergence values and average characteristic values"""
    #####################################################
        
        dif_T = 0.0
        av_v = 0.0
        av_T = 0.0
        vel_new = 0.0
        for k in range(1,self.nr+1):
            for i in range(1,self.nl+1):
                dif_T = dif_T + abs(self.T[k,i]-self.Titer[k,i])
                #av_v = av_v+self.vel[k,i]
                av_T = av_T+self.T[k,i]
                vel_new = vel_new + abs(self.u[k,i]+self.u[k,i-1])/2.0 \
                + abs(self.w[k,i]+self.w[k-1,i])/2.0
                #vel_new = vel_new + abs(self.u[k,i]-self.u[k,i-1])/self.dl \
                #+ abs(self.w[k,i]-self.w[k-1,i])/self.dr

        dif_T = dif_T/float(self.nl*self.nr)
        #av_v = av_v/float(self.nl*self.nr)
        av_T = av_T/float(self.nl*self.nr)

        vel_new = vel_new/float(self.nl*self.nr)
        dif_vel = abs(vel_new-self.vel_old)
        self.vel_old = vel_new

        ##################################################
        # rms vel -> sqrt(sum_i sum_k u^2+w^2 / (nl*nr)) #
        ##################################################
        av_v = 0
        for k in range(self.nr):
          for i in range(self.nl):
            if ((self.geom>0) and (self.plot_vel=='ang')):
              # lateral velocity is just angular velocity
              av_v += ((0.5*(self.u[k,i]+self.u[k,i+1])/self.rc[k])**2 
                      +(0.5*(self.w[k,i]+self.w[k+1,i]))**2)
            else:  
              av_v += ((0.5*(self.u[k,i]+self.u[k,i+1]))**2 
                      +(0.5*(self.w[k,i]+self.w[k+1,i]))**2)

        av_v = math.sqrt(av_v/float(self.nl*self.nr))

        return dif_T,dif_vel,av_T,av_v


    #####################################################
    def get_Nu(self):
        """Get top and bottom Nusselt numbers"""
    #####################################################

        # calculation works for geom=0 and geom=1/2 since mesh is uniform
        Nu_t = 0.0
        Nu_b = 0.0
        Nu_t = - (self.T[self.nr+1]-self.T[self.nr])/(self.rb[self.nr]-self.rc[self.nr])
        Nu_b = - (self.T[1]-self.T[0])/(self.rc[1]-self.rb[0])
#        Nu_t = - (self.T[self.nr+1]-self.T[self.nr])/(self.rc[self.nr+1]-self.rc[self.nr])
#        Nu_b = - (self.T[1]-self.T[0])/(self.rc[1]-self.rc[0])
        Nu_t = np.sum(Nu_t[1:-1])/float(self.nl)
        Nu_b = np.sum(Nu_b[1:-1])/float(self.nl)
        
        return Nu_t,Nu_b


    #####################################################
    def get_next_dt(self):
        """Get new timestep length depending on strength of convection"""
    #####################################################

        if self.Cdt>0:    
            # Courant criterium, Cdt ~ 1
            if (self.geom==0):
              maxvel = max([np.max(self.u)/self.dl,np.max(self.w)/self.dr])
            else:
              maxvel = max([np.max(self.u)/self.dl*2*math.pi,np.max(self.w)/self.dr])
            if maxvel==0:
                maxvel=1
            self.dt = self.Cdt/maxvel
        else:
            # Delta criterium, Cdt ~ -1
            maxvel=0.0
            for k in range(1,self.nr+1):
                for i in range(1,self.nl+1):
                  if (self.geom==0):
                    maxvel=max([maxvel,abs(self.u[k,i]-self.uold[k,i])/self.dl 
                               + abs(self.w[k,i]-self.wold[k,i])/self.dr])
                  else:
                    maxvel=max([maxvel,abs(self.u[k,i]-self.uold[k,i])/self.dl*2*math.pi 
                               + abs(self.w[k,i]-self.wold[k,i])/self.dr])

            if maxvel>0:
                dt_old = self.dt
                self.dt = -self.Cdt/maxvel
                self.dt = 0.5*(self.dt+dt_old)
#                self.dt = -self.Cdt*self.dt/maxdt # units don't match
            else:
                self.dt = -self.Cdt*self.dt_ini

            #print(self.dt,self.dt_min,self.dt_max)
        self.dt = max(min(self.dt,self.dt_max),self.dt_min)


    #####################################################
    def add_bnd(self,mat):
      """Add boundaries to field for full contourf"""
    #####################################################
      #can in principle also be writte with np.hstack() or similar functions
      mat_full = np.zeros((self.nr+2,self.nl+2))
      mat_full[1:self.nr+1,1:self.nl+1] = mat
      mat_full[0,1:self.nl+1] = mat_full[1,1:self.nl+1]
      mat_full[-1,1:self.nl+1] = mat_full[-2,1:self.nl+1]
      mat_full[:,0] = mat_full[:,1]
      mat_full[:,-1] = mat_full[:,-2]
      return mat_full


    #####################################################
    def create_plot(self):
        """Create figure with relevant information"""
    #####################################################

        if (self.showplot or self.saveplot):
          plt.figure() 

          # read-in string (e.g. 'TvuwCV') and arrange image automatically
          if self.geom==0:
              ar = (self.lmax-self.lmin)/(self.rmax-self.rmin)
          else:
              ar = 1.0
          nrPlots = len(self.plot)
          if nrPlots == 1:
              Plx = 1; Ply = 1
              figsize=(5,4/ar)
          elif nrPlots == 2:
              Plx = 1; Ply = 2
              figsize=(10,4/ar)
          elif nrPlots == 3:
              Plx = 1; Ply = 3
              figsize=(15,4/ar)
          elif nrPlots == 4:
              Plx = 2; Ply = 2
              figsize=(10,8/ar)
          elif nrPlots == 5:
              Plx = 1; Ply = 5
              figsize=(25,4/ar)
          elif nrPlots == 6:
              Plx = 2; Ply = 3
              figsize=(15,8/ar)
          elif nrPlots == 7:
              Plx = 1; Ply = 7
              figsize=(35,4/ar)
          else: # 8 subplots
              Plx = 2; Ply = 4
              figsize=(20,8/ar)

          if self.figsize != (1,1):
              figsize = self.figsize
          if self.Plx > 0:
              Plx = self.Plx
          if self.Ply > 0:
              Ply = self.Ply

          if (self.geom>0):
            fig,ax = plt.subplots(Plx,Ply,figsize=figsize,subplot_kw=dict(projection='polar'))

            lci = self.lc#*180/math.pi
            rc = self.rc.copy()
            lbi = self.lb#*180/math.pi
            rb = self.rb.copy()
          else:
            fig,ax = plt.subplots(Plx,Ply,figsize=figsize)
            lci = self.lc.copy()
            rc = self.rc.copy()
            lbi = self.lb.copy()
            rb = self.rb.copy()

          fig.suptitle("Time: %7.4f" % self.t)


          ind = 0        
          for i in range(Plx):
              for k in range(Ply):
                  if Plx==1:
                      it = k
                  elif Ply==1:
                      it = i
                  else:
                      it = (i,k) # index tuple

                  if (self.geom>0):
                    ax[it].set_rorigin(0)
                    ax[it].set_theta_zero_location('E')

                    ax[it].grid(alpha = 0.2)
                    
                    if self.regional == 1: 
                        full = True                
                        angle_offset = 0                
                    elif self.regional == 0.5:
                        angle_offset = 0 
                        ax[it].set_thetamin(-90)
                        ax[it].set_thetamax(90)
                        ax[it].set_xticks(np.pi/180. * np.linspace(90,  -90, 3))#, endpoint=False))
                    elif self.regional == 0.25:
                        angle_offset = np.pi/4#180/4#0
                        ax[it].set_thetamin(0)
                        ax[it].set_thetamax(90)
#                        ax[it].set_thetamin(-45)
#                        ax[it].set_thetamax(45)
#                        ax[it].set_thetamin(90-self.regional*180)
#                        ax[it].set_thetamax(90+self.regional*180)
                        #ax[it].set_xticks(np.pi/180. * np.linspace(90,0,3))
                    else:
                        angle_offset = 180/2#np.pi/2
                        ax[it].set_thetamin(90-self.regional*180)
                        ax[it].set_thetamax(90+self.regional*180)
                    
                  else:
                    angle_offset = 0
                    
                  lc = lci + angle_offset
                  lb = lbi + angle_offset
                  #print(np.min(lc),np.max(lc),np.min(rc),np.max(rc))

                  if self.plot[ind]=='T':
                      #print('T',lc.shape,rc.shape,(self.T).shape,(self.lc).shape,(self.rc).shape)
                      #print(self.T)
                      #LC_,RC_ = np.meshgrid(lc,rc)
                      #x = (RC_ * np.cos(LC_)).flatten()
                      #y = (RC_ * np.sin(LC_)).flatten()
                      #triang = tri.Triangulation(x, y)
                      #im=ax[it].tricontourf(triang,(self.T).flatten(),cmap='magma',levels=20)
                      im=ax[it].contourf(lc, rc,self.T,cmap='magma',levels=20)
                      plt.colorbar(im,ax=ax[it],label="Temperature")
                  elif self.plot[ind]=='t':
                      ax[it].plot(np.min(self.T, axis=1),rc,label="min")
                      ax[it].plot(np.mean(self.T, axis=1),rc,label="mean")
                      ax[it].plot(np.max(self.T, axis=1),rc,label="max")
                      if self.Tsol0>0: # melt
                          ax[it].plot(self.Tsol0+self.Tsolz*(self.rmax-rc),rc,'k:',label='Tsol' )
                          ax[it].plot(self.Tliq0+self.Tliqz*(self.rmax-rc),rc,'k--',label='Tliq')
                      ax[it].legend()
                  elif self.plot[ind]=='v':
                      im=ax[it].contourf(lc, rc,self.vel,cmap='jet',levels=20)
                      plt.colorbar(im,ax=ax[it],label="RMS velocity")
                  elif self.plot[ind]=='u':
                      if ((self.geom==0) or (self.plot_vel=='std')):
                        vmin = np.min(self.u); vmax = np.max(self.u); bnd = max([-vmin,vmax])
                        # lateral velocity in length/time instead of angular velocity
                        im=ax[it].contourf(lb, rc,self.u[:,0:self.nl+1],cmap='bwr',vmin=-bnd,vmax=bnd,levels=21)
                      else:
                        vmin = np.min(self.u[:,0:self.nl+1]/(self.rc).reshape((self.u).shape[0],1))
                        vmax = np.max(self.u[:,0:self.nl+1]/(self.rc).reshape((self.u).shape[0],1)); bnd = max([-vmin,vmax])
                        im=ax[it].contourf(lb, rc,self.u[:,0:self.nl+1]/(self.rc).reshape((self.u).shape[0],1),cmap='bwr',vmin=-bnd,vmax=bnd,levels=21)
                      plt.colorbar(im,ax=ax[it],label="Lateral vel")
                      #print('u std:',self.u[0,1],self.u[self.nr+1,1])
                      #print('u ang:',self.u[0,1]/self.rc[0],self.u[self.nr+1,1]/self.rc[self.nr+1])
                  elif self.plot[ind]=='w':
                      vmin = np.min(self.w); vmax = np.max(self.w); bnd = max([-vmin,vmax])
                      im=ax[it].contourf(lc, rb,self.w,cmap='bwr',vmin=-bnd,vmax=bnd,levels=21)
                      plt.colorbar(im,ax=ax[it],label="Radial vel")
                  elif self.plot[ind]=='V':
                      im=ax[it].contourf(lc, rc,np.log10(self.eta),cmap='magma',levels=20)
                      plt.colorbar(im,ax=ax[it],label="Log10 viscosity")
                  elif self.plot[ind]=='C':
                      im=ax[it].contourf(lc, rc,self.C,cmap=self.Cmap,levels=20)
                      plt.colorbar(im,ax=ax[it],label="Composition interpolated")
                  elif self.plot[ind]=='c': # composition grid
                      im=ax[it].imshow(np.flipud(self.C),cmap=self.Cmap,vmin=self.Cmin,vmax=self.Cmax)
                      plt.colorbar(ScalarMappable(norm=im.norm,cmap=im.cmap),ax=ax[it],label="Composition grid")
                      im.set_clim(self.Cmin,self.Cmax)
                  elif self.plot[ind]=='P': # particle composition scatter
                      im=ax[it].scatter(self.p_pos[:,0],self.p_pos[:,1],c=self.p_C,s=50,cmap=self.Cmap,vmin=self.Cmin,vmax=self.Cmax)
                      plt.colorbar(ScalarMappable(norm=im.norm,cmap=im.cmap),ax=ax[it],label="Composition particles")
                      im.set_clim(self.Cmin,self.Cmax)
                  elif self.plot[ind]=='p':
                      #im=ax[it].contourf(lc[1:-1],rc[1:-1],self.p,cmap='ocean',levels=20)
                      im=ax[it].contourf(lc,rc,self.add_bnd(self.p),cmap='ocean',levels=20)
                      plt.colorbar(im,ax=ax[it],label="Convective pressure")                    
                  elif self.plot[ind]=='R': # particle trace to see Runge-Kutta method
                      if self.it>0:
                          plot_rk = np.array(self.rk_part)
                          im=ax[it].scatter(plot_rk[:,0],plot_rk[:,1],c=plot_rk[:,2],s=50,cmap='jet',vmin=0,vmax=self.nbiter)
                          plt.colorbar(ScalarMappable(norm=im.norm,cmap=im.cmap),ax=ax[it],label="Trace particle over time")
                          im.set_clim(0,self.nbiter)
                          ax[it].set_xlim(self.lmin,self.lmax)
                          ax[it].set_ylim(self.rmin,self.rmax)
                  elif self.plot[ind]=='s': # strain rate
                      plot_strR = np.where(self.strR<1.0e-20,1.0e-20,self.strR)
                      im=ax[it].contourf(lc,rc,np.log10(self.add_bnd(plot_strR)),cmap='ocean',levels=20)
#                      im=ax[it].contourf(lc,rc,np.log10(self.add_bnd(self.strR)),cmap='ocean',levels=20)
                      plt.colorbar(im,ax=ax[it],label="Log10 Strain rate")                    
                  elif self.plot[ind]=='S': # stress
                      plot_stress = np.where(self.stress<1.0e-20,1.0e-20,self.stress)
                      im=ax[it].contourf(lc,rc,np.log10(self.add_bnd(plot_stress)),cmap='ocean',levels=20)
                      plt.colorbar(im,ax=ax[it],label="Log10 Stress")
                  elif self.plot[ind] == 'F':
                      im = ax[it].contourf(lc, rc, self.meltF, cmap='magma', levels=20)
                      plt.colorbar(im, ax=ax[it], label="Melt fraction")
                  elif self.plot[ind] == 'd':
                      im = ax[it].contourf(lc, rc, self.depl, cmap='magma', levels=20)
                      plt.colorbar(im, ax=ax[it], label="Melt depletion")
                  elif self.plot[ind] == 'D':
                      im=ax[it].scatter(self.p_pos[:,0],self.p_pos[:,1],c=self.p_D,s=50,cmap='magma',vmin=0, vmax=0.3)
                      plt.colorbar(ScalarMappable(norm=im.norm,cmap=im.cmap),ax=ax[it],label="Depletion particles")
                      im.set_clim(0,0.3)
                  elif self.plot[ind] == 'g': # g=graphite/carbon dioxide
                      im = ax[it].contourf(lc, rc, self.CO2*1e6, cmap='magma', levels=20)
                      plt.colorbar(im, ax=ax[it], label="Carbon storage [ppm CO2]")
                  elif self.plot[ind] == 'h':  # h=hydrogen/water
                      im = ax[it].contourf(lc, rc, self.H2O * 1e6, cmap='magma', levels=20)
                      plt.colorbar(im, ax=ax[it], label="Hydrogen storage [ppm H2O]")
                  else:
                      print('No plot routine implemented yet for letter "'+self.plot[ind]+'"!')
                  ind = ind + 1

          if self.saveplot:
            plt.savefig(self.output+'/plot_'+str(self.it)+'.png', dpi=100, transparent=False)
          if self.showplot:
            plt.show()
          #else:
          
          plt.close('all')
          #fig.tight_layout()

    
    
    #####################################################
    def write_output(self):
        """Create txt file with output data"""
    #####################################################
        
        # Write timeseries on file
        if self.output!='':
            #data.charval = [it,step_OI,data.t,data.dt,av_v,av_T,Nu_t,Nu_b,ViscContrast]
            if self.it==0:
                if (os.path.exists(self.output)==False):
                    os.makedirs(self.output)
                header = ["TS", "OI", "t", "dt", "|v|", "|T|", "|Nu_t|", "|Nu_b|", "VC"]
                with open(self.output+'/output.txt', 'w', newline='') as out:
                    writefile = csv.writer(out, delimiter=',')
                    writefile.writerow(header)
                    writefile.writerow(self.charval)
            else:
                with open(self.output+'/output.txt', 'a', newline='') as out:
                    writefile = csv.writer(out, delimiter=',')
                    writefile.writerow(self.charval)

    #####################################################
    def write_snapshot(self):
        """Write fields in snapshot file to restart simulations"""
    #####################################################

        if os.path.isfile(self.output + '/snapshot.npz'):
            # Store snapshot as old snapshot in case an error occurs now in writing the current snapshot
            try:
                os.rename(self.output + '/snapshot.npz', self.output + '/snapshot_old.npz')
            except FileExistsError:
                os.remove(self.output + '/snapshot_old.npz')
                os.rename(self.output + '/snapshot.npz', self.output + '/snapshot_old.npz')

        # store relevant data
        # mesh data as well as eta/vel/strR/stress/VD can be set based on input/snapshot data and do not need to be saved in file
        if self.output!='':
            np.savez(self.output + '/snapshot', T=self.T, T1=self.Told1, T2=self.Told2, p=self.p, u=self.u, w=self.w,
                     C=self.C, C1=self.Cold1, C2=self.Cold2, V=self.Vi, V1=self.Viold1, V2=self.Viold2,
                     depl=self.depl, CO2=self.CO2, H2O=self.H2O, TS=self.it, t=self.t, dt=self.dt, dto = self.dt_old)
        return
        # add Told as energy solver used that?

    #####################################################
    def read_snapshot(self):
        """Write fields in snapshot file to restart simulations"""
    #####################################################

        # read relevant data
        if self.output!='':
            loaded_data = np.load(self.output + '/snapshot.npz')
            self.T = loaded_data['T']
            self.Told1 = loaded_data['T1']
            self.Told2 = loaded_data['T2']
            self.p = loaded_data['p']
            self.u = loaded_data['u']
            self.w = loaded_data['w']
            self.C = loaded_data['C']
            self.Cold1 = loaded_data['C1']
            self.Cold2 = loaded_data['C2']
            try:
                self.Vi = loaded_data['V']
                self.Viold1 = loaded_data['V1']
                self.Viold2 = loaded_data['V2']
            except:
                print('No V field detected in snapshot, continue without')
            try:
                self.depl = loaded_data['depl']
            except:
                print('No depl field detected in snapshot, continue without')
            try:
                self.CO2 = loaded_data['CO2']
            except:
                print('No CO2 field detected in snapshot, continue without')
            try:
                self.H2O = loaded_data['H2O']
            except:
                print('No H2O field detected in snapshot, continue without')
            self.it = loaded_data['TS']
            self.t = loaded_data['t']
            self.dt = loaded_data['dt']
            self.dt_old = loaded_data['dto']

            # update dependent data
            self.get_rms_velocity() # get root-mean-square velocity field (data.vel)
            self.update_strR_stress()
            self.update_viscosity()
            if self.ys_0 > 0 or self.ys_z > 0:
              self.plasticity()

        return



