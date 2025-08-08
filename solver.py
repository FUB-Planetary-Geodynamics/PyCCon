# -*- coding: utf-8 -*-
"""
Created in Dec 2022
Update Sep 2024 (sph. annulus)

@author: Lena Noack
"""
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve, bicg, bicgstab
import math

def Ti(self,k,i):
    return i+k*(self.nl+2)


'''
Energy solver -> returns T
'''

def solver_e(self): 
    # second-order implicit energy solver

    # solver matrix and RHS vector
    num_equ = (self.nl+2)*(self.nr+2)
    A = lil_matrix((num_equ,num_equ))
    b = np.zeros(num_equ)

    # boundary conditions are already included in equations system

    # 2nd-order accuracy: dT/dt = dt_a*T^{n,i} - dt_b*T^{n-1} + dt_c*T^{n-2}
    dt_a = 1.0/self.dt + 1.0/(self.dt+self.dt_old)      # prefactor of the new T^{n,1} (temperature to be solved for: sys_en%x)
    dt_b = 1.0/self.dt + 1.0/self.dt_old                # prefactor of T^{n-1} (stored in field%Told1)
    dt_c = self.dt/(self.dt_old*(self.dt+self.dt_old))  # prefactor of T^{n-2} (stored in field%Told2)

    row = 0

    ###################
    # bottom boundary #
    ###################
    k = 0
    for i in range(self.nl+2):
        A[row,Ti(self,k,i)] = 1.0
        if (self.inner_bound):
            A[row,Ti(self,k+1,i)] = -1.0
            b[row] = 0.0
        else:
            A[row,Ti(self,k+1,i)] = 1.0
            b[row] = 2.0*self.Tbot # since Tbot is set at cell boundary, not cell center
        row = row+1
    
    for k in range(1,self.nr+1):
        #################
        # left boundary #
        #################
        i = 0
        A[row,Ti(self,k,0)] = 1.0
        if self.periodic==0:
          A[row,Ti(self,k,1)] = -1.0 # reflective boundary
        else:
          A[row,Ti(self,k,self.nl)] = -1.0 # periodic boundary
        b[row] = 0.0
        row = row+1

        #################
        # center domain #        
        #################
        for i in range(1,self.nl+1):

          # Density is approximated via the Adams–Williamson relation: 
          if self.compress>0:
            rho = np.exp(self.Di/self.Gr*(self.rmax-self.rc[k])) # rho = rho_ref*exp(Di/Gr); rho_ref is set to one (nondim. value), Gruneisen Gr ~ 1.2
          else:
            rho = 1

          # EBA terms scaled via self.Di; if Di=0 then BA is used
          DiTerm1 = self.Di * 0.5*(self.w[k-1,i]+self.w[k,i])
          DiTerm2 = self.Di * self.ViscD[k-1,i-1] / (self.Ra*rho) - DiTerm1*self.T0

          #   |------w(k,i)-----|
          #   |                 |
          # u(k,i-1) T(k,i)   u(k,i)
          #   |                 |
          #   |-----w(k-1,i)----|

          if (self.geom==0):
            dVi = 1.0/(self.dl*self.dr)
            Lb = self.dl
            Lt = self.dl
            Ll = self.dr
            Lr = self.dr
            dri = 1.0/self.dr
            dli = 1.0/self.dl
            rC = 1.0 # join different geometries in one formulation
            rB = 1.0
            rT = 1.0
          else:
            dVi = 1.0/(self.dl*(self.rc[k]**self.geom)*self.dr) 
            Lb = self.dl*(self.rb[k-1]**self.geom)
            Lt = self.dl*(self.rb[k]**self.geom)
            Ll = self.dr*(self.rc[k]**(self.geom-1))
            Lr = self.dr*(self.rc[k]**(self.geom-1))
            dri = 1.0/self.dr
            dli = 1.0/self.dl
            rC = self.rc[k] # radii ad center, bottom and top of cell
            rB = self.rb[k-1]
            rT = self.rb[k]

          # Solve rho(dT/dt + v*nabla T) = -Di rho w (T+T0) + nabla^2 T + Phi*Di/Ra + rho*H
          #   <=> (dT/dt + v*nabla T) = -Di w (T+T0) + (nabla^2 T)/rho + Phi*Di/(Ra*rho) + H
          # If incompressible, then v*nabla T = nabla(v*T) and FVM would be possible, but for compressible we use FDM for this term
          if (self.solver==1): #FVM
            A[row,Ti(self,k-1,i)] = -0.25*(self.w[k,i]+self.w[k-1,i])*dri - dVi*Lb*dri/rho
            A[row,Ti(self,k,i-1)] = -0.25*(self.u[k,i]+self.u[k,i-1])*dli/rC - dVi*Ll*dli/(rC*rho)
            A[row,Ti(self,k,i)]   = dt_a + DiTerm1 + dVi*(Lr*dli/rC + Ll*dli/rC + Lt*dri + Lb*dri)/rho
            A[row,Ti(self,k,i+1)] = 0.25*(self.u[k,i]+self.u[k,i-1])*dli/rC - dVi*Lr*dli/(rC*rho)
            A[row,Ti(self,k+1,i)] = 0.25*(self.w[k,i]+self.w[k-1,i])*dri - dVi*Lt*dri/rho
          else: # FDM
            A[row,Ti(self,k-1,i)] = -0.25*(self.w[k,i]+self.w[k-1,i])*dri - (rB/rC)**self.geom*dri**2/rho
            A[row,Ti(self,k,i-1)] = -0.25*(self.u[k,i]+self.u[k,i-1])*dli/rC - dli**2/rC/rho
            A[row,Ti(self,k,i)]   = dt_a + DiTerm1 + (2*dli**2/rC + (rT/rC)**self.geom*dri**2 + (rB/rC)**self.geom*dri**2)/rho
            A[row,Ti(self,k,i+1)] = 0.25*(self.u[k,i]+self.u[k,i-1])*dli/rC - dli**2/rC/rho
            A[row,Ti(self,k+1,i)] = 0.25*(self.w[k,i]+self.w[k-1,i])*dri - (rT/rC)**self.geom*dri**2/rho

          heat = self.H0*math.exp(-self.Hlambda*self.t)

          b[row] = dt_b*self.Told1[k,i] - dt_c*self.Told2[k,i] + heat +  DiTerm2
          row = row+1

        ##################
        # right boundary #
        ##################
        i = self.nl+1
        A[row,Ti(self,k,i)] = 1.0
        if self.periodic==0:
          A[row,Ti(self,k,i-1)] = -1.0 # reflective boundary
        else:
          A[row,Ti(self,k,1)] = -1.0 # periodic boundary
        b[row] = 0.0
        row = row+1

    ################
    # top boundary #
    ################
    k = self.nr+1
    for i in range(self.nl+2):
        A[row,Ti(self,k,i)] = 1.0
        A[row,Ti(self,k-1,i)] = 1.0
        b[row] = 2.0*self.Ttop # since Ttop is set at cell boundary, not cell center
        row = row+1

    A = A.tocsr() # convert to CSR format needed to solve sparse linear equation system
    if self.lin_solver == 0:# 0-spsolve, 1-bicg, 2-bicgstab
      x = spsolve(A, b)
    else:
      # get x0 initial solution (from last time/iteration step)
      x0 = np.zeros(num_equ)
      row = 0
      for k in range(self.nr+2):
          for i in range(self.nl+2):
              x0[row] = self.T[k,i]
              row = row+1
      if self.lin_solver == 1:
        x = bicg(A, b, x0)[0]
      else:
        x = bicgstab(A, b, x0)[0]
    #print(x.shape)

    # extract solution
    row = 0
    for k in range(self.nr+2):
        for i in range(self.nl+2):
            self.T[k,i] = x[row]
            row = row+1

    # set back Ttop/Tbot
    if (self.inner_bound==False):
        self.T[0,:] = self.Tbot
    self.T[self.nr+1,:] = self.Ttop

    return

def iu(self,k,i):
    ii = 0
    if self.periodic:
      ii = 1
    return i+k*(self.nl+1+ii)

def iw(self,k,i):
    ii = 0
    if self.periodic:
      ii = 1
    return i+k*(self.nl+2) + (self.nl+1+ii)*(self.nr+2)

def ip(self,k,i):
    ii = 0
    if self.periodic:
      ii = 1
    return i+k*self.nl + (self.nl+1+ii)*(self.nr+2) + (self.nl+2)*(self.nr+1)


'''
Momentum-mass solver -> return u,w and p
'''

def solver_mm(self): 
    # solver matrix: u+w+p 
    if self.periodic==0:
      num_equ = (self.nl+1)*(self.nr+2)+(self.nl+2)*(self.nr+1)+self.nl*self.nr
      #print(num_equ)
      # nl=nr=4
      # mom-l: 0...29
      # mom-r: 30...59
      # mass: 60...75
    else: # extra u bnd cell to the right
      num_equ = (self.nl+2)*(self.nr+2)+(self.nl+2)*(self.nr+1)+self.nl*self.nr
    A = lil_matrix((num_equ,num_equ))
    b = np.zeros(num_equ)

    ##############################
    # momentum lateral direction #
    ##############################
    
    bnd_i = 1
    bnd_o = 1
    if (self.geom>0):
      bnd_i = (self.rc[0]/self.rc[1])#**self.geom   
      bnd_o = (self.rc[self.nr+1]/self.rc[self.nr])#**self.geom

    # values for side conditions for subduction simulations
    v_out = 0.0
    if (self.velr != 0):
#      v_out = abs(self.velr*(self.z_in+0.5*(self.z_out-self.z_in))/((1.0-self.z_out) + 0.5*(self.z_out-self.z_in)))
      v_out = abs(self.velr*self.z_in)/(1.0-self.z_out)

    row=0
    if self.periodic==0:
      ii=0
    else:
      ii=1 # shift max indices by 1 for periodic extra u bnd cell on the right side
    
    # bottom boundary
    k = 0
    for i in range(0,self.nl+1+ii):
        if self.free_slip_b==1:
            A[row,iu(self,0,i)] = 1.0
            A[row,iu(self,1,i)] = -bnd_i
            b[row] = 0.0
        elif self.free_slip_b==0: # no-slip boundary
            A[row,iu(self,0,i)] = 1.0
            A[row,iu(self,1,i)] = bnd_i
            b[row] = 0.0
        else: # fix velocity at bottom
            A[row,iu(self,0,i)] = self.rb[0]/self.rc[0] # 1.0
            A[row,iu(self,1,i)] = 0.0
            b[row] = self.botv
        row = row+1

    # k is 1:nr
    for k in range(1,self.nr+1):            
        # left boundary
        i = 0
        A[row,iu(self,k,i)] = 1.0
        if self.periodic==1:
          A[row,iu(self,k,self.nl)] = -1.0
        if (self.velr==0) or (self.z_right):
            b[row] = 0.0
        else:
          depth = self.rmax-self.rc[k]
          if depth > self.z_out: # incoming plate
            b[row] = - v_out
        row = row+1

        # inside domain, i is 1:nl-1 / for periodic 1:nl
        for i in range(1,self.nl+ii):
            # -----------------
            # |       |       |
            # |   i   |  i+1  |
            # |       |       |
            # -----------------
            #       u(i,k)
            dVi = 1.0/(self.dl*self.dr) # inverse of cell volume
            Lb = self.dl # length of bottom side
            Lt = self.dl # length of top side
            Ll = self.dr # length of left side
            Lr = self.dr # length of right side
            if (self.geom>0):
              dVi = dVi / self.rc[k]**self.geom
              Lb = Lb*self.rb[k-1]**self.geom
              Lt = Lt*self.rb[k]**self.geom
              Ll = Ll*self.rc[k]**(self.geom-1)
              Lr = Lr*self.rc[k]**(self.geom-1)
            dli = 1/self.dl 
            dri = 1/self.dr

            # Density is approximated via the Adams–Williamson relation: 
            if self.compress>0: # rho = rho_ref*exp(Di/Gr); rho_ref is set to one (nondim. value), Gruneisen Gr ~ 1.2
              rhoT = np.exp(self.Di/self.Gr*(self.rmax-self.rb[k])) 
              rho  = np.exp(self.Di/self.Gr*(self.rmax-self.rc[k])) 
              rhoB = np.exp(self.Di/self.Gr*(self.rmax-self.rb[k-1])) 
            else:
              rhoT = 1
              rho  = 1
              rhoB = 1

            if (self.geom>0):
              rTT= self.rc[k+1]
              rT = self.rb[k]
              rC = self.rc[k]
              rB = self.rb[k-1]
              rBB= self.rc[k-1]
              sph = 1 # multiplied to terms that vanish if sph==0 (not the same as self.geom, that can be 1 or 2!)
            else:
              rTT = 1; rT = 1; rC = 1; rB = 1; rBB = 1; sph = 0

            # characteristic viscosity to scale all lines of equation system
            av_eta = 1.0/self.etaA[k,i]
            d = self.geom # 0 for box, 1 for cylinder, 2 for spherical annulus

            #     etaN(k,i)
            #  w(k,i)  w(k,i+1)
            # ----|---*---|----
            # |       |       |
            # |   i   u  i+1  |   -> u(k,i)
            # |       |       |
            # ----|-------|----
            #  w(k-1,i)   

            if (self.solver==1): #FVM
              A[row,iu(self,k-1,i)] = av_eta*( dVi*Lb*self.etaN[k-1,i]*dri )
              A[row,iu(self,k,i-1)] = av_eta*( dVi*Ll*self.eta[k,i]*2.0*dli/rC )
              A[row,iu(self,k,i  )] = av_eta*(-dVi*Lr*self.eta[k,i+1]*2.0*dli/rC
                                              -dVi*Ll*self.eta[k,i]*2.0*dli/rC
                                              -dVi*Lt*self.etaN[k,i]*dri
                                              -dVi*Lb*self.etaN[k-1,i]*dri )
              A[row,iu(self,k,i+1)] = av_eta*( dVi*Lr*self.eta[k,i+1]*2.0*dli/rC )
              A[row,iu(self,k+1,i)] = av_eta*( dVi*Lt*self.etaN[k,i]*dri )

              A[row,iw(self,k-1,i  )] = av_eta*(-dVi*Ll*self.eta[k,i]/3.0*dri * (rhoT-rhoB)/rho # if incompr. rhoT-rhoB=0
                                                +dVi*Lb*self.etaN[k-1,i]*dli/rB )
              A[row,iw(self,k-1,i+1)] = av_eta*( dVi*Lr*self.eta[k,i+1]/3.0*dri * (rhoT-rhoB)/rho
                                                -dVi*Lb*self.etaN[k-1,i]*dli/rB )
              A[row,iw(self,k  ,i  )] = av_eta*(-dVi*Ll*self.eta[k,i]/3.0*dri * (rhoT-rhoB)/rho
                                                -dVi*Lt*self.etaN[k,i]*dli/rT )
              A[row,iw(self,k  ,i+1)] = av_eta*( dVi*Lr*self.eta[k,i+1]/3.0*dri * (rhoT-rhoB)/rho
                                                +dVi*Lt*self.etaN[k,i]*dli/rT )

              A[row,ip(self,k-1,i-1)] = av_eta*dli/rC
              if ((self.periodic==1) and (i==self.nl)):
                A[row,ip(self,k-1,0)] = -av_eta*dli/rC # from left boundary since periodic
              else:
                A[row,ip(self,k-1,i)] = -av_eta*dli/rC

            else: # FDM
              # geom=0:
              #   0 = -dp/dx + d/dx(2 eta du/dx) + d/dz(eta(dw/dx+du/dz))
              # geom>0:
              #   tau_rr = 2*eta * dw/dr
              #   tau_phiphi = 2*eta/r * ( du/dl + w )
              #   tau_thethe = (d-1)*2*eta w/r
              #   tau_rphi = eta ( 1/r dw/dl + r d/dr (u/r) )
              #   0 =  - 1/r dp/dl + 1/r^d d/dr (r^d tau_(rphi)) + 1/r d/dl tau_(phiphi) + tau_rphi/r

              A[row,iu(self,k-1,i)] = av_eta*( rB**(d+1)/rC**d * self.etaN[k-1,i]*dri**2/rBB )
              A[row,iu(self,k,i-1)] = av_eta*( 2.0*self.eta[k,i]*dli**2/rC**2 )
              A[row,iu(self,k,i  )] = av_eta*(-rT**(d+1)/rC**d * self.etaN[k,i]*dri**2/rC 
                                              -rB**(d+1)/rC**d * self.etaN[k-1,i]*dri**2/rC
                                              -2.0*self.eta[k,i+1]*dli**2/rC**2
                                              -2.0*self.eta[k,i]*dli**2/rC**2 )
              A[row,iu(self,k,i+1)] = av_eta*( 2.0*self.eta[k,i+1]*dli**2/rC**2 )
              A[row,iu(self,k+1,i)] = av_eta*( rT**(d+1)/rC**d * self.etaN[k,i]*dri**2/rTT )

              A[row,iw(self,k-1,i  )] = av_eta*( rB**(d-1)/rC**d * self.etaN[k-1,i]*dri*dli
                                                -sph*self.eta[k,i]*dli/rC**2
                                                -self.eta[k,i]/3.0 * (rhoT-rhoB)/rho * dli*dri/rC ) # if incompr, then rhoT-rhoB=0
              A[row,iw(self,k-1,i+1)] = av_eta*(-rB**(d-1)/rC**d * self.etaN[k-1,i]*dri*dli
                                                +sph*self.eta[k,i+1]*dli/rC**2
                                                +self.eta[k,i+1]/3.0 * (rhoT-rhoB)/rho * dli*dri/rC )
              A[row,iw(self,k  ,i  )] = av_eta*(-rT**(d-1)/rC**d * self.etaN[k,i]*dri*dli
                                                -sph*self.eta[k,i]*dli/rC**2
                                                -self.eta[k,i]/3.0 * (rhoT-rhoB)/rho * dli*dri/rC )
              A[row,iw(self,k  ,i+1)] = av_eta*( rT**(d-1)/rC**d * self.etaN[k,i]*dri*dli 
                                                +sph*self.eta[k,i+1]*dli/rC**2
                                                +self.eta[k,i+1]/3.0 * (rhoT-rhoB)/rho * dli*dri/rC )

              A[row,ip(self,k-1,i-1)] = av_eta*dli/rC
              if ((self.periodic==1) and (i==self.nl)):
                A[row,ip(self,k-1,0)] = -av_eta*dli/rC # from left boundary since periodic
              else:
                A[row,ip(self,k-1,i)] = -av_eta*dli/rC

            b[row] = 0.0
            row = row+1


        # right boundary cell
        i = self.nl+ii
        A[row,iu(self,k,i)] = 1.0
        if self.periodic==1:
          #if (k==1) and (self.botv == 0):
          if (k==self.nr) and (self.botv == 0):
            A[row,iu(self,k,1)] = 0.0 # ToDO trick to avoid lateral angular rotation of entire sphere
          else:
            A[row,iu(self,k,1)] = -1.0
        if (self.velr==0):
            b[row] = 0.0
        else:
          depth = self.rmax-self.rc[k]
          if depth < self.z_in: # incoming plate
            b[row] = - self.velr
          if self.z_right:
            if depth > self.z_out:  # outgoing mantle flow
              b[row] = v_out
              #print(k,v_out,-self.velr)
        row = row+1

    # top boundary
    k = self.nr+1
    for i in range(0,self.nl+1+ii):
        if self.free_slip_t:
            A[row,iu(self,k,i)] = 1.0
            A[row,iu(self,k-1,i)] = -bnd_o
            b[row] = 0.0
        else: # no-slip boundary
            A[row,iu(self,k,i)] = 1.0
            A[row,iu(self,k-1,i)] = bnd_o
            b[row] = 0.0
        row = row+1


    #############################
    # momentum radial direction #
    #############################

    # bottom boundary
    k = 0
    for i in range(0,self.nl+2):
        A[row,iw(self,0,i)] = 1.0 # for both free-slip and no-slip, radial velocity is zero at boundaries
        b[row] = 0.0
        row = row+1

    # k is 1:nr-1
    for k in range(1,self.nr):            
        # left boundary cell, reflective boundary
        i = 0
        A[row,iw(self,k,0)] = 1.0
        if self.periodic==0:
          A[row,iw(self,k,1)] = -1.0
        else:
          A[row,iw(self,k,self.nl)] = -1.0
        b[row] = 0.0
        row = row+1

        # inside domain, i is 1:nl
        for i in range(1,self.nl+1):
            # ---------
            # |       |
            # |  k+1  |
            # |       |
            # ----w---- <- w(i,k)
            # |       |
            # |   k   |
            # |       |
            # ---------
            dVi = 1.0/(self.dl*self.dr) # inverse of cell volume
            Lb = self.dl # length of bottom side
            Lt = self.dl # length of top side
            Ll = self.dr # length of left side
            Lr = self.dr # length of right side
            if (self.geom>0):
              dVi = dVi / self.rb[k]**self.geom
              Lb = Lb*self.rc[k]**self.geom
              Lt = Lt*self.rc[k+1]**self.geom
              Ll = Ll*self.rb[k]**(self.geom-1)
              Lr = Lr*self.rb[k]**(self.geom-1)
            dli = 1/self.dl 
            dri = 1/self.dr

            # Density is approximated via the Adams–Williamson relation: 
            if self.compress>0: # rho = rho_ref*exp(Di/Gr); rho_ref is set to one (nondim. value), Gruneisen Gr ~ 1.2
              rhoTT = np.exp(self.Di/self.Gr*(self.rmax-self.rb[k+1])) 
              rhoT = np.exp(self.Di/self.Gr*(self.rmax-self.rc[k+1])) 
              rho  = np.exp(self.Di/self.Gr*(self.rmax-self.rb[k])) 
              rhoB = np.exp(self.Di/self.Gr*(self.rmax-self.rc[k])) 
              rhoBB = np.exp(self.Di/self.Gr*(self.rmax-self.rb[k-1])) 
            else:
              rhoTT = 1; rhoT = 1; rho  = 1; rhoB = 1; rhoBB = 1

            if self.geom>0:
              rT = self.rc[k+1]
              rC = self.rb[k]
              rB = self.rc[k]
              sph = 1 # multiplied to terms that vanish if sph==0 (not the same as self.geom, that can be 1 or 2!)
            else:
              rT = 1; rC = 1; rB = 1
              sph = 0


            # characteristic viscosity to scale all lines of equation system
            av_eta = 1.0/self.etaC[k,i]
            d = self.geom

            # ---------
            # |       |
            # |  k+1  |
            # |       |
            # ----w---* etaN(k,i)      <- w(k,i)
            # |       |
            # |   k  u(k,i)
            # |       |
            # ---------

            if self.solver==1: #FVM
              A[row,iu(self,k  ,i-1)] = av_eta*( dVi*Ll*self.etaN[k,i-1]*dri )
              A[row,iu(self,k  ,i  )] = av_eta*(-dVi*Lr*self.etaN[k,i]*dri )
              A[row,iu(self,k+1,i-1)] = av_eta*(-dVi*Ll*self.etaN[k,i-1]*dri )
              A[row,iu(self,k+1,i  )] = av_eta*( dVi*Lr*self.etaN[k,i]*dri )

              A[row,iw(self,k-1,i  )] = av_eta*( dVi*Lb*self.eta[k,i]*2.0*dri
                                                -dVi*Lb*self.eta[k,i]/3.0*dri * (rho-rhoBB)/rhoB )
              A[row,iw(self,k  ,i-1)] = av_eta*( dVi*Ll*self.etaN[k,i-1]*dli/rC )
              A[row,iw(self,k  ,i  )] = av_eta*(-dVi*Lr*self.etaN[k,i]*dli/rC
                                                -dVi*Ll*self.etaN[k,i-1]*dli/rC
                                                -dVi*Lt*self.eta[k+1,i]*2.0*dri
                                                -dVi*Lb*self.eta[k,i]*2.0*dri
                                                +dVi*Lt*self.eta[k+1,i]/3.0*dri * (rhoTT-rho)/rhoT
                                                -dVi*Lb*self.eta[k,i]/3.0*dri * (rho-rhoBB)/rhoB )
              A[row,iw(self,k  ,i+1)] = av_eta*( dVi*Lr*self.etaN[k,i]*dli/rC )
              A[row,iw(self,k+1,i  )] = av_eta*( dVi*Lt*self.eta[k+1,i]*2.0*dri
                                                +dVi*Lt*self.eta[k+1,i]/3.0*dri * (rhoTT-rho)/rhoT )

              A[row,ip(self,k-1,i-1)] = av_eta*dri 
              A[row,ip(self,k,i-1)] = -av_eta*dri

            else: # FDM
              # geom=0:
              #   -RaT = -dp/dz + d/dx(eta*(dw/dx+du/dz)) + d/dz(2eta dw/dz)
              # geom>0:
              #   tau_rr = 2*eta * dw/dr
              #   tau_phiphi = 2*eta/r * ( du/dl + w )
              #   tau_thethe = (d-1)*2*eta w/r
              #   tau_rphi = eta ( 1/r dw/dl + r d/dr (u/r) )
              #   -RaT = 1/r^d d/dr (r^d tau_(rr)) + 1/r d/dl tau_(rphi) - (tau_thethe+tau_rphi)/r - dp/dr
              A[row,iu(self,k  ,i-1)] = av_eta*( self.eta[k,i-1]*dli*dri/rB
                                                +sph*self.etaC[k,i]*dli/rC**2 )
              A[row,iu(self,k  ,i  )] = av_eta*(-self.eta[k,i]*dli*dri/rB
                                                -sph*self.etaC[k,i]*dli/rC**2 )
              A[row,iu(self,k+1,i-1)] = av_eta*(-self.eta[k,i-1]*dli*dri/rT
                                                +sph*self.etaC[k,i]*dli/rC**2 )
              A[row,iu(self,k+1,i  )] = av_eta*( self.eta[k,i]*dli*dri/rT
                                                -sph*self.etaC[k,i]*dli/rC**2 )

              A[row,iw(self,k-1,i  )] = av_eta*( 2.0*self.eta[k,i]*(rB/rC)**d * dri**2
                                                -self.eta[k,i]/3.0*(rB/rC)**d * dri**2 * (rho-rhoBB)/rhoB )
              A[row,iw(self,k  ,i-1)] = av_eta*( self.etaN[k,i-1]*dli**2/rC**2 )
              A[row,iw(self,k  ,i  )] = av_eta*(-2.0*self.eta[k+1,i]*(rT/rC)**d * dri**2
                                                -2.0*self.eta[k,i]*(rB/rC)**d * dri**2
                                                -self.eta[k,i]/3.0*(rB/rC)**d * dri**2 * (rho-rhoBB)/rhoB
                                                +self.eta[k+1,i]/3.0*(rT/rC)**d * dri**2 * (rhoTT-rho)/rhoT
                                                -self.etaN[k,i]*dli**2/rC**2
                                                -self.etaN[k,i-1]*dli**2/rC**2
                                                -sph*d*2.0*self.etaC[k,i]*(1.0/rC**2+dri/(3.0*rC)*(rhoT-rhoB)/rho) )
              A[row,iw(self,k  ,i+1)] = av_eta*( self.etaN[k,i]*dli**2/rC**2 )
              A[row,iw(self,k+1,i  )] = av_eta*( 2.0*self.eta[k+1,i]*(rT/rC)**d * dri**2
                                                +self.eta[k+1,i]/3.0*(rT/rC)**d * dri**2 * (rhoTT-rho)/rhoT )

              A[row,ip(self,k-1,i-1)] = av_eta*dri 
              A[row,ip(self,k,i-1)] = -av_eta*dri

            # Still needs checking: with or without T0? Though results are not affected at least for isoviscous simulations
#            BoussT = self.Ra*(0.5*(self.T[k,i]+self.T[k+1,i])-self.T0) # T - (Tref+T0); Tref not seperately defined, assume here Tref==0 => surf temp
            BoussT = self.Ra*(0.5*(self.T[k,i]+self.T[k+1,i])) # T - (Tref+T0); Tref not seperately defined, assume here Tref==-T0 => 0K
            BoussC = self.B*self.Ra*(0.5*(self.C[k,i]+self.C[k+1,i])-self.Cref)
            b[row] = av_eta*rho*(BoussC - BoussT)
            row = row+1

        # right boundary cell, reflective boundary
        i = self.nl+1
        A[row,iw(self,k,i)] = 1.0
        if self.periodic==0:
          A[row,iw(self,k,i-1)] = -1.0
        else:
          A[row,iw(self,k,1)] = -1.0
        b[row] = 0.0
        row = row+1

    # top boundary
    k = self.nr
    for i in range(0,self.nl+2):
        A[row,iw(self,k,i)] = 1.0 # for both free-slip and no-slip, radial velocity is zero at boundaries
        b[row] = 0.0
        row = row+1

    
    #####################
    # mass conservation #
    #####################

    # continuity equation
    # d(u*rho)/dl + d(w*rho)/dr = 0

    for k in range(1,self.nr+1):
        for i in range(1,self.nl+1):
            # --------- 
            # |       |
            # |  k,i  |
            # |       |
            # ---------
            dVi = 1.0/(self.dl*self.dr) # inverse of cell volume
            Lb = self.dl # length of bottom side
            Lt = self.dl # length of top side
            Ll = self.dr # length of left side
            Lr = self.dr # length of right side
            if (self.geom>0):
              dVi = dVi/(self.rc[k]**self.geom) 
              Lb = Lb*(self.rb[k-1]**self.geom)
              Lt = Lt*(self.rb[k]**self.geom)
              Ll = Ll*(self.rc[k]**(self.geom-1))
              Lr = Lr*(self.rc[k]**(self.geom-1))

            # Density is approximated via the Adams–Williamson relation: 
            if self.compress>0: # rho = rho_ref*exp(Di/Gr); rho_ref is set to one (nondim. value), Gruneisen Gr ~ 1.2
              rhoT = np.exp(self.Di/self.Gr*(self.rmax-self.rb[k])) 
              rho  = np.exp(self.Di/self.Gr*(self.rmax-self.rc[k])) 
              rhoB = np.exp(self.Di/self.Gr*(self.rmax-self.rb[k-1])) 
            else:
              rhoT = 1; rho  = 1; rhoB = 1

            if self.geom>0:
              rT = self.rb[k]
              rC = self.rc[k]
              rB = self.rb[k-1]
            else:
              rT = 1; rC = 1; rB = 1

            # characteristic viscosity to scale all lines of equation system
            av_eta = 1.0/self.eta[k,i]

            if self.solver==1: #FVM
              A[row,iu(self,k,i-1)] = -av_eta*Ll*dVi * rho
              A[row,iu(self,k,i)] = av_eta*Lr*dVi * rho
              A[row,iw(self,k-1,i)] = -av_eta*Lb*dVi * rhoB
              A[row,iw(self,k,i)] = av_eta*Lt*dVi * rhoT
            else:
              # geom=0:
              #   0 = du/dx + dw/dz
              # geom>0:
              #   0 = 1/r du/dl + 1/r**d d/dr (r**d w)
              A[row,iu(self,k,i-1)] = -av_eta*dli/rC * rho
              A[row,iu(self,k,i)] = av_eta*dli/rC * rho
              A[row,iw(self,k-1,i)] = -av_eta*dri*(rB/rC)**self.geom * rhoB
              A[row,iw(self,k,i)] = av_eta*dri*(rT/rC)**self.geom * rhoT

              # penalty term for incompressible fluids, Zhong et al., 2007, Treatise
            A[row,ip(self,k-1,i-1)] = self.penalty/self.eta[k,i]

            b[row] = 0.0
            row = row+1

    #print(A)

    A = A.tocsr() # convert to CSR format needed to solve sparse linear equation system
    if self.lin_solver == 0:# 0-spsolve, 1-bicg, 2-bicgstab
      x = spsolve(A, b) # x includes u,w,p
    else:
      # get x0 initial solution (from last time/iteration step)
      x0 = np.zeros(num_equ)
      row = 0
      for k in range(self.nr+2):
        for i in range(self.nl+1+ii):
          x0[row] = self.u[k,i]
          row = row+1
            
      for k in range(self.nr+1):
        for i in range(self.nl+2):
          x0[row] = self.w[k,i]
          row = row+1

      for k in range(self.nr):
        for i in range(self.nl):
          x0[row] = self.p[k,i]
          row = row+1
      if self.lin_solver == 1:
        x = bicg(A, b, x0)[0]
      else:
        x = bicgstab(A, b, x0)[0]



    # extract solution
    row = 0
    for k in range(self.nr+2):
        for i in range(self.nl+1+ii):
            self.u[k,i] = x[row]
            row = row+1
        
    for k in range(self.nr+1):
        for i in range(self.nl+2):
            self.w[k,i] = x[row]
            row = row+1

    for k in range(self.nr):
        for i in range(self.nl):
            self.p[k,i] = x[row]
            row = row+1


    if (self.corrRot):
      # correct net rotation
      sum_v = 0
      vol = 0
      for k in range(self.nr):
        for i in range(self.nl):
          if self.geom==0:
            sum_v += 0.5*(self.u[k+1,i]+self.u[k+1,i+1])*self.dr*self.dl
            vol += self.dr*self.dl
          else:
            sum_v += 0.5*(self.u[k+1,i]+self.u[k+1,i+1])/self.rc[k+1]*self.dr*self.dl*self.rc[k+1]
            vol += self.dr*self.dl*self.rc[k+1]
      sum_v = sum_v/vol

      for k in range(self.nr+2):
        for i in range(self.nl+1+ii):
          if self.geom==0:
            self.u[k,i] -= sum_v
          else:
            self.u[k,i] -= sum_v*self.rc[k]

    return


'''
Compositional solver -> returns C
'''

def solver_c(self): 
    # second-order implicit energy solver
    # ToDo: add FVM as above for eneryg solver

    # solver matrix and RHS vector
    num_equ = (self.nl+2)*(self.nr+2)
    A = lil_matrix((num_equ,num_equ))
    b = np.zeros(num_equ)

    # boundary conditions are already included in equations system

    # 2nd-order accuracy: dC/dt = dt_a*C^{n,i} - dt_b*C^{n-1} + dt_c*C^{n-2}
    # for constant dt: dC/dt = 1/dt * [ 3/2*C^{n,i} - 2*C^{n-1} + 1/2*C^{n-2} ]
    dt_a = 1.0/self.dt + 1.0/(self.dt+self.dt_old)      # prefactor of the new C^{n,1} (composition to be solved for: sys_en%x)
    dt_b = 1.0/self.dt + 1.0/self.dt_old                # prefactor of C^{n-1} (stored in field%Cold1)
    dt_c = self.dt/(self.dt_old*(self.dt+self.dt_old))  # prefactor of C^{n-2} (stored in field%Cold2)

    row = 0
    
    #below keep Ti() for indices, since the same field size is used for compositional field
    
    ###################
    # bottom boundary #
    ###################
    k = 0
    for i in range(self.nl+2):
        A[row,Ti(self,k,i)] = 1.0 # uses the value from above cell, since here no boundary values are given for composition
        A[row,Ti(self,k+1,i)] = -1.0
        b[row] = 0.0
        row = row+1
    
    for k in range(1,self.nr+1):
        #################
        # left boundary #
        #################
        i = 0
        A[row,Ti(self,k,0)] = 1.0
        if self.periodic==0:
          A[row,Ti(self,k,1)] = -1.0 # reflective boundary
        else:
          A[row,Ti(self,k,self.nl)] = -1.0 # periodic boundary
        b[row] = 0.0
        row = row+1

        #################
        # center domain #        
        #################
        for i in range(1,self.nl+1):
          if (self.geom==0):
            dVi = 1.0/(self.dl*self.dr)
            Lb = self.dl
            Lt = self.dl
            Ll = self.dr
            Lr = self.dr
            dri = 1.0/self.dr
            dli = 1.0/self.dl
            Lei = 1.0/self.Le
            rT = 1; rC = 1; rB = 1

            '''
            A[row,Ti(self,k-1,i)] = -Lb*dri*dVi*Lei - 0.5*self.w[k-1,i]*Lb*dVi
            A[row,Ti(self,k,i-1)] = -Ll*dli*dVi*Lei - 0.5*self.u[k,i-1]*Ll*dVi
            A[row,Ti(self,k,i)] = dt_a + (Lr*dli+Ll*dli+Lt*dri+Lb*dri)*dVi*Lei \
              - 0.5*(self.u[k,i-1]*Ll-self.w[k,i]*Lt-self.u[k,i]*Lr+self.w[k-1,i]*Lb)*dVi # changed analogously to energy solver
            A[row,Ti(self,k,i+1)] = -Lr*dli*dVi*Lei + 0.5*self.u[k,i]*Lr*dVi
            A[row,Ti(self,k+1,i)] = -Lt*dri*dVi*Lei + 0.5*self.w[k,i]*Lt*dVi
            '''

          else:
            dVi = 1.0/(self.dl*(self.rc[k]**self.geom)*self.dr) 
            Lb = self.dl*(self.rb[k-1]**self.geom)
            Lt = self.dl*(self.rb[k]**self.geom)
            Ll = self.dr*(self.rc[k]**(self.geom-1))
            Lr = self.dr*(self.rc[k]**(self.geom-1))
            dri = 1.0/self.dr
            dli = 1.0/self.dl
            Lei = 1.0/self.Le
            rT = self.rb[k]
            rC = self.rc[k]
            rB = self.rb[k-1]

            '''
            # here v*nabla T exchanged by nabla (v*T), but this only works for incompressible flows with nabla v = 0
            A[row,Ti(self,k-1,i)] = -Lb*dri*dVi*Lei - 0.5*self.w[k-1,i]*Lb*dVi
            A[row,Ti(self,k,i-1)] = -Ll*dli*dVi*Lei*rc_inv - 0.5*self.u[k,i-1]*Ll*dVi
            A[row,Ti(self,k,i)] = dt_a + (Lr*dli*rc_inv+Ll*dli*rc_inv+Lt*dri+Lb*dri)*dVi*Lei \
              - 0.5*(self.u[k,i-1]*Ll-self.w[k,i]*Lt-self.u[k,i]*Lr+self.w[k-1,i]*Lb)*dVi 
            A[row,Ti(self,k,i+1)] = -Lr*dli*dVi*Lei*rc_inv + 0.5*self.u[k,i]*Lr*dVi
            A[row,Ti(self,k+1,i)] = -Lt*dri*dVi*Lei + 0.5*self.w[k,i]*Lt*dVi
            '''

          if (self.solver==1): #FVM
            A[row,Ti(self,k-1,i)] = -0.25*(self.w[k,i]+self.w[k-1,i])*dri - dVi*Lb*dri*Lei
            A[row,Ti(self,k,i-1)] = -0.25*(self.u[k,i]+self.u[k,i-1])*dli/rC - dVi*Ll*dli*Lei/(rC)
            A[row,Ti(self,k,i)]   = dt_a + dVi*(Lr*dli/rC + Ll*dli/rC + Lt*dri + Lb*dri)*Lei
            A[row,Ti(self,k,i+1)] = 0.25*(self.u[k,i]+self.u[k,i-1])*dli/rC - dVi*Lr*dli*Lei/(rC)
            A[row,Ti(self,k+1,i)] = 0.25*(self.w[k,i]+self.w[k-1,i])*dri - dVi*Lt*dri*Lei
          else: # FDM
            A[row,Ti(self,k-1,i)] = -0.25*(self.w[k,i]+self.w[k-1,i])*dri - (rB/rC)**self.geom*dri**2*Lei
            A[row,Ti(self,k,i-1)] = -0.25*(self.u[k,i]+self.u[k,i-1])*dli/rC - dli**2*Lei/rC
            A[row,Ti(self,k,i)]   = dt_a + (2*dli**2/rC + (rT/rC)**self.geom*dri**2 + (rB/rC)**self.geom*dri**2)*Lei
            A[row,Ti(self,k,i+1)] = 0.25*(self.u[k,i]+self.u[k,i-1])*dli/rC - dli**2*Lei/rC
            A[row,Ti(self,k+1,i)] = 0.25*(self.w[k,i]+self.w[k-1,i])*dri - (rT/rC)**self.geom*dri**2*Lei

          b[row] = dt_b*self.Cold1[k,i] - dt_c*self.Cold2[k,i]
          row = row+1

        ##################
        # right boundary #
        ##################
        i = self.nl+1
        A[row,Ti(self,k,i)] = 1.0
        if self.periodic==0:
          A[row,Ti(self,k,i-1)] = -1.0 # reflective boundary
        else: 
          A[row,Ti(self,k,1)] = -1.0 # periodic boundary
        b[row] = 0.0
        row = row+1

    ################
    # top boundary #
    ################
    k = self.nr+1
    for i in range(self.nl+2):
        A[row,Ti(self,k,i)] = 1.0
        A[row,Ti(self,k-1,i)] = -1.0
        b[row] = 0.0
        row = row+1
    

    # ToDo: add different solver routines here as in energy solver
    A = A.tocsr() # convert to CSR format needed to solve sparse linear equation system
    x = spsolve(A, b)

    # extract solution
    row = 0
    for k in range(self.nr+2):
        for i in range(self.nl+2):
            self.C[k,i] = x[row]
            row = row+1

    return
