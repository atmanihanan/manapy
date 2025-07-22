from numpy import zeros
from mpi4py import MPI
import numpy as np

from manapy.solvers.combustion import (update_euler, time_step_euler, explicitscheme_convective_euler,explicitscheme_dissipative_NS,term_source)


from manapy.comms import Iall_to_all
from manapy.comms import define_halosend

from manapy.ast import Variable
from manapy.base.base import Struct
from manapy.base.base import make_get_conf


class EulerSolver():
    
    _parameters = [('dt', float, 0., 0., 'time step'),
                    ('order', int, 1, 2,'order of the convective scheme'),
                    ('cfl', float, .5, .5,'cfl of the explicit scheme'),
                     ('gamma', float, 1.4, 1.4,'Adiabatic index'),   
                      ('Cp',float,1004.5, 1004.5  ,'specific heat at constant pressure'),
                      ('Cv', float,  717.0,717.0, 'specific heat at constant volume '),
                      ('Pr',float,0.5,0.5,'Prandtl number'),
                      ('R',int,287,287,'perfect gas constant '),
                      ('T_ref', float , 293,293,'reference temperature '),
                      ('mu_ref', float , 1.8e-5,3e-2,'reference viscosity '),
                      ('C_zero', float , 110.4, 110.4,'Sutherland constant '),
                    #   ('mu',float,0.,1.,'dynamc viscosity'),
                      ('Ta', float, 1e4, 1e4, 'Activation Temperature'),
                      ('B', float,3.3e7, 1.5e11, 'Facteur B(T)'),
                      ('deltaQ', float,4e4,4e7,'Heat of Reaction'),
                      ('alpha', float,0, 0,'la puissance de T dan la loi arrhenius'),
                      ('M1', float , 16e-3, 16, 'masse molaire espece1'),
                      ('M2', float , 32e-3, 32, 'masse molaire espece2'),
                      ('mu1', int, 0,1, 'coeff stochimetrique  mu1'),
                      ('mu2', int, 0,2, 'coeff stochimetrique  mu2'),
                      ('nu1', int, 0, 1, 'coeff stochimetrique  nu1'),
                      ('nu2', int, 0,1, 'coeff stochimetrique  nu2'),
                      ('Sc1',float, 1,1,'Nombre de Schmidt pour l espice1'),
                      ('Sc2',float, 1,1,'Nombre de Schmidt pour l espice2'),

                       
    ]
    
    @classmethod
    def process_conf(cls, conf, kwargs):
        """                              
        # print(F" I am the process {self.RANK}, I received data {self.scheme_to_exec}  from 0")
                                                                                                                                                                       
        Process configuration parameters.                                                                                                                                                                  
        """
        get = make_get_conf(conf, kwargs)
        
        if len(cls._parameters) and cls._parameters[0][0] != 'name':
            options = EulerSolver._parameters + cls._parameters
        else:
            options = EulerSolver._parameters
            
        opts = Struct()
        allow_extra = False
        for name, _, default, required, _ in options:
            if name == '*':
                allow_extra = True
                continue

            msg = ('missing "%s" in options!' % name) if required else None
            setattr(opts, name, get(name, default, msg))

        if allow_extra:
            all_keys = set(conf.to_dict().keys())
            other = all_keys.difference(list(opts.to_dict().keys()))
            for name in other:
                setattr(opts, name, get(name, None, None))
                
        return opts
    
    def __init__(self, rho=None, rhovel=(None, None), rhoE=None,e_internal=None, P=None, T=None, Y1=None, Y2=None, rhoY1=None, rhoY2=None, conf=None, RANK=None, COMM=None,**kwargs):
        if conf is None:
            conf = Struct()
            
        new_conf = self.process_conf(conf, kwargs)
        self.conf = new_conf
        get = make_get_conf(self.conf, kwargs)
        
        if not isinstance(rho, Variable):
            raise ValueError("rho must be a Variable type")
        
        if not isinstance(rhovel[0], Variable):
            raise ValueError("rhou must be a Variable type")
        
        if not isinstance(rhovel[1], Variable):
            raise ValueError("rhov must be a Variable type")
        if not isinstance(rhoE, Variable):
            raise ValueError("rhoE must be a Variable type")
        if not isinstance(P, Variable):
            raise ValueError("P must be a Variable type")
        if not isinstance(e_internal, Variable):
            raise ValueError("e_internal must be a Variable type")
        
        self.rho = rho
        self.comm = self.rho.comm
        self.domain = self.rho.domain
        self.dim = self.rho.dim
        self.float_precision = self.domain.float_precision

        self.rhou  = rhovel[0]
        self.rhov   = rhovel[1]
        self.rhoE   = rhoE
        self.P = P
        self.T= T
        self.e_internal=e_internal
        self.Y1 = Y1
        self.Y2 = Y2
        self.rhoY1 = rhoY1
        self.rhoY2 = rhoY2
        
        self.varbs = {}
        self.varbs['rho'] = self.rho
        self.varbs['rhou'] = self.rhou
        self.varbs['rhov'] = self.rhov
        self.varbs['rhoE'] = self.rhoE
        self.varbs['P'] = self.P
        self.varbs['T'] = self.T
        self.varbs['e_internal'] = self.e_internal
        self.varbs['rhoY1'] = self.rhoY1
        self.varbs['rhoY2'] = self.rhoY2
        self.varbs['Y1'] = self.Y1
        self.varbs['Y2'] = self.Y2

        
        terms = [ 'source','dissipation','convective']
        for var in self.varbs.values():
            for term in terms:
                var.__dict__[term] = zeros(self.domain.nbcells, dtype=self.float_precision)
         
        
        # les paramétres
        self.dt    = get("dt")
        self.order = get("order")
        self.cfl   = get("cfl")
        self.gamma  = get("gamma")
        self.T_ref=get("T_ref")
        self.mu_ref=get("mu_ref")
        self.Pr=get("Pr")
        self.Cp=get("Cp")
        self.C_zero=get("C_zero")
        self.R=get("R")
        self.Cv=get("Cv")
        self.M1=get("M1")
        self.M2=get("M2")
        self.mu1=get("mu1")
        self.mu2=get("mu2")
        self.nu1=get("nu1")
        self.nu2=get("nu2")
        self.deltaQ=get("deltaQ")
        self.B=get("B")
        self.alpha=get("alpha")
        self.Ta=get("Ta")
        self.Sc1=get("Sc1")
        self.Sc2=get("Sc2")


        self.backend = self.domain.backend
        self.signature = self.domain.signature
        
        self._explicitscheme_convective  = self.backend.compile(explicitscheme_convective_euler, signature=self.signature)
        self._explicit_dissipative  = self.backend.compile(explicitscheme_dissipative_NS, signature=self.signature)
        self._time_step_euler= self.backend.compile(time_step_euler, signature=self.signature)
        self._update_new_value = self.backend.compile(update_euler, signature=self.signature)
       # self._update_new_value = self.backend.compile(update_SW, signature=self.signature)
        self.RANK=RANK
        self.COMM=COMM
        self.limiter_to_exec=10
        # ==========================
        if self.order==2:
            if self.RANK==0:
                self.available_limiters = {
                    "1": "Minmodf",
                    "2": "VanAlbada",
                    "3": "Barth",
                    "4": "Minmodn",
                    
                    # "5": "lax_friedrichs",
                    # "6": "Roe_version2
                    # "7": "AUSM"
                }
                print("*"*29, "Limiter Choice", "*"*28)
                print("Available Limiters: ")
                for i, name in self.available_limiters.items():
                    print(f'{i} - {name}')
                limiter = input('Choose the scheme (integer): ')
                while limiter not in self.available_limiters.keys():
                    print(">> Wrong choice")
                    limiter= input('Choose the scheme (integer): ')
                sendbuf_limiter=int(limiter)
            else:
                sendbuf_limiter = None
            self.limiter_to_exec=self.COMM.bcast(sendbuf_limiter,root=0)
            # print(F" I am the process {self.RANK}, I received data {self.limiter_to_exec}  from 0")
        
        
        self.comm.Barrier()


        if self.RANK==0:
            
            self.available_schemes = {
                "1": "rusanov",
                "2": "Roe",
                "3": "HLL",
                "4": "SRNH, VFRoe",
                # "5": "lax_friedrichs",
                # "6": "Roe_version2
                # "7": "AUSM"
            }
            print("*"*29, "Scheme Choice", "*"*28)
            print("Available schemes: ")
            for i, name in self.available_schemes.items():
                print(f'{i} - {name}')
            scheme = input('Choose the scheme (integer): ')
            while scheme not in self.available_schemes.keys():
                print(">> Wrong choice")
                scheme = input('Choose the scheme (integer): ')
          
            sendbuf=int(scheme)
        else:
          sendbuf = None
          

        
        
        self.scheme_to_exec=self.COMM.bcast(sendbuf,root=0)
        
        # print(F" I am the process {self.RANK}, I received data {self.scheme_to_exec}  from 0")
    
    
        # ==========================
     
    def explicit_convective(self):
        if self.order == 2:
            self.rho.compute_cell_gradient(self.limiter_to_exec)
            self.rhou.compute_cell_gradient(self.limiter_to_exec)
            self.rhov.compute_cell_gradient(self.limiter_to_exec)
            self.rhoE.compute_cell_gradient(self.limiter_to_exec)
            self.P.compute_cell_gradient(self.limiter_to_exec)
            self.rhoY1.compute_cell_gradient(self.limiter_to_exec)
            self.rhoY2.compute_cell_gradient(self.limiter_to_exec)
   

    
        explicitscheme_convective_euler(self.rho.convective, self.rhou.convective, self.rhov.convective, self.rhoE.convective, self.rhoY1.convective, self.rhoY2.convective, 
                                     self.rho.cell, self.rhou.cell, self.rhov.cell, self.rhoE.cell, self.P.cell,self.rhoY1.cell, self.rhoY2.cell,
                                     self.rho.ghost, self.rhou.ghost, self.rhov.ghost, self.rhoE.ghost,self.P.ghost,self.rhoY1.ghost,self.rhoY2.ghost,
                                     self.rho.halo, self.rhou.halo, self.rhov.halo, self.rhoE.halo,self.P.halo,self.rhoY1.halo, self.rhoY2.halo,
                                     self.rho.gradcellx, self.rho.gradcelly, self.rho.gradhalocellx, self.rho.gradhalocelly,
                                     self.rhou.gradcellx, self.rhou.gradcelly, self.rhou.gradhalocellx, self.rhou.gradhalocelly,
                                     self.rhov.gradcellx, self.rhov.gradcelly, self.rhov.gradhalocellx, self.rhov.gradhalocelly,
                                     self.rhoE.gradcellx, self.rhoE.gradcelly, self.rhoE.gradhalocellx, self.rhoE.gradhalocelly,
                                     self.P.gradcellx, self.P.gradcelly, self.P.gradhalocellx, self.P.gradhalocelly,self.rhoY1.gradcellx, self.rhoY1.gradcelly, self.rhoY1.gradhalocellx, self.rhoY1.gradhalocelly,self.rhoY2.gradcellx, self.rhoY2.gradcelly, self.rhoY2.gradhalocellx, self.rhoY2.gradhalocelly
                                     ,self.rho.psi,self.rho.psihalo,self.rhou.psi,self.rhou.psihalo,self.rhov.psi,self.rhov.psihalo,self.rhoE.psi,self.rhoE.psihalo,
                                    self.P.psi,self.P.psihalo,self.rhoY1.psi,self.rhoY1.psihalo,self.rhoY2.psi,self.rhoY2.psihalo,
                                    self.domain.cells.center,self.domain.faces.center, self.domain.halos.centvol, self.domain.faces.ghostcenter,
                                     self.domain.faces.cellid, self.domain.faces.mesure, self.domain.faces.normal,self.domain.faces.tangent, 
                                     self.domain.faces.halofid,self.domain.innerfaces, self.domain.halofaces, self.domain.boundaryfaces, self.gamma, 
                                     self.order, self.scheme_to_exec, self.limiter_to_exec, self.rho.minmod_x
                                    ,self.rho.minmod_y,self.rho.minmod_halo_x,self.rho.minmod_halo_y,self.rhou.minmod_x
                                    ,self.rhou.minmod_y,self.rhou.minmod_halo_x,self.rhou.minmod_halo_y,self.rhov.minmod_x
                                    ,self.rhov.minmod_y,self.rhov.minmod_halo_x,self.rhov.minmod_halo_y,self.rhoE.minmod_x
                                    ,self.rhoE.minmod_y,self.rhoE.minmod_halo_x,self.rhoE.minmod_halo_y,self.P.minmod_x
                                    ,self.P.minmod_y,self.P.minmod_halo_x,self.P.minmod_halo_y,self.rhoY1.minmod_x
                                    ,self.rhoY1.minmod_y,self.rhoY1.minmod_halo_x,self.rhoY1.minmod_halo_y,self.rhoY2.minmod_x
                                    ,self.rhoY2.minmod_y,self.rhoY2.minmod_halo_x,self.rhoY2.minmod_halo_y,
                                    self.rho.vanalbada_limiter,self.rhou.vanalbada_limiter,self.rhov.vanalbada_limiter,self.rhoE.vanalbada_limiter,self.P.vanalbada_limiter, self.rho.vanalbada_halo_limiter,self.rhou.vanalbada_halo_limiter,self.rhov.vanalbada_halo_limiter,self.rhoE.vanalbada_halo_limiter
                                    ,self.P.vanalbada_halo_limiter,self.rhoY1.vanalbada_limiter,self.rhoY2.vanalbada_limiter,self.rhoY1.vanalbada_halo_limiter,self.rhoY2.vanalbada_halo_limiter)
    def explicit_dissipative(self):
    
        self.rho.compute_face_gradient()
        self.rhou.compute_face_gradient()
        self.rhov.compute_face_gradient()
        self.rhoE.compute_face_gradient()
        self.P.compute_face_gradient()
        self.Y1.compute_face_gradient()
        self.Y2.compute_face_gradient()

        explicitscheme_dissipative_NS(self.rho.cell, self.rhou.cell, self.rhov.cell,self.rhoE.cell, self.P.cell, self.T.cell,self.Y1.cell, self.Y2.cell,
                                    self.rho.ghost, self.rhou.ghost, self.rhov.ghost, self.rhoE.ghost, self.P.ghost,self.T.ghost, self.Y1.ghost, self.Y1.ghost, 
                                    self.rho.halo, self.rhou.halo, self.rhov.halo,self.rhoE.halo,self.P.halo,self.T.halo,self.Y1.halo,self.Y2.halo,
                                    self.rho.gradfacex,self.rho.gradfacey,self.rhou.gradfacex,self.rhou.gradfacey,
                                    self.rhov.gradfacex,self.rhov.gradfacey, self.rhoE.gradfacex,self.rhoE.gradfacey,self.P.gradfacex,self.P.gradfacey,self.T.gradfacex,self.T.gradfacey,
                                    self.Y1.gradfacex,self.Y1.gradfacey,self.Y2.gradfacex,self.Y2.gradfacey,
                                    self.rhou.dissipation,self.rhov.dissipation ,self.rhoE.dissipation, self.rhoY1.dissipation, self.rhoY2.dissipation, 
                                    self.domain.innerfaces, self.domain.halofaces, self.domain.boundaryfaces,
                                    self.domain.faces.normal,self.domain.faces.mesure,self.domain.faces.cellid,self.domain.faces.halofid,
                                    self.T_ref,self.mu_ref,self.C_zero,self.Cp,self.Pr,self.R,self.gamma, self.domain.faces.name, self.Sc1, self.Sc2)
    
        
                                    
                                        
    
    def stepper(self):
        ######calculation of the time step

        dt_c = self._time_step_euler(self.rho.cell, self.rhou.cell, self.rhov.cell,self.T.cell,self.P.cell, self.cfl, self.domain.faces.normal, self.domain.faces.mesure, 
                                  self.domain.cells.volume, self.domain.cells.faceid, self.gamma,self.Pr,self.Sc1,self.Sc1, self.T_ref, self.mu_ref, self.C_zero)
        
        self.dt = self.comm.allreduce(dt_c, MPI.MIN)
        return  self.dt
       
       
    def update_halo_values(self):
        requests = []
        for var in self.varbs.values():
            define_halosend(var.cell, var.halotosend, var.domain.halos.indsend)
            req = Iall_to_all(var.halotosend, var.nbhalos, var.domain.halos.scount, var.domain.halos.rcount, var.halo, 
                              var.comm)
            requests.append(req)
        MPI.Request.Waitall( requests )


    def update_ghost_values(self):
        for var in self.varbs.values():
            var.update_ghost_value()
    def update_term_source(self):
        term_source(self.rho.source, self.rhou.source, self.rhov.source,self.rhoE.source,self.rhoY1.source,self.rhoY2.source, self.T.cell,self.rhoY1.cell,self.rhoY2.cell, self.B, self.Ta, self.alpha, self.deltaQ, self.M1, self.M2,self.mu1, self.mu2, self.nu1, self.nu2)

    def compute_new_val(self):
        # update of the variables
        self._update_new_value(self.rho.cell, self.rhou.cell, self.rhov.cell, self.rhoE.cell,self.e_internal.cell,self.P.cell ,self.T.cell,self.rhoY1.cell ,self.rhoY2.cell ,self.Y1.cell ,self.Y2.cell ,
                               self.rho.convective, self.rhou.convective, self.rhov.convective, self.rhoE.convective, self.rhoY1.convective, self.rhoY2.convective,self.rhou.dissipation,
                               self.rhov.dissipation ,self.rhoE.dissipation, self.rhoY1.dissipation, self.rhoY2.dissipation, 
                                self.dt, self.domain.cells.volume, self.gamma,self.rho.source,self.rhou.source,self.rhov.source,self.rhoE.source, self.rhoY1.source,self.rhoY2.source, self.R)
    


    def compute_fluxes(self):
        
        #update halos
        self.update_halo_values()
        
        #update boundary conditions
        self.update_ghost_values()
        
        #convective flux
        self.explicit_convective()

        self.update_term_source()
        #  #dissipative flux
        # if self.diffusion: 
        # self.var.interpolate_celltonode()
        self.explicit_dissipative()
    
        
