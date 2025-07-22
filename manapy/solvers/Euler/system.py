from numpy import zeros
from mpi4py import MPI
import numpy as np

from manapy.solvers.Euler import (update_euler, time_step_euler, explicitscheme_convective_euler)


from manapy.comms import Iall_to_all
from manapy.comms import define_halosend

from manapy.ast import Variable
from manapy.base.base import Struct
from manapy.base.base import make_get_conf


class EulerSolver():
    
    _parameters = [('dt', float, 0., 0.,
                    'time step'),
                    ('order', int, 1, 1,
                     'order of the convective scheme'),
                    ('cfl', float, .5, .5,
                     'cfl of the explicit scheme'),
                     ('gamma', float, 1.4, 1.4,
                      'Adiabatic index')
    ]
    
    @classmethod
    def process_conf(cls, conf, kwargs):
        """                                                                                                                                                                                                
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
    
    def __init__(self, rho=None, rhovel=(None, None), rhoE=None,e_internal=None, P=None, conf=None, RANK=None, COMM=None,**kwargs):
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
        self.e_internal=e_internal
        
        self.varbs = {}
        self.varbs['rho'] = self.rho
        self.varbs['rhou'] = self.rhou
        self.varbs['rhov'] = self.rhov
        self.varbs['rhoE'] = self.rhoE
        self.varbs['P'] = self.P
        self.varbs['e_internal'] = self.e_internal
        
        terms = ["convective"]
        for var in self.varbs.values():
            for term in terms:
                var.__dict__[term] = zeros(self.domain.nbcells, dtype=self.float_precision)
        
        
        # Constants
        self.dt    = get("dt")
        self.order = get("order")
        self.cfl   = get("cfl")
        self.gamma  = get("gamma")
        
        
        self.backend = self.domain.backend
        self.signature = self.domain.signature
        
        self._explicitscheme_convective  = self.backend.compile(explicitscheme_convective_euler, signature=self.signature)
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
        # computing the flux
        # explicitscheme_convective_euler(self.rho.convective, self.rhou.convective, self.rhov.convective, self.rhoE.convective, 
        #                              self.rho.cell, self.rhou.cell, self.rhov.cell, self.rhoE.cell, self.P.cell,
        #                              self.rho.ghost, self.rhou.ghost, self.rhov.ghost, self.rhoE.ghost,self.P.ghost, self.rho.halo, 
        #                              self.rhou.halo, self.rhov.halo, self.rhoE.halo,self.P.halo,
        #                              self.rho.gradcellx, self.rho.gradcelly, self.rho.gradhalocellx, self.rho.gradhalocelly,
        #                              self.rhou.gradcellx, self.rhou.gradcelly, self.rhou.gradhalocellx, self.rhou.gradhalocelly,
        #                              self.rhov.gradcellx, self.rhov.gradcelly, self.rhov.gradhalocellx, self.rhov.gradhalocelly,
        #                              self.rhoE.gradcellx, self.rhoE.gradcelly, self.rhoE.gradhalocellx, self.rhoE.gradhalocelly,
        #                              self.P.gradcellx, self.P.gradcelly, self.P.gradhalocellx, self.P.gradhalocelly,self.rho.psi, 
        #                              self.rho.psihalo,self.rhou.psi, self.rhou.psihalo, self.rhov.psi, self.rhov.psihalo,  
        #                             self.rhoE.psi, self.rhoE.psihalo, self.P.psi, self.P.psihalo, self.domain.cells.center, 
        #                              self.domain.faces.center, self.domain.halos.centvol, self.domain.faces.ghostcenter,
        #                              self.domain.faces.cellid, self.domain.faces.mesure, self.domain.faces.normal,self.domain.faces.tangent, 
        #                              self.domain.faces.halofid,self.domain.innerfaces, self.domain.halofaces, self.domain.boundaryfaces, 
        #                              self.domain.cells.center,self.domain.faces.nodeid, self.domain.nodes.vertex, self.dt, self.gamma, 
        #                              self.order, self.scheme_to_exec, self.domain.faces.lf_volums, self.domain.cells.nodeid,self.domain.nodes.cellid,self.domain.cells.faceid,self.domain.faces.name, self.domain.cells.volume)
    

    
        explicitscheme_convective_euler(self.rho.convective, self.rhou.convective, self.rhov.convective, self.rhoE.convective, 
                                     self.rho.cell, self.rhou.cell, self.rhov.cell, self.rhoE.cell, self.P.cell,
                                     self.rho.ghost, self.rhou.ghost, self.rhov.ghost, self.rhoE.ghost,self.P.ghost, self.rho.halo, 
                                     self.rhou.halo, self.rhov.halo, self.rhoE.halo,self.P.halo,
                                     self.rho.gradcellx, self.rho.gradcelly, self.rho.gradhalocellx, self.rho.gradhalocelly,
                                     self.rhou.gradcellx, self.rhou.gradcelly, self.rhou.gradhalocellx, self.rhou.gradhalocelly,
                                     self.rhov.gradcellx, self.rhov.gradcelly, self.rhov.gradhalocellx, self.rhov.gradhalocelly,
                                     self.rhoE.gradcellx, self.rhoE.gradcelly, self.rhoE.gradhalocellx, self.rhoE.gradhalocelly,
                                     self.P.gradcellx, self.P.gradcelly, self.P.gradhalocellx, self.P.gradhalocelly,self.rho.psi,
                                     self.rho.psihalo,self.rhou.psi,self.rhou.psihalo,self.rhov.psi,self.rhov.psihalo,self.rhoE.psi,self.rhoE.psihalo,self.P.psi,self.P.psihalo,self.domain.cells.center, 
                                     self.domain.faces.center, self.domain.halos.centvol, self.domain.faces.ghostcenter,
                                     self.domain.faces.cellid, self.domain.faces.mesure, self.domain.faces.normal,self.domain.faces.tangent, 
                                     self.domain.faces.halofid,self.domain.innerfaces, self.domain.halofaces, self.domain.boundaryfaces, self.gamma, 
                                     self.order, self.scheme_to_exec, self.limiter_to_exec, self.rho.minmod_x
                                    ,self.rho.minmod_y,self.rho.minmod_halo_x,self.rho.minmod_halo_y,self.rhou.minmod_x
                                    ,self.rhou.minmod_y,self.rhou.minmod_halo_x,self.rhou.minmod_halo_y,self.rhov.minmod_x
                                    ,self.rhov.minmod_y,self.rhov.minmod_halo_x,self.rhov.minmod_halo_y,self.rhoE.minmod_x
                                    ,self.rhoE.minmod_y,self.rhoE.minmod_halo_x,self.rhoE.minmod_halo_y,self.P.minmod_x
                                    ,self.P.minmod_y,self.P.minmod_halo_x,self.P.minmod_halo_y,
                                    self.rho.vanalbada_limiter,self.rhou.vanalbada_limiter,self.rhov.vanalbada_limiter,self.rhoE.vanalbada_limiter,self.P.vanalbada_limiter, self.rho.vanalbada_halo_limiter,self.rhou.vanalbada_halo_limiter,self.rhov.vanalbada_halo_limiter,self.rhoE.vanalbada_halo_limiter,self.P.vanalbada_halo_limiter)
    
    def stepper(self):
        ######calculation of the time step
        dt_c = self._time_step_euler(self.rho.cell, self.rhou.cell, self.rhov.cell, self.rhoE.cell,self.P.cell, self.cfl, self.domain.faces.normal, self.domain.faces.mesure, 
                                  self.domain.cells.volume, self.domain.cells.faceid, self.gamma)
        
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


    def compute_new_val(self):
        # update of the variables
        self._update_new_value(self.rho.cell, self.rhou.cell, self.rhov.cell, self.rhoE.cell,self.e_internal.cell,self.P.cell ,
                               self.rho.convective, self.rhou.convective, self.rhov.convective, self.rhoE.convective,
                                self.dt, self.domain.cells.volume, self.gamma)
           

    def compute_fluxes(self):
        
        #update halos
        self.update_halo_values()
        
        #update boundary conditions
        self.update_ghost_values()
        
        #convective flux
        self.explicit_convective()
        
        
        

