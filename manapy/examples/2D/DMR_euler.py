
from mpi4py import MPI
import timeit
import numpy as np
import pandas as pd
import os
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

from manapy.partitions import MeshPartition
from manapy.ddm import Domain

from manapy.solvers.Euler import EulerSolver
from manapy.solvers.Euler.tools_utils import initialisation_DMR,ghost_value_slip,ghost_value_dirichlet, ghost_value_neumann,ghost_value_DoubleMach, MN,haloghost_value_DoubleMach
from manapy.ast import Variable
from manapy.base.base import Struct
import matplotlib.pyplot as plt
import os
start = timeit.default_timer()

#get the mesh directory

try:
    MESH_DIR = os.environ['MESH_DIR']
 
except:
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    BASE_DIR = os.path.join(BASE_DIR , '..', '..','..')
    MESH_DIR = os.path.join(BASE_DIR, 'mesh')
 
#MESH_DIR = "/home/asus/manapy/mesh"
#filename = "choc_4.msh"
filename="DMP.msh"
#filename="choc_fin.msh"
#File name
filename = os.path.join(MESH_DIR, filename)
dim = 2
#print(MESH_DIR)

###Config###
#backend numba or python
#signature: add types to functions (make them faster) but compilation take time
#cache: avoid recompilation in the next run
running_conf = Struct(backend="numba", signature=True, cache=True, float_precision="double")#, int_precision="signed")
mesh = MeshPartition(filename, dim=dim, conf=running_conf, periodic=[0,0,0])

#Create the informations about cells, faces and nodes
domain = Domain(dim=dim, conf=running_conf)
faces = domain.faces
cells = domain.cells
halos = domain.halos
nodes = domain.nodes

nbnodes = domain.nbnodes
nbfaces = domain.nbfaces
nbcells = domain.nbcells
faces_bottom = np.array(domain.get_bottom_faces())
faces_upper = np.array(domain.get_upper_faces())
faces_in = np.array(domain.get_in_faces())
faces_out =np.array(domain.get_out_faces())
end = timeit.default_timer()

tt = COMM.reduce(end -start, op=MPI.MAX, root=0)
if RANK == 0:
    print("Time to create the domain", tt)

#TODO tfinal
if RANK == 0: print("Start Computation ...")
time = 0
tfinal =0.2
miter = 0
niter = 1
saving_at_node = 0
order=2
CFL=0.05
# CFL=0.02
#initialisation 
rl=8.0
rr=1.4
pl=116.5
pr=1.0
ru_l=8.0*8.25*(np.sqrt(3)/2)
rv_l=-8.0*8.25*0.5
ru_r=0.0
rv_r=0.0

'''
boundaries1 = {"in" : "neumann",
              "out" : "neumann",
              "upper":"neumann",
              "bottom":"neumann"
              }

boundaries2 = {"in" : "dirichlet",
              "out" : "dirichlet",
              "upper":"neumann",
              "bottom":"neumann"
              }

values_rho = {"in" : rl, "out" :rr
          
          }
values_p={"in" : pl, "out" :pr
          
          }


rho   = Variable(domain=domain,BC=boundaries2,values=values_rho)
rhou  = Variable(domain=domain,BC=boundaries1)
rhov  = Variable(domain=domain,BC=boundaries1)
P     = Variable(domain=domain,BC=boundaries2,values=values_p)
rhoE  = Variable(domain=domain,BC=boundaries1)
e_internal=Variable(domain=domain,BC=boundaries1)
'''
rho   = Variable(domain=domain)
rhou  = Variable(domain=domain)
rhov  = Variable(domain=domain)
P     = Variable(domain=domain)
rhoE  = Variable(domain=domain)
e_internal=Variable(domain=domain)

initialisation_DMR(rho.cell, rhou.cell, rhov.cell, rhoE.cell,e_internal.cell,
                    P.cell, cells.center, gamma=1.4)



#Call the transport solver
conf = Struct(order=order, cfl= CFL)
S = EulerSolver(rho=rho, rhovel=(rhou, rhov), rhoE=rhoE,e_internal=e_internal, P=P, conf=conf,RANK=RANK, COMM=COMM  )
domain.save_on_cell_multi(0, time, niter, miter, variables=["rho", "u","v", "E","e_internal","p"],values=[rho.cell, rhou.cell/rho.cell, rhov.cell/rho.cell, rhoE.cell/rho.cell,e_internal.cell, P.cell])

ts = MPI.Wtime()

if RANK == 0: print("Start While loop ...")



#loop over time
while time <= tfinal:
    d_t = S.stepper()
    tot = int(tfinal/d_t/50)+1

    time = time + d_t

    print(time)

    S.update_halo_values()

    ghost_value_DoubleMach(rho.ghost,P.ghost,rhou.ghost,rhov.ghost,
                    rhoE.ghost, rho.cell,P.cell,rhou.cell,rhov.cell,rhoE.cell,
                     domain.faces.cellid,domain.faces.normal,domain.faces.mesure,
                     domain.faces.ghostcenter, time,faces_in,faces_out,faces_upper,faces_bottom)
     
    haloghost_value_DoubleMach(rho.halo,rhou.halo, rhov.halo, rhoE.halo, P.halo, 
                            rho.haloghost, rhou.haloghost,rhov.haloghost,rhoE.haloghost,P.haloghost
                            ,domain.nodes.haloghostcenter, domain.halonodes, domain.nodes.haloghostfaceinfo, domain.faces.ghostcenter, time)
  
    S.explicit_convective()
    S.compute_new_val()
   

    '''
    if niter== 1 or niter%tot == 0  or time >= tfinal:
        
        if saving_at_node:

            rho.update_halo_value()
            rhou.update_halo_value()
            rhov.update_halo_value()
            P.update_halo_value()
            rhoE.update_halo_value()
            e_internal.update_halo_value()

       
            rho.interpolate_celltonode()
            rhou.interpolate_celltonode()
            rhov.interpolate_celltonode()
            P.interpolate_celltonode()
            rhoE.interpolate_celltonode()
            e_internal.interpolate_celltonode()

        
            domain.save_on_node_multi(d_t, time, niter, miter,RANK, variables=["rho", "u","v", "E","e_internal","p"],values=[rho.node, rhou.node/rho.node, rhov.node/rho.node, rhoE.node/rho.node,e_internal.node, P.node])
            
        else:
            domain.save_on_cell_multi(d_t, time, niter, miter, RANK,variables=["rho", "u","v", "E","e_internal","p"],values=[rho.cell, rhou.cell/rho.cell, rhov.cell/rho.cell, rhoE.cell/rho.cell,e_internal.cell, P.cell])
        '''
    miter += 1
    niter += 1

te = MPI.Wtime()

tt = COMM.reduce(te-ts, op=MPI.MAX, root=0)

if RANK == 0:
    print("Time to do calculation", tt)

MN = MN(rho.cell, rhou.cell, rhov.cell, P.cell, gamma=1.4)




senddbuf_rho = np.array(rho.cell)
senddbuf_P = np.array(P.cell)
senddbuf_x = np.array(cells.center[:, 0])
senddbuf_y = np.array(cells.center[:, 1])

# Récupérer les tailles des buffers depuis chaque processus
sendcounts_rho = np.array(COMM.gather(len(senddbuf_rho), root=0))
sendcounts_P = np.array(COMM.gather(len(senddbuf_P), root=0))
sendcounts_x = np.array(COMM.gather(len(senddbuf_x), root=0))
sendcounts_y = np.array(COMM.gather(len(senddbuf_y), root=0))
#COMM.Barrier()
if RANK == 0:
    total_size_rho = sum(sendcounts_rho)
    total_size_P = sum(sendcounts_P)
    total_size_x = sum(sendcounts_x)
    total_size_y = sum(sendcounts_y)

    # recvbuf_rho = np.empty(total_size_rho, dtype=float)
    # recvbuf_x = np.empty(total_size_x, dtype=float)
    # recvbuf_y = np.empty(total_size_y, dtype=float)
    recvbuf_rho = np.empty(total_size_rho, dtype=float)
    recvbuf_P = np.empty(total_size_P, dtype=float)
    recvbuf_x = np.empty(total_size_x, dtype=float)
    recvbuf_y = np.empty(total_size_y, dtype=float)

else:
    recvbuf_rho = None
    recvbuf_P = None
    recvbuf_x = None
    recvbuf_y = None
COMM.Gatherv(sendbuf=senddbuf_rho, recvbuf=(recvbuf_rho,sendcounts_rho), root=0)
COMM.Gatherv(sendbuf=senddbuf_P, recvbuf=(recvbuf_P,sendcounts_P), root=0)
COMM.Gatherv(sendbuf=senddbuf_x, recvbuf=(recvbuf_x,sendcounts_x), root=0)
COMM.Gatherv(sendbuf=senddbuf_y, recvbuf=(recvbuf_y,sendcounts_y), root=0)


if RANK == 0:
    from matplotlib.tri import Triangulation
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable 

    def plot_isovaleurs(x, y, values, title, filename, cmap="turbo", label="Variable"):
        triang = Triangulation(x, y)
        
        fig, ax = plt.subplots(figsize=(12, 5)) 
        contourf = ax.tricontourf(triang, values, levels=50, cmap=cmap)  
        # contour = ax.tricontour(triang, values, levels=np.linspace(8.5, np.max(values),50), colors='k', linewidths=0.5)  # Lignes de contour
        # # contour = ax.tricontour(triang, values, levels=np.linspace(8.5, np.max(values), 50), cmap=cmap)
       
        niveaux_bas = np.linspace(np.min(values), 7, 20)
        niveaux_haut = np.linspace(8.4, np.max(values), 30)

        niveaux_total = np.concatenate((niveaux_bas, niveaux_haut))
        contour =  ax.tricontour(triang, values, levels=niveaux_total, colors='k', linewidths=0.5)

      
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.4) 
        cbar = plt.colorbar(contourf, cax=cax, orientation="horizontal")  
        cbar.set_label(label) 

        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.set_xlim(0, 3)  
        ax.set_ylim(0, 1)  
        ax.set_aspect(1)  

        plt.savefig(filename, dpi=300)
        plt.show()

    Orthocentre_x = recvbuf_x  
    Orthocentre_y = recvbuf_y  
    rho_vals = recvbuf_rho  
    P_vals = recvbuf_P  

    plot_isovaleurs(Orthocentre_x, Orthocentre_y, rho_vals, "Isovalues of Density (rho)", "rhorusanov_barth.png", cmap="turbo", label="Density")
    plot_isovaleurs(Orthocentre_x, Orthocentre_y, P_vals, "Isovalues of Pressure (P)", "p_srnh_rusanov.png", cmap="turbo", label="Pressure")
















































































































































































































































