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
from manapy.solvers.Euler.tools_utils import initialisation_RS,ghost_value_RS,haloghost_value_RS
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
filename="RS.msh"
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
tfinal = 0.13
miter = 0
niter = 1
saving_at_node = 1
order=1
CFL=0.7
#initialisation 
rl=2
rr=1
ul=0.0
ur=0.0
pl=15
pr=1



b1 = {"in" : "neumann",
              "out" : "neumann",
              "upper":"neumann",
              "bottom":"neumann"}

rho   = Variable(domain=domain)
rhou  = Variable(domain=domain)
rhov  = Variable(domain=domain)
P     = Variable(domain=domain)
rhoE  = Variable(domain=domain)
e_internal=Variable(domain=domain)



initialisation_RS(rho.cell, rhou.cell, rhov.cell, rhoE.cell,e_internal.cell, P.cell, cells.center, gamma=1.4)


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
    
    haloghost_value_RS(rho.halo,rhou.halo, rhov.halo, rhoE.halo, P.halo, 
                            rho.haloghost, rhou.haloghost,rhov.haloghost,rhoE.haloghost,P.haloghost
                            ,domain.nodes.haloghostcenter, domain.halonodes, domain.nodes.haloghostfaceinfo,cells.center, time)
    
    
    ghost_value_RS(rho.ghost,P.ghost,rhou.ghost,rhov.ghost,
                    rhoE.ghost, rho.cell,P.cell,rhou.cell,rhov.cell,rhoE.cell,
                     domain.faces.cellid, faces_in,faces_out,faces_upper,faces_bottom)
     
    S.explicit_convective()
    S.compute_new_val()
   


    
    if niter== 1 or niter%tot == 0  or time >= tfinal:
        
        if saving_at_node:
            rho.interpolate_celltonode()
            rhou.interpolate_celltonode()
            rhov.interpolate_celltonode()
            P.interpolate_celltonode()
            rhoE.interpolate_celltonode()
            e_internal.interpolate_celltonode()

            domain.save_on_node_multi(d_t, time, niter, miter, variables=["rho", "u","v", "E","e_internal","p"],values=[rho.node, rhou.node/rho.node, rhov.node/rho.node, rhoE.node/rho.node,e_internal.node, P.node])
            
        else:
            domain.save_on_cell_multi(d_t, time, niter, miter, variables=["rho", "u","v", "E","e_internal","p"],values=[rho.cell, rhou.cell/rho.cell, rhov.cell/rho.cell, rhoE.cell/rho.cell,e_internal.cell, P.cell])
        
    miter += 1
  
    niter += 1

te = MPI.Wtime()

tt = COMM.reduce(te-ts, op=MPI.MAX, root=0)
if RANK == 0:
    print("Time to do calculation", tt)
    from matplotlib.tri import Triangulation
    import matplotlib.colors as mcolors
    

    def plot_isovaleurs(Orthocentre, values, title, filename, cmap="rainbow"):
        """
        Fonction pour tracer les isovaleurs des variables sur un maillage non structuré.

        Parameters:
        Orthocentre : np.array
            Tableau des coordonnées des centres des cellules (x, y).
        values : np.array
            Valeurs scalaires associées aux cellules (ex: rho ou P).
        title : str
            Titre du graphique.
        filename : str
            Nom du fichier pour sauvegarder l'image.
        cmap : str
            Colormap à utiliser pour l'affichage.
        """
        triang = Triangulation(Orthocentre[:, 0], Orthocentre[:, 1])
    
        norm = mcolors.Normalize(vmin=np.min(values), vmax=np.max(values))

        plt.figure(figsize=(10, 6))
        contour = plt.tricontourf(triang, values, levels=50, cmap=cmap, norm=norm)  # 50 niveaux pour plus de détails
        plt.colorbar(contour)
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(filename, dpi=300)
        plt.show()

 
    Orthocentre = np.array(cells.center[:, :2])  
    rho_vals = np.array(rho.cell) 
    P_vals = np.array(P.cell) 

    # Utilisation d'une meilleure colormap
    plot_isovaleurs(Orthocentre, rho_vals, "Isovalues of Density (rho)", "rho_isovaleurs.png", cmap="nipy_spectral")
    plot_isovaleurs(Orthocentre, P_vals, "Isovalues of Pressure (P)", "p_isovaleurs.png", cmap="nipy_spectral")


