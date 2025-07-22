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
from manapy.solvers.Euler.tools_utils import initialisation_tubeChok2, initialisation_tubeChok2exac, Exact_Euler, step, compute_p,ghost_value_test,haloghost_value_test
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
filename="rectangle.msh"
#filename="rectangle.msh"
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
tfinal = 0.06
miter = 0
niter = 1
saving_at_node = 1
order=2
CFL=0.6
#initialisation 
rl=1
rr=0.125
ul=0.0
ur=0.0
pl=10
pr=1.0
xmin=0
xmax=1
x_0=0.5

rho   = Variable(domain=domain)
rhou  = Variable(domain=domain)
rhov  = Variable(domain=domain)
P     = Variable(domain=domain)
rhoE  = Variable(domain=domain)
e_internal=Variable(domain=domain)

initialisation_tubeChok2(rho.cell, rhou.cell, rhov.cell, rhoE.cell,e_internal.cell, P.cell, cells.center, gamma=1.4)

##### exact variables#####""


rho_exact  = Variable(domain=domain)
u_exact= Variable(domain=domain)
P_exact = Variable(domain=domain)
E_exact = Variable(domain=domain)
e_internal_exact=Variable(domain=domain)
initialisation_tubeChok2exac(rho_exact.cell, u_exact.cell, E_exact.cell,e_internal_exact.cell, P_exact.cell, cells.center, gamma=1.4)

#Call the transport solver
conf = Struct(order=order, cfl=CFL)
S = EulerSolver(rho=rho, rhovel=(rhou, rhov), rhoE=rhoE,e_internal=e_internal, P=P, conf=conf,RANK=RANK, COMM=COMM  )
domain.save_on_cell_multi(0, time, 0, miter, variables=["rho", "u","v", "E","e_internal","p"],values=[rho.cell, rhou.cell/rho.cell, rhov.cell/rho.cell, rhoE.cell/rho.cell,e_internal.cell, P.cell])
#ghost_value_test2=domain.backend.compile( ghost_value_test, signature=domain.signature)

haloghost_value_test = domain.backend.compile(haloghost_value_test)#, signature=domain.signature)
ghost_value_test = domain.backend.compile(ghost_value_test)#, signature=domain.signature)


ts = MPI.Wtime()
#if RANK == 0: 
    
print("Start While loop ...", RANK, SIZE)

#loop over time
while time <= tfinal:
    d_t = S.stepper()
    tot = int(tfinal/d_t/50)+1

    time = time + d_t
    
    # S.update_halo_values()
    
    # haloghost_value_test(rho.halo,rhou.halo, rhov.halo, rhoE.halo, P.halo, 
    #                         rho.haloghost, rhou.haloghost,rhov.haloghost,rhoE.haloghost,P.haloghost
    #                         ,domain.nodes.haloghostcenter, domain.halonodes, domain.nodes.haloghostfaceinfo,cells.center, time, rl,rr,pl,pr)
    
    
    ghost_value_test(rho.ghost,P.ghost,rhou.ghost,rhov.ghost,
                    rhoE.ghost, rho.cell,P.cell,rhou.cell,rhov.cell,rhoE.cell,
                     domain.faces.cellid,domain.faces.normal,domain.faces.mesure, faces_in,faces_out,faces_upper,faces_bottom, rl, rr, pl, pr)
     
    S.explicit_convective()
    S.compute_new_val()
   
    
    if niter== 1 or niter%tot == 0  or time >= tfinal:
      
        if saving_at_node:

            # rho.update_halo_value()
            # rho.update_ghost_value()
            # rho.interpolate_celltonode()

            # rhou.update_halo_value()
            # rhou.update_ghost_value()
            # rhou.interpolate_celltonode()

            # rhov.update_halo_value()
            # rhov.update_ghost_value()
            # rhov.interpolate_celltonode()
            
            # P.update_halo_value()
            # P.update_ghost_value()
            # P.interpolate_celltonode()

            # rhoE.update_halo_value()
            # rhoE.update_ghost_value()
            # rhoE.interpolate_celltonode()

            # e_internal.update_halo_value()
            # e_internal.update_ghost_value()
            # e_internal.interpolate_celltonode()


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
    #import sys; sys.exit()

te = MPI.Wtime()

tt = COMM.reduce(te-ts, op=MPI.MAX, root=0)
if RANK == 0:
    print("Time to do calculation", tt)

Exact_Euler(rho_exact.cell, u_exact.cell, E_exact.cell,e_internal_exact.cell, P_exact.cell, cells.center,rl,pl, ul, rr, pr, ur, x_0, xmin, xmax, time, gamma=1.4)
rho_exact.update_halo_value()
rho_exact.update_ghost_value()  

#save vtk files for the solution
u_exact.update_halo_value()
u_exact.update_ghost_value()  


#save vtk files for the solution
P_exact.update_halo_value()
P_exact.update_ghost_value()  
E_exact.update_halo_value()
E_exact.update_ghost_value() 
e_internal_exact.update_halo_value()
e_internal_exact.update_ghost_value() 


'''
rho_exact.interpolate_celltonode()
u_exact.interpolate_celltonode()
P_exact.interpolate_celltonode()
E_exact.interpolate_celltonode()
e_internal_exact.interpolate_celltonode()
domain.save_on_node_multi(d_t, time, niter, miter, variables=["rho_exact", "u_exact", "E_exact","e_internal_exact","p_exact"], values=[rho_exact.node, u_exact.node,E_exact.node,e_internal_exact.node, P_exact.node])
'''
ord=1
Error_rho=rho.norml2(rho_exact.cell, order=ord)
Error_rhou=rhou.norml2(rho_exact.cell*u_exact.cell, order=ord)
Error_rhoE=rhoE.norml2(rho_exact.cell*E_exact.cell, order=ord)
Error_e_internal=e_internal.norml2(e_internal_exact.cell, order=ord)
Error_p=P.norml2(P_exact.cell, order=ord)
h=step(domain.cells.volume, nbcells )


# recvbuf_rho=COMM.gather(Error_rho, root=0)
# recvbuf_rhou=COMM.gather(Error_rhou, root=0)
# recvbuf_rhoE=COMM.gather(Error_rhoE, root=0)

# recvbuf_rho=COMM.gather(Error_rho, root=0)
# recvbuf_rhou=COMM.gather(Error_rhou, root=0)
# recvbuf_rhoE=COMM.gather(Error_rhoE, root=0)
# recvbuf_P=COMM.gather(Error_p, root=0)
# recvbuf_e=COMM.gather(Error_e_internal, root=0)

# recvbuf_P=COMM.gather(Error_p, root=0)
# recvbuf_e=COMM.gather(Error_e_internal, root=0)

if RANK==0:
    # Error_rho=sum(recvbuf_rho)
    # Error_rhou=sum(recvbuf_rhou)
    # Error_rhoE=sum(recvbuf_rhoE)
    # Error_p=sum(recvbuf_P)
    # Error_e_internal=sum(recvbuf_e)
    def save_results(nbcells, h, Error_rho, Error_rhou, Error_rhoE, Error_e_internal, Error_p, ordre, CPU_time, filename="results2.csv"):
        """
        Enregistre les résultats dans un fichier CSV en ajoutant une nouvelle ligne à chaque exécution.
        Inclut une colonne `h` pour le pas de discrétisation.
        """
        file_exists = os.path.exists(filename)

        # Création du dataframe avec les résultats
        new_data = pd.DataFrame([{
            "nbcells": nbcells,
            "h": h,  
            "Error_rho": Error_rho,
            "Error_rhou": Error_rhou,
            "Error_rhoE": Error_rhoE,
            "Error_e_internal": Error_e_internal,
            "Error_p": Error_p,
            "ordre": ordre,
            "CPU_time": CPU_time
        }])

        # Si le fichier existe, on ajoute une ligne, sinon on crée un nouveau fichier
        if file_exists:
            existing_data = pd.read_csv(filename)
            df = pd.concat([existing_data, new_data], ignore_index=True)
        else:
            df = new_data

        # Sauvegarde dans le fichier CSV
        df.to_csv(filename, index=False)
        print(f"Résultats ajoutés au fichier {filename}")


    csv_file = "results2.csv"
    ordre=None
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        if len(df) >= 1:
            last_row = df.iloc[-1]
            h1 = last_row["h"]
            error1 = last_row["Error_rho"]
            ordre = compute_p(h1, h, error1, Error_rho)

    # Enregistrement des résultats avec `h`
    save_results(nbcells, h, Error_rho, Error_rhou, Error_rhoE, Error_e_internal, Error_p, ordre, tt)



