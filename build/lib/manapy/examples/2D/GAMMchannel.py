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
from manapy.solvers.Euler.tools_utils import GAMM_channel, ghost_value_gamm, MN,haloghost_value_Gamm2d,residual_L2
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
filename="Gamm2D_2.msh"
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
if RANK == 0: 
    print("Start Computation ...")
    Error_rho=[]
    Error_u=[]
    Error_v=[]
    Error_E=[]

time = 0
tfinal =1.1
miter = 0
niter = 1
saving_at_node = 1
order=1
CFL=0.5
gamma=1.4
#initialisation 




rho   = Variable(domain=domain)
rhou  = Variable(domain=domain)
rhov  = Variable(domain=domain)
P     = Variable(domain=domain)
rhoE  = Variable(domain=domain)
e_internal=Variable(domain=domain)


GAMM_channel(rho.cell, rhou.cell, rhov.cell, rhoE.cell,e_internal.cell, P.cell, cells.center, gamma)

domain.save_on_cell_multi(0, time, niter, miter, variables=["rho", "u","v", "E","e_internal","p"],values=[rho.cell, rhou.cell/rho.cell, rhov.cell/rho.cell, rhoE.cell/rho.cell,e_internal.cell, P.cell])
        
#Call the transport solver
conf = Struct(order=order, cfl= CFL)
S = EulerSolver(rho=rho, rhovel=(rhou, rhov), rhoE=rhoE,e_internal=e_internal, P=P, conf=conf, RANK=RANK, COMM=COMM )


ts = MPI.Wtime()

if RANK == 0: print("Start While loop ...")

rho1=np.zeros(len(rho.cell))
u1=np.zeros(len(rhou.cell/rho.cell))
v1=np.zeros(len(rhov.cell/rho.cell))
E1=np.zeros(len(rhoE.cell/rho.cell))
iterations=[]

#loop over time
while time <= tfinal:
    d_t = S.stepper()
    tot = int(tfinal/d_t/50)+1
    time = time + d_t
    rho1=rho1 = np.copy(rho.cell)
    u1=np.copy(rhou.cell/rho.cell)
    v1=np.copy(rhov.cell/rho.cell)
    E1=np.copy(rhoE.cell/rho.cell)



    S.update_halo_values()
    ghost_value_gamm(rho.ghost,P.ghost,rhou.ghost,rhov.ghost,
                    rhoE.ghost, rho.cell,P.cell,rhou.cell,rhov.cell,rhoE.cell,
                     domain.faces.cellid,domain.faces.normal,domain.faces.mesure,faces_in,faces_out,faces_upper,faces_bottom)
     
       
    haloghost_value_Gamm2d(rho.halo,rhou.halo, rhov.halo, rhoE.halo, P.halo, 
                            rho.haloghost, rhou.haloghost,rhov.haloghost,rhoE.haloghost,P.haloghost
                            ,domain.nodes.haloghostcenter, domain.halonodes, domain.nodes.haloghostfaceinfo)
  
    S.explicit_convective()
    S.compute_new_val()
    
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
            MN_n=np.sqrt((rhou.node*rhou.node + rhov.node*rhov.node)/(rho.node*rho.node))/np.sqrt(gamma*P.node/rho.node)
            MN_c=np.sqrt((rhou.cell*rhou.cell + rhov.cell*rhov.cell)/(rho.cell*rho.cell))/np.sqrt(gamma*P.cell/rho.cell)

            domain.save_on_node_multi(d_t, time, niter, miter, variables=["rho", "u","v", "E","e_internal","p", "MN"],values=[rho.node, rhou.node/rho.node, rhov.node/rho.node, rhoE.node/rho.node,e_internal.node, P.node, MN_n])
            
        else:
            domain.save_on_cell_multi(d_t, time, niter, miter, variables=["rho", "u","v", "E","e_internal","p", "MN"],values=[rho.cell, rhou.cell/rho.cell, rhov.cell/rho.cell, rhoE.cell/rho.cell,e_internal.cell, P.cell, MN_c])
        

    senddbuf_rho2 = np.array(rho.cell)
    senddbuf_rho1 = np.array(rho1)
    senddbuf_u2 = np.array(rhou.cell/rho.cell)
    senddbuf_u1 = np.array(u1)
    senddbuf_v2 = np.array(rhov.cell/rho.cell)
    senddbuf_v1 = np.array(v1)
    senddbuf_E2 = np.array(rhoE.cell/rho.cell)
    senddbuf_E1 = np.array(E1)
    iterations.append(niter)

    miter += 1
    niter += 1



  

    # Récupérer les tailles des buffers depuis chaque processus
    sendcounts_rho1 = np.array(COMM.gather(len(senddbuf_rho1), root=0))
    sendcounts_rho2 = np.array(COMM.gather(len(senddbuf_rho2), root=0))
    sendcounts_u1 = np.array(COMM.gather(len(senddbuf_u1), root=0))
    sendcounts_u2 = np.array(COMM.gather(len(senddbuf_u2), root=0))
    sendcounts_v2 = np.array(COMM.gather(len(senddbuf_v2), root=0))
    sendcounts_v1 = np.array(COMM.gather(len(senddbuf_v1), root=0))
    sendcounts_E1 = np.array(COMM.gather(len(senddbuf_E1), root=0))
    sendcounts_E2 = np.array(COMM.gather(len(senddbuf_E2), root=0))
  
    #COMM.Barrier()
    if RANK == 0:
        total_size_rho1 = sum(sendcounts_rho1)
        total_size_rho2 = sum(sendcounts_rho2)
        total_size_u1 = sum(sendcounts_u1)
        total_size_u2 = sum(sendcounts_u2)
        total_size_v1 = sum(sendcounts_v1)
        total_size_v2 = sum(sendcounts_v2)
        total_size_E1 = sum(sendcounts_E1)
        total_size_E2 = sum(sendcounts_E2)
     

 
        recvbuf_rho1 = np.empty(total_size_rho1, dtype=float)
        recvbuf_rho2 = np.empty(total_size_rho2, dtype=float)
        recvbuf_u1 = np.empty(total_size_u1, dtype=float)
        recvbuf_u2 = np.empty(total_size_u2, dtype=float)
        recvbuf_v1 = np.empty(total_size_v1, dtype=float)
        recvbuf_v2 = np.empty(total_size_v2, dtype=float)
        recvbuf_E1 = np.empty(total_size_E1, dtype=float)
        recvbuf_E2 = np.empty(total_size_E2, dtype=float)

    else:
        recvbuf_rho1 = None
        recvbuf_rho2= None
        recvbuf_u1 = None
        recvbuf_u2= None
        recvbuf_v1 = None
        recvbuf_v2= None 
        recvbuf_E1 = None
        recvbuf_E2= None

    COMM.Gatherv(sendbuf=senddbuf_rho1, recvbuf=(recvbuf_rho1,sendcounts_rho1), root=0)
    COMM.Gatherv(sendbuf=senddbuf_rho2, recvbuf=(recvbuf_rho2,sendcounts_rho2), root=0)

    COMM.Gatherv(sendbuf=senddbuf_u1, recvbuf=(recvbuf_u1,sendcounts_u1), root=0)
    COMM.Gatherv(sendbuf=senddbuf_u2, recvbuf=(recvbuf_u2,sendcounts_u2), root=0)
   
    COMM.Gatherv(sendbuf=senddbuf_v1, recvbuf=(recvbuf_v1,sendcounts_v1), root=0)
    COMM.Gatherv(sendbuf=senddbuf_v2, recvbuf=(recvbuf_v2,sendcounts_v2), root=0)
   
    COMM.Gatherv(sendbuf=senddbuf_E1, recvbuf=(recvbuf_E1,sendcounts_E1), root=0)
    COMM.Gatherv(sendbuf=senddbuf_E2, recvbuf=(recvbuf_E2,sendcounts_E2), root=0)
   
   

    if RANK==0:

        Error_rho.append(np.log(residual_L2(recvbuf_rho1,recvbuf_rho2, time)))
        Error_u.append(np.log(residual_L2(recvbuf_u1,recvbuf_u2, time)))
        Error_v.append(np.log(residual_L2(recvbuf_v1,recvbuf_v2, time)))
        Error_E.append(np.log(residual_L2(recvbuf_E1,recvbuf_E2, time)))


te = MPI.Wtime()

tt = COMM.reduce(te-ts, op=MPI.MAX, root=0)
if RANK == 0:
    print("Time to do calculation", tt)
    plt.figure(figsize=(10, 6))

    plt.plot(iterations, Error_rho, 'r-', label=r'$\rho$')  
    plt.plot(iterations, Error_u, 'b-', label=r'$u$')  
    plt.plot(iterations, Error_v, 'g-', label=r'$v$')  
    plt.plot(iterations, Error_E, 'k-', label=r'$E$')  

    plt.xlim(left=0)
    plt.ylim(min(Error_rho), max(Error_rho))  # Échelle normale en Y
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.minorticks_on()

    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Log(Error)", fontsize=14)  # Supprime log(Error)
    plt.title("Residuals", fontsize=16)

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(fontsize=12)

    plt.savefig("residuals_plot.png", dpi=300)
    plt.show()






senddbuf_rho = np.array(rho.cell)
senddbuf_P = np.array(P.cell)
senddbuf_E=np.array(rhoE.cell/rho.cell)
senddbuf_MN = np.array(np.sqrt((rhou.cell**2 + rhov.cell**2) / rho.cell**2) / np.sqrt(gamma * P.cell / rho.cell)) 
senddbuf_x = np.array(cells.center[:, 0])
senddbuf_y = np.array(cells.center[:, 1])

# Récupérer les tailles des buffers depuis chaque processus
sendcounts_rho = np.array(COMM.gather(len(senddbuf_rho), root=0))
sendcounts_P = np.array(COMM.gather(len(senddbuf_P), root=0))
sendcounts_E = np.array(COMM.gather(len(senddbuf_E), root=0))
sendcounts_MN = np.array(COMM.gather(len(senddbuf_MN), root=0))
sendcounts_x = np.array(COMM.gather(len(senddbuf_x), root=0))
sendcounts_y = np.array(COMM.gather(len(senddbuf_y), root=0))
#COMM.Barrier()
if RANK == 0:
    total_size_rho = sum(sendcounts_rho)
    total_size_P = sum(sendcounts_P)
    total_size_E = sum(sendcounts_E)
    total_size_MN = sum(sendcounts_MN)
    total_size_x = sum(sendcounts_x)
    total_size_y = sum(sendcounts_y)

    # recvbuf_rho = np.empty(total_size_rho, dtype=float)
    # recvbuf_x = np.empty(total_size_x, dtype=float)
    # recvbuf_y = np.empty(total_size_y, dtype=float)
    recvbuf_rho = np.empty(total_size_rho, dtype=float)
    recvbuf_P = np.empty(total_size_P, dtype=float)
    recvbuf_E = np.empty(total_size_E, dtype=float)
    recvbuf_MN = np.empty(total_size_MN, dtype=float)
    recvbuf_x = np.empty(total_size_x, dtype=float)
    recvbuf_y = np.empty(total_size_y, dtype=float)

else:
    recvbuf_rho = None
    recvbuf_P = None
    recvbuf_E = None
    recvbuf_MN = None
    recvbuf_x = None
    recvbuf_y = None
COMM.Gatherv(sendbuf=senddbuf_rho, recvbuf=(recvbuf_rho,sendcounts_rho), root=0)
COMM.Gatherv(sendbuf=senddbuf_P, recvbuf=(recvbuf_P,sendcounts_P), root=0)
COMM.Gatherv(sendbuf=senddbuf_E, recvbuf=(recvbuf_E,sendcounts_E), root=0)
COMM.Gatherv(sendbuf=senddbuf_MN, recvbuf=(recvbuf_MN,sendcounts_MN), root=0)
COMM.Gatherv(sendbuf=senddbuf_x, recvbuf=(recvbuf_x,sendcounts_x), root=0)
COMM.Gatherv(sendbuf=senddbuf_y, recvbuf=(recvbuf_y,sendcounts_y), root=0)


if RANK == 0:
    from matplotlib.tri import Triangulation
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable  

    def plot_isovaleurs(x, y, values, title, filename, cmap="turbo", label="Variable"):
        triang = Triangulation(x, y)
        
        fig, ax = plt.subplots(figsize=(12, 5))  # Création d'une figure et d'un axe
        contourf = ax.tricontourf(triang, values, levels=50, cmap=cmap)  # Isovaleurs
        contour = ax.tricontour(triang, values, levels=50, colors='k', linewidths=0.5)  # Lignes de contour
        
        # 🔹 Utilisation de `make_axes_locatable` pour forcer la colorbar en bas
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.4)  # Ajuste la taille et l'écart de la colorbar
        
        cbar = plt.colorbar(contourf, cax=cax, orientation="horizontal")  
        cbar.set_label(label)  # Label dynamique

        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.set_xlim(0, 3)  
        ax.set_ylim(0, 1)  
        ax.set_aspect(1)  

        plt.savefig(filename, dpi=300)
        plt.show()

    # Extraction des données après Gatherv
    Orthocentre_x = recvbuf_x  
    Orthocentre_y = recvbuf_y  
    rho_vals = recvbuf_rho  
    P_vals = recvbuf_P  
    E_vals = recvbuf_E
    MN_vals = recvbuf_MN 

    # Tracer rho et P avec la bonne colorbar en bas
    plot_isovaleurs(Orthocentre_x, Orthocentre_y, rho_vals, "Isovalues of Density (rho)", "rho_isovaleurs.png", cmap="turbo", label="Density")
    plot_isovaleurs(Orthocentre_x, Orthocentre_y, P_vals, "Isovalues of Pressure (P)", "p_isovaleurs.png", cmap="turbo", label="Pressure")
    plot_isovaleurs(Orthocentre_x, Orthocentre_y, E_vals, "Isovalues of Energy (E)", "E_isovaleurs.png", cmap="turbo", label="Energy")
    plot_isovaleurs(Orthocentre_x, Orthocentre_y, MN_vals, "Isovalues of Mach number (M)", "MN_isovaleurs.png", cmap="turbo", label="Mach number")
