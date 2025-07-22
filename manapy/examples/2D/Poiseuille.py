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

from manapy.solvers.NavierStokes import EulerSolver
from manapy.solvers.NavierStokes.tools_utils import haloghost_value_poiseuille,ghost_value_poiseuille,initialisation_poiseuille,exact_poiseuille,residual,norm_L2


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
filename="rectangle_e3.msh"
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
tfinal = 2
#3.5
miter = 0
niter = 1
saving_at_node = 1
order=2
CFL=0.5
#paramétres de probléme
height=1
L=1*height
Re=1
gamma=1.4
Ma=0.2
V_in=1
rho_in=1
nu=V_in*height/Re
mu=V_in*nu
C=V_in/Ma
P_in=C**2*rho_in/gamma 


convergane=[]
converganeP=[]
Error_rho=[]
Error_u=[]
T=[]
iterations=[]

rho   = Variable(domain=domain)
rhou  = Variable(domain=domain)
rhov  = Variable(domain=domain)
P     = Variable(domain=domain)
rhoE  = Variable(domain=domain)
e_internal=Variable(domain=domain)

u_exact=Variable(domain=domain)
P_exact=Variable(domain=domain)

initialisation_poiseuille(rho.cell, rhou.cell, rhov.cell, rhoE.cell,e_internal.cell, P.cell, cells.center, gamma=1.4, V_in=V_in,P_in=P_in, rho_in=rho_in)

#Call the transport solver
conf = Struct(order=order, Pr=0.72,cfl=CFL, mu=mu)
S = EulerSolver(rho=rho, rhovel=(rhou, rhov), rhoE=rhoE,e_internal=e_internal, P=P, conf=conf,RANK=RANK, COMM=COMM ,viscosity=1 )
domain.save_on_cell_multi(0, 0, 0, 0, variables=["rho", "u","v", "E","e_internal","p"],values=[rho.cell, rhou.cell/rho.cell, rhov.cell/rho.cell, rhoE.cell/rho.cell,e_internal.cell, P.cell])

rho.interpolate_celltonode()
rhou.interpolate_celltonode()
rhov.interpolate_celltonode()
P.interpolate_celltonode()
rhoE.interpolate_celltonode()
e_internal.interpolate_celltonode()
domain.save_on_node_multi(0, 0, 0, 0, variables=["rho", "u","v", "E","e_internal","p"],values=[rho.node, rhou.node/rho.node, rhov.node/rho.node, rhoE.node/rho.node,e_internal.node, P.node])

ts = MPI.Wtime()
if RANK == 0: print("Start While loop ...")

# #loop over time
# exact_poiseuille(cells.center, u_exact.cell,P_exact.cell, H, mu, GP)
# u_exact.update_halo_value()
# u_exact.update_ghost_value()


# senddbuf_exact = np.array(u_exact.cell)
# senddbuf_exact_P = np.array(P_exact.cell)

# sendcounts_exact= np.array(COMM.gather(len(senddbuf_exact), root=0))
# sendcounts_exact_P= np.array(COMM.gather(len(senddbuf_exact_P), root=0))
# #COMM.Barrier()
# if RANK == 0:
#     total_size_exact= sum(sendcounts_exact)
#     recvbuf_exact= np.empty(total_size_exact, dtype=float)
#     total_size_exact_P= sum(sendcounts_exact_P)
#     recvbuf_exact_P= np.empty(total_size_exact_P, dtype=float)
    
# else:
#     recvbuf_exact= None
#     recvbuf_exact_P= None

# COMM.Gatherv(sendbuf=senddbuf_exact, recvbuf=(recvbuf_exact,sendcounts_exact), root=0)
# COMM.Gatherv(sendbuf=senddbuf_exact_P, recvbuf=(recvbuf_exact_P,sendcounts_exact_P), root=0)
# Error=5
seul=3
rho1=np.zeros(len(rho.cell))
u1=np.zeros(len(rho.cell))
n_saves = 60
save_interval = tfinal / n_saves  
next_save_time = 0.0
test = 0  

while time<=tfinal:
# while Error>10**(-4):
# while seul > 3e-9:
    d_t = S.stepper()
    tot = int(tfinal/d_t/80)+1
    # d_t=min(d_t,tfinal-time)

    time = time + d_t
    rho1= np.copy(rho.cell)
    u1= np.copy(rhou.cell/rho.cell)

    S.update_halo_values()  
         
    ghost_value_poiseuille(rho.ghost,P.ghost,rhou.ghost,rhov.ghost,
                    rhoE.ghost, rho.cell,P.cell,rhou.cell,rhov.cell,rhoE.cell,
                     domain.faces.cellid,domain.faces.normal,domain.faces.mesure, faces_in,faces_out,faces_upper,faces_bottom,domain.faces.ghostcenter, V_in, P_in, rho_in,mu)
  
    
    haloghost_value_poiseuille(rho.halo,rhou.halo, rhov.halo, rhoE.halo, P.halo, 
                            rho.haloghost, rhou.haloghost,rhov.haloghost,rhoE.haloghost,P.haloghost
                            ,domain.nodes.haloghostcenter, domain.halonodes,domain.cells.haloghostcenter,V_in, P_in, rho_in,mu)

    S.explicit_convective()
    S.update_term_source()

    rho.interpolate_celltonode()
    rhou.interpolate_celltonode()
    rhov.interpolate_celltonode()
    rhoE.interpolate_celltonode()
    P.interpolate_celltonode()
    e_internal.interpolate_celltonode()

    S.explicit_dissipative()

    S.compute_new_val()


    if  niter==0 or  niter%tot==0 or time>=tfinal:
        if saving_at_node:

            rho.update_halo_value()
            # rho.update_ghost_value()
            # rho.interpolate_celltonode()

            rhou.update_halo_value()
            # rhou.update_ghost_value()
            # rhou.interpolate_celltonode()
            

            rhov.update_halo_value()
            # rhov.update_ghost_value()
            # rhov.interpolate_celltonode()
            
            P.update_halo_value()
            # P.update_ghost_value()
            # P.interpolate_celltonode()

            rhoE.update_halo_value()
            # rhoE.update_ghost_value()
            # rhoE.interpolate_celltonode()

            e_internal.update_halo_value()
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
        

      
    senddbuf_rho2 = np.array(rho.cell)
    senddbuf_rho1 = np.array(rho1)
    senddbuf_u2 = np.array(rhou.cell/rho.cell)
    senddbuf_u1 = np.array(u1)
    senddbuf_volume= np.array(domain.cells.volume)
    # senddbuf_u = np.array(rhou.cell/rho.cell)
    # senddbuf_P = np.array(P.cell)
    iterations.append(niter)
 
    
    if RANK==0:
        miter += 1
        niter += 1
        T.append(time)
       
    else:
        miter =None
        niter = None
       
    
    miter=COMM.bcast(miter,root=0)
    niter=COMM.bcast(niter,root=0)

        
    sendcounts_rho1 = np.array(COMM.gather(len(senddbuf_rho1), root=0))
    sendcounts_rho2 = np.array(COMM.gather(len(senddbuf_rho2), root=0))
    sendcounts_u1 = np.array(COMM.gather(len(senddbuf_u1), root=0))
    sendcounts_u2 = np.array(COMM.gather(len(senddbuf_u2), root=0))
    sendcounts_volume = np.array(COMM.gather(len(senddbuf_volume), root=0))
    # sendcounts_u= np.array(COMM.gather(len(senddbuf_u), root=0))
    # sendcounts_P= np.array(COMM.gather(len(senddbuf_P), root=0))
 
    #COMM.Barrier()
    if RANK == 0:
        total_size_rho1 = sum(sendcounts_rho1)
        total_size_rho2 = sum(sendcounts_rho2)

        total_size_u1 = sum(sendcounts_u1)
        total_size_u2 = sum(sendcounts_u2)
     
        total_size_volume= sum(sendcounts_volume)

        recvbuf_rho1 = np.empty(total_size_rho1, dtype=float)
        recvbuf_rho2 = np.empty(total_size_rho2, dtype=float)

        recvbuf_u1 = np.empty(total_size_u1, dtype=float)
        recvbuf_u2 = np.empty(total_size_u2, dtype=float)
       
        recvbuf_volume = np.empty(total_size_volume, dtype=float)
      
        # total_size_u= sum(sendcounts_u)
        # recvbuf_u = np.empty(total_size_u, dtype=float)
        # total_size_P= sum(sendcounts_P)
        # recvbuf_P = np.empty(total_size_P, dtype=float)
     
    else:
        recvbuf_rho1 = None
        recvbuf_rho2= None
        recvbuf_u1 = None
        recvbuf_u2= None
    
        recvbuf_volume= None
 
 
 
        # recvbuf_u= None
        # recvbuf_P= None

    # COMM.Gatherv(sendbuf=senddbuf_u, recvbuf=(recvbuf_u,sendcounts_u), root=0)
    # COMM.Gatherv(sendbuf=senddbuf_P, recvbuf=(recvbuf_P,sendcounts_P), root=0)

    COMM.Gatherv(sendbuf=senddbuf_rho1, recvbuf=(recvbuf_rho1,sendcounts_rho1), root=0)
    COMM.Gatherv(sendbuf=senddbuf_rho2, recvbuf=(recvbuf_rho2,sendcounts_rho2), root=0)
   
    COMM.Gatherv(sendbuf=senddbuf_u1, recvbuf=(recvbuf_u1,sendcounts_u1), root=0)
    COMM.Gatherv(sendbuf=senddbuf_u2, recvbuf=(recvbuf_u2,sendcounts_u2), root=0)
   
    COMM.Gatherv(sendbuf=senddbuf_volume, recvbuf=(recvbuf_volume,sendcounts_volume), root=0)
  
    if RANK==0:
        # print( "la forme de residu",residual(recvbuf_rho1,recvbuf_rho2, time))
        # send_Error=poiseuille_error(recvbuf_exact,recvbuf_u,u_max)
        # send_Error_P=poiseuille_error(recvbuf_exact_P,recvbuf_P,u_max)
        # Error1 = residual(recvbuf_rho1,recvbuf_rho2,  d_t)
        Error1=norm_L2(recvbuf_rho2,recvbuf_rho1,recvbuf_volume)
        
        Error2=norm_L2(recvbuf_u2,recvbuf_u1,recvbuf_volume)
        send_seul=Error2
        Error_rho.append(Error1)
        Error_u.append(Error2)
           
    # else:
    #     # send_Error=None
    #     # send_Error_P=None
    else:
        send_seul=None
    
    seul=COMM.bcast(send_seul,root=0)

    if RANK==0:
        print("seul", seul)
        print("time step", d_t)
        print("time", time)

    # Error=COMM.bcast(send_Error,root=0)
    # Error_P=COMM.bcast(send_Error_P,root=0)
    # if RANK==0:
    #     print("Error", Error)
    #     # print("time", time)
    #     convergane.append(Error)
    #     converganeP.append(Error_P)
        



te = MPI.Wtime()

tt = COMM.reduce(te-ts, op=MPI.MAX, root=0)
if RANK == 0:
    print("Time to do calculation", tt)
    plt.figure(figsize=(10, 6))

    # plt.plot(T,Error_rho, 'r-', label=r'$\rho$') 
    plt.plot(T, Error_u, 'b-', label=r'$u$')  
   

    plt.xlim(left=0)
    plt.yscale("log") 
     # Échelle normale en Y
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.minorticks_on()

    plt.xlabel("Time", fontsize=14)
    plt.ylabel("L2 Error ", fontsize=14)  
 

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(fontsize=12)

    plt.savefig("resi_time_hll.png", dpi=300)
    plt.show()

if RANK == 0:
    print("Time to do calculation", tt)
    plt.figure(figsize=(10, 6))

    # plt.plot(iterations, Error_rho, 'r-', label=r'$\rho$')  
    plt.plot(iterations, Error_u, 'b-', label=r'$u$')  
   

    plt.xlim(left=0)
    plt.yscale("log") 
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.minorticks_on()

    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("L2 Error", fontsize=14)  


    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(fontsize=12)

    plt.savefig("resi_iteration_hll.png", dpi=300)
    plt.show()



senddbuf_u = np.array(rhou.cell/rho.cell)
senddbuf_P= np.array(P.cell)
senddbuf_x = np.array(cells.center[:, 0])
senddbuf_y = np.array(cells.center[:, 1])
sendcounts_u = np.array(COMM.gather(len(senddbuf_u), root=0))
sendcounts_P = np.array(COMM.gather(len(senddbuf_P), root=0))
sendcounts_x = np.array(COMM.gather(len(senddbuf_x), root=0))
sendcounts_y = np.array(COMM.gather(len(senddbuf_y), root=0))

#COMM.Barrier()
if RANK == 0:
    total_size_u= sum(sendcounts_u)
    total_size_P= sum(sendcounts_P)
    total_size_x = sum(sendcounts_x)
    total_size_y = sum(sendcounts_y)

    # recvbuf_rho = np.empty(total_size_rho, dtype=float)
    # recvbuf_x = np.empty(total_size_x, dtype=float)
    # recvbuf_y = np.empty(total_size_y, dtype=float)
    recvbuf_u = np.empty(total_size_u, dtype=float)
    recvbuf_P = np.empty(total_size_P, dtype=float)
    recvbuf_x = np.empty(total_size_x, dtype=float)
    recvbuf_y = np.empty(total_size_y, dtype=float)

else:
    recvbuf_u = None
    recvbuf_P = None
    recvbuf_x = None
    recvbuf_y = None
COMM.Gatherv(sendbuf=senddbuf_u, recvbuf=(recvbuf_u,sendcounts_u), root=0)
COMM.Gatherv(sendbuf=senddbuf_P, recvbuf=(recvbuf_P,sendcounts_P), root=0)
COMM.Gatherv(sendbuf=senddbuf_x, recvbuf=(recvbuf_x,sendcounts_x), root=0)
COMM.Gatherv(sendbuf=senddbuf_y, recvbuf=(recvbuf_y,sendcounts_y), root=0)


if RANK == 0:
    from matplotlib.tri import Triangulation
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable  
    

    def plot_isovaleurs(x, y, values, title, filename, cmap="turbo", label="Variable"):
        triang = Triangulation(x, y)
        
        fig, ax = plt.subplots(figsize=(12, 5))  # Création d'une figure et d'un axe
        contourf = ax.tricontourf(triang, values, levels=50, cmap=cmap)  
        # contour = ax.tricontour(triang, values, levels=np.linspace(8.5, np.max(values),50), colors='k', linewidths=0.5)  # Lignes de contour
        # # contour = ax.tricontour(triang, values, levels=np.linspace(8.5, np.max(values), 50), cmap=cmap)
       
        niveaux_total = np.linspace(np.min(values), np.max(values), 50)


        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.4) 
        
        cbar = plt.colorbar(contourf, cax=cax, orientation="horizontal")  
        cbar.set_label(label)  

        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.set_xlim(0, 1)  
        ax.set_ylim(0, 1)  
        ax.set_aspect(1)  

        plt.savefig(filename, dpi=300)
        plt.show()

 
    Orthocentre_x = recvbuf_x  
    Orthocentre_y = recvbuf_y  
    u_vals = recvbuf_u  
    P_vals = recvbuf_P


    # Tracer rho et P avec la bonne colorbar en bas
    plot_isovaleurs(Orthocentre_x, Orthocentre_y, u_vals, "u", "isovaleur_u.png", cmap="turbo", label="Velocity")
    plot_isovaleurs(Orthocentre_x, Orthocentre_y, P_vals, "P", "isovaleur_P.png", cmap="turbo", label="P")
   




'''
u_exact.interpolate_celltonode() 
P_exact.interpolate_celltonode()   
domain.save_on_node_multi(0, tfinal, niter, miter, variables=[ "u_Exact","P_Exact"],values=[u_exact.node,P_exact.node])
'''
