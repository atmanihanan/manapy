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
from manapy.solvers.NavierStokes.tools_utils import initialisation_Cavity,residual,norm_L2,ghost_value_Cavity,haloghost_value_Cavity
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
filename="rectangle_e4.msh"
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
tfinal = 28
#1
miter = 0
niter = 1
saving_at_node = 1
order=2
CFL=0.4
#paramétres de probléme
height=1
L=1*height
Re=1000
gamma=1.4
Ma=0.2
V_in=1
rho_in=1
nu=V_in*height/Re
mu=rho_in*nu
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

initialisation_Cavity(rho.cell, rhou.cell, rhov.cell, rhoE.cell,e_internal.cell, P.cell, cells.center, gamma=1.4, V_in=V_in,P_in=P_in, rho_in=rho_in)

#Call the transport solver
conf = Struct(order=order, Pr=0.72,cfl=CFL, mu=mu)
S = EulerSolver(rho=rho, rhovel=(rhou, rhov), rhoE=rhoE,e_internal=e_internal, P=P, conf=conf,RANK=RANK, COMM=COMM ,viscosity=1 )
Vitesse_c=np.sqrt((rhou.cell/rho.cell)**2+(rhov.cell/rho.cell)**2)
domain.save_on_cell_multi(0, 0, 0, 0, variables=["rho", "u","v", "E","e_internal","p","Velocity"],values=[rho.cell, rhou.cell/rho.cell, rhov.cell/rho.cell, rhoE.cell/rho.cell,e_internal.cell, P.cell, Vitesse_c])

rho.interpolate_celltonode()
rhou.interpolate_celltonode()
rhov.interpolate_celltonode()
P.interpolate_celltonode()
rhoE.interpolate_celltonode()
e_internal.interpolate_celltonode()
Vitesse_n=np.sqrt((rhou.node/rho.node)**2+(rhov.node/rho.node)**2)
domain.save_on_node_multi(0, 0, 0, 0, variables=["rho", "u","v", "E","e_internal","p","Velocity"],values=[rho.node, rhou.node/rho.node, rhov.node/rho.node, rhoE.node/rho.node,e_internal.node, P.node, Vitesse_n])

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
n_saves = 280
save_interval = tfinal / n_saves  
next_save_time = 0.0
test = 0  

# while time<=tfinal:
# while Error>10**(-4):
while seul > 4.8e-6:
    d_t = S.stepper()
    tot = int(tfinal/d_t/50)+1
    # d_t=min(d_t,tfinal-time)

    time = time + d_t
    rho1= np.copy(rho.cell)
 
    uu= np.copy(rhou.cell/rho.cell)
    vv=np.copy(rhov.cell/rho.cell)
    u1=np.sqrt(uu**2+vv**2)

    S.update_halo_values()  
         
    ghost_value_Cavity(rho.ghost,P.ghost,rhou.ghost,rhov.ghost,
                    rhoE.ghost, rho.cell,P.cell,rhou.cell,rhov.cell,rhoE.cell,
                     domain.faces.cellid,domain.faces.normal,domain.faces.mesure, faces_in,faces_out,faces_upper,faces_bottom,domain.faces.ghostcenter, V_in, P_in, rho_in)
       
    haloghost_value_Cavity(rho.halo,rhou.halo, rhov.halo, rhoE.halo, P.halo, 
                            rho.haloghost, rhou.haloghost,rhov.haloghost,rhoE.haloghost,P.haloghost
                            ,domain.nodes.haloghostcenter, domain.halonodes,domain.cells.haloghostcenter,V_in, P_in, rho_in)

  
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

    # if seul<=1e-15:
    #     test=1
    # error=poiseuille_error(rhou.cell/rho.cell, u_exact.cell,u_max)
    # print('error',error)
    
    # if niter== 1 or niter%tot == 0  or time >= tfinal:
    # # if save_interval >= next_save_time or test == 1:
    #     next_save_time +=1

    #     if saving_at_node:

    #         rho.update_halo_value()
    #         # rho.update_ghost_value()
    #         # rho.interpolate_celltonode()

    #         rhou.update_halo_value()
    #         # rhou.update_ghost_value()
    #         # rhou.interpolate_celltonode()
            

    #         rhov.update_halo_value()
    #         # rhov.update_ghost_value()
    #         # rhov.interpolate_celltonode()
            
    #         P.update_halo_value()
    #         # P.update_ghost_value()
    #         # P.interpolate_celltonode()

    #         rhoE.update_halo_value()
    #         # rhoE.update_ghost_value()
    #         # rhoE.interpolate_celltonode()

    #         e_internal.update_halo_value()
    #         # e_internal.update_ghost_value()
    #         # e_internal.interpolate_celltonode()

        
    #         rho.interpolate_celltonode()
    #         rhou.interpolate_celltonode()
    #         rhov.interpolate_celltonode()
    #         P.interpolate_celltonode()
    #         rhoE.interpolate_celltonode()
    #         e_internal.interpolate_celltonode()

    #         Vitesse_n=np.sqrt((rhou.node/rho.node)**2+(rhov.node/rho.node)**2)
    #         Vitesse_c=np.sqrt((rhou.cell/rho.cell)**2+(rhov.cell/rho.cell)**2)
    #         domain.save_on_node_multi(d_t, time, niter, miter, variables=["rho", "u","v", "E","e_internal","p","Velocity"],values=[rho.node, rhou.node/rho.node, rhov.node/rho.node, rhoE.node/rho.node,e_internal.node, P.node, Vitesse_n])
            
    #     else:
    #         domain.save_on_cell_multi(d_t, time, niter, miter, variables=["rho", "u","v", "E","e_internal","p", "Velocity"],values=[rho.cell, rhou.cell/rho.cell, rhov.cell/rho.cell, rhoE.cell/rho.cell,e_internal.cell, P.cell,Vitesse_c])
        

      
    senddbuf_rho2 = np.array(rho.cell)
    senddbuf_rho1 = np.array(rho1)
    # senddbuf_u2 = np.array(rhou.cell/rho.cell)
    senddbuf_u2 =  senddbuf_u2 =np.array(np.sqrt((rhou.cell/rho.cell)**2+(rhov.cell/rho.cell)**2))
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
    # plt.ylim(min(Error_u), max(Error_u))
    plt.yscale("log")
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.minorticks_on()

    plt.xlabel("Time", fontsize=14)
    plt.ylabel("L2 Error ", fontsize=14)  # Supprime log(Error)
    # plt.title("Residuals", fontsize=16)

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(fontsize=12)

    plt.savefig("resi_time_cavity.png", dpi=300)
    plt.show()

if RANK == 0:
    print("Time to do calculation", tt)
    plt.figure(figsize=(10, 6))
    # plt.plot(iterations, Error_rho, 'r-', label=r'$\rho$')  
    plt.plot(iterations, Error_u, 'b-', label=r'$u$')    
    plt.xlim(left=0)
    plt.yscale("log") # Échelle normale en Y
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.minorticks_on()
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("L2 Error", fontsize=14)  # Supprime log(Error)
    # plt.title("Residuals", fontsize=16)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(fontsize=12)
    plt.savefig("resi_iteration_cavityl.png", dpi=300)
    plt.show()

senddbuf_u = np.array(np.sqrt((rhou.cell/rho.cell)**2+(rhov.cell/rho.cell)**2))
senddbuf_x = np.array(cells.center[:, 0])
senddbuf_y = np.array(cells.center[:, 1])
senddbuf_U= np.array(rhou.cell/rho.cell)
senddbuf_V = np.array(rhov.cell/rho.cell)

# Récupérer les tailles des buffers depuis chaque processus
sendcounts_u = np.array(COMM.gather(len(senddbuf_u), root=0))
sendcounts_U= np.array(COMM.gather(len(senddbuf_U), root=0))
sendcounts_V = np.array(COMM.gather(len(senddbuf_V), root=0))
sendcounts_x = np.array(COMM.gather(len(senddbuf_x), root=0))
sendcounts_y = np.array(COMM.gather(len(senddbuf_y), root=0))
#COMM.Barrier()
if RANK == 0:
    total_size_u= sum(sendcounts_u)
    total_size_U= sum(sendcounts_U)
    total_size_V= sum(sendcounts_V)
    total_size_x = sum(sendcounts_x)
    total_size_y = sum(sendcounts_y)

    # recvbuf_rho = np.empty(total_size_rho, dtype=float)
    # recvbuf_x = np.empty(total_size_x, dtype=float)
    # recvbuf_y = np.empty(total_size_y, dtype=float)
    recvbuf_u = np.empty(total_size_u, dtype=float)
    recvbuf_U = np.empty(total_size_U, dtype=float)
    recvbuf_V = np.empty(total_size_V, dtype=float)
    recvbuf_x = np.empty(total_size_x, dtype=float)
    recvbuf_y = np.empty(total_size_y, dtype=float)
else:
    recvbuf_u = None
    recvbuf_U = None
    recvbuf_V = None
    recvbuf_x = None
    recvbuf_y = None
COMM.Gatherv(sendbuf=senddbuf_u, recvbuf=(recvbuf_u,sendcounts_u), root=0)
COMM.Gatherv(sendbuf=senddbuf_U, recvbuf=(recvbuf_U,sendcounts_U), root=0)
COMM.Gatherv(sendbuf=senddbuf_V, recvbuf=(recvbuf_V,sendcounts_V), root=0)
COMM.Gatherv(sendbuf=senddbuf_x, recvbuf=(recvbuf_x,sendcounts_x), root=0)
COMM.Gatherv(sendbuf=senddbuf_y, recvbuf=(recvbuf_y,sendcounts_y), root=0)

if RANK == 0:
    from matplotlib.tri import Triangulation
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable  # Import pour positionner la colorbar

    def plot_isovaleurs(x, y, values, title, filename, cmap="turbo", label="Variable"):
        triang = Triangulation(x, y)
        
        fig, ax = plt.subplots(figsize=(6, 6))  # Création d'une figure et d'un axe
        contourf = ax.tricontourf(triang, values, levels=50, cmap=cmap)  
        # contour = ax.tricontour(triang, values, levels=np.linspace(8.5, np.max(values),50), colors='k', linewidths=0.5)  # Lignes de contour
        # # contour = ax.tricontour(triang, values, levels=np.linspace(8.5, np.max(values), 50), cmap=cmap)
       
       # Une seule plage régulière entre min et max — propre et sûr
        niveaux_total = np.linspace(np.min(values), np.max(values), 50)

        # Tracer les lignes noires uniquement dans ces plages
        # contour =  ax.tricontour(triang, values, levels=niveaux_total, colors='k', linewidths=0.5)
        # 🔹 Utilisation de `make_axes_locatable` pour forcer la colorbar en bas
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.4)  
        
        cbar = plt.colorbar(contourf, cax=cax, orientation="horizontal")  
        cbar.set_label(label)  # Label dynamique

        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.set_xlim(0, 1)  
        ax.set_ylim(0, 1)  
        ax.set_aspect(1)  

        plt.savefig(filename, dpi=300)
        plt.show()

    # Extraction des données après Gatherv
    Orthocentre_x = recvbuf_x  
    Orthocentre_y = recvbuf_y  
    u_vals = recvbuf_u  

    # Tracer rho et P avec la bonne colorbar en bas
   
    plot_isovaleurs(Orthocentre_x, Orthocentre_y, u_vals, "Velocity", "isovaleur_cavity.png",cmap= "coolwarm", label="Velocity")

if RANK==0:
    from scipy.interpolate import griddata

    Orthocentre_x = recvbuf_x  
    Orthocentre_y = recvbuf_y  
    u_vals = recvbuf_U 
    v_vals = recvbuf_V 
    x = np.linspace(0, 1, 300)
    y = np.linspace(0, 1, 300)
    X, Y = np.meshgrid(x, y)
    # Interpolation sur la grille
    U = griddata((Orthocentre_x, Orthocentre_y), u_vals, (X, Y), method='linear')
    V = griddata((Orthocentre_x, Orthocentre_y), v_vals, (X, Y), method='linear')


    plt.figure(figsize=(6, 6))
    plt.streamplot(X, Y, U, V, density=1.5, color='k', linewidth=1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Streamlines of the velocity field")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect(1)
    plt.grid(True)
    plt.savefig("streamlines_cavity.png", dpi=300, bbox_inches='tight')

    plt.show()

    dy, dx = Y[1, 0] - Y[0, 0], X[0, 1] - X[0, 0]
    dV_dx = np.gradient(V, dx, axis=1)
    dU_dy = np.gradient(U, dy, axis=0)
    vorticity = dV_dx - dU_dy

    plt.figure(figsize=(6, 6))

    #  On change uniquement la colormap ici (choisis celle que tu préfères)
    cf = plt.contourf(X, Y, vorticity, levels=50, cmap="jet")#ou 'jet', 'seismic', 'Spectral'

    #  Contours noirs conservés
    plt.contour(X, Y, vorticity, levels=50, colors='k', linewidths=0.8)

    #  Barre colorée correcte (liée à cf)
    cbar = plt.colorbar(cf)
    cbar.set_label("Vorticity ω")

    #  Mise en forme inchangée
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Vorticity field ")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal')
    plt.grid(True)
    plt.savefig("vorticity_field.png", dpi=300, bbox_inches='tight')
    plt.show()


if RANK == 0:

    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    def plot_cross_profiles(y, ux, x, vy, y_ghia, ux_ghia, x_ghia, vy_ghia, title="Velocity Profiles", filename="cross_plot.png"):
        fig, ax = plt.subplots(figsize=(8, 8))

        # u_x(y) - courbe verticale à gauche (x bas / y gauche)
        ax.plot(ux_ghia, y_ghia, 'bo', label='Ghia and al $u(y)$', markerfacecolor='none')
        ax.plot(ux, y, 'r-', label='compressible $u(y)$')

        # v_y(x) - courbe horizontale (x haut / y droite)
        ax2 = ax.twinx().twiny()
        ax2.plot(x_ghia, vy_ghia, 'ko', label='Ghia and al $v(x)$', markerfacecolor='none')
        ax2.plot(x, vy, 'g-', label='compressible $v(x)$')

        # Axes et labels
        ax.set_ylabel("y", fontsize=12)
        ax.set_xlabel(r"$u(y)$", fontsize=12)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(0, 1)
        ax.grid(True)

        ax2.set_xlabel("x", fontsize=12)
        ax2.set_ylabel(r"$v(x)$", fontsize=12)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(-1.0, 1.0)

        # Légende combinée
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=10)

        plt.title(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()





    # def plot_profile(x, u, label="velocity", xlabel="x [m]", ylabel="u [m/s]", filename="output.png"):
    #     plt.figure(figsize=(6, 5))
    #     plt.plot(x, u, 'b-', linewidth=2, label=label)  # bleu + épais
    #     plt.xlabel(xlabel)
    #     plt.ylabel(ylabel)
    #     plt.grid(True)
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(filename, dpi=300)
    #     plt.show()

    # Données extraites
    Ux, Vx, Uy, Vy, y_raw, x_raw = [], [], [], [], [], []
    epsilon = 1e-2

    for i in range(len(recvbuf_U)):
        if abs(Orthocentre_y[i] - 0.5) < epsilon:
            Ux.append(recvbuf_U[i])
            Vx.append(recvbuf_V[i])
            x_raw.append(Orthocentre_x[i])
        if abs(Orthocentre_x[i] - 0.5) < epsilon:
            Uy.append(recvbuf_U[i])
            Vy.append(recvbuf_V[i])
            y_raw.append(Orthocentre_y[i])

    # Interpolation sur une grille régulière
    x_uniform = np.linspace(0, 1, 120)
    y_uniform = np.linspace(0, 1, 120)

    if x_raw:
        x_sorted, Ux_sorted = zip(*sorted(zip(x_raw, Ux)))
        x_sorted, Vx_sorted = zip(*sorted(zip(x_raw, Vx)))
        Ux_interp = np.interp(x_uniform, x_sorted, Ux_sorted)
        Vx_interp = np.interp(x_uniform, x_sorted, Vx_sorted)
        # plot_profile(x_uniform, Ux_interp, label="u(x) Compressible", xlabel='x [m]', ylabel="u [m/s]", filename="Ux.png")
        # plot_profile(x_uniform, Vx_interp, label="v(x) Compressible", xlabel='x [m]', ylabel="v [m/s]", filename="Vx.png")

    if y_raw:
        y_sorted, Uy_sorted = zip(*sorted(zip(y_raw, Uy)))
        y_sorted, Vy_sorted = zip(*sorted(zip(y_raw, Vy)))
        Uy_interp = np.interp(y_uniform, y_sorted, Uy_sorted)
        Vy_interp = np.interp(y_uniform, y_sorted, Vy_sorted)
        # plot_profile(y_uniform, Uy_interp, label="u(y) Compressible", xlabel='y [m]', ylabel="u [m/s]", filename="Uy.png")
        # plot_profile(y_uniform, Vy_interp, label="v(y) Compressible", xlabel='y [m]', ylabel="v [m/s]", filename="Vy.png")
   


        # Chargement des données Ghia
    df_u = pd.read_csv("ghia_u_y_Re.csv")
    df_v = pd.read_csv("ghia_v_x_Re.csv")

    Re = 1000
    ux_ghia = df_u[f"u_x_Re{Re}"]
    y_ghia = df_u["y"]
    vy_ghia = df_v[f"v_y_Re{Re}"]
    x_ghia = df_v["x"]

    # Tes propres données
    # y_raw = [y1, y2, ...], Uy = [u1, u2, ...]
    # x_raw = [x1, x2, ...], Vx = [v1, v2, ...]

    # Interpolation si besoin
    y_sorted, Uy_sorted = zip(*sorted(zip(y_raw, Uy)))
    x_sorted, Vx_sorted = zip(*sorted(zip(x_raw, Vx)))

    y_uniform = np.linspace(0, 1, 200)
    x_uniform = np.linspace(0, 1, 200)
    Uy_interp = np.interp(y_uniform, y_sorted, Uy_sorted)
    Vx_interp = np.interp(x_uniform, x_sorted, Vx_sorted)

    # Plot
    plot_cross_profiles(
        y=y_uniform, ux=Uy_interp,
        x=x_uniform, vy=Vx_interp,
        y_ghia=y_ghia, ux_ghia=ux_ghia,
        x_ghia=x_ghia, vy_ghia=vy_ghia,
        title="Re=1000",
        filename="uv_plot_1000.png"
    )





   
   