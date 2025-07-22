
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

from manapy.solvers.combustion import EulerSolver
from manapy.solvers.combustion.tools_utils import initialisation_Droplet, ghost_value_Droplet,haloghost_value_Droplet
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
filename="drop3.msh"
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
# tfinal =0.1
miter = 0
niter = 1
saving_at_node = 0

#initialisation 
rho   = Variable(domain=domain)
rhou  = Variable(domain=domain)
rhov  = Variable(domain=domain)
P     = Variable(domain=domain)
T    = Variable(domain=domain)
Y1   = Variable(domain=domain)
Y2    = Variable(domain=domain)
rhoY1    = Variable(domain=domain)
rhoY2    = Variable(domain=domain)
rhoE  = Variable(domain=domain)
e_internal=Variable(domain=domain)


#paramtres
# order=1
# CFL=0.5
order=1
CFL=0.1
R_gaz=287


###parametre1###
Ta=1e4
B=1.5e13
mu_ref=0.03
T_ref=293
Sc1=1
Sc2=1
alpha=0
M1=16
M2=32
mu1=1
mu2=2
nu1=0
nu2=0
deltaQ=4e7
Pr=0.5
###parametre2###
# Ta=1e4
# B=1.5e11
# mu_ref=1.8e-5
# T_ref=293
# Sc1=1
# Sc2=1
# alpha=0
# M1=16e-3
# M2=32e-3
# mu1=1
# mu2=2
# nu1=0
# nu2=0
# deltaQ=4e4
# Pr=0.5
####parametres 3########
# Ta=1e4
# B=3.3e7
# mu_ref=0.03
# T_ref=293
# Sc1=1
# Sc2=1
# alpha=0
# M1=16
# M2=32
# mu1=1
# mu2=2
# nu1=0
# nu2=0
# deltaQ=4e7
# Pr=0.5


##paramétres initialisation
T_out=1100
U_0=0.1
T_0=300
P_max=1*R_gaz*T_0



initialisation_Droplet(rho.cell, rhou.cell, rhov.cell, rhoE.cell,e_internal.cell, 
                    P.cell,T.cell,Y1.cell, Y2.cell, rhoY1.cell, rhoY2.cell , cells.center, gamma=1.4, R_gaz=R_gaz, T_0=T_0)


#Call the transport solver
conf = Struct(order=order, cfl= CFL,Pr=Pr, Ta=Ta, B=B, mu_ref=mu_ref, T_ref=T_ref ,Sc1=Sc1,Sc2=Sc2,alpha=alpha, M1=M1, M2=M2,
               mu1=mu1, nu1=nu1, mu2=mu2, nu2=nu2, deltaQ=deltaQ )
S = EulerSolver(rho=rho, rhovel=(rhou, rhov), rhoE=rhoE,e_internal=e_internal, P=P,T=T, Y1=Y1, Y2=Y2, rhoY1=rhoY1, rhoY2=rhoY2, conf=conf,RANK=RANK, COMM=COMM  )
Wc=B*np.exp(-Ta/T.cell)*(((rhoY1.cell/M1)**mu1)*((rhoY2.cell/M2)**mu2))

 
domain.save_on_cell_multi(0, time, niter, miter, variables=["rho", "u","v", "E","P","T", "Y1", "Y2", "reaction rate"],values=[rho.cell, rhou.cell/rho.cell, rhov.cell/rho.cell, rhoE.cell/rho.cell,P.cell, T.cell,Y1.cell,Y2.cell,Wc])

ts = MPI.Wtime()

if RANK == 0: print("Start While loop ...")

tfinal=0.02
#loop over time
infi=0
while infi==0:
# while time < tfinal:
    d_t = S.stepper()
    tot = int(tfinal/d_t/135)+1

    time = time + d_t
    # if RANK==0:
    #     print("time", time)

    S.update_halo_values()

  
    ghost_value_Droplet(rho.ghost,P.ghost,rhou.ghost,rhov.ghost,
                    rhoE.ghost,T.ghost,rhoY1.ghost,rhoY2.ghost,Y1.ghost,Y2.ghost,
                    rho.cell,P.cell,rhou.cell,rhov.cell,T.cell,rhoY1.cell,rhoY2.cell,
                    Y1.cell,Y2.cell,domain.faces.cellid,domain.faces.normal,domain.faces.mesure,
                    domain.faces.ghostcenter,faces_in,faces_out,faces_upper ,faces_bottom, R_gaz,T_out, U_0,P_max,T_0) 
        
#cell
    haloghost_value_Droplet(rho.halo,rhou.halo, rhov.halo, rhoE.halo, P.halo, T.halo,rhoY1.halo,rhoY2.halo, Y1.halo,Y2.halo, 
                            rho.haloghost, rhou.haloghost,rhov.haloghost,rhoE.haloghost,P.haloghost,T.haloghost,rhoY1.haloghost,rhoY2.haloghost, Y1.haloghost,Y2.haloghost
                            ,domain.nodes.haloghostcenter, domain.halonodes, domain.nodes.haloghostfaceinfo,domain.cells.haloghostcenter, R_gaz, T_out, U_0, P_max, T_0)
  
    # for i in range( len(T.cell)):
    #     if T.cell[i]>1000:
    #         print(T.cell[i])

    
    S.explicit_convective()
    S.update_term_source()

    rho.interpolate_celltonode()
    rhou.interpolate_celltonode()
    rhov.interpolate_celltonode()
    rhoE.interpolate_celltonode()
    P.interpolate_celltonode()
    T.interpolate_celltonode()
    Y1.interpolate_celltonode()
    Y2.interpolate_celltonode()
    rhoY1.interpolate_celltonode()
    rhoY2.interpolate_celltonode()

    e_internal.interpolate_celltonode()

    S.explicit_dissipative()

    S.compute_new_val()

    send_P= np.array(P.cell)
    sendc_P = np.array(COMM.gather(len(send_P), root=0))
    if RANK == 0:
        TZ_P = sum(sendc_P)
        recv_P= np.empty(TZ_P, dtype=float)
    else:
        recv_P=None
    

    COMM.Gatherv(sendbuf=send_P, recvbuf=(recv_P,sendc_P), root=0)
    if RANK==0:
        P_max=np.max(recv_P)
    else:
        P_max = None   

    P_max=COMM.bcast(P_max,root=0)
 
    if niter%10 == 0 :
    # if niter== 1 or niter%tot == 0  or time >= tfinal:
            
        if saving_at_node:

            rho.update_halo_value()
            rhou.update_halo_value()
            rhov.update_halo_value()
            P.update_halo_value()
            rhoE.update_halo_value()
            T.update_halo_value()
            Y1.update_halo_value()
            Y2.update_halo_value()
            e_internal.update_halo_value()
            rhoY1.update_halo_value()
            rhoY2.update_halo_value()

        
            rho.interpolate_celltonode()
            rhou.interpolate_celltonode()
            rhov.interpolate_celltonode()
            P.interpolate_celltonode()
            rhoE.interpolate_celltonode()
            T.interpolate_celltonode()
            Y1.interpolate_celltonode()
            Y2.interpolate_celltonode()

            # rhoY1.interpolate_celltonode()
            # rhoY2.interpolate_celltonode()
            # e_internal.interpolate_celltonode()

            Wn=B*np.exp(-Ta/T.node)*(((rhoY1.node/M1)**mu1)*((rhoY2.node/M2)**mu2))


            domain.save_on_node_multi(d_t, time, niter, miter,RANK, variables=["rho", "u","v", "E","P","T","Y1", "Y2", "reaction rate"],values=[rho.node, rhou.node/rho.node, rhov.node/rho.node, rhoE.node/rho.node,P.node, T.node,Y1.node, Y2.node, Wn])
            
        else:
            Wc=B*np.exp(-Ta/T.cell)*(((rhoY1.cell/M1)**mu1)*((rhoY2.cell/M2)**mu2))
            
            # for k in range(len(T.cell)):
            #     if Wc[k]>1e-5:
            #         print("Wc",Wc[k])
            domain.save_on_cell_multi(d_t, time, niter, miter, variables=["rho", "u","v", "E","P","T", "Y1", "Y2", "reaction rate"],values=[rho.cell, rhou.cell/rho.cell, rhov.cell/rho.cell, rhoE.cell/rho.cell,P.cell, T.cell,Y1.cell,Y2.cell,Wc])

    miter += 1
    niter += 1

te = MPI.Wtime()

tt = COMM.reduce(te-ts, op=MPI.MAX, root=0)

if RANK == 0:
    print("Time to do calculation", tt)


senddbuf_T = np.array(T.cell)
senddbuf_Y1=np.array(Y1.cell)
senddbuf_Y2=np.array(Y2.cell)
senddbuf_rho = np.array(rho.cell)
senddbuf_P = np.array(P.cell)
senddbuf_E=np.array(rhoE.cell/rho.cell)
senddbuf_MN = np.array(B*np.exp(-Ta/T.cell)*(((rhoY1.cell/M1)**mu1)*((rhoY2.cell/M2)**mu2)))
senddbuf_x = np.array(cells.center[:, 0])
senddbuf_y = np.array(cells.center[:, 1])
senddbuf_MN2 = np.array(B*np.exp(-Ta/T.cell)*(((rhoY1.cell/(16e-3))**mu1)*((rhoY2.cell/(32e-3))**mu2)))

# Récupérer les tailles des buffers depuis chaque processus
sendcounts_T = np.array(COMM.gather(len(senddbuf_T), root=0))
sendcounts_Y1 = np.array(COMM.gather(len(senddbuf_Y1), root=0))
sendcounts_Y2 = np.array(COMM.gather(len(senddbuf_Y2), root=0))
sendcounts_rho = np.array(COMM.gather(len(senddbuf_rho), root=0))
sendcounts_P = np.array(COMM.gather(len(senddbuf_P), root=0))
sendcounts_E = np.array(COMM.gather(len(senddbuf_E), root=0))
sendcounts_MN = np.array(COMM.gather(len(senddbuf_MN), root=0))
sendcounts_x = np.array(COMM.gather(len(senddbuf_x), root=0))
sendcounts_y = np.array(COMM.gather(len(senddbuf_y), root=0))
sendcounts_MN2 = np.array(COMM.gather(len(senddbuf_MN2), root=0))
#COMM.Barrier()
if RANK == 0:
    total_size_T = sum(sendcounts_T)
    total_size_Y1 = sum(sendcounts_Y1)
    total_size_Y2 = sum(sendcounts_Y2)
    total_size_rho = sum(sendcounts_rho)
    total_size_P = sum(sendcounts_P)
    total_size_E = sum(sendcounts_E)
    total_size_MN = sum(sendcounts_MN)
    total_size_x = sum(sendcounts_x)
    total_size_y = sum(sendcounts_y)
    total_size_MN2 = sum(sendcounts_MN2)
    total_size_T = sum(sendcounts_T)
    total_size_Y1 = sum(sendcounts_Y1)
    total_size_Y2 = sum(sendcounts_Y2)

    # recvbuf_rho = np.empty(total_size_rho, dtype=float)
    # recvbuf_x = np.empty(total_size_x, dtype=float)
    # recvbuf_y = np.empty(total_size_y, dtype=float)
   

    recvbuf_rho = np.empty(total_size_rho, dtype=float)
    recvbuf_P = np.empty(total_size_P, dtype=float)
    recvbuf_E = np.empty(total_size_E, dtype=float)
    recvbuf_MN = np.empty(total_size_MN, dtype=float)
    recvbuf_x = np.empty(total_size_x, dtype=float)
    recvbuf_y = np.empty(total_size_y, dtype=float)
    recvbuf_MN2 = np.empty(total_size_MN2, dtype=float)
    recvbuf_T = np.empty(total_size_T, dtype=float)
    recvbuf_Y1 = np.empty(total_size_Y1, dtype=float)
    recvbuf_Y2 = np.empty(total_size_Y2, dtype=float)

else:
    recvbuf_T=None
    recvbuf_Y1=None
    recvbuf_Y2=None
    recvbuf_rho = None
    recvbuf_P = None
    recvbuf_E = None
    recvbuf_MN = None
    recvbuf_x = None
    recvbuf_y = None
    recvbuf_MN2 = None

COMM.Gatherv(sendbuf=senddbuf_T, recvbuf=(recvbuf_T,sendcounts_T), root=0)
COMM.Gatherv(sendbuf=senddbuf_Y1, recvbuf=(recvbuf_Y1,sendcounts_Y1), root=0)
COMM.Gatherv(sendbuf=senddbuf_Y2, recvbuf=(recvbuf_Y2,sendcounts_Y2), root=0)
COMM.Gatherv(sendbuf=senddbuf_rho, recvbuf=(recvbuf_rho,sendcounts_rho), root=0)
COMM.Gatherv(sendbuf=senddbuf_P, recvbuf=(recvbuf_P,sendcounts_P), root=0)
COMM.Gatherv(sendbuf=senddbuf_E, recvbuf=(recvbuf_E,sendcounts_E), root=0)
COMM.Gatherv(sendbuf=senddbuf_MN, recvbuf=(recvbuf_MN,sendcounts_MN), root=0)
COMM.Gatherv(sendbuf=senddbuf_x, recvbuf=(recvbuf_x,sendcounts_x), root=0)
COMM.Gatherv(sendbuf=senddbuf_y, recvbuf=(recvbuf_y,sendcounts_y), root=0)
COMM.Gatherv(sendbuf=senddbuf_MN2, recvbuf=(recvbuf_MN2,sendcounts_MN2), root=0)

if RANK == 0:
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import numpy as np

    def plot_isovaleurs(x, y, values, title, filename, cmap="seismic", label="Variable",nb=6, mi=0, ma=1):
        triang = Triangulation(x, y)

        fig, ax = plt.subplots(figsize=(5, 3.5))

        tpc = ax.tripcolor(triang, values, shading='gouraud', cmap=cmap)

        if mi==0:
            mi=np.min(values)
        if ma==1:
            ma=np.max(values)

        levels = np.linspace(mi, ma, nb)
        # levels = plt.MaxNLocator().tick_values(mi, ma)

        contours = ax.tricontour(triang, values, levels=levels, colors='k', linewidths=0.5)

        # 🔹 Barre de couleur en bas avec make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.4)
        cbar = plt.colorbar(tpc, cax=cax, orientation="horizontal")
        cbar.set_label(label)

        # 🔹 Mise en forme de l'axe
        # ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.5)
        ax.set_aspect(1)

        plt.savefig(filename, dpi=300)
        plt.show()

    # Extraction des données après Gatherv
    Orthocentre_x = recvbuf_x  
    Orthocentre_y = recvbuf_y  
    P_vals = recvbuf_P
    T_vals=recvbuf_T
    Y1_vals = recvbuf_Y1
    Y2_vals=recvbuf_Y2
    W_vals = recvbuf_MN
    W2_vals = recvbuf_MN2
    rho_vals = recvbuf_rho
    E_vals = recvbuf_E
   

    # Tracer rho et P avec la bonne colorbar en bas
    plot_isovaleurs(Orthocentre_x, Orthocentre_y, P_vals, "P", "isovaleur_preussure.png", cmap="turbo", label="Pressure", nb=8)
    plot_isovaleurs(Orthocentre_x, Orthocentre_y, T_vals, "T", "isovaleur_T.png", cmap="turbo", label="Temperature",nb=8)
    plot_isovaleurs(Orthocentre_x, Orthocentre_y, Y1_vals, "Y1", "isovaleur_Y1.png", cmap="turbo", label="Fuel", nb=8, mi=0.02, ma=1)
    plot_isovaleurs(Orthocentre_x, Orthocentre_y, Y2_vals, "Y2", "isovaleur_Y2.png", cmap="turbo", label="Oxygen",nb=8, mi=0, ma=0.97)
    plot_isovaleurs(Orthocentre_x, Orthocentre_y, W_vals, "W", "isovaleur_w.png", cmap="turbo", label="reaction rate",nb=14) 
    plot_isovaleurs(Orthocentre_x, Orthocentre_y, W2_vals, "W", "isovaleur_w2.png", cmap="turbo", label="reaction rate",nb=14)
    
    plot_isovaleurs(Orthocentre_x, Orthocentre_y, rho_vals, "rho", "isovaleur_RHO.png", cmap="turbo", label="Density",nb=8) 
    plot_isovaleurs(Orthocentre_x, Orthocentre_y, E_vals, "E", "isovaleur_E.png", cmap="turbo", label="Energy",nb=8)































































