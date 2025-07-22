from numpy import zeros, fabs, sqrt, int32, uint32, array, arccos, cos, pi
import numpy as np
from numba import njit

'''
def time_step_euler(rho_c:'float[:]', rhou_c:'float[:]', rhov_c:'float[:]',rhoE_c:'float[:]',P_c:'float[:]', cfl:'float', normal:'float[:,:]', mesure:'float[:]', volume:'float[:]', 
                 faceid:'int32[:,:]', gamma:'float'):
    nbelement =  len(faceid)
    u_n = 0.
    dt_c=np.zeros(nbelement)
    norm = zeros(3)

    for i in range(nbelement):
        lam = 0.
        rho_c_safe = max(rho_c[i], 1e-12)
        c = np.sqrt(gamma*np.abs(P_c[i]/rho_c_safe ))
      
        #print(velson)
        for j in range(faceid[i][-1]):
            norm[:] = normal[faceid[i][j]][:]
            nx=norm[0]
            ny=norm[1] 
            u=rhou_c[i]/rho_c_safe 
            v=rhov_c[i]/rho_c_safe
            #print(normal)
            #u_n = (rhou_c[i]*nx+ rhov_c[i]*ny)/rho_c_safe 
            u_n=u*nx+v*ny
            lam_convect = max(np.abs(u_n + c), np.abs(u_n -c),np.abs(u_n ))
            #print(lam_convect)
            lam += lam_convect * mesure[faceid[i][j]]   
        
        dt_c[i]  = cfl*volume[i]/lam 
    dt =  np.min(dt_c)
   # dt= max(dt,0.00001)
    return dt
'''
def time_step_euler(rho_c:'float[:]', rhou_c:'float[:]', rhov_c:'float[:]',rhoE_c:'float[:]',P_c:'float[:]', cfl:'float', normal:'float[:,:]', mesure:'float[:]', volume:'float[:]', 
                 faceid:'int32[:,:]', gamma:'float',Pr:'float',mu:'float'):
    nbelement =  len(faceid)
    u_n = 0.
    dt_c=np.zeros(nbelement)
    norm = zeros(3)
    mu=mu

    for i in range(nbelement):
        lam = 0.
        rho_c_safe = max(rho_c[i], 1e-12)
        c = np.sqrt(gamma*np.abs(P_c[i]/rho_c_safe ))
      
        #print(velson)
        for j in range(faceid[i][-1]):
            norm[:] = normal[faceid[i][j]][:]
            nx=norm[0]/ mesure[faceid[i][j]]   
            ny=norm[1]/ mesure[faceid[i][j]]   
            u=rhou_c[i]/rho_c_safe 
            v=rhov_c[i]/rho_c_safe
            #print(normal)
            #u_n = (rhou_c[i]*nx+ rhov_c[i]*ny)/rho_c_safe 
            u_n=u*nx+v*ny
            lam_convect = max(np.abs(u_n + c), np.abs(u_n -c),np.abs(u_n ))
            #print(lam_convect)
            lam += lam_convect * mesure[faceid[i][j]]   

            ###diffusion
            Dxx=max(4/3*mu,mu, 4/3*mu*u,v*mu,mu*gamma/Pr)
            Dyy=max(4/3*mu,mu, 4/3*mu*v,u*mu,mu*gamma/Pr)
            mes = sqrt(norm[0]*norm[0] + norm[1]*norm[1])
            lam_diff = Dxx * mes**2 + Dyy * mes**2
            lam += lam_diff/volume[i]
        
        dt_c[i]  = cfl*volume[i]/lam 
    dt =  np.min(dt_c)
    # dt=0.000000001
   
    return dt

def update_euler(rho_c:'float[:]', rhou_c:'float[:]', rhov_c:'float[:]', rhoE_c:'float[:]',e_interanl_c:'float[:]', P_c:'float [:]',
              rez_rho:'float[:]', rez_rhou:'float[:]', rez_rhov:'float[:]', rez_rhoE:'float[:]',diss_rhou:'float[:]',diss_rhov:'float[:]',diss_rhoE:'float[:]',
              dtime:'float', vol:'float[:]',gamma:'float', src_rho:'float[:]', src_rhou:'float[:]', src_rhov:'float[:]', src_rhoE:'float[:]'):

    for i in range(len(rho_c)):
        rho_c[i]   = rho_c[i] + dtime  * (rez_rho[i] +src_rho[i])/vol[i]
        rhou_c[i]  = rhou_c[i]+ dtime  *( rez_rhou[i]+src_rhou[i]-diss_rhou[i] )/vol[i]
        rhov_c[i]  = rhov_c[i]+ dtime  * (rez_rhov[i]+src_rhov[i]-diss_rhov[i])/vol[i]
        rhoE_c[i]  = rhoE_c[i] +dtime  * (rez_rhoE[i]+src_rhoE[i]-diss_rhoE[i])/vol[i]
        
        #print(rho_c[i])
        if rho_c[i]!=0:
            P_c[i]= (gamma - 1) * (rhoE_c[i] - 0.5 * (rhou_c[i]**2 + rhov_c[i]**2) / rho_c[i])
            e_interanl_c[i]=P_c[i] / (rho_c[i] * (gamma - 1.))
        else: 
            P_c[i]=0
            e_interanl_c[i]=0
def term_source_poiseuille(src_rho:'float[:]', src_rhou:'float[:]', src_rhov:'float[:]', src_rhoE:'float[:]'):
    
    nbelement = len(src_rho)

    for i in range(nbelement):
     
        src_rho[i]   = 0.0
        src_rhou[i]  = 0.0
        src_rhov[i]  = 0.0
        src_rhoE[i]  = 0.0
     
 
# def explicitscheme_dissipative_NS(rho_c:'float[:]', rhou_c:'float[:]', rhov_c:'float[:]',  rhoE_c:'float[:]',P_c:'float[:]',
#                                rho_ghost:'float[:]', rhou_ghost:'float[:]', rhov_ghost:'float[:]', rhoE_ghost:'float[:]',P_ghost:'float[:]',
#                                rho_halo:'float[:]', rhou_halo:'float[:]', rhov_halo:'float[:]',rhoE_halo:'float[:]',P_halo:'float[:]',rhox_face:'float[:]',rhoy_face:'float[:]',  rhoux_face:'float[:]',rhouy_face:'float[:]', rhovx_face:'float[:]',
#                                 rhovy_face:'float[:]',rhoEx_face:'float[:]',rhoEy_face:'float[:]',Px_face:'float[:]',Py_face:'float[:]',
#                                 dissip_rhou:'float[:]', dissip_rhov:'float[:]', dissip_rhoE:'float[:]',
#                                innerfaces:'int32[:]', halofaces:'int32[:]', boundaryfaces:'int32[:]',normalf:'float[:,:]', mesuref:'float[:]', cellidf:'int32[:,:]',halofid:'int32[:]',
#                                 T_ref:'float', mu_ref:'float',C_zero:'float',Cp:'float', Pr:'float',R:'int32',gamma:'float',mu:'float',namef:'uint32[:]'):
    
#     dissip_rhou[:] = 0.; dissip_rhov[:] = 0.; dissip_rhoE[:] = 0.
 
#     for i in innerfaces:
#       rho_l  = rho_c[cellidf[i][0]]
#       rhou_l = rhou_c[cellidf[i][0]]
#       rhov_l = rhov_c[cellidf[i][0]]
#       P_l    = P_c[cellidf[i][0]]

#       rho_r  = rho_c[cellidf[i][1]]
#       rhou_r = rhou_c[cellidf[i][1]]
#       rhov_r = rhov_c[cellidf[i][1]]
#       P_r    = P_c[cellidf[i][1]]

#       rhoE_l=rhoE_c[cellidf[i][0]]
#       rhoE_r=rhoE_c[cellidf[i][1]]
#       ####appproximation des varables sur les noeuds
#       rho_f=(rho_r+rho_l)/2
#       u_f=(rhou_r/rho_r+rhou_l/rho_l)/2
#       v_f=(rhov_r/rho_r+rhov_l/rho_l)/2
#       P_f=(P_r+P_l)/2
#       T_f=P_f/(rho_f*R)
#       # mu_f=mu_ref*np.sqrt(T_f/T_ref)*(1+C_zero/T_ref)/1+C_zero/T_f
#       mu_f=mu
#       normal = normalf[i]
#       mesure=mesuref[i]
#       nx=normal[0]/mesure
#       ny=normal[1]/mesure
#       u_x=(rhoux_face[i]-rhox_face[i]*u_f)/rho_f
#       v_x=(rhovx_face[i]-rhox_face[i]*v_f)/rho_f
#       u_y=(rhouy_face[i]-rhoy_face[i]*u_f)/rho_f
#       v_y=(rhovy_face[i]-rhoy_face[i]*v_f)/rho_f
#       E_f=(rhoE_r/rho_r+rhoE_l/rho_l)/2
#       E_x=(rhoEx_face[i]-rhox_face[i]*E_f)/rho_f
#       E_y=(rhoEy_face[i]-rhoy_face[i]*E_f)/rho_f
#       tau_xx=2/3*mu_f*(2*u_x-v_y)
#       tau_yy=2/3*mu_f*(2*v_y-u_x)
#       tau_xy=mu_f*(u_y+v_x)
#       q_x=-Cp/(Pr*R)*mu_f*((Px_face[i]*rho_f-P_f*rhox_face[i])/(rho_f**2))
#       q_y=-Cp/(Pr*R)*mu_f*((Py_face[i]*rho_f-P_f*rhoy_face[i])/(rho_f**2))

#       diss_rhou=(tau_xx*nx+tau_xy*ny)*mesure
#       diss_rhov=(tau_xy*nx+tau_yy*ny)*mesure
#       diss_rhoE=((u_f*tau_xx+v_f*tau_xy-q_x)*nx + (u_f*tau_xy+v_f*tau_yy-q_y)*ny)*mesure
#       #diss_rhoE=((2/3*mu_f*(2*u_x-v_y)*u_f+mu_f*(u_y+v_x)*v_f +gamma/(Pr)*mu_f*E_x)*nx+(2/3*mu_f*(2*v_y-u_x)*v_f+mu_f*(u_y+v_x)*u_f+gamma/Pr*mu_f*E_y)*ny)*mesure
#       if namef[i] == 0:
#         dissip_rhou[cellidf[i][0]] += diss_rhou
#         dissip_rhov[cellidf[i][0]] += diss_rhov
#         dissip_rhoE[cellidf[i][0]] += diss_rhoE
        
#         dissip_rhou[cellidf[i][1]] -= diss_rhou
#         dissip_rhov[cellidf[i][1]] -=diss_rhov
#         dissip_rhoE[cellidf[i][1]] -=diss_rhoE
#       else:
#         dissip_rhou[cellidf[i][0]] += diss_rhou
#         dissip_rhov[cellidf[i][0]] += diss_rhov
#         dissip_rhoE[cellidf[i][0]] += diss_rhoE
      

 
def explicitscheme_dissipative_NS(rho_c:'float[:]', rhou_c:'float[:]', rhov_c:'float[:]',  rhoE_c:'float[:]',P_c:'float[:]',
                               rho_ghost:'float[:]', rhou_ghost:'float[:]', rhov_ghost:'float[:]', rhoE_ghost:'float[:]',P_ghost:'float[:]',
                               rho_halo:'float[:]', rhou_halo:'float[:]', rhov_halo:'float[:]',rhoE_halo:'float[:]',P_halo:'float[:]',rhox_face:'float[:]',rhoy_face:'float[:]',  rhoux_face:'float[:]',rhouy_face:'float[:]', rhovx_face:'float[:]',
                                rhovy_face:'float[:]',rhoEx_face:'float[:]',rhoEy_face:'float[:]',Px_face:'float[:]',Py_face:'float[:]',
                                dissip_rhou:'float[:]', dissip_rhov:'float[:]', dissip_rhoE:'float[:]',
                               innerfaces:'int32[:]', halofaces:'int32[:]', boundaryfaces:'int32[:]',normalf:'float[:,:]', mesuref:'float[:]', cellidf:'int32[:,:]',halofid:'int32[:]',
                                T_ref:'float', mu_ref:'float',C_zero:'float',Cp:'float', Pr:'float',R:'int32',gamma:'float',mu:'float',namef:'uint32[:]'):
    


    # print("Cp", Cp)
    # print("mu", mu)
    dissip_rhou[:] = 0.; dissip_rhov[:] = 0.; dissip_rhoE[:] = 0.
    # print("mu",mu)
    for i in innerfaces:
      rho_l  = rho_c[cellidf[i][0]]
      rhou_l = rhou_c[cellidf[i][0]]
      rhov_l = rhov_c[cellidf[i][0]]
      P_l    = P_c[cellidf[i][0]]

      rho_r  = rho_c[cellidf[i][1]]
      rhou_r = rhou_c[cellidf[i][1]]
      rhov_r = rhov_c[cellidf[i][1]]
      P_r    = P_c[cellidf[i][1]]

      rhoE_l=rhoE_c[cellidf[i][0]]
      rhoE_r=rhoE_c[cellidf[i][1]]
      ####appproximation des varables sur les noeuds
      rho_f=(rho_r+rho_l)/2
      u_f=(rhou_r/rho_r+rhou_l/rho_l)/2
      v_f=(rhov_r/rho_r+rhov_l/rho_l)/2
      P_f=(P_r+P_l)/2
      T_f=P_f/(rho_f*R)
      # mu_f=mu_ref*np.sqrt(T_f/T_ref)*(1+C_zero/T_ref)/1+C_zero/T_f
      mu_f=mu
      normal = normalf[i]
      mesure=mesuref[i]
      nx=normal[0]/mesure
      ny=normal[1]/mesure
      u_x=(rhoux_face[i]-rhox_face[i]*u_f)/rho_f
      v_x=(rhovx_face[i]-rhox_face[i]*v_f)/rho_f
      u_y=(rhouy_face[i]-rhoy_face[i]*u_f)/rho_f
      v_y=(rhovy_face[i]-rhoy_face[i]*v_f)/rho_f
      E_f=(rhoE_r/rho_r+rhoE_l/rho_l)/2
      E_x=(rhoEx_face[i]-rhox_face[i]*E_f)/rho_f
      E_y=(rhoEy_face[i]-rhoy_face[i]*E_f)/rho_f
      tau_xx=2/3*mu_f*(2*u_x-v_y)
      tau_yy=2/3*mu_f*(2*v_y-u_x)
      tau_xy=mu_f*(u_y+v_x)
      q_x=-Cp/(Pr*R)*mu_f*((Px_face[i]*rho_f-P_f*rhox_face[i])/(rho_f**2))
      q_y=-Cp/(Pr*R)*mu_f*((Py_face[i]*rho_f-P_f*rhoy_face[i])/(rho_f**2))

      diss_rhou=(tau_xx*nx+tau_xy*ny)*mesure
      diss_rhov=(tau_xy*nx+tau_yy*ny)*mesure
      diss_rhoE=((u_f*tau_xx+v_f*tau_xy-q_x)*nx + (u_f*tau_xy+v_f*tau_yy-q_y)*ny)*mesure
      #diss_rhoE=((2/3*mu_f*(2*u_x-v_y)*u_f+mu_f*(u_y+v_x)*v_f +gamma/(Pr)*mu_f*E_x)*nx+(2/3*mu_f*(2*v_y-u_x)*v_f+mu_f*(u_y+v_x)*u_f+gamma/Pr*mu_f*E_y)*ny)*mesure
   
      dissip_rhou[cellidf[i][0]] -= diss_rhou
      dissip_rhov[cellidf[i][0]] -= diss_rhov
      dissip_rhoE[cellidf[i][0]] -= diss_rhoE
      
      dissip_rhou[cellidf[i][1]] += diss_rhou
      dissip_rhov[cellidf[i][1]] +=diss_rhov
      dissip_rhoE[cellidf[i][1]] +=diss_rhoE
      
      
    for i in halofaces :
    
      rho_l  = rho_c[cellidf[i][0]]
      rhou_l = rhou_c[cellidf[i][0]]
      rhov_l = rhov_c[cellidf[i][0]]
      P_l    = P_c[cellidf[i][0]]
 
      rho_r  = rho_halo[halofid[i]]
      rhou_r = rhou_halo[halofid[i]]
      rhov_r = rhov_halo[halofid[i]]
      P_r = P_halo[halofid[i]]

      rhoE_l=rhoE_c[cellidf[i][0]]
      rhoE_r=rhoE_halo[halofid[i]]
      ####appproximation des varables sur les noeuds
      rho_f=(rho_r+rho_l)/2
      u_f=(rhou_r/rho_r+rhou_l/rho_l)/2
      v_f=(rhov_r/rho_r+rhov_l/rho_l)/2
      P_f=(P_r+P_l)/2
      T_f=P_f/(rho_f*R)
      #mu_f=mu_ref*np.sqrt(T_f/T_ref)*(1+C_zero/T_ref)/1+C_zero/T_f
      mu_f=mu
      mesure=mesuref[i]
      normal = normalf[i]
      nx=normal[0]/mesure
      ny=normal[1]/mesure
      u_x=(rhoux_face[i]-rhox_face[i]*u_f)/rho_f
      v_x=(rhovx_face[i]-rhox_face[i]*v_f)/rho_f
      u_y=(rhouy_face[i]-rhoy_face[i]*u_f)/rho_f
      v_y=(rhovy_face[i]-rhoy_face[i]*v_f)/rho_f
      E_f=(rhoE_r/rho_r+rhoE_l/rho_l)/2
      E_x=(rhoEx_face[i]-rhox_face[i]*E_f)/rho_f
      E_y=(rhoEy_face[i]-rhoy_face[i]*E_f)/rho_f


      tau_xx=2/3*mu_f*(2*u_x-v_y)
      tau_yy=2/3*mu_f*(2*v_y-u_x)
      tau_xy=mu_f*(u_y+v_x)
      q_x=-Cp/(Pr*R)*mu_f*((Px_face[i]*rho_f-P_f*rhox_face[i])/(rho_f**2))
      q_y=-Cp/(Pr*R)*mu_f*((Py_face[i]*rho_f-P_f*rhoy_face[i])/(rho_f**2))

      diss_rhou=(tau_xx*nx+tau_xy*ny)*mesure
      diss_rhov=(tau_xy*nx+tau_yy*ny)*mesure
      diss_rhoE=((u_f*tau_xx+v_f*tau_xy-q_x)*nx + (u_f*tau_xy+v_f*tau_yy-q_y)*ny)*mesure

 
      #diss_rhoE=((2/3*mu_f*(2*u_x-v_y)*u_f+mu_f*(u_y+v_x)*v_f +gamma/(Pr)*mu_f*E_x)*nx+(2/3*mu_f*(2*v_y-u_x)*v_f+mu_f*(u_y+v_x)*u_f+gamma/Pr*mu_f*E_y)*ny)*mesure
  
      dissip_rhou[cellidf[i][0]] -= diss_rhou
      dissip_rhov[cellidf[i][0]] -= diss_rhov
      dissip_rhoE[cellidf[i][0]] -= diss_rhoE
      
     
                
    for i in boundaryfaces:  
      rho_l  = rho_c[cellidf[i][0]]
      rhou_l = rhou_c[cellidf[i][0]]
      rhov_l = rhov_c[cellidf[i][0]]
      P_l    = P_c[cellidf[i][0]]

      rho_r  = rho_ghost[i]
      rhou_r = rhou_ghost[i]
      rhov_r = rhov_ghost[i]
      P_r    = P_ghost[i]

      rhoE_l=rhoE_c[cellidf[i][0]]
      rhoE_r=rhoE_ghost[i]

      ####appproximation des varables sur les noeuds
      rho_f=(rho_r+rho_l)/2
      u_f=(rhou_r/rho_r+rhou_l/rho_l)/2
      v_f=(rhov_r/rho_r+rhov_l/rho_l)/2
      P_f=(P_r+P_l)/2
      E_f=(rhoE_r/rho_r+rhoE_l/rho_l)/2
      T_f=P_f/(rho_f*R)
      #mu_f=mu_ref*np.sqrt(T_f/T_ref)*(1+C_zero/T_ref)/1+C_zero/T_f
      mu_f=mu
      normal = normalf[i]
      mesure=mesuref[i]
      nx=normal[0]/mesure
      ny=normal[1]/mesure
      u_x=(rhoux_face[i]-rhox_face[i]*u_f)/rho_f
      v_x=(rhovx_face[i]-rhox_face[i]*v_f)/rho_f
      u_y=(rhouy_face[i]-rhoy_face[i]*u_f)/rho_f
      v_y=(rhovy_face[i]-rhoy_face[i]*v_f)/rho_f
      
      E_x=(rhoEx_face[i]-rhox_face[i]*E_f)/rho_f
      E_y=(rhoEy_face[i]-rhoy_face[i]*E_f)/rho_f

      tau_xx=2/3*mu_f*(2*u_x-v_y)
      tau_yy=2/3*mu_f*(2*v_y-u_x)
      tau_xy=mu_f*(u_y+v_x)
      q_x=-Cp/(Pr*R)*mu_f*((Px_face[i]*rho_f-P_f*rhox_face[i])/(rho_f**2))
      q_y=-Cp/(Pr*R)*mu_f*((Py_face[i]*rho_f-P_f*rhoy_face[i])/(rho_f**2))

      diss_rhou=(tau_xx*nx+tau_xy*ny)*mesure
      diss_rhov=(tau_xy*nx+tau_yy*ny)*mesure
      diss_rhoE=((u_f*tau_xx+v_f*tau_xy-q_x)*nx + (u_f*tau_xy+v_f*tau_yy-q_y)*ny)*mesure

      #diss_rhoE=((2/3*mu_f*(2*u_x-v_y)*u_f+mu_f*(u_y+v_x)*v_f +gamma/(Pr)*mu_f*E_x)*nx+(2/3*mu_f*(2*v_y-u_x)*v_f+mu_f*(u_y+v_x)*u_f+gamma/Pr*mu_f*E_y)*ny)*mesure
      dissip_rhou[cellidf[i][0]] -= diss_rhou
      dissip_rhov[cellidf[i][0]] -= diss_rhov
      dissip_rhoE[cellidf[i][0]] -= diss_rhoE
  

    
    # print("dissip_rhou",dissip_rhou)
    # print("dissip_rhov",dissip_rhov)
    # print("dissip_rhoE",dissip_rhoE)

def explicitscheme_convective_euler(rez_rho:'float[:]', rez_rhou:'float[:]', rez_rhov:'float[:]', rez_rhoE:'float[:]', 
                                 rho_c:'float[:]', rhou_c:'float[:]', rhov_c:'float[:]', rhoE_c:'float[:]', P_c:'float[:]',
                                 rho_ghost:'float[:]', rhou_ghost:'float[:]', rhov_ghost:'float[:]', rhoE_ghost:'float[:]',P_ghost:'float[:]',
                                 rho_halo:'float[:]', rhou_halo:'float[:]', rhov_halo:'float[:]', rhoE_halo:'float[:]',P_halo:'float[:]',
                                 rho_x:'float[:]', rho_y:'float[:]', rhox_halo:'float[:]', rhoy_halo:'float[:]', 
                                 rhou_x:'float[:]', rhou_y:'float[:]', rhoux_halo:'float[:]', rhouy_halo:'float[:]',
                                 rhov_x:'float[:]', rhov_y:'float[:]', rhovx_halo:'float[:]', rhovy_halo:'float[:]',
                                 rhoE_x:'float[:]', rhoE_y:'float[:]', rhoEx_halo:'float[:]', rhoEy_halo:'float[:]',
                                 P_x:'float[:]', P_y:'float[:]', Px_halo:'float[:]', Py_halo:'float[:]',rho_psi:'float[:]',psirho_halo:'float[:]',rhou_psi:'float[:]',
                                 psirhou_halo:'float[:]',rhov_psi:'float[:]',psirhov_halo:'float[:]',rhoE_psi:'float[:]',psirhoE_halo:'float[:]',P_psi:'float[:]',psiP_halo:'float[:]',
                                 centerc:'float[:,:]', centerf:'float[:,:]', centerh:'float[:,:]', centerg:'float[:,:]',
                                 cellidf:'int32[:,:]', mesuref:'float[:]', normalf:'float[:,:]', tangentf:'float[:,:]', halofid:'int32[:]',
                                 innerfaces:'int32[:]', halofaces:'int32[:]', boundaryfaces:'int32[:]', gamma:'float', order:'int32', scheme_to_exec: 'int32'
                                 ,limiter: 'int32',minmodrho_x:'float[:]',minmodrho_y:'float[:]',minmodrho_halo_x:'float[:]'
                                  ,minmodrho_halo_y:'float[:]',minmodrhou_x:'float[:]'
                                  ,minmodrhou_y:'float[:]',minmodrhou_halo_x:'float[:]',minmodrhou_halo_y:'float[:]',minmodrhov_x:'float[:]'
                                  ,minmodrhov_y:'float[:]',minmodrhov_halo_x:'float[:]',minmodrhov_halo_y:'float[:]',minmodrhoE_x:'float[:]'
                                  ,minmodrhoE_y:'float[:]',minmodrhoE_halo_x:'float[:]',minmodrhoE_halo_y:'float[:]',minmodP_x:'float[:]'
                                  ,minmodP_y:'float[:]',minmodP_halo_x:'float[:]',minmodP_halo_y:'float[:]',
                                  vanalbadarho_limiter:'float[:,:]',vanalbadarhou_limiter:'float[:,:]',vanalbadarhov_limiter:'float[:,:]',vanalbadarhoE_limiter:'float[:,:]',vanalbadaP_limiter:'float[:,:]',vanalbadarho_halo:'float[:,:]',vanalbadarhou_halo:'float[:,:]',vanalbadarhov_halo:'float[:,:]',vanalbadarhoE_halo:'float[:,:]',vanalbadaP_halo:'float[:,:]' ):
  
    

            
            #print("le nombre d'element au bord:",boundaryfaces)barthlimiter_2d
            rez_rho[:] = 0.; rez_rhou[:] = 0.; rez_rhov[:] = 0.; rez_rhoE[:] = 0.
            
            flux = zeros(4)
            r_l = zeros(2)
            r_r = zeros(2)
            
            for i in innerfaces:
                rho_l  = rho_c[cellidf[i][0]]
                rhou_l = rhou_c[cellidf[i][0]]
                rhov_l = rhov_c[cellidf[i][0]]
                rhoE_l = rhoE_c[cellidf[i][0]]
                P_l    = P_c[cellidf[i][0]]
        
                
                tangent = tangentf[i]
                normal = normalf[i]
                mesure = mesuref[i]
               
              
                rho_r  = rho_c[cellidf[i][1]]
                rhou_r = rhou_c[cellidf[i][1]]
                rhov_r = rhov_c[cellidf[i][1]]
                rhoE_r = rhoE_c[cellidf[i][1]]
                P_r    = P_c[cellidf[i][1]]
      
                # V_l    = lf_volums[i][0]
                # V_r    = lf_volums[i][1]
              
               
              ####ordre2####
                center_left = centerc[cellidf[i][0]]
                center_right = centerc[cellidf[i][1]]

              # rho
                rho_x_left = rho_x[cellidf[i][0]] 
                rho_y_left = rho_y[cellidf[i][0]]  
                rho_y_right = rho_y[cellidf[i][1]]
                rho_x_right = rho_x[cellidf[i][1]]
              # rhou
                rhou_x_left = rhou_x[cellidf[i][0]]
                rhou_y_left = rhou_y[cellidf[i][0]]
                rhou_x_right = rhou_x[cellidf[i][1]]
                rhou_y_right = rhou_y[cellidf[i][1]]

                rhov_x_left = rhov_x[cellidf[i][0]]
                rhov_y_left = rhov_y[cellidf[i][0]]
                rhov_x_right =  rhov_x[cellidf[i][1]]
                rhov_y_right = rhov_y[cellidf[i][1]]
                
              # rhoE
                rhoE_x_left = rhoE_x[cellidf[i][0]]
                rhoE_y_left = rhoE_y[cellidf[i][0]]
                rhoE_x_right = rhoE_x[cellidf[i][1]]
                rhoE_y_right = rhoE_y[cellidf[i][1]]
                # p
                P_x_left = P_x[cellidf[i][0]] 
                P_y_left = P_y[cellidf[i][0]]
                P_x_right = P_x[cellidf[i][1]]
                P_y_right = P_y[cellidf[i][1]]
               
                r_l[0] = centerf[i][0] - center_left[0]; r_r[0] = centerf[i][0] - center_right[0]; 
                r_l[1] = centerf[i][1] - center_left[1]; r_r[1] = centerf[i][1] - center_right[1]; 
                
     
                ##########barth limiter#####################

                if order==2:
        
                  if limiter==3:
                    rho_l = rho_l +rho_psi[cellidf[i][0]]*(rho_x_left * r_l[0]  + rho_y_left * r_l[1]  )
                    rho_r = rho_r+rho_psi[cellidf[i][1]]*(rho_x_right * r_r[0]  + rho_y_right * r_r[1]  )

                    rhou_l = rhou_l +rhou_psi[cellidf[i][0]]*(rhou_x_left * r_l[0]  + rhou_y_left * r_l[1]  )
                    rhou_r = rhou_r+rhou_psi[cellidf[i][1]]*(rhou_x_right* r_r[0]  + rhou_y_right * r_r[1]  )

                    rhov_l = rhov_l +rhov_psi[cellidf[i][0]]*(rhov_x_left * r_l[0]  + rhov_y_left * r_l[1]  )
                    rhov_r = rhov_r+rhov_psi[cellidf[i][1]]*(rhov_x_right * r_r[0]  + rhov_y_right * r_r[1]  )

                    rhoE_l = rhoE_l +rhoE_psi[cellidf[i][0]]*(rhoE_x_left * r_l[0]  + rhoE_y_left * r_l[1]  )
                    rhoE_r = rhoE_r+rhoE_psi[cellidf[i][1]]*(rhoE_x_right* r_r[0]  + rhoE_y_right * r_r[1]  )

                    P_l = P_l +P_psi[cellidf[i][0]]*(P_x_left * r_l[0]  + P_y_left * r_l[1]  )
                    P_r = P_r+P_psi[cellidf[i][1]]*(P_x_right * r_r[0]  + P_y_right * r_r[1]  )
                  
                      
                    ######vanalbada
                  elif limiter==2:
                     
                    rho_l= rho_l +0.5*vanalbadarho_limiter[i][0]
                    rho_r= rho_r -0.5*vanalbadarho_limiter[i][1]
                    rhou_l= rhou_l +0.5*vanalbadarhou_limiter[i][0]
                    rhou_r= rhou_r -0.5*vanalbadarhov_limiter[i][1]
                    rhov_l= rhov_l +0.5*vanalbadarhov_limiter[i][0]
                    rhov_r= rhov_r -0.5*vanalbadarhov_limiter[i][1]
                    rhoE_l= rhoE_l +0.5*vanalbadarhoE_limiter[i][0]
                    rhoE_r= rhoE_r -0.5*vanalbadarhoE_limiter[i][1]
                    P_l= P_l +0.5*vanalbadaP_limiter[i][0]
                    P_r= P_r -0.5*vanalbadaP_limiter[i][1]

                  
                      
                  elif limiter==1:
         
                    rho_l =rho_l +(minmodrho_x[cellidf[i][0]]*r_l[0]+minmodrho_y[cellidf[i][0]]*r_l[1])
                    rho_r =rho_r +(minmodrho_x[cellidf[i][1]]*r_r[0]+minmodrho_y[cellidf[i][1]]*r_r[1])
                    rhou_l =rhou_l +(minmodrhou_x[cellidf[i][0]]*r_l[0]+minmodrhou_y[cellidf[i][0]]*r_l[1])
                    rhou_r =rhou_r +(minmodrhou_x[cellidf[i][1]]*r_r[0]+minmodrhou_y[cellidf[i][1]]*r_r[1])
                    rhov_l =rhov_l +(minmodrhov_x[cellidf[i][0]]*r_l[0]+minmodrhov_y[cellidf[i][0]]*r_l[1])
                    rhov_r =rhov_r +(minmodrhov_x[cellidf[i][1]]*r_r[0]+minmodrhov_y[cellidf[i][1]]*r_r[1])
                    rhoE_l =rhoE_l +(minmodrhoE_x[cellidf[i][0]]*r_l[0]+minmodrhoE_y[cellidf[i][0]]*r_l[1])
                    rhoE_r =rhoE_r +(minmodrhoE_x[cellidf[i][1]]*r_r[0]+minmodrhoE_y[cellidf[i][1]]*r_r[1])
                    P_l =P_l +(minmodP_x[cellidf[i][0]]*r_l[0]+minmodP_y[cellidf[i][0]]*r_l[1])
                    P_r =P_r +(minmodP_x[cellidf[i][1]]*r_r[0]+minmodP_y[cellidf[i][1]]*r_r[1])


              ################ available schemes################""
                if scheme_to_exec == 1:
                    rusanov_flux_2d(rho_l, rhou_l, rhov_l, rhoE_l, P_l, rho_r, rhou_r, rhov_r, rhoE_r, P_r, normal, gamma, mesure, flux)
                elif scheme_to_exec == 2: #"Roe_version1"
                    Roe_version2(rho_l,rhou_l,rhov_l, rhoE_l,P_l,rho_r,rhou_r, rhov_r,rhoE_r,P_r ,gamma, normal, tangent,flux,mesure)
                elif scheme_to_exec == 3: #"HLL"
                    HLL_euler_2d(rho_l,rhou_l,rhov_l, rhoE_l,P_l,rho_r,rhou_r, rhov_r,rhoE_r,P_r ,gamma, normal, tangent,flux,mesure)
                elif scheme_to_exec==4: #'SRNH, VFRoe'
                    SRNH_2d_euler(rho_l,rhou_l,rhov_l, rhoE_l,P_l,rho_r,rhou_r, rhov_r,rhoE_r,P_r ,gamma, normal, tangent,flux,mesure)
                
                # elif scheme_to_exec == 3: #"Roe_version2"
                #     Roe_version2(rho_l,rhou_l,rhov_l, rhoE_l,P_l,rho_r,rhou_r, rhov_r,rhoE_r,P_r ,gamma, normal, tangent,flux,mesure)
                 # elif scheme_to_exec == 1: #"lax_friedrichs"
                #   #lax_friedrichs(rho_l,rhou_l,rhov_l, rhoE_l,P_l,rho_r,rhou_r, rhov_r,rhoE_r,P_r ,center_l,center_r,nodes,gamma, normal,flux,mesure,dt)
                #     lax_friedrichs(rho_l,rhou_l,rhov_l, rhoE_l,P_l,rho_r,rhou_r, rhov_r,rhoE_r,P_r ,V_l,V_r, normal,flux,mesure,dt)
                # elif scheme_to_exec==6:#'AUSM'
                #     AUSM_scheme(rho_l, rhou_l, rhov_l, rhoE_l, P_l, rho_r, rhou_r, rhov_r, rhoE_r, P_r, normal, gamma, mesure, flux)

              
                rez_rho[cellidf[i][0]]  -= flux[0]
                rez_rhou[cellidf[i][0]] -= flux[1]
                rez_rhov[cellidf[i][0]] -= flux[2]
                rez_rhoE[cellidf[i][0]] -= flux[3]
                
                rez_rho[cellidf[i][1]]  += flux[0]
                rez_rhou[cellidf[i][1]] += flux[1]
                rez_rhov[cellidf[i][1]] += flux[2]
                rez_rhoE[cellidf[i][1]] += flux[3]
                

            for i in halofaces :
                
                rho_l  = rho_c[cellidf[i][0]]
                rhou_l = rhou_c[cellidf[i][0]]
                rhov_l = rhov_c[cellidf[i][0]]
                rhoE_l = rhoE_c[cellidf[i][0]]
                P_l    = P_c[cellidf[i][0]]
                #center_l=center[cellidf[i][0]][:]
                
                
                tangent = tangentf[i]
                normal = normalf[i]
                mesure = mesuref[i]
                #nodes= [vertex[nodeid[i][0]],vertex[nodeid[i][1]]]

                rho_r  = rho_halo[halofid[i]]
                rhou_r = rhou_halo[halofid[i]]
                rhov_r = rhov_halo[halofid[i]]
                rhoE_r = rhoE_halo[halofid[i]]
                P_r = P_halo[halofid[i]]


                # V_l  = lf_volums[i][0]
                # V_r=lf_volums[i][1]
                #center_r=center[cellidf[i][1]][:]
           
              ####ordre2####
                
                center_left = centerc[cellidf[i][0]]
                center_right = centerh[halofid[i]]
                r_l[0] = centerf[i][0] - center_left[0]; r_r[0] = centerf[i][0] - center_right[0]; 
                r_l[1] = centerf[i][1] - center_left[1]; r_r[1] = centerf[i][1] - center_right[1]; 
              # rho
                rho_x_left = rho_x[cellidf[i][0]]
                rho_y_left = rho_y[cellidf[i][0]]
                rho_x_right = rhox_halo[halofid[i]]
                rho_y_right = rhoy_halo[halofid[i]]  
              # rhou
                rhou_x_left = rhou_x[cellidf[i][0]];  rhou_x_right = rhoux_halo[halofid[i]]
                rhou_y_left = rhou_y[cellidf[i][0]];  rhou_y_right = rhouy_halo[halofid[i]]
                #rhov
                rhov_x_left = rhov_x[cellidf[i][0]];  rhov_x_right = rhovx_halo[halofid[i]]
                rhov_y_left = rhov_y[cellidf[i][0]];  rhov_y_right = rhovy_halo[halofid[i]]
              # rhoE
                rhoE_x_left = rhoE_x[cellidf[i][0]];  rhoE_x_right = rhoEx_halo[halofid[i]]
                rhoE_y_left = rhoE_y[cellidf[i][0]];  rhoE_y_right = rhoEy_halo[halofid[i]]
                # p
                P_x_left = P_x[cellidf[i][0]];  P_x_right = Px_halo[halofid[i]]
                P_y_left = P_y[cellidf[i][0]];  P_y_right = Py_halo[halofid[i]]
         
                ##########barth limiter#####################
                if order==2:
                  if limiter==3:
                    rho_l = rho_l +rho_psi[cellidf[i][0]]*(rho_x_left * r_l[0]  + rho_y_left * r_l[1]  )
                    rho_r = rho_r+psirho_halo[halofid[i]]*(rho_x_right * r_r[0]  + rho_y_right* r_r[1]  )
                    rhou_l = rhou_l +rhou_psi[cellidf[i][0]]*(rhou_x_left * r_l[0]  + rhou_y_left * r_l[1]  )
                    rhou_r = rhou_r+psirhou_halo[halofid[i]]*(rhou_x_right * r_r[0]  + rhou_y_right * r_r[1]  )
                    rhov_l = rhov_l +rhov_psi[cellidf[i][0]]*(rhov_x_left * r_l[0]  + rhov_y_left * r_l[1]  )
                    rhov_r = rhov_r+psirhov_halo[halofid[i]]*(rhov_x_right * r_r[0]  + rhov_y_right * r_r[1]  )
                    rhoE_l = rhoE_l +rhoE_psi[cellidf[i][0]]*(rhoE_x_left * r_l[0]  + rhoE_y_left * r_l[1]  )
                    rhoE_r = rhoE_r+psirhoE_halo[halofid[i]]*(rhoE_x_right* r_r[0]  + rhoE_y_right* r_r[1]  )
                    P_l = P_l +P_psi[cellidf[i][0]]*(P_x_left * r_l[0]  + P_y_left * r_l[1]  )
                    P_r = P_r+psiP_halo[halofid[i]]*(P_x_right * r_r[0]  + P_y_right* r_r[1]  )

           
               
                  #####Vanalbada
                  elif limiter==2:
          
                    rho_l= rho_l +0.5*vanalbadarho_limiter[i][0]
                    rho_r= rho_r -0.5*vanalbadarho_halo[i][1]
                    rhou_l= rhou_l +0.5*vanalbadarhou_limiter[i][0]
                    rhou_r= rhou_r -0.5*vanalbadarhou_halo[i][1]
                    rhov_l= rhov_l +0.5*vanalbadarhov_limiter[i][0]
                    rhov_r= rhov_r -0.5*vanalbadarhov_halo[i][1]
                    rhoE_l= rhoE_l +0.5*vanalbadarhoE_limiter[i][0]
                    rhoE_r= rhoE_r -0.5*vanalbadarhoE_halo[i][1]
                    P_l= P_l +0.5*vanalbadaP_limiter[i][0]
                    P_r= P_r -0.5*vanalbadaP_halo[i][1]


                  elif limiter==1:
             
                                  
                    rho_l =rho_l +(minmodrho_x[cellidf[i][0]]*r_l[0]+minmodrho_y[cellidf[i][0]]*r_l[1])
                    rho_r =rho_r +(minmodrho_halo_x[halofid[i]]*r_r[0]+minmodrho_halo_y[halofid[i],]*r_r[1])
                    rhou_l =rhou_l +(minmodrhou_x[cellidf[i][0]]*r_l[0]+minmodrhou_y[cellidf[i][0]]*r_l[1])
                    rhou_r =rhou_r +(minmodrhou_halo_x[halofid[i]]*r_r[0]+minmodrhou_halo_y[halofid[i],]*r_r[1])
                    rhov_l =rhov_l +(minmodrhov_x[cellidf[i][0]]*r_l[0]+minmodrhov_y[cellidf[i][0]]*r_l[1])
                    rhov_r =rhov_r +(minmodrhov_halo_x[halofid[i]]*r_r[0]+minmodrhov_halo_y[halofid[i],]*r_r[1])
                    rhoE_l =rhoE_l +(minmodrhoE_x[cellidf[i][0]]*r_l[0]+minmodrhoE_y[cellidf[i][0]]*r_l[1])
                    rhoE_r =rhoE_r +(minmodrhoE_halo_x[halofid[i]]*r_r[0]+minmodrhoE_halo_y[halofid[i],]*r_r[1])
                    P_l =P_l +(minmodP_x[cellidf[i][0]]*r_l[0]+minmodP_y[cellidf[i][0]]*r_l[1])
                    P_r =P_r +(minmodP_halo_x[halofid[i]]*r_r[0]+minmodP_halo_y[halofid[i],]*r_r[1])

           

  
  #######################" available schemes#####################""
                if scheme_to_exec == 1:
                    rusanov_flux_2d(rho_l, rhou_l, rhov_l, rhoE_l, P_l, rho_r, rhou_r, rhov_r, rhoE_r, P_r, normal, gamma, mesure, flux)
                elif scheme_to_exec == 2: #"Roe_version1"
                    Roe_version2(rho_l,rhou_l,rhov_l, rhoE_l,P_l,rho_r,rhou_r, rhov_r,rhoE_r,P_r ,gamma, normal, tangent,flux,mesure)
                elif scheme_to_exec == 3: #"HLL"
                    HLL_euler_2d(rho_l,rhou_l,rhov_l, rhoE_l,P_l,rho_r,rhou_r, rhov_r,rhoE_r,P_r ,gamma, normal, tangent,flux,mesure)
                elif scheme_to_exec==4: #'SRNH, VFRoe'
                    SRNH_2d_euler(rho_l,rhou_l,rhov_l, rhoE_l,P_l,rho_r,rhou_r, rhov_r,rhoE_r,P_r ,gamma, normal, tangent,flux,mesure)
                
                # elif scheme_to_exec == 3: #"Roe_version2"
                #     Roe_version2(rho_l,rhou_l,rhov_l, rhoE_l,P_l,rho_r,rhou_r, rhov_r,rhoE_r,P_r ,gamma, normal, tangent,flux,mesure)
                 # elif scheme_to_exec == 1: #"lax_friedrichs"
                #   #lax_friedrichs(rho_l,rhou_l,rhov_l, rhoE_l,P_l,rho_r,rhou_r, rhov_r,rhoE_r,P_r ,center_l,center_r,nodes,gamma, normal,flux,mesure,dt)
                #     lax_friedrichs(rho_l,rhou_l,rhov_l, rhoE_l,P_l,rho_r,rhou_r, rhov_r,rhoE_r,P_r ,V_l,V_r, normal,flux,mesure,dt)
                # elif scheme_to_exec==6:#'AUSM'
                #     AUSM_scheme(rho_l, rhou_l, rhov_l, rhoE_l, P_l, rho_r, rhou_r, rhov_r, rhoE_r, P_r, normal, gamma, mesure, flux)

              
                rez_rho[cellidf[i][0]]  -= flux[0]
                rez_rhou[cellidf[i][0]] -= flux[1]
                rez_rhov[cellidf[i][0]] -= flux[2]
                rez_rhoE[cellidf[i][0]] -= flux[3]
            
            for i in boundaryfaces:
              
                rho_l  = rho_c[cellidf[i][0]]
                rhou_l = rhou_c[cellidf[i][0]]
                rhov_l = rhov_c[cellidf[i][0]]
                rhoE_l = rhoE_c[cellidf[i][0]]
                P_l    = P_c[cellidf[i][0]]
               
                
                #center_l=center[cellidf[i][0]][:]
                tangent = tangentf[i]
                normal = normalf[i]
                mesure = mesuref[i]
                #nodes= [vertex[nodeid[i][0]],vertex[nodeid[i][1]]]
                rho_r  = rho_ghost[i]
                rhou_r = rhou_ghost[i]
                rhov_r = rhov_ghost[i]
                rhoE_r = rhoE_ghost[i]
                P_r    = P_ghost[i]
        
                center_left = centerc[cellidf[i][0]]
                center_right = centerg[i]
          
                r_l[0] = centerf[i][0] - center_left[0]; 
                r_l[1] = centerf[i][1] - center_left[1]; 
                                        
                rho_x_left = rho_x[cellidf[i][0]]
                rho_y_left = rho_y[cellidf[i][0]]
     
              
                rhou_x_left = rhou_x[cellidf[i][0]]
                rhou_y_left = rhou_y[cellidf[i][0]]
            
                #rhov
                rhov_x_left = rhov_x[cellidf[i][0]]
                rhov_y_left = rhov_y[cellidf[i][0]]
              
              # rhoE
                rhoE_x_left = rhoE_x[cellidf[i][0]]
                rhoE_y_left = rhoE_y[cellidf[i][0]]
              
                # p
                P_x_left = P_x[cellidf[i][0]]
                P_y_left = P_y[cellidf[i][0]]
              
           
#####################barth#######################################
                if order==2:
                  if limiter==3:
                     
                    rho_l = rho_l +rho_psi[cellidf[i][0]]*(rho_x_left * r_l[0]  + rho_y_left * r_l[1]  )                  
                    rhou_l = rhou_l +rhou_psi[cellidf[i][0]]*(rhou_x_left * r_l[0]  + rhou_y_left * r_l[1]  )                  
                    rhov_l = rhov_l +rhov_psi[cellidf[i][0]]*(rhov_x_left * r_l[0]  + rhov_y_left * r_l[1]  )                 
                    rhoE_l = rhoE_l +rhoE_psi[cellidf[i][0]]*(rhoE_x_left * r_l[0]  + rhoE_y_left * r_l[1]  )
                    P_l = P_l +P_psi[cellidf[i][0]]*(P_x_left * r_l[0]  + P_y_left * r_l[1]  )
                   
               
               
                  elif limiter==2:  
           
                    rho_l= rho_l +0.5*vanalbadarho_limiter[i][0]                   
                    rhou_l= rhou_l +0.5*vanalbadarhou_limiter[i][0]            
                    rhov_l= rhov_l +0.5*vanalbadarhov_limiter[i][0]                   
                    rhoE_l= rhoE_l +0.5*vanalbadarhoE_limiter[i][0]
                    P_l= P_l +0.5*vanalbadaP_limiter[i][0]
              
                 
                  elif limiter==1:  
                    
                    rho_l =rho_l +(minmodrho_x[cellidf[i][0]]*r_l[0]+minmodrho_y[cellidf[i][0]]*r_l[1])
                    rhou_l =rhou_l +(minmodrhou_x[cellidf[i][0]]*r_l[0]+minmodrhou_y[cellidf[i][0]]*r_l[1])
                    rhov_l =rhov_l +(minmodrhov_x[cellidf[i][0]]*r_l[0]+minmodrhov_y[cellidf[i][0]]*r_l[1])
                    rhoE_l =rhoE_l +(minmodrhoE_x[cellidf[i][0]]*r_l[0]+minmodrhoE_y[cellidf[i][0]]*r_l[1])                  
                    P_l =P_l +(minmodP_x[cellidf[i][0]]*r_l[0]+minmodP_y[cellidf[i][0]]*r_l[1])
              

           #############avialable schemes##############
                if scheme_to_exec == 1:
                    rusanov_flux_2d(rho_l, rhou_l, rhov_l, rhoE_l, P_l, rho_r, rhou_r, rhov_r, rhoE_r, P_r, normal, gamma, mesure, flux)
                elif scheme_to_exec == 2: #"Roe_version1"
                    Roe_version2(rho_l,rhou_l,rhov_l, rhoE_l,P_l,rho_r,rhou_r, rhov_r,rhoE_r,P_r ,gamma, normal, tangent,flux,mesure)
                elif scheme_to_exec == 3: #"HLL"
                    HLL_euler_2d(rho_l,rhou_l,rhov_l, rhoE_l,P_l,rho_r,rhou_r, rhov_r,rhoE_r,P_r ,gamma, normal, tangent,flux,mesure)
                elif scheme_to_exec==4: #'SRNH, VFRoe'
                    SRNH_2d_euler(rho_l,rhou_l,rhov_l, rhoE_l,P_l,rho_r,rhou_r, rhov_r,rhoE_r,P_r ,gamma, normal, tangent,flux,mesure)
                
                # elif scheme_to_exec == 3: #"Roe_version2"
                #     Roe_version2(rho_l,rhou_l,rhov_l, rhoE_l,P_l,rho_r,rhou_r, rhov_r,rhoE_r,P_r ,gamma, normal, tangent,flux,mesure)
                 # elif scheme_to_exec == 1: #"lax_friedrichs"
                #   #lax_friedrichs(rho_l,rhou_l,rhov_l, rhoE_l,P_l,rho_r,rhou_r, rhov_r,rhoE_r,P_r ,center_l,center_r,nodes,gamma, normal,flux,mesure,dt)
                #     lax_friedrichs(rho_l,rhou_l,rhov_l, rhoE_l,P_l,rho_r,rhou_r, rhov_r,rhoE_r,P_r ,V_l,V_r, normal,flux,mesure,dt)
                # elif scheme_to_exec==6:#'AUSM'
                #     AUSM_scheme(rho_l, rhou_l, rhov_l, rhoE_l, P_l, rho_r, rhou_r, rhov_r, rhoE_r, P_r, normal, gamma, mesure, flux)

              

                rez_rho[cellidf[i][0]]  -= flux[0]
                rez_rhou[cellidf[i][0]] -= flux[1]
                rez_rhov[cellidf[i][0]] -= flux[2]
                rez_rhoE[cellidf[i][0]] -= flux[3]



@njit
def Roe_version1(rho_l:'float',rhou_l:'float',rhov_l:'float',rhoE_l:'float',P_l:'float',rho_r:'float',rhou_r:'float',rhov_r:'float',rhoE_r:'float',P_r:'float', gamma:'float', normal:'float[:]',tangent:'float[:]',flux:'float[:]',mesure:'float'):
    
    #the unit t and tangent normal
    nx=normal[0]/mesure
    ny=normal[1]/mesure
    tx=tangent[0]/mesure
    ty=tangent[1]/mesure
   

   #initialisation of matrix
    P=np.ones((4,4))
    P_inv=np.zeros((4,4))
    D=np.zeros((4,4))

    #H_l=rhoE_l/rho_l +(gamma-1)*(rhoE_l/rho_l-(0.5*(rhou_l**2+rhov_l**2))/(rho_l**2))
    #H_r=rhoE_r/rho_r +(gamma-1)*(rhoE_r/rho_r-(0.5*(rhou_r**2+rhov_r**2))/(rho_r**2))

   
    d= np.sqrt(rho_l)+np.sqrt(rho_r)
    u_l=rhou_l/rho_l
    v_l=rhov_l/rho_l
    u_r=rhou_r/rho_r
    v_r=rhov_r/rho_r
    E_l=rhoE_l/rho_l
    E_r=rhoE_r/rho_r

    H_l=E_l+P_l/rho_l
    H_r=E_r+ P_r/rho_r
     
    c_l = np.sqrt(gamma * P_l / rho_l)  
    c_r = np.sqrt(gamma * P_r / rho_r) 
    #les etats de roe
    rhotilde= 0.5*(rho_r + rho_l)
    utilde=((np.sqrt(rho_l)*u_l)+(np.sqrt(rho_r)*u_r))/d
    vtilde=(np.sqrt(rho_l)*v_l +np.sqrt(rho_r)*v_r)/d
    Htilde=(np.sqrt(rho_l)*H_l+np.sqrt(rho_r)*H_r)/d
    ptilde=(np.sqrt(rho_l)*P_l+np.sqrt(rho_r)*P_r)/d
    #ctilde=np.sqrt(gamma*ptilde/rhotilde)
    
    ctilde=np.sqrt((gamma-1)*(Htilde-0.5*(utilde**2+vtilde**2)))
 
    # les valeurs propres de la matrice jacob
    lamd1= utilde*nx+ vtilde*ny - ctilde
    lamd2=utilde*nx+ vtilde*ny 
    lamd3=lamd2
    lamd4= utilde*nx+ vtilde*ny + ctilde


# les valeurs propres left and right
    lamd1_l= u_l*nx+ v_l*ny - c_l
    lamd2_l=u_l*nx+ v_l*ny 
    lamd3_l=lamd2_l
    lamd4_l= u_l*nx+ v_l*ny + c_l

    lamd1_r= u_r*nx+ v_r*ny - c_r
    lamd2_r=u_r*nx+ v_r*ny 
    lamd3_r=lamd2_r
    lamd4_r= u_r*nx+ v_r*ny + c_r
    epsilon_1=max(lamd1_r-lamd1_l,0)
    epsilon_2=max(lamd2_r-lamd2_l,0)
    epsilon_3=max(lamd3_r-lamd3_l,0)
    epsilon_4=max(lamd4_r-lamd4_l,0)
    
# corrction entropique
    if lamd1>0:
      if np.abs(lamd1)<epsilon_1:
          lamd1= (lamd1**2+epsilon_1**2)/(2*epsilon_1)
      else:
          lamd1= np.abs(lamd1)
    else:
      if np.abs(lamd1)<epsilon_1:
          lamd1= -(lamd1**2+epsilon_1**2)/(2*epsilon_1)
      else:
          lamd1= - np.abs(lamd1)
    if lamd2>0:
      if np.abs(lamd2)<epsilon_2:
        lamd2= (lamd2**2+epsilon_2**2)/(2*epsilon_2)
      else:
        lamd2= np.abs(lamd2)
    else:
      if np.abs(lamd2)<epsilon_2:
        lamd2= -(lamd2**2+epsilon_2**2)/(2*epsilon_2)
      else:
        lamd2= -np.abs(lamd2)
    if lamd3>0:
      if np.abs(lamd3)<epsilon_3:
        lamd3= (lamd3**2+epsilon_3**2)/(2*epsilon_3)
      else:
        lamd3= np.abs(lamd3)
    else: 
      if np.abs(lamd3)<epsilon_3:
        lamd3= -(lamd3**2+epsilon_3**2)/(2*epsilon_3)
      else:
        lamd3= -np.abs(lamd3)
    if lamd4>0:
      if np.abs(lamd4)<epsilon_4:
        lamd4= (lamd4**2+epsilon_4**2)/(2*epsilon_4)
      else:
        lamd4= np.abs(lamd4)
    else:
      if np.abs(lamd4)<epsilon_4:
        lamd4= -(lamd4**2+epsilon_4**2)/(2*epsilon_4)
      else:
        lamd4= -np.abs(lamd4)
          
    
    ### F1(w_left)

    F1_rho_l= rhou_l
    F1_rhou_l= rhou_l*u_l+P_l
    F1_rhov_l= rhou_l*v_l
    F1_rhoE_l= (rhoE_l+P_l)*u_l

  ### F1(w_right)

    F1_rho_r= rhou_r
    F1_rhou_r= rhou_r*u_r+P_r
    F1_rhov_r=rhou_r*v_r
    F1_rhoE_r=(rhoE_r+P_r)*u_r


  ### F2(w_left)

    F2_rho_l= rhov_l
    F2_rhou_l= rhov_l*u_l
    F2_rhov_l= rhov_l*v_l+P_l
    F2_rhoE_l= (rhoE_l+P_l)*v_l
  ### F2(w_right)

    F2_rho_r= rhov_r
    F2_rhou_r=rhov_r*u_r
    F2_rhov_r=rhov_r*v_r+P_r
    F2_rhoE_r= (rhoE_r+P_r)*v_r


    W_right_left=np.array([rho_r-rho_l, rhou_r-rhou_l, rhov_r-rhov_l, rhoE_r-rhoE_l])
    

    D[0][0]=np.abs(lamd1)
    D[1][1]=np.abs(lamd2)
    D[2][2]=np.abs(lamd3)
    D[3][3]=np.abs(lamd4 )

    P[0][2]=0
    P[1][0]=utilde-ctilde*nx
    P[1][1]=utilde 
    P[1][2]=tx
    P[1][3]=utilde+ctilde*nx
    P[2][0]=vtilde-ctilde*ny
    P[2][1]=vtilde 
    P[2][2]=ty
    P[2][3]=vtilde+ctilde*ny
    P[3][0]=Htilde-ctilde*(utilde*nx+vtilde*ny)
    P[3][1]=(utilde**2+vtilde**2)/2
    P[3][2]=utilde*tx+vtilde*ty
    P[3][3]=Htilde+ctilde*(utilde*nx+vtilde*ny)
    P_inv=np.linalg.inv(P)
    
    diff=(np.dot(np.dot(np.dot(P,D),P_inv),W_right_left))

    flux[0]=(0.5*(F1_rho_r*nx+F2_rho_r*ny+F1_rho_l*nx+F2_rho_l*ny )-0.5*diff[0])*mesure
    flux[1]=(0.5*(F1_rhou_r*nx+ F2_rhou_r*ny+F1_rhou_l*nx+ F2_rhou_l*ny)-0.5*diff[1])*mesure
    flux[2]=(0.5*(F1_rhov_r*nx+ F2_rhov_r*ny+F1_rhov_l*nx+ F2_rhov_l*ny)-0.5*diff[2])*mesure
    flux[3]=(0.5*(F1_rhoE_r*nx+ F2_rhoE_r*ny+F1_rhoE_l*nx+ F2_rhoE_l*ny)-0.5*diff[3])*mesure
 
@njit
def Roe_version2(rho_l:'float',rhou_l:'float',rhov_l:'float',rhoE_l:'float',P_l:'float',rho_r:'float',rhou_r:'float',rhov_r:'float',rhoE_r:'float',P_r:'float', gamma:'float', normal:'float[:]',tangent:'float[:]',flux:'float[:]',mesure:'float'):
    
    #the unit t and tangent normal
    nx=normal[0]/mesure
    ny=normal[1]/mesure
    tx=tangent[0]/mesure
    ty=tangent[1]/mesure
   

  #initialisation of matrix
    R1=np.zeros((4,1))
    R4=np.zeros((4,1))
    L1=np.zeros((1,4))
    L4=np.zeros((1,4))

    #H_l=rhoE_l/rho_l +(gamma-1)*(rhoE_l/rho_l-(0.5*(rhou_l**2+rhov_l**2))/(rho_l**2))
    #H_r=rhoE_r/rho_r +(gamma-1)*(rhoE_r/rho_r-(0.5*(rhou_r**2+rhov_r**2))/(rho_r**2))

   
    d= np.sqrt(rho_l)+np.sqrt(rho_r)
    u_l=rhou_l/rho_l
    v_l=rhov_l/rho_l
    u_r=rhou_r/rho_r
    v_r=rhov_r/rho_r
    E_l=rhoE_l/rho_l
    E_r=rhoE_r/rho_r

    H_l=E_l+P_l/rho_l
    H_r=E_r+ P_r/rho_r
    c_l = np.sqrt(gamma * P_l / rho_l)  
    c_r = np.sqrt(gamma * P_r / rho_r) 
    #les etats de roe
    rhotilde= 0.5*(rho_r + rho_l)
    utilde=((np.sqrt(rho_l)*u_l)+(np.sqrt(rho_r)*u_r))/d
    vtilde=(np.sqrt(rho_l)*v_l +np.sqrt(rho_r)*v_r)/d
    Htilde=(np.sqrt(rho_l)*H_l+np.sqrt(rho_r)*H_r)/d
    ptilde=(np.sqrt(rho_l)*P_l+np.sqrt(rho_r)*P_r)/d
    #ctilde=np.sqrt(gamma*ptilde/rhotilde)
    
    ctilde=np.sqrt((gamma-1)*(Htilde-0.5*(utilde**2+vtilde**2)))
   
    # les valeurs propres de la matrice jacob
    lamd1= utilde*nx+ vtilde*ny - ctilde
    lamd2=utilde*nx+ vtilde*ny 
    lamd3=lamd2
    lamd4= utilde*nx+ vtilde*ny + ctilde





  # les valeurs propres left and right
    lamd1_l= u_l*nx+ v_l*ny - c_l
    lamd2_l=u_l*nx+ v_l*ny 
    lamd3_l=lamd2_l
    lamd4_l= u_l*nx+ v_l*ny + c_l

    lamd1_r= u_r*nx+ v_r*ny - c_r
    lamd2_r=u_r*nx+ v_r*ny 
    lamd3_r=lamd2_r
    lamd4_r= u_r*nx+ v_r*ny + c_r
    epsilon_1=max(lamd1_r-lamd1_l,0)
    epsilon_2=max(lamd2_r-lamd2_l,0)
    epsilon_3=max(lamd3_r-lamd3_l,0)
    epsilon_4=max(lamd4_r-lamd4_l,0)
    
  # corrction entropique
    if lamd1>0:
      if np.abs(lamd1)<epsilon_1:
          lamd1= (lamd1**2+epsilon_1**2)/(2*epsilon_1)
      else:
          lamd1= np.abs(lamd1)
    else:
      if np.abs(lamd1)<epsilon_1:
          lamd1= -(lamd1**2+epsilon_1**2)/(2*epsilon_1)
      else:
          lamd1= - np.abs(lamd1)
    if lamd2>0:
      if np.abs(lamd2)<epsilon_2:
        lamd2= (lamd2**2+epsilon_2**2)/(2*epsilon_2)
      else:
        lamd2= np.abs(lamd2)
    else:
      if np.abs(lamd2)<epsilon_2:
        lamd2= -(lamd2**2+epsilon_2**2)/(2*epsilon_2)
      else:
        lamd2= -np.abs(lamd2)
    if lamd3>0:
      if np.abs(lamd3)<epsilon_3:
        lamd3= (lamd3**2+epsilon_3**2)/(2*epsilon_3)
      else:
        lamd3= np.abs(lamd3)
    else: 
      if np.abs(lamd3)<epsilon_3:
        lamd3= -(lamd3**2+epsilon_3**2)/(2*epsilon_3)
      else:
        lamd3= -np.abs(lamd3)
    if lamd4>0:
      if np.abs(lamd4)<epsilon_4:
        lamd4= (lamd4**2+epsilon_4**2)/(2*epsilon_4)
      else:
        lamd4= np.abs(lamd4)
    else:
      if np.abs(lamd4)<epsilon_4:
        lamd4= -(lamd4**2+epsilon_4**2)/(2*epsilon_4)
      else:
        lamd4= -np.abs(lamd4)
    
    
    ### F1(w_left)

    F1_rho_l= rhou_l
    F1_rhou_l= rhou_l*u_l+P_l
    F1_rhov_l= rhou_l*v_l
    F1_rhoE_l= (rhoE_l+P_l)*u_l

  ### F1(w_right)

    F1_rho_r= rhou_r
    F1_rhou_r= rhou_r*u_r+P_r
    F1_rhov_r=rhou_r*v_r
    F1_rhoE_r=(rhoE_r+P_r)*u_r


  ### F2(w_left)

    F2_rho_l= rhov_l
    F2_rhou_l= rhov_l*u_l
    F2_rhov_l= rhov_l*v_l+P_l
    F2_rhoE_l= (rhoE_l+P_l)*v_l
  ### F2(w_right)

    F2_rho_r= rhov_r
    F2_rhou_r=rhov_r*u_r
    F2_rhov_r=rhov_r*v_r+P_r
    F2_rhoE_r= (rhoE_r+P_r)*v_r


    W_right_left=np.array([rho_r-rho_l, rhou_r-rhou_l, rhov_r-rhov_l, rhoE_r-rhoE_l])
    
    
    R1[0][0]=1/ctilde
    R1[1][0]=utilde/ctilde -nx
    R1[2][0]=vtilde/ctilde -ny
    R1[3][0]=(vtilde**2+utilde**2)/(2*ctilde) - vtilde*ny-utilde*nx + ctilde/(gamma-1)
   
  

    L1[0][0]=0.5*(utilde*nx+vtilde*ny+(gamma-1)*(utilde**2+vtilde**2)/(2*ctilde))
    L1[0][1]=-0.5*(nx+(gamma-1)*utilde/ctilde)
    L1[0][2]=-0.5*(ny+(gamma-1)*vtilde/ctilde)
    L1[0][3]=0.5*(gamma-1)/ctilde


    R4[0][0]=1/ctilde
    R4[1][0]=utilde/ctilde +nx
    R4[2][0]=vtilde/ctilde +ny
    R4[3][0]=(vtilde**2+utilde**2)/(2*ctilde) + vtilde*ny+utilde*nx + ctilde/(gamma-1)


    L4[0][0]=0.5*(-utilde*nx-vtilde*ny+(gamma-1)*(utilde**2+vtilde**2)/(2*ctilde))
    L4[0][1]=0.5*(nx-(gamma-1)*utilde/ctilde)
    L4[0][2]=0.5*(ny-(gamma-1)*vtilde/ctilde)
    L4[0][3]=0.5*((gamma-1)/ctilde)

  
    diff1=np.dot(np.dot(R1,L1),W_right_left)
    diff2=np.dot(np.dot(R4,L4),W_right_left)

    if lamd2<0:
      if lamd4<=0:
        flux[0]=(F1_rho_r*nx+F2_rho_r*ny)*mesure
        flux[1]=(F1_rhou_r*nx+ F2_rhou_r*ny)*mesure
        flux[2]=(F1_rhov_r*nx+ F2_rhov_r*ny)*mesure
        flux[3]=(F1_rhoE_r*nx+ F2_rhoE_r*ny)*mesure
      
      else: 
        flux[0]=(F1_rho_r*nx+F2_rho_r*ny-lamd4*diff2[0])*mesure
        flux[1]=(F1_rhou_r*nx+ F2_rhou_r*ny-lamd4*diff2[1])*mesure
        flux[2]=(F1_rhov_r*nx+ F2_rhov_r*ny-lamd4*diff2[2])*mesure
        flux[3]=(F1_rhoE_r*nx+ F2_rhoE_r*ny-lamd4*diff2[3])*mesure
       
    else:
      if lamd1>=0:
        flux[0]=(F1_rho_l*nx+F2_rho_l*ny)*mesure
        flux[1]=(F1_rhou_l*nx+ F2_rhou_l*ny)*mesure
        flux[2]=(F1_rhov_l*nx+ F2_rhov_l*ny)*mesure
        flux[3]=(F1_rhoE_l*nx+ F2_rhoE_l*ny)*mesure
 
      else:
        flux[0]=(F1_rho_l*nx+F2_rho_l*ny +lamd1*diff1[0])*mesure
        flux[1]=(F1_rhou_l*nx+ F2_rhou_l*ny+lamd1*diff1[1])*mesure
        flux[2]=(F1_rhov_l*nx+ F2_rhov_l*ny+lamd1*diff1[2])*mesure
        flux[3]=(F1_rhoE_l*nx+ F2_rhoE_l*ny+ lamd1*diff1[3])*mesure
       


    # if lamd2<0:
    #   if lamd4<=0:
    #     flux[0]=(F1_rho_r*nx+F2_rho_r*ny)*mesure
    #     flux[1]=(F1_rhou_r*nx+ F2_rhou_r*ny)*mesure
    #     flux[2]=(F1_rhov_r*nx+ F2_rhov_r*ny)*mesure
    #     flux[3]=(F1_rhoE_r*nx+ F2_rhoE_r*ny)*mesure
    

    #   else: 
    #     flux[0]=(F1_rho_r*nx+F2_rho_r*ny-lamd4*diff2[0])*mesure
    #     flux[1]=(F1_rhou_r*nx+ F2_rhou_r*ny-lamd4*diff2[1])*mesure
    #     flux[2]=(F1_rhov_r*nx+ F2_rhov_r*ny-lamd4*diff2[2])*mesure
    #     flux[3]=(F1_rhoE_r*nx+ F2_rhoE_r*ny-lamd4*diff2[3])*mesure
  
    # else:
    #   if lamd1>=0:
    #     flux[0]=(F1_rho_l*nx+F2_rho_l*ny)*mesure
    #     flux[1]=(F1_rhou_l*nx+ F2_rhou_l*ny)*mesure
    #     flux[2]=(F1_rhov_l*nx+ F2_rhov_l*ny)*mesure
 
    #   else:
    #     flux[0]=(F1_rho_l*nx+F2_rho_l*ny +lamd1*diff1[0])*mesure
    #     flux[1]=(F1_rhou_l*nx+ F2_rhou_l*ny+lamd1*diff1[1])*mesure
    #     flux[2]=(F1_rhov_l*nx+ F2_rhov_l*ny+lamd1*diff1[2])*mesure
    #     flux[3]=(F1_rhoE_l*nx+ F2_rhoE_l*ny+ lamd1*diff1[3])*mesure
      

    # if lamd1>=0:
    #     flux[0]=(F1_rho_l*nx+F2_rho_l*ny)*mesure
    #     flux[1]=(F1_rhou_l*nx+ F2_rhou_l*ny)*mesure
    #     flux[2]=(F1_rhov_l*nx+ F2_rhov_l*ny)*mesure
    #     flux[3]=(F1_rhoE_l*nx+ F2_rhoE_l*ny)*mesure
    # elif lamd4<=0:
    #     flux[0]=(F1_rho_r*nx+F2_rho_r*ny)*mesure
    #     flux[1]=(F1_rhou_r*nx+ F2_rhou_r*ny)*mesure
    #     flux[2]=(F1_rhov_r*nx+ F2_rhov_r*ny)*mesure
    #     flux[3]=(F1_rhoE_r*nx+ F2_rhoE_r*ny)*mesure
    # elif lamd2>=0:
    #     flux[0]=(F1_rho_l*nx+F2_rho_l*ny +lamd1*diff1[0])*mesure
    #     flux[1]=(F1_rhou_l*nx+ F2_rhou_l*ny+lamd1*diff1[1])*mesure
    #     flux[2]=(F1_rhov_l*nx+ F2_rhov_l*ny+lamd1*diff1[2])*mesure
    #     flux[3]=(F1_rhoE_l*nx+ F2_rhoE_l*ny+ lamd1*diff1[3])*mesure
    # elif lamd2<=0:
    #     flux[0]=(F1_rho_r*nx+F2_rho_r*ny-lamd4*diff2[0])*mesure
    #     flux[1]=(F1_rhou_r*nx+ F2_rhou_r*ny-lamd4*diff2[1])*mesure
    #     flux[2]=(F1_rhov_r*nx+ F2_rhov_r*ny-lamd4*diff2[2])*mesure
    #     flux[3]=(F1_rhoE_r*nx+ F2_rhoE_r*ny-lamd4*diff2[3])*mesure
   
@njit
def lax_friedrichs(rho_l:'float',rhou_l:'float',rhov_l:'float', rhoE_l:'float',P_l:'float',rho_r:'float',rhou_r:'float', rhov_r:'float',rhoE_r:'float',P_r:'float' ,V_l:'float',V_r:'float', normal:'float[:]',flux:'float[:]',mesure:'float',dt:'float'):
    nx=normal[0]/mesure
    ny=normal[1]/mesure
    
    '''
    node1=nodes[0]
    node2=nodes[1]

    a_l=np.sqrt((node1[0]-center_l[0])**2+(node1[1]-center_l[1])**2)
    b_l=np.sqrt((node2[0]-center_l[0])**2+(node2[1]-center_l[1])**2)
    c_l=mesure

    a_r=np.sqrt((node1[0]-center_r[0])**2+(node1[1]-center_r[1])**2)
    b_r=np.sqrt((node2[0]-center_r[0])**2+(node2[1]-center_r[1])**2)
    c_r=mesure

    s_l=(a_l+b_l+c_l)/2
    s_r=(a_r+b_r+c_r)/2

    V_l=np.sqrt(s_l*(s_l-a_l)*(s_l-b_l)*(s_l-c_l))
    V_r=np.sqrt(s_r*(s_r-a_r)*(s_r-b_r)*(s_r-c_r))
    '''

 
  ### F1(w_left)
    F1_rho_l= rhou_l
    F1_rhou_l= (rhou_l**2)/rho_l +P_l
    F1_rhov_l= (rhou_l*rhov_l)/rho_l
    F1_rhoE_l= (rhoE_l+P_l)*rhou_l/rho_l

  ### F1(w_right)

    F1_rho_r= rhou_r
    F1_rhou_r= (rhou_r**2)/rho_r +P_r
    F1_rhov_r= (rhou_r*rhov_r)/rho_r
    F1_rhoE_r= (rhoE_r+P_r)*rhou_r/rho_r


  ### F2(w_left)

    F2_rho_l= rhov_l
    F2_rhou_l= (rhou_l*rhov_l)/rho_l 
    F2_rhov_l= (rhov_l**2)/rho_l +P_l
    F2_rhoE_l= (rhoE_l+P_l)*rhov_l/rho_l 

  ### F2(w_right)

    F2_rho_r= rhov_r
    F2_rhou_r=(rhou_r*rhov_r)/rho_r
    F2_rhov_r=(rhov_r**2)/rho_r +P_r
    F2_rhoE_r= (rhoE_r+P_r)*rhov_r/rho_r 

    diff=(2*V_l*V_r)/((V_l+V_r)*dt*mesure)
    
    
    flux[0]=((V_l*(F1_rho_r*nx+ F2_rho_r*ny)+V_r*(F1_rho_l*nx+ F2_rho_l*ny))/(V_l+V_r) -diff*(rho_r-rho_l))*mesure
    flux[1]=((V_l*(F1_rhou_r*nx+ F2_rhou_r*ny)+V_r*(F1_rhou_l*nx+ F2_rhou_l*ny))/(V_l+V_r) -diff*(rhou_r-rhou_l))*mesure
    flux[2]=((V_l*(F1_rhov_r*nx+ F2_rhov_r*ny)+V_r*(F1_rhov_l*nx+ F2_rhov_l*ny))/(V_l+V_r) -diff*(rhov_r-rhov_l))*mesure
    flux[3]=((V_l*(F1_rhoE_r*nx+ F2_rhoE_r*ny)+V_r*(F1_rhoE_l*nx+ F2_rhoE_l*ny))/(V_l+V_r) -diff*(rhoE_r-rhoE_l))*mesure


@njit
def rusanov_flux_2d(rho_l:'float', rhou_l:'float', rhov_l:'float', rhoE_l:'float', P_l:'float',
                    rho_r:'float', rhou_r:'float', rhov_r:'float', rhoE_r:'float', P_r:'float',
                    normal:'float[:]', gamma:'float',mesure:'float', flux:'float[:]'):
 
    nx = normal[0]/mesure 
    ny = normal[1] /mesure
   
    # Vitesse à gauche
    u_l = rhou_l / rho_l
    v_l = rhov_l / rho_l
    un_l = u_l * nx + v_l * ny  
    c_l = np.sqrt(gamma * P_l / rho_l)  

    # Vitesse à droite
    u_r = rhou_r / rho_r
    v_r = rhov_r / rho_r
    un_r = u_r * nx + v_r * ny  
    c_r = np.sqrt(gamma * P_r / rho_r) 

    # Vitesse caractéristique maximale
    alpha = max(np.abs(un_l+ c_l),np.abs(un_l- c_l),abs(un_l), np.abs(un_r+ c_r),np.abs(un_r- c_r),np.abs(un_r))

    # Flux à gauche (F1 et F2)
    F1_rho_l = rhou_l
    F1_rhou_l = rhou_l * u_l + P_l
    F1_rhov_l = rhou_l * v_l
    F1_rhoE_l = (rhoE_l + P_l) * u_l

    F2_rho_l = rhov_l
    F2_rhou_l = rhov_l * u_l
    F2_rhov_l = rhov_l * v_l + P_l
    F2_rhoE_l = (rhoE_l + P_l) * v_l
    
    flux_rho_l=F1_rho_l * nx + F2_rho_l * ny
    flux_rhou_l=F1_rhou_l * nx + F2_rhou_l * ny
    flux_rhov_l= F1_rhov_l * nx + F2_rhov_l * ny
    flux_rhoE_l= F1_rhoE_l * nx + F2_rhoE_l * ny
    
    # Flux à droite (F1 et F2)
    F1_rho_r = rhou_r
    F1_rhou_r = rhou_r * u_r + P_r
    F1_rhov_r = rhou_r * v_r
    F1_rhoE_r = (rhoE_r + P_r) * u_r

    F2_rho_r = rhov_r
    F2_rhou_r = rhov_r * u_r
    F2_rhov_r = rhov_r * v_r + P_r
    F2_rhoE_r = (rhoE_r + P_r) * v_r

    flux_rho_r=F1_rho_r* nx + F2_rho_r * ny
    flux_rhou_r=F1_rhou_r * nx + F2_rhou_r * ny
    flux_rhov_r= F1_rhov_r * nx + F2_rhov_r * ny
    flux_rhoE_r= F1_rhoE_r * nx + F2_rhoE_r* ny
   
    flux[0]= (0.5 * (flux_rho_l + flux_rho_r) - 0.5 * alpha * (rho_r - rho_l))*mesure
    flux[1]=(0.5 * (flux_rhou_l + flux_rhou_r) - 0.5 * alpha * (rhou_r - rhou_l))*mesure
    flux[2]=(0.5 * (flux_rhov_l + flux_rhov_r) - 0.5 * alpha * (rhov_r - rhov_l))*mesure
    flux[3]=(0.5 * (flux_rhoE_l + flux_rhoE_r) - 0.5 * alpha * (rhoE_r - rhoE_l))*mesure


@njit
def HLL_euler_2d(rho_l:'float',rhou_l:'float',rhov_l:'float',rhoE_l:'float',P_l:'float',rho_r:'float',rhou_r:'float',rhov_r:'float',rhoE_r:'float',P_r:'float', gamma:'float', normal:'float[:]',tangent:'float[:]',flux:'float[:]',mesure:'float'):
    
    #the unit t and tangent normal
    nx=normal[0]/mesure
    ny=normal[1]/mesure

    u_l=rhou_l/rho_l
    v_l=rhov_l/rho_l
    u_r=rhou_r/rho_r
    v_r=rhov_r/rho_r
    E_l=rhoE_l/rho_r
    E_r=rhoE_r/rho_r

    c_r = np.sqrt(gamma * P_r / rho_r) 
    c_l = np.sqrt(gamma * P_l / rho_l) 
    
    lamd1_l= u_l*nx+ v_l*ny - c_l
    lamd4_l= u_l*nx+ v_l*ny + c_l
    
    lamd1_r= u_r*nx+ v_r*ny - c_r
    lamd4_r= u_r*nx+ v_r*ny + c_r
  
  #speed
    SL= min(lamd1_l, lamd1_r)
    SR=max(lamd4_l, lamd4_r)
   
  ### F1(w_left)
    F1_rho_l= rhou_l
    F1_rhou_l= rhou_l*u_l+P_l
    F1_rhov_l= rhou_l*v_l
    F1_rhoE_l= (rhoE_l+P_l)*u_l

  ### F1(w_right)
    F1_rho_r= rhou_r
    F1_rhou_r= rhou_r*u_r+P_r
    F1_rhov_r=rhou_r*v_r
    F1_rhoE_r=(rhoE_r+P_r)*u_r


  ### F2(w_left)
    F2_rho_l= rhov_l
    F2_rhou_l= rhov_l*u_l
    F2_rhov_l= rhov_l*v_l+P_l
    F2_rhoE_l= (rhoE_l+P_l)*v_l
 
  ### F2(w_right)
    F2_rho_r= rhov_r
    F2_rhou_r=rhov_r*u_r
    F2_rhov_r=rhov_r*v_r+P_r
    F2_rhoE_r= (rhoE_r+P_r)*v_r


    if SL>0:
      flux[0]=(F1_rho_l*nx+ F2_rho_l*ny)*mesure
      flux[1]=(F1_rhou_l*nx + F2_rhou_l*ny)*mesure
      flux[2]=(F1_rhov_l*nx + F2_rhov_l*ny)*mesure
      flux[3]=(F1_rhoE_l*nx + F2_rhoE_l*ny)*mesure
    elif SL<=0 and SR>=0:
      flux[0]=(((F1_rho_l*nx+ F2_rho_l*ny)*SR-SL*(F1_rho_r*nx+ F2_rho_r*ny)+SL*SR*(rho_r-rho_l))/(SR-SL))*mesure
      flux[1]=(((F1_rhou_l*nx + F2_rhou_l*ny)*SR-SL*(F1_rhou_r*nx+ F2_rhou_r*ny)+SL*SR*(rhou_r-rhou_l))/(SR-SL))*mesure
      flux[2]=(((F1_rhov_l*nx + F2_rhov_l*ny)*SR-SL*(F1_rhov_r*nx+ F2_rhov_r*ny)+SL*SR*(rhov_r-rhov_l))/(SR-SL))*mesure
      flux[3]=(((F1_rhoE_l*nx + F2_rhoE_l*ny)*SR-SL*(F1_rhoE_r*nx+ F2_rhoE_r*ny)+SL*SR*(rhoE_r-rhoE_l))/(SR-SL))*mesure
    elif SR<0:
      flux[0]=(F1_rho_r*nx+ F2_rho_r*ny)*mesure
      flux[1]=(F1_rhou_r*nx + F2_rhou_r*ny)*mesure
      flux[2]=(F1_rhov_r*nx + F2_rhov_r*ny)*mesure
      flux[3]=(F1_rhoE_r*nx + F2_rhoE_r*ny)*mesure
    
@njit
def lax_Wendroff(rho_l:'float',rhou_l:'float',rhov_l:'float', rhoE_l:'float',P_l:'float',rho_r:'float',rhou_r:'float', rhov_r:'float',rhoE_r:'float',P_r:'float' ,center_l:'float[:]',center_r:'float[:]',nodes:'float[:,:]',gamma:'float', normal:'float[:]',flux:'float[:]',mesure:'float',dt:'float'):
    
    
    nx=normal[0]/mesure
    ny=normal[1]/mesure
    node1=nodes[0]
    node2=nodes[1]

    a_l=np.sqrt((node1[0]-center_l[0])**2+(node1[1]-center_l[1])**2)
    b_l=np.sqrt((node2[0]-center_l[0])**2+(node2[1]-center_l[1])**2)
    c_l=mesure

    a_r=np.sqrt((node1[0]-center_r[0])**2+(node1[1]-center_r[1])**2)
    b_r=np.sqrt((node2[0]-center_r[0])**2+(node2[1]-center_r[1])**2)
    c_r=mesure

    s_l=(a_l+b_l+c_l)/2
    s_r=(a_r+b_r+c_r)/2

    V_l=np.sqrt(s_l*(s_l-a_l)*(s_l-b_l)*(s_l-c_l))
    V_r=np.sqrt(s_r*(s_r-a_r)*(s_r-b_r)*(s_r-c_r))

 
    F1_rho_l= rhou_l
    F1_rhou_l= (rhou_l**2)/rho_l +P_l
    F1_rhov_l= (rhou_l*rhov_l)/rho_l
    F1_rhoE_l= (rhoE_l+P_l)*rhou_l/rho_l

  ### F1(w_right)

    F1_rho_r= rhou_r
    F1_rhou_r= (rhou_r**2)/rho_r +P_r
    F1_rhov_r= (rhou_r*rhov_r)/rho_r
    F1_rhoE_r= (rhoE_r+P_r)*rhou_r/rho_r


  ### F2(w_left)

    F2_rho_l= rhov_l
    F2_rhou_l= (rhou_l*rhov_l)/rho_l 
    F2_rhov_l= (rhov_l**2)/rho_l +P_l
    F2_rhoE_l= (rhoE_l+P_l)*rhov_l/rho_l 

  ### F2(w_right)

    F2_rho_r= rhov_r
    F2_rhou_r=(rhou_r*rhov_r)/rho_r
    F2_rhov_r=(rhov_r**2)/rho_r +P_r
    F2_rhoE_r= (rhoE_r+P_r)*rhov_r/rho_r 
    
    #diff=(2*dt*mesure)*(V_l+V_r)/(V_l*V_r)
    diff=0.5*dt*mesure/(V_l+V_r)
   
    SW_rho=(rho_l*V_l+rho_r*V_r)/(V_l+V_r)-diff*((F1_rho_r*nx+ F2_rho_r*ny)-(F1_rho_l*nx+ F2_rho_l*ny))
    SW_rhou=(rhou_l*V_l+rhou_r*V_r)/(V_l+V_r)-diff*((F1_rhou_r*nx+ F2_rhou_r*ny)-(F1_rhou_l*nx+ F2_rhou_l*ny))
    SW_rhov=(rhov_l*V_l+rhov_r*V_r)/(V_l+V_r)-diff*((F1_rhov_r*nx+ F2_rhov_r*ny)-(F1_rhov_l*nx+ F2_rhov_l*ny))
    SW_rhoE=(rhoE_l*V_l+rhoE_r*V_r)/(V_l+V_r)-diff*((F1_rhoE_r*nx+ F2_rhoE_r*ny)-(F1_rhoE_l*nx+ F2_rhoE_l*ny))
 
    w=[SW_rho,SW_rhou,SW_rhov,SW_rhoE]
      
  ### F1(State of lax_Wendroff)

    F1_SW_rho= w[1]
    F1_SW_rhou= w[1]**2/w[0]+(gamma-1)*w[3]-((gamma-1)/2)*((w[1]**2+w[2]**2)/w[0])
    F1_SW_rhov= (w[1]*w[2])/w[0]
    F1_SW_rhoE= (w[3]*w[1])/w[0]+(gamma-1)*((w[3]*w[1])/w[0]-0.5*((w[1]**3)/(w[0]**2)+(w[1]*w[2]**2)/(w[0]**2)))

   
  ### F2(State of lax_Wendroff)

    F2_SW_rho= w[2]
    F2_SW_rhou=(w[1]*w[2])/w[0]
    F2_SW_rhov= w[2]**2/w[0]+(gamma-1)*w[3]-((gamma-1)/2)*((w[1]**2+w[2]**2)/w[0])
    F2_SW_rhoE= (w[3]*w[2])/w[0]+(gamma-1)*((w[3]*w[2])/w[0]-0.5*((w[2]**3)/(w[0]**2)+(w[2]*w[1]**2)/(w[0]**2)))
    
    flux[0]=(F1_SW_rho*nx+ F2_SW_rho*ny)*mesure
    flux[1]=(F1_SW_rhou*nx+ F2_SW_rhou*ny)*mesure
    flux[2]=(F1_SW_rhov*nx+ F2_SW_rhov*ny)*mesure
    flux[3]=(F1_SW_rhoE*nx+ F2_SW_rhoE*ny)*mesure

@njit
def FORCE(rho_l:'float',rhou_l:'float',rhov_l:'float', rhoE_l:'float',P_l:'float',rho_r:'float',rhou_r:'float', rhov_r:'float',rhoE_r:'float',P_r:'float' ,center_l:'float[:]',center_r:'float[:]',nodes:'float[:,:]',gamma:'float', normal:'float[:]',flux:'float[:]',mesure:'float',dt:'float'):
    #copier of flux
    
    flux_friedrichs= flux
    flux_wendroff=flux
    #comput  flux_wendroff and flux_friedrichs
    lax_friedrichs(rho_l,rhou_l,rhov_l, rhoE_l,P_l,rho_r,rhou_r, rhov_r,rhoE_r,P_r ,center_l,center_r,nodes,gamma, normal, flux_friedrichs, mesure,dt)
    lax_Wendroff(rho_l,rhou_l,rhov_l, rhoE_l,P_l,rho_r,rhou_r, rhov_r,rhoE_r,P_r ,center_l,center_r,nodes,gamma, normal, flux_wendroff, mesure,dt)
    #the weighted average
    w=0.5
    #flux
    flux[0]=w*flux_wendroff[0]+(w-1)*flux_friedrichs[0]
    flux[1]=w*flux_wendroff[1]+(w-1)*flux_friedrichs[1]
    flux[2]=w*flux_wendroff[2]+(w-1)*flux_friedrichs[2]
    flux[3]=w*flux_wendroff[3]+(w-1)*flux_friedrichs[3]


@njit
def SRNH_2d_euler(rho_l:'float',rhou_l:'float',rhov_l:'float',rhoE_l:'float',P_l:'float',rho_r:'float',rhou_r:'float',rhov_r:'float',rhoE_r:'float',P_r:'float', gamma:'float', normal:'float[:]',tangent:'float[:]',flux:'float[:]',mesure:'float'): 
    
    #the unit and tangent normal
    nx=normal[0]/mesure
    ny=normal[1]/mesure
    tx=tangent[0]/mesure
    ty=tangent[1]/mesure
   

  #initialisation of matrix
    P=np.ones((4,4))
    P_inv=np.zeros((4,4))
    sign_D=np.zeros((4,4))
    w=zeros(4)
    diff=zeros(4)

    #H_l=rhoE_l/rho_l +(gamma-1)*(rhoE_l/rho_l-(0.5*(rhou_l**2+rhov_l**2))/(rho_l**2))
    #H_r=rhoE_r/rho_r +(gamma-1)*(rhoE_r/rho_r-(0.5*(rhou_r**2+rhov_r**2))/(rho_r**2))

   
    d= np.sqrt(rho_l)+np.sqrt(rho_r)
    u_l=rhou_l/rho_l
    v_l=rhov_l/rho_l
    u_r=rhou_r/rho_r
    v_r=rhov_r/rho_r
    E_l=rhoE_l/rho_l
    E_r=rhoE_r/rho_r

    H_l=E_l+P_l/rho_l
    H_r=E_r+ P_r/rho_r
   
    #les etats de roe
    rhotilde= 0.5*(rho_r + rho_l)
    utilde=((np.sqrt(rho_l)*u_l)+(np.sqrt(rho_r)*u_r))/d
    vtilde=(np.sqrt(rho_l)*v_l +np.sqrt(rho_r)*v_r)/d
    Htilde=(np.sqrt(rho_l)*H_l+np.sqrt(rho_r)*H_r)/d
    ptilde=(np.sqrt(rho_l)*P_l+np.sqrt(rho_r)*P_r)/d
    #ctilde=np.sqrt(gamma*ptilde/rhotilde)
    
    ctilde=np.sqrt((gamma-1)*(Htilde-0.5*(utilde**2+vtilde**2)))
 
    # les valeurs propres de la matrice jacob
    lamd1= utilde*nx+ vtilde*ny - ctilde
    lamd2=utilde*nx+ vtilde*ny 
    lamd3=lamd2
    lamd4= utilde*nx+ vtilde*ny + ctilde
    

    W_right_left=np.array([rho_r-rho_l, rhou_r-rhou_l, rhov_r-rhov_l, rhoE_r-rhoE_l])
    
    epsilon = 1e-3
    
    if fabs(lamd1) < epsilon:
        sign1 = 0.
    else:
        sign1 =lamd1/np.abs(lamd1)
        
    if  fabs(lamd2) < epsilon:
        sign2 = 0.
    else:
        sign2 = lamd2/np.abs(lamd2)
    
    if   fabs(lamd3) < epsilon:
        sign3 = 0.
    else:
        sign3 = lamd3/np.abs(lamd3)
    
    if  fabs(lamd4) < epsilon:
        sign4 = 0.
    else:
        sign4 = lamd4/np.abs(lamd4 )
  
    

    sign_D[0][0]=sign1
    sign_D[1][1]=sign2
    sign_D[2][2]=sign3
    sign_D[3][3]=sign4

    P[0][2]=0
    P[1][0]=utilde-ctilde*nx
    P[1][1]=utilde 
    P[1][2]=tx
    P[1][3]=utilde+ctilde*nx
    P[2][0]=vtilde-ctilde*ny
    P[2][1]=vtilde 
    P[2][2]=ty
    P[2][3]=vtilde+ctilde*ny
    P[3][0]=Htilde-ctilde*(utilde*nx+vtilde*ny)
    P[3][1]=(utilde**2+vtilde**2)/2
    P[3][2]=utilde*tx+vtilde*ty
    P[3][3]=Htilde+ctilde*(utilde*nx+vtilde*ny)

    P_inv=np.linalg.inv(P)
    
    diff=(np.dot(np.dot(np.dot(P,sign_D),P_inv),W_right_left))

    SR_rho= 0.5*(rho_r+rho_l)-0.5*diff[0]
    SR_rhou= 0.5*(rhou_r+rhou_l)-0.5*diff[1]
    SR_rhov= 0.5*(rhov_r+rhov_l)-0.5*diff[2]
    SR_rhoE= 0.5*(rhoE_r+rhoE_l)-0.5*diff[3]
    w[0]=SR_rho
    w[1]=SR_rhou
    w[2]=SR_rhov
    w[3]=SR_rhoE
  ### F1(solution of riemann problem in the face)

    F1_SR_rho= w[1]
    F1_SR_rhou= w[1]**2/w[0]+(gamma-1)*w[3]-((gamma-1)/2)*((w[1]**2+w[2]**2)/w[0])
    F1_SR_rhov= (w[1]*w[2])/w[0]
    F1_SR_rhoE= (w[3]*w[1])/w[0]+(gamma-1)*((w[3]*w[1])/w[0]-0.5*((w[1]**3)/(w[0]**2)+(w[1]*w[2]**2)/(w[0]**2)))

   
  ### F2(solution of riemann problem in the face)
    F2_SR_rho= w[2]
    F2_SR_rhou=(w[1]*w[2])/w[0]
    F2_SR_rhov= w[2]**2/w[0]+(gamma-1)*w[3]-((gamma-1)/2)*((w[1]**2+w[2]**2)/w[0])
    F2_SR_rhoE= (w[3]*w[2])/w[0]+(gamma-1)*((w[3]*w[2])/w[0]-0.5*((w[2]**3)/(w[0]**2)+(w[2]*w[1]**2)/(w[0]**2)))
    

    flux[0]=(F1_SR_rho*nx+F2_SR_rho*ny)*mesure
    flux[1]=(F1_SR_rhou*nx+F2_SR_rhou*ny)*mesure
    flux[2]=(F1_SR_rhov*nx+F2_SR_rhov*ny)*mesure
    flux[3]=(F1_SR_rhoE*nx+F2_SR_rhoE*ny)*mesure

@njit
def AUSM_scheme(rho_l:'float', rhou_l:'float', rhov_l:'float', rhoE_l:'float', P_l:'float',
                    rho_r:'float', rhou_r:'float', rhov_r:'float', rhoE_r:'float', P_r:'float',
                    normal:'float[:]', gamma:'float',mesure:'float', flux:'float[:]'):

    nx = normal[0]/mesure 
    ny = normal[1] /mesure
    flux_c=np.zeros(4)
    p_c=np.zeros(4)
   
    # Vitesse à gauche
    u_l = rhou_l / rho_l
    v_l = rhov_l / rho_l
    un_l = u_l * nx + v_l * ny  
    c_l = np.sqrt(gamma * P_l / rho_l)  

    # Vitesse à droite
    u_r = rhou_r / rho_r
    v_r = rhov_r / rho_r
    un_r = u_r * nx + v_r * ny  
    c_r = np.sqrt(gamma * P_r / rho_r) 

    E_l=rhoE_l/rho_l
    E_r=rhoE_r/rho_r

    H_l=E_l+P_l/rho_l
    H_r=E_r+ P_r/rho_r
    #mach number
    M_left=un_l/c_l
    M_right=un_r/c_r

    if np.abs( M_left)<=1:
      M_left_posi=1/4*(M_left+1)**2
    else:
      M_left_posi=0.5*(M_left+np.abs(M_left))

    if np.abs( M_right)<=1:
      M_right_nega=-1/4*(M_right-1)**2
    else:
      M_right_nega=0.5*(M_right-np.abs(M_right))


    if np.abs( M_left)<=1:
       p_left_posi=(P_l/4)*(M_left +1)**2*(2+M_left)
    else:
       p_left_posi=(P_l/2)*(M_left +np.abs(M_left))/M_left

    if np.abs( M_right)<=1:
       p_right_nega=(P_r/4)*(M_right -1)**2*(2-M_right)
    else:
       p_right_nega=(P_r/2)*(M_right - np.abs(M_right))/M_right
       
    
    p_lf=p_left_posi +p_right_nega
    M_lf= M_left_posi +M_right_nega
    '''
    if M_lf>=0:
       rhoc= rho_l*c_l
       rhouc= rhou_l*c_l
       rhovc= rhov_l*c_l
       rhoHc= rho_l*H_l*c_l
    else:
       rhoc= rho_r*c_r
       rhouc= rhou_r*c_r
       rhovc= rhov_r*c_r
       rhoHc= rho_r*H_r*c_r


    flux_c[0]=M_lf*rhoc
    flux_c[1]=M_lf*rhouc
    flux_c[2]=M_lf*rhovc
    flux_c[3]=M_lf*rhoHc
    '''
    flux_c[0]=0.5*M_lf*(rho_l*c_l+rho_r*c_r) -0.5*np.abs(M_lf)*(rho_r*c_r-rho_l*c_l)
    flux_c[1]=0.5*M_lf*(rhou_l*c_l+rhou_r*c_r) -0.5*np.abs(M_lf)*(rhou_r*c_r-rhou_l*c_l)
    flux_c[2]=0.5*M_lf*(rhov_l*c_l+rhov_r*c_r) -0.5*np.abs(M_lf)*(rhov_r*c_r-rhov_l*c_l)
    flux_c[3]=0.5*M_lf*(rho_l*H_l*c_l+rho_r*H_r*c_r) -0.5*np.abs(M_lf)*(rho_r*H_r*c_r-rho_l*H_l*c_l)
    p_c[1]=p_lf*nx
    p_c[2]=p_lf*ny


    flux[0]=(flux_c[0]+p_c[0])*mesure
    flux[1]=(flux_c[1]+p_c[1])*mesure
    flux[2]=(flux_c[2]+p_c[2])*mesure
    flux[3]=(flux_c[3]+p_c[3])*mesure


       

@njit   
def lim_valbada(x:'float', y:'float', seuil=1e-4):
    if x * y >= 0.0:
        return ((x * x + seuil) * y + (y * y + seuil) * x) / (x * x + y * y + 2.0 * seuil)
    else:
        return 0.0
@njit   
def valbada(Z_l:'float' ,Z_x_l:'float',Z_y_l:'float',BARYCENTRE_l:'float[:]', Z_r:'float',Z_x_r:'float' ,Z_y_r:'float', BARYCENTRE_r:'float[:]'):
    
      vect=np.zeros(2)

      vect[0] = BARYCENTRE_r[0]- BARYCENTRE_l[0]
      vect[1]= BARYCENTRE_r[1]- BARYCENTRE_l[1]

      beta = 2.0 / 3.0
      yy = Z_r - Z_l
      xx = beta * (Z_x_l * vect[0] + Z_y_l* vect[1]) + (1.0 - beta) * yy
      zleft = Z_l+ 0.5 * lim_valbada(xx, yy)

      xx = beta * (Z_x_r * vect[0] + Z_y_r* vect[1]) + (1.0 - beta) * yy
      zright = Z_r - 0.5 * lim_valbada(xx, yy)

      return zleft, zright
@njit   
def valbada_boundary(Z_l:'float' ,Z_x_l:'float',Z_y_l:'float',BARYCENTRE_l:'float[:]', Z_r:'float', BARYCENTRE_r:'float[:]'):
    
      vect=np.zeros(2)

      vect[0] = BARYCENTRE_r[0]- BARYCENTRE_l[0]
      vect[1]= BARYCENTRE_r[1]- BARYCENTRE_l[1]

      beta = 2.0 / 3.0
      yy = Z_r - Z_l
      xx = beta * (Z_x_l * vect[0] + Z_y_l* vect[1]) + (1.0 - beta) * yy
      zleft = Z_l+ 0.5 * lim_valbada(xx, yy)

      return zleft

###############Barth#######################"




# def barthlimiter_2d(w_c:'float[:]', w_ghost:'float[:]', w_halo:'float[:]',
#                     w_x:'float[:]', w_y:'float[:]',r:'float[:]',cellid:'int32[:,:]', faceid:'int32[:,:]', namef:'uint32[:]',
#                     halofid:'int32[:]', centerc:'float[:,:]', centerf:'float[:,:]',i:'int32',innerfaces:'int32[:]' ,boundaryfaces: 'int32[:]', halofaces: 'int32[:]', h:'float'):
@njit
def barthlimiter_2d(w_c:'float[:]', w_ghost:'float[:]', w_halo:'float[:]',
                    w_x:'float[:]', w_y:'float[:]',r:'float[:]',cellid:'int32[:,:]', faceid:'int32[:,:]', namef:'uint32[:]',
                    halofid:'int32[:]', centerc:'float[:,:]', centerf:'float[:,:]',i:'int32'):
   
   
    val  = 1.
    w_max = w_c[i]
    w_min = w_c[i]
    psi=val
   

    for j in range(faceid[i][-1]):
        face = faceid[i][j]
        # if face in innerfaces:#
        # #11 or namef[face] == 22 or namef[face] == 33 or namef[face] == 44:
        #     w_max = max(w_max, w_c[cellid[face][0]], w_c[cellid[face][1]])
        #     w_min = min(w_min, w_c[cellid[face][0]], w_c[cellid[face][1]])
        # elif face in boundaryfaces :
        #     w_max = max(w_max,  w_c[cellid[face][0]], w_ghost[face])
        #     w_min = min(w_min,  w_c[cellid[face][0]], w_ghost[face])
        # elif face in  halofaces:
        #     w_max = max(w_max,  w_c[cellid[face][0]], w_halo[halofid[face]])
        #     w_min = min(w_min,  w_c[cellid[face][0]], w_halo[halofid[face]])
                    
        if namef[face] == 0 or namef[face] > 10:#
        #11 or namef[face] == 22 or namef[face] == 33 or namef[face] == 44:
            w_max = max(w_max, w_c[cellid[face][0]], w_c[cellid[face][1]])
            w_min = min(w_min, w_c[cellid[face][0]], w_c[cellid[face][1]])
        elif namef[face] == 1 or namef[face] == 2 or namef[face] == 3 or namef[face] == 4:
            w_max = max(w_max,  w_c[cellid[face][0]], w_ghost[face])
            w_min = min(w_min,  w_c[cellid[face][0]], w_ghost[face])
        else:
            w_max = max(w_max,  w_c[cellid[face][0]], w_halo[halofid[face]])
            w_min = min(w_min,  w_c[cellid[face][0]], w_halo[halofid[face]])
  
    for j in range(faceid[i][-1]):
      face = faceid[i][j]

      r_xyz1 = centerf[face][0] - centerc[i][0] 
      r_xyz2 = centerf[face][1] - centerc[i][1]
      
      delta2 = w_x[i] * r_xyz1 +w_y[i] * r_xyz2 

      if np.abs(delta2) < 1e-12:
          psi_ij = 1.
      else:
          if delta2 > 0.:
              value = (w_max - w_c[i]) / delta2
              psi_ij = min(val, value)
          elif delta2 < 0.:
              value = (w_min - w_c[i]) / delta2
              psi_ij = min(val, value)
      psi= min(psi, psi_ij)
  
    # print(psi)
    w_cr=w_c[i] + psi*w_x[i]*r[0] + psi*w_y[i]*r[1]
    return w_cr


# @njit

# def minmodlimiter_2d(w_c:'float', w_x:'float[:]', w_y:'float[:]',nodecid:'int32[:]',  cellnid:'int32[:,:]' , vect:'float[:]',i:"int32"):
    
#     nbcells=len(w_x)
    # vois=[]
    # for j in nodecid: 
    #     for v in cellnid[j]:
    #         if i!=v :
    #             if v not in vois:
    #                vois.append(v)

    # k0 = vois[0]
    # sgn_x=[]
    # sgn_y=[]
    # wx=[]
    # wy=[]
    # epsilon=1e-12
    # for k in vois:
    #   if np.abs(w_x[k]) < epsilon:
    #         sgnx = 0.
    #   else:
    #         sgnx =w_x[k]/np.abs(w_x[k])
    #   sgn_x.append(sgnx )
    #   wx.append(np.abs(w_x[k]))
    #   if np.abs(w_y[k]) < epsilon:
    #         sgny = 0.
    #   else:
    #         sgny =w_y[k]/np.abs(w_y[k])
    #   sgn_y.append(sgny )
    #   wy.append(np.abs(w_y[k]))


    # minsgn_x=min(sgn_x)
    # minsgn_y=min(sgn_y)
    # maxsgn_x=max(sgn_x)
    # maxsgn_y=max(sgn_y)
    # min_wx=min(wx)
    # min_wy=min(wy)
    

    # psi_x=0.5*(minsgn_x+maxsgn_x)*min_wx
    # psi_y=0.5*(minsgn_y+maxsgn_y)*min_wy

    # print("psi_x", psi_x)
      
    # w_limiter=w_c +  (psi_x* vect[0] + psi_y* vect[1])
    # return w_limiter


@njit
# def Venhtatakrishnanlimiter_2d(w_c:'float[:]', w_ghost:'float[:]', w_halo:'float[:]',
#                     w_x:'float[:]', w_y:'float[:]',r:'float[:]',cellid:'int32[:,:]', faceid:'int32[:,:]', namef:'uint32[:]',
#                     halofid:'int32[:]', centerc:'float[:,:]', centerf:'float[:,:]',i:'int32',innerfaces:'int32[:]' ,boundaryfaces: 'int32[:]', halofaces: 'int32[:]',vol:'float'):
def Venkatakrishnanlimiter_2d(w_c:'float[:]', w_ghost:'float[:]', w_halo:'float[:]',
                    w_x:'float[:]', w_y:'float[:]',r:'float[:]',cellid:'int32[:,:]', faceid:'int32[:,:]', namef:'uint32[:]',
                    halofid:'int32[:]', centerc:'float[:,:]', centerf:'float[:,:]',i:'int32',vol:'float'):
    # val  = 1.
    w_max = w_c[i]
    w_min = w_c[i]
    psi=1


    for j in range(faceid[i][-1]):
        face = faceid[i][j]
        # if face in innerfaces:#
        # #11 or namef[face] == 22 or namef[face] == 33 or namef[face] == 44:
        #     w_max = max(w_max, w_c[cellid[face][0]], w_c[cellid[face][1]])
        #     w_min = min(w_min, w_c[cellid[face][0]], w_c[cellid[face][1]])
        # elif face in boundaryfaces :
        #     w_max = max(w_max,  w_c[cellid[face][0]], w_ghost[face])
        #     w_min = min(w_min,  w_c[cellid[face][0]], w_ghost[face])
        # elif face in  halofaces:
        #     w_max = max(w_max,  w_c[cellid[face][0]], w_halo[halofid[face]])
        #     w_min = min(w_min,  w_c[cellid[face][0]], w_halo[halofid[face]])
                    
        if namef[face] == 0 or namef[face] > 10:#
        #11 or namef[face] == 22 or namef[face] == 33 or namef[face] == 44:
            w_max = max(w_max, w_c[cellid[face][0]], w_c[cellid[face][1]])
            w_min = min(w_min, w_c[cellid[face][0]], w_c[cellid[face][1]])
        elif namef[face] == 1 or namef[face] == 2 or namef[face] == 3 or namef[face] == 4:
            w_max = max(w_max,  w_c[cellid[face][0]], w_ghost[face])
            w_min = min(w_min,  w_c[cellid[face][0]], w_ghost[face])
        else:
            w_max = max(w_max,  w_c[cellid[face][0]], w_halo[halofid[face]])
            w_min = min(w_min,  w_c[cellid[face][0]], w_halo[halofid[face]])
  
    for j in range(faceid[i][-1]):
        face = faceid[i][j]

        r_xyz1 = centerf[face][0] - centerc[i][0] 
        r_xyz2 = centerf[face][1] - centerc[i][1]
        
        delta= w_x[i] * r_xyz1 +w_y[i] * r_xyz2 
        delta_max=w_max - w_c[i]
        delta_min=w_min - w_c[i]
    
        beta=np.sqrt((0.5*vol)**3)

        if np.fabs(delta) < 1e-8:
            psi_ij = 1.
        else:
            if delta > 0.:
                limiter=(((delta_max**2 +beta**2)*delta+2*(delta**2)*(delta_max**2))/ (delta_max**2 +2*delta**2+delta*delta_max+beta**2))*1/delta
                psi_ij  =min(1,limiter)
                
            elif delta < 0.:
                limiter=(((delta_min**2 +beta**2)*delta+2*(delta**2)*(delta_min**2))/ (delta_min**2 +2*delta**2+delta*delta_min+beta**2))*1/delta
                
                psi_ij  =min(1,limiter)
         
        psi=min(psi, psi_ij)
    
        
    w_cr=w_c[i] + psi*w_x[i]*r[0] + psi*w_y[i]*r[1]
    return w_cr



########minmod limiter using faces
# @njit   
# def minmodlimiter_2d(w_c:'float', w_x:'float[:]', w_y:'float[:]',faceid:'int32[:,:]',  cellfid:'int32[:,:]' , vect:'float[:]',i:"int32", halofid:'int32[:]',innerfaces:'int32[:]' ,boundaryfaces:'int32[:]', halofaces: 'int32[:]'):
#         epsilon=1e-12
#         min_sgnx=1
#         max_sgnx=-1
#         min_x=np.abs(w_x[i])
#         min_sgny=1
#         max_sgny=-1
#         min_y=np.abs(w_y[i])
#         for j in range(faceid[i][-1]):
#             face = faceid[i][j]
#             if face in innerfaces:
#               left=cellfid[face][0]
#               right=cellfid[face][1]
#             elif face in boundaryfaces :
#               left=cellfid[face][0]
#               right= face
#             elif face in halofaces:
#               left= cellfid[face][0]
#               right=halofid[face]
               
#             if left==i:
#                 k=right
#             else:
#                 k=left
            
#             if np.abs(w_x[k]) < epsilon:
#                 sgnx = 0.
#             else:
#                 sgnx =w_x[k]/np.abs(w_x[k])
#             if sgnx<=min_sgnx:
#                 min_sgnx=sgnx
#             if sgnx>= max_sgnx:
#                 max_sgnx=sgnx
#             if min_x>=np.abs(w_x[k]):
#                 min_x=np.abs(w_x[k])
            
#             if np.abs(w_y[k]) < epsilon:
#                 sgny = 0.
#             else:
#                 sgny=w_y[k]/np.abs(w_y[k])
#             if sgny<=min_sgny:
#                 min_sgny=sgny
#             if sgny>= max_sgny:
#                 max_sgny=sgny
#             if min_y>=np.abs(w_y[k]):
#                 min_y=np.abs(w_y[k])

      
#         psi_x=0.5*(min_sgnx+max_sgnx)*min_x
#         psiy=0.5*(min_sgny+max_sgny)*min_y
#         w_limiter=w_c +  (psi_x* vect[0] + psiy* vect[1])
#         return w_limiter
    

# @njit   
# def minmodlimiter_2d(w_c:'float', w_x:'float[:]', w_y:'float[:]',nodecid:'int32[:]',  cellnid:'int32[:,:]' , vect:'float[:]',i:"int32"):
    
#     nbcells=len(w_x)
#     vois=[]
#     for j in nodecid: 
#         for v in cellnid[j]:
#             # if i!=v and v!=-1 and v!=1 and v!=0 :
#             if i!=v:
#                 if v not in vois:
#                    vois.append(v)

#     # print("les voisinages de ", i)
#     # print("sont:", vois)
#     minwx=w_x[vois[0]]
#     maxwx=w_x[vois[0]]
#     minabswx=np.abs(w_x[vois[0]])

#     minwy=w_y[vois[0]]
#     maxwy=w_y[vois[0]]
#     minabswy=np.abs(w_y[vois[0]])

#     for k in vois:
#       if np.abs(w_x[k])<minabswx:
#         minabswx=np.abs(w_x[k])
#       if w_x[k]<minwx:
#         minwx=w_x[k]
#       if w_x[k]>=maxwx:
#         maxwx=w_x[k]

#       if np.abs(w_y[k])<minabswy:
#         minabswy=np.abs(w_y[k])
#       if w_y[k]<minwy:
#         minwy=w_y[k]
#       if w_y[k]>=maxwy:
#         maxwy=w_y[k]
    

#     # sgn_minwx=np.sign(minwx)
#     # sgn_maxwx=np.sign(maxwx)
#     # sgn_minwy=np.sign(minwy)
#     # sgn_maxwy=np.sign(maxwy)
    
#     # print("minwx:",minwx, sgn(minwx))
    
#     sgn_minwx=sgn(minwx)
#     sgn_maxwx=sgn(maxwx)
#     sgn_minwy=sgn(minwy)
#     sgn_maxwy=sgn(maxwy)

#     psi_x=0.5*(sgn_minwx+sgn_maxwx)*minabswx
#     psi_y=0.5*(sgn_minwy+sgn_maxwy)*minabswy
#     # if psi_x != 0:
#     #   # print("psi_x", psi_x)
      
#     w_limiter=w_c +  (psi_x* vect[0] + psi_y* vect[1])
#     return w_limiter


@njit
def minmodlimiter_2d(w_c:'float', w_x:'float[:]', w_y:'float[:]',faceid:'int32[:,:]',  cellfid:'int32[:,:]' , vect:'float[:]',i:"int32", halofid:'int32[:]',innerfaces:'int32[:]' ,boundaryfaces:'int32[:]', halofaces: 'int32[:]'):
    
    nbcells=len(w_x)
    vois=[]
    for j in range(faceid[i][-1]):
      face = faceid[i][j]
      if face in innerfaces:
        left=cellfid[face][0]
        right=cellfid[face][1]
      elif face in boundaryfaces :
        left=cellfid[face][0]
        right= face
      elif face in halofaces:
        left= cellfid[face][0]
        right=halofid[face]
          

      if left==i:
          v=right
      else:
          v=left
      vois.append(v)

    # print("les voisinages de ", i)
    # print("sont:", vois)
    minwx=w_x[vois[0]]
    maxwx=w_x[vois[0]]
    minabswx=np.abs(w_x[vois[0]])

    minwy=w_y[vois[0]]
    maxwy=w_y[vois[0]]
    minabswy=np.abs(w_y[vois[0]])

    for k in vois:
      if np.abs(w_x[k])<minabswx:
        minabswx=np.abs(w_x[k])
      if w_x[k]<minwx:
        minwx=w_x[k]
      if w_x[k]>=maxwx:
        maxwx=w_x[k]

      if np.abs(w_y[k])<minabswy:
        minabswy=np.abs(w_y[k])
      if w_y[k]<minwy:
        minwy=w_y[k]
      if w_y[k]>=maxwy:
        maxwy=w_y[k]
    

    # sgn_minwx=np.sign(minwx)
    # sgn_maxwx=np.sign(maxwx)
    # sgn_minwy=np.sign(minwy)
    # sgn_maxwy=np.sign(maxwy)
    
    # print("minwx:",minwx, sgn(minwx))
    
    sgn_minwx=sgn(minwx)
    sgn_maxwx=sgn(maxwx)
    sgn_minwy=sgn(minwy)
    sgn_maxwy=sgn(maxwy)

    psi_x=0.5*(sgn_minwx+sgn_maxwx)*minabswx
    psi_y=0.5*(sgn_minwy+sgn_maxwy)*minabswy
    # if psi_x != 0:
    #   # print("psi_x", psi_x)
      
    w_limiter=w_c +  (psi_x* vect[0] + psi_y* vect[1])
    return w_limiter
      
@njit
def sgn(x:'float'):
    epsilon=1e-6
    if np.abs(x) < epsilon:
            sgnx = 0.
    else:
            sgnx =x/np.abs(x)
    return sgnx


       

       

       
    
       
    

    

