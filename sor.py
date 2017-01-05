#from  import 
import numpy as np
import scipy.signal as sc
def sid_m(u_cur,v_cur,u_d,v_d):
 #print u_cur
 #print v_cur
 #print (u_cur-u_d)**2[u_d<>0]
 #print (v_cur-v_d)[v_d<>0]
 a= np.longdouble(2.0*np.sqrt((u_cur-u_d)**2.0+(v_cur-v_d)**2.0+0.001**2.0))
 #print np.where(a==0)
 #print a
 #a=1.0/a

 #print 1/a
 #print a[np.where(a==np.nan)]
 #exit()
 #print "sid_m"
 return 1/a #1/(2*np.sqrt((u_cur-u_d)**2-(v_cur-v_d)**2+0.001**2))

def sid_s(u_cur,v_cur,u_0,v_0):
 x = np.longdouble(np.array([[-1.0,1.0],[-1.,1.]]))
 #x = np.longdouble(np.array([-1.,8.,0.,-8.,1.]))/12.0
 #x = np.array([[0,0,0,0,0],[-1.,8.,0.,-8.,1.],[0,0,0,0,0]])/12.0
 t1 = np.longdouble(-np.ones((2.0,2.0)))
 t2 = -t1

 ux = sc.convolve2d(u_0,x,mode='same')+sc.convolve2d(u_cur,x,mode='same')
 uy =  sc.convolve2d(u_0,x.T,mode='same')+sc.convolve2d(u_cur,x.T,mode='same')
 ut = sc.convolve2d(u_0,t1,mode='same')+sc.convolve2d(u_cur,t2,mode='same')
 #print ux[ux==0]
 #print uy[uy==0]
 #print ut[ut==0]
 vx = sc.convolve2d(v_0,x,mode='same')+sc.convolve2d(v_cur,x,mode='same')
 vy = sc.convolve2d(v_0,x.T,mode='same')+sc.convolve2d(v_cur,x.T,mode='same')
 vt = sc.convolve2d(v_0,t1,mode='same')+sc.convolve2d(v_cur,t2,mode='same')
 #print vx[vx==0]
 #print vy[vy==0]
 #print vt[vt==0]
 #print "sid_s"
 #ut=0
 #vt=0
 a = np.longdouble(2.0*np.sqrt(ux**2.0+uy**2.0+ut**2.0 + vx**2.0 + vy**2.0 + vt**2.0 + 0.001**2.0))
 #print np.where(a==0)
 return 1/a

def sid_data(J,u,v):
 surd = np.longdouble(J[0,0,:,:]*u**2.0 + 2.0*J[0,1,:,:]*v*u+ 2.0*J[0,2,:,:]*u + J[1,1,:,:]*v**2.0 + J[2,2,:,:] + 2.0*J[1,2,:,:]*v)
 a = np.longdouble(2.0*np.sqrt(surd**2.0 + 0.001**2.0))
 #print "sid_data"
 #print a[a==np.nan]
 #print np.where(a==0)
 return 1/a


def sor_25(phi,J0,Jxy,c,alpha,beta,uc,vc,ud,vd,up,vp):
 #alpha=12
 w = np.longdouble(1.6)
 avg_kernel = np.longdouble(np.array([[0,1,0],[1,0,1],[0,1,0]]))
 du = np.longdouble(np.zeros_like(uc))
 dv = np.longdouble(np.zeros_like(vc))
 delta =0.0
 gamma = 0.8
 for k in range(0,5):
  #alpha = 12.0
  #print (u[u==np.nan])
  #print "ITERATION ",k
  si_od = np.longdouble(sid_data(J0,uc,vc))
  si_xy = np.longdouble(sid_data(Jxy,uc,vc))
  si_m = np.longdouble(sid_m(uc,vc,ud,vd))
  #si_m=0
  si_s = np.longdouble(sid_s(uc,vc,up,vp))
  #si_m=0
  #print np.allclose(np.nan,np.nan)
  
  '''
  print "si_od=",si_od.shape
  print "si_xy=",si_xy.shape
  print "J0=",J0.shape,Jxy.shape
  print "si_m",si_m.shape
  print "c=",c.shape
  print "phi=",phi.shape
  print "beta=",beta
  print "alpha=",alpha.shape
  print "si_s=",si_s.shape
  ''' 
  ubar = np.longdouble(sc.convolve2d(uc, avg_kernel, boundary='wrap', mode='same')/avg_kernel.sum())
  vbar = np.longdouble(sc.convolve2d(vc, avg_kernel, boundary='wrap', mode='same')/avg_kernel.sum())
  #print np.allclose(si_s[:,:],np.zeros_like(J0[0,0,:,:]))
  #print np.allclose(c[:,:],np.zeros_like(Jxy[0,0,:,:]))
  #exit()
  A11 = np.longdouble(np.longdouble(delta*si_od*2.0*J0[0,0,:,:]) + np.longdouble(gamma*si_xy*2.0*Jxy[0,0,:,:]) + np.longdouble(2.0*si_m*c*phi*beta+np.longdouble(alpha*si_s)))
  #print A11
  #exit()
  #print A11[A11==0]
  #exit()
  #print A11[A11==0.]
  
  A12 = np.longdouble(np.longdouble(delta*si_od*J0[1,0,:,:])+np.longdouble(delta*si_od*J0[0,1,:,:])+np.longdouble(gamma*si_xy*(Jxy[1,0,:,:]+Jxy[0,1,:,:])))
  #print A12[A12==0.]
  A22 = np.longdouble(delta*si_od*2.0*J0[1,1,:,:] + np.longdouble(gamma*si_xy*2.0*Jxy[1,1,:,:]) + np.longdouble(2.0*si_m*c*phi*beta)+np.longdouble(alpha*si_s))
  #print A22[A22==0.]
  #exit()
  b_inter1 = np.longdouble(-delta*si_od*(2.0*J0[0,0,:,:]*uc + np.longdouble(J0[2,0,:,:]) + np.longdouble(J0[0,2,:,:]) + np.longdouble((J0[1,0,:,:] + J0[0,1,:,:]))*vc))
  b_inter2 = np.longdouble(-gamma*si_xy*(2.0*Jxy[0,0,:,:]*uc + Jxy[2,0,:,:] + Jxy[0,2,:,:] + (Jxy[1,0,:,:] + Jxy[0,1,:,:])*vc))
  b_inter3 = np.longdouble(-2.0*si_m*np.longdouble(c*phi)*beta*(uc-ud) + np.longdouble(alpha*si_s*(ubar-uc)))
  B1 = np.longdouble(b_inter1 + b_inter2 + b_inter3)

  b2_inter1 = np.longdouble(-np.longdouble(delta*si_od)*(2*np.longdouble(J0[1,1,:,:]*vc) + np.longdouble(J0[1,2,:,:]) + np.longdouble(J0[2,1,:,:]) + np.longdouble((J0[1,0,:,:] + J0[0,1,:,:])*uc)))
  b2_inter2 = np.longdouble(-gamma*si_xy*(2*Jxy[1,1,:,:]*vc + np.longdouble(Jxy[1,2,:,:]) + np.longdouble(Jxy[2,1,:,:]) + np.longdouble((Jxy[1,0,:,:] + Jxy[0,1,:,:]))*uc))
  b2_inter3 = np.longdouble(-2.0*si_m*np.longdouble(c*phi)*beta*(vc-vd) + np.longdouble(alpha*si_s)*(vbar-vc))
  B2 = np.longdouble(b2_inter1 + b2_inter2 + b2_inter3)
  #print si_xy
  
  det = np.longdouble((A11*A22-A12*A12)) #.astype(float)
  #print len(det[det==0.])
  #det[det==0]=np.longdouble(det.min())
  #exit()
  du = np.longdouble((1-w)*du + w*(B1-A12*dv)/np.longdouble(A11)) 
  dv = np.longdouble((1-w)*dv + w*(B2-A12*du)/np.longdouble(A22))
  #print du
  #print dv
  uc = np.longdouble(np.longdouble(uc) + du)
  vc = np.longdouble(np.longdouble(vc) + dv)
 return uc,vc
