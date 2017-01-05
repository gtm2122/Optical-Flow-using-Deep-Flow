from deepmatching import deepmatching as dm
import matplotlib.pyplot as plt
import numpy as np
import math
from colorsys import hsv_to_rgb
from PIL import Image as im
from scipy import signal as sc
import scipy
from sor import *



def combine_img(im1,im2):                                                    
 images = map(im.open, [im1,im2])                                          
 widths, heights = zip(*(i.size for i in images))                           
 total_width = sum(widths)                                                
 max_height = max(heights)                                                   
 new_im = im.new('RGB', (total_width, max_height))                          
 x_offset = 0                                                                
 for img in images:                                                     
  new_im.paste(img, (x_offset,0))                                     
  x_offset += img.size[0]                                                
 new_im.save(im1+im2)          

## Testing the dm algorithm

'''

count = 0
fig1 = plt.figure()
fig2 = plt.figure()

ax1 = fig1.add_subplot(1,1,1)
ax2 = fig2.add_subplot(1,1,1)

im1 = np.array(im.open('liberty1.png'))
im2 = np.array(im.open('liberty2.png'))

colors = ['red','green','blue','yellow']
#print ind
#exit()
f = dm(im1,im2)[:,0:2]
s = dm(im1,im2)[:,2:4]

for i in f:
 ax1.imshow(im1)
 #print ind[count]
 a = count%3
 b= colors[a] 
 #print a
 
 #exit()
 ax1.scatter(i[0],i[1],color = b)
 ax2.imshow(im2)
 ax2.scatter(s[count,0],s[count,1],color =b)
 
 count+=1
  
fig1.savefig('one.png')
fig2.savefig('two.png')
combine_img('one.png','two.png')

exit()
'''



 
 
im1 = np.array(im.open('liberty1.png'))
im2 = np.array(im.open('liberty2.png'))
#print np.array(np.array(im1.shape).shape)
#exit()

#print sc.convolve2d(im1[:,:,1],x,mode='same').shape
#exit()
def find_ders(im1,im2): 
 x = (np.array([[-1.,1.],[-1.,1.]]))
 #x = np.array([[0,0,0,0,0],[-1.,8.,0.,-8.,1.],[0,0,0,0,0]])/12.0
 #x = np.array([[-1.0,0,1.0],[-2.,0.,2.],[-1.,0,1.]])

 t1 = (-np.ones((2,2)))
 t2 = -t1
#im1 = np.zeros((3,3)) + 1 * np.eye(3)
#im2 = np.ones((3,3))  - np.eye(3)
 dx = (np.zeros_like(im1))
 dy = (np.zeros_like(im1))
 dt = (np.zeros_like(im1))

 dxx = (np.zeros_like(im1))
 dxy = (np.zeros_like(im1))
 dxt = (np.zeros_like(im1))

 dyy = (np.zeros_like(im1))
 dyx = (np.zeros_like(im1))
 dyt = (np.zeros_like(im1))


 for i in range(0,3):
  # For getting to J0
  dx[:,:,i] = sc.convolve2d(im1[:,:,i],x,mode='same')+sc.convolve2d(im2[:,:,i],x,mode='same')
  dy[:,:,i] = sc.convolve2d(im1[:,:,i],x.T,mode='same')+sc.convolve2d(im2[:,:,i],x.T,mode='same')
  dt[:,:,i] = sc.convolve2d(im1[:,:,i],t1,mode='same')+sc.convolve2d(im2[:,:,i],t2,mode='same')

  # For getting to Jxy

  dxx[:,:,i] = sc.convolve2d(dx[:,:,i],x,mode='same')+sc.convolve2d(dx[:,:,i],x,mode='same')
  dxy[:,:,i] = sc.convolve2d(dx[:,:,i],x.T,mode='same')+sc.convolve2d(dx[:,:,i],x.T,mode='same')
  dxt1 = sc.convolve2d(sc.convolve2d(im1[:,:,i],x,mode='same'),t1,mode='same')
  dxt2 = sc.convolve2d(sc.convolve2d(im2[:,:,i],x,mode='same'),t2,mode='same')
  dxt[:,:,i] = dxt1+dxt2
  
  dyy[:,:,i] = sc.convolve2d(dy[:,:,i],x,mode='same')+sc.convolve2d(dy[:,:,i],x,mode='same')
  dyx[:,:,i] = sc.convolve2d(dy[:,:,i],x.T,mode='same')+sc.convolve2d(dy[:,:,i],x.T,mode='same')
  dyt1 = sc.convolve2d(sc.convolve2d(im1[:,:,i],x.T,mode='same'),t1,mode='same')
  dyt2 = sc.convolve2d(sc.convolve2d(im2[:,:,i],x.T,mode='same'),t2,mode='same')
 
  dyt[:,:,i] = dyt1+dyt2
   
  
 return dx,dy,dt,dxx,dxy,dxt,dyy,dyx,dyt  
 


#print dx[0,0,:]

# This function can be used to calculate J0 and Jxy depending on the input

def J0_calc(dx,dy,dt):
 J0 = (np.zeros((3,3)))
 for k in range(0,3):
  a = np.vstack((dx[k],dy[k],dt[k]))
  J0 += (np.matmul(a,a.T))
 return (J0/(dx**2.0+dy**2.0+0.1*0.1))

#im1 = np.array(im.open('liberty1.png'))
#im2 = np.array(im.open('liberty2.png'))



#udash,vdash = wdash(f,s)

def wdash(f,s,R,C):
 #print s.shape
 #print f.shape 
 w_d = np.zeros((R,C,2))
 
 udash = (s-f)[:,0]
 vdash = (s-f)[:,1]
 c = np.zeros((R,C))
 #print s[:,0]
 #print s[:,1]
 w_d[s[:,0].astype(int),s[:,1].astype(int),0]=(s-f)[:,0]
 w_d[s[:,0].astype(int),s[:,1].astype(int),1]=(s-f)[:,1]
 c[s[:,0].astype(int),s[:,1].astype(int)]=1.
 #print np.where(c==1)
 #print w_d[s[:,0].astype(int),s[:,1].astype(int),0]
 #print udash[0]
 #print w_d[s[:,0].astype(int),s[:,1].astype(int),1]
 #print vdash[0]
 #print w_d
 #return
 return c,w_d

def delta_x(im1,im2,wu,wv,x,y,mode=0,dx2=0,dy2=0):
 deltax=0.
 x_t = int(x-wu)
 y_t = int(y-wv)
 #print x_t
 #print y_t
 #print y_t
 #return
 if(mode==0):
  for i in xrange(0,3):
   if(x_t<0 or y_t<0):
    x_tt = np.arange(0,64.0)
    y_tt = np.arange(0,64.0)
    z = scipy.interpolate.interp2d(y_tt,x_tt,im2[:,:,i])
    deltax += np.abs(im1[x,y,i]-z(y_t,x_t))
    #exit()
   else:
    #print x,y,i
    #print x_t,y_t,i
    deltax += (np.abs(float(im1[x,y,i])-float(im2[x_t,y_t,i])))
 else:
  dx1=im1
  dy1=im2
  for i in xrange(0,3):
   if(x_t<0 or y_t<0):
    x_tt = np.arange(0,64)
    y_tt = np.arange(0,64)
    z = scipy.interpolate.interp2d(y_tt,x_tt,dx2[:,:,i])
    z2 = scipy.interpolate.interp2d(y_tt,x_tt,dy2[:,:,i])

    deltax += np.sqrt(((dx1[x,y,i]-z(y_t,x_t)))**2.0+(np.abs(im2[x,y,i]-z2(y_t,x_t)))**2.0)
   else:
    deltax +=( np.sqrt(((dx1[x,y,i]-dx2[y_t,x_t,i]))**2.0+(np.abs(dy1[x,y,i]-dy2[y_t,x_t,i]))**2.0))  
 #print deltax
 #return
 #print deltax

 return (np.abs(deltax))



def get_alpha(del_I2,x,y):
 return (np.exp((-5.0*del_I2[x,y])))


def get_pyramid(im1,im2,list_coord):
 im1_temp = (scipy.ndimage.filters.gaussian_filter(im1,sigma=0.5))
 im2_temp = (scipy.ndimage.filters.gaussian_filter(im2,sigma=0.5))
 #temp_pyr = np.zeros((im1_temp.shape[0],im1_temp.shape[1],3,2))
 #temp_pyr[:,:,:,0]=im1_temp
 #temp_pyr[:,:,:,1]=im2_temp
 pyr = {}
 eta = 0.95
 for i in range(0,26):
  
  if i not in pyr:
   pyr[i]={}
   
  a=np.array(scipy.misc.imresize(im1_temp,(list_coord[25-i])))
  #print a.shape
  pyr[i][0]=a
  a=np.array(scipy.misc.imresize(im2_temp,(list_coord[25-i])))
  #print a.shape
  pyr[i][1]=a
  #print pyr[i][0].shape
 return pyr
##### MAIN STARTS HERE

def sidash(x):
 return 0.5/np.sqrt(0.001**2 + x**2)

#print coord_list
#exit()
 
def get_lambda(im1,im2): 
 #print w.shape
 #exit()
 #exit()
 #print w.shape
 #exit()
 dx,dy,dt,dxx,dxy,dxt,dyy,dyx,dyt=find_ders(im1,im2)
 dy1=np.zeros_like(dx)
 dx1=np.zeros_like(dy1)
 dy2=np.zeros_like(dy1)
 dx2=np.zeros_like(dy1)

 ## Getting Autocorrelation matrix 

 ac_dx=np.zeros((im1.shape[0],im1.shape[1]))
 ac_dxy=np.zeros_like(ac_dx)
 ac_dy=np.zeros_like(ac_dx)
 for i in xrange(0,3):
  dy1[:,:,i],dx1[:,:,i] = np.gradient(im1[:,:,i])
  dy2[:,:,i],dx2[:,:,i] = np.gradient(im2[:,:,i])
  ac_dx+=(dx1[:,:,i]**2.0)
  ac_dxy+=(dx1[:,:,i]*dy1[:,:,i])
  ac_dy+=(dy1[:,:,i]**2.0)
 #print ac_dy
 #exit()
 t1= (0.5*ac_dy+0.5*ac_dx)
 
 t2= (t1**2.0) + (ac_dxy**2.0) -(ac_dx*ac_dy)
 t2[t2<0]=(0.0)
 #print np.sqrt(t2)
 #print t1
 #print np.allclose(t1,np.sqrt(t2))
 t_inter = t1 - (np.sqrt(t2))
 t_inter[t_inter<=0.0]=(0.0) 
 lam2 = (np.sqrt(np.abs(10*t_inter)))
 #print lam2.shape
 #lam2[lam2==0]=np.inf
 
 #print lam2
 #lam = np.sqrt(np.min(np.abs(lam2),axis=0))
 #print lam2
 return (lam2) 
 #exit()  
#print eigs[eigs>0]
#exit()

def get_deltax(im1,im2,w_precomp):

 deltax = (np.zeros((im1.shape[0],im1.shape[1])))
 #x = np.array([[0,0,0,0,0],[-1.,8.,0.,-8.,1.],[0,0,0,0,0]])/12.0
 #x = np.array([[-1.0,0,1.0],[-2.,0.,2.],[-1.,0,1.]])
 x = np.array([[-1.,1],[-1.,1.]])
 dy1_delta =(np.zeros((im1.shape[0],im1.shape[1],im1.shape[2])))
 dx1_delta = np.zeros_like(dy1_delta)
 dy2_delta = np.zeros_like(dy1_delta)
 dx2_delta = np.zeros_like(dy1_delta)
 for k in range(0,3):
  dx1_delta[:,:,k] = sc.convolve2d(im1[:,:,k],x,mode='same')
  dy1_delta[:,:,k] = sc.convolve2d(im1[:,:,k],x.T,mode='same')
  dx2_delta[:,:,k] = sc.convolve2d(im2[:,:,k],x,mode='same')
  dy2_delta[:,:,k] = sc.convolve2d(im2[:,:,k],x.T,mode='same')

  #dy1_delta[:,:,k],dx1_delta[:,:,k] = (np.gradient(im1[:,:,k]))
 
  #dy2_delta[:,:,k],dx2_delta[:,:,k] = (np.gradient(im2[:,:,k]))
 
 for i in range(im1.shape[0]):
  for j in range(im2.shape[0]):

   deltax[i,j] =  delta_x(im1,im2,w_precomp[i,j,0],w_precomp[i,j,1],i,j)
   deltax[i,j] += delta_x(dx1_delta,dy1_delta,w_precomp[i,j,0],w_precomp[i,j,1],i,j,mode=1,dx2=dx2_delta,dy2=dy2_delta)
 return (deltax)
  
def EM_const(eigs,deltax):
 return (eigs/(np.sqrt(2.0*np.pi)*50.0))*np.exp(-deltax/(100.0))
#phi = EM_const(eigs,deltax)


def ES_const(im1,im2):
 x = np.longdouble(np.array([[-1.,1.],[-1.,1.]]))
 #x = np.longdouble(np.array([-1,8,0,-8,1]))/12.0
 #dx = (np.zeros_like(im1))
 #x = np.array([[0,0,0,0,0],[-1.,8.,0.,-8.,1.],[0,0,0,0,0]])/12.0
 #x = np.array([[-1.0,0,1.0],[-2.,0.,2.],[-1.,0,1.]])

 #dy = np.zeros_like(dx)
 
 #for i in range(0,3):
  #dx[:,:,i],dy[:,:,i] = np.gradient(im1[:,:,i])
  
 
  #dx[:,:,i] = (sc.convolve2d(im1[:,:,0],x,mode='same')+sc.convolve2d(im2[:,:,0],x,mode='same'))
  #dy[:,:,i] = (sc.convolve2d(im1[:,:,0],x.T,mode='same')+sc.convolve2d(im2[:,:,0],x.T,mode='same'))
 interm1 = (1/255.0)*(0.2989*im1[:,:,0] + 0.587*im1[:,:,1] + 0.1140*im1[:,:,2])
 interm2 = (1/255.0)*(0.2989*im2[:,:,0] + 0.587*im2[:,:,1] + 0.1140*im2[:,:,2])
 
 dx = (sc.convolve2d(interm1,x,mode='same')+sc.convolve2d(interm2,x,mode='same'))
 dy = (sc.convolve2d(interm1,x.T,mode='same')+sc.convolve2d(interm2,x.T,mode='same'))
 
 #0.2989 * R + 0.5870 * G + 0.1140 * B
 #ax = (dx[:,:,0]**2.0 + dx[:,:,1]**2.0+ dx[:,:,2]**2.0)
 #ay = (dy[:,:,0]**2.0 + dy[:,:,1]**2.0 + dy[:,:,2]**2.0)
 #print np.exp(-5*np.sqrt(ax+ay))
 #exp(5*del2 of I1 and I2)
 return np.longdouble(np.exp(np.longdouble(-5.0*np.sqrt(dx**2+dy**2))))
 #return 12*np.exp(-(np.sqrt(ax+ay))**0.8)
#alpha = ED_const(im1,im2)
#print alpha.shape
#print EM_constant.shape
#print EM_constant[EM_constant>0]
#exit()
#print pyr[0][1].shape
def alg(pyramid,b,u0,v0):
 u_new = (np.zeros((pyramid[25][1].shape[0],pyramid[25][1].shape[1])))
 v_new = np.zeros_like(u_new)
 counter = -1
 for k in range(25,-1,-1):
  counter+=1
  #pyramid[0][1].shape
  #exit()
  #if(k==0):
   #beta = 0.0
  beta = (300.0)*(k/25.0)**0.6
  im1=pyramid[k][0]
  im2=pyramid[k][1]
  matches = dm(im1,im2)
  f = matches[:,0:2].astype(int)
  s = matches[:,2:4].astype(int)
  c_m,w_pr = wdash(f,s,im1.shape[0],im1.shape[1])
  u_new = scipy.misc.imresize(u_new,tuple(im1.shape))
  #print u.shape
  #exit()
  print k
  v_new = scipy.misc.imresize(v_new,tuple(u_new.shape))
  #print v_new.shape
  deltax = get_deltax(im1,im2,w_pr)
  eigs = get_lambda(im1,im2)
  p = EM_const(eigs,deltax) 
  #p = scipy.misc.imresize(p,tuple(u_new.shape))
  J0_l = np.zeros((3,3,im1.shape[0],im2.shape[1]))
  Jxy_l = np.zeros_like(J0_l)
  a = ES_const(im1,im2)
  #print a[a==0].shape
  #exit()
  #c_m = scipy.misc.imresize(c,tuple(im1.shape))
  u0 = scipy.misc.imresize(u0,tuple(im1.shape))
  v0 = scipy.misc.imresize(v0,tuple(im1.shape))
  ## HERE
  dx,dy,dt,dxx,dxy,dxt,dyy,dyx,dyt = find_ders(im1,im2)  
  #print "p=",p.shape
  for i in range(0,im1.shape[0]):
   for j in range(0,im1.shape[1]):
    J0_l[:,:,i,j] = J0_calc(dx[i,j,:],dy[i,j,:],dt[i,j,:])
    Jxy_l[:,:,i,j] = J0_calc(dxx[i,j,:],dxy[i,j,:],dxt[i,j,:])+J0_calc(dyy[i,j,:],dyx[i,j,:],dyt[i,j,:])
  u_new,v_new = sor_25(p,J0_l,Jxy_l,c_m,a,b,u_new,v_new,w_pr[:,:,0],w_pr[:,:,1],u0,v0) 
  #print u_new[-1,-1],v_new[-1,-1] #u[u==np.nan]
  #print v_new[-1,-1] #v[u==np.nan]
  #exit()
  print "coarse to fine ",100*counter/25," done"
 return u_new,v_new
def deepflow(im1,im2,u_0,v_0):
 #im1 = np.array(im.open('liberty1.png'))
 #im2 = np.array(im.open('liberty2.png'))


 eigs = get_lambda(im1,im2)
 beta = 300.0
 u = np.zeros((im1.shape[0],im2.shape[1]))
 v = np.zeros_like(u)
 #u_0 = np.zeros_like(u)
 #v_0 = np.zeros_like(u) 
 matches = dm(im1,im2)
 f = matches[:,0:2].astype(int)
 s = matches[:,2:4].astype(int)
 
 c,w_precomp = (wdash(f,s,im1.shape[0],im1.shape[1]))
 coord_list = []

 mult=0.95**25.0

 for i in xrange(25,-1,-1):
  if i == 0:
   #print "here"
   #print im1.shape
   coord_list.append(tuple([im1.shape[0],im1.shape[1]]))
  else:
   coord_list.append(tuple([int(math.floor(mult*im1.shape[0])),int(math.floor(mult*im1.shape[1]))]))
  mult/=0.95
 #print coord_list
 #exit()

 pyr=get_pyramid(im1,im2,coord_list)
 u,v = alg(pyr,beta,u_0,v_0)
 return u,v



