import pickle
from project import *
from PIL import Image as im
from glob import glob
from scipy.misc import imresize
from time import time
import matplotlib.pyplot as plt
from colorsys import hsv_to_rgb
filelist = sorted(glob('sintelmove/0000441[0-9].png'))
print filelist

imgs = np.zeros((len(filelist),128,128,3),dtype='int')
print len(filelist)
count=0

for i in filelist:
 xyz = im.open(i).resize((128,128),im.ANTIALIAS)
 xyz.save('source_'+str(count)+'.png')
 count+=1
count = 0

for i in range(0,len(filelist)):
 print i
 imgs[i,:,:,:]=im.open('source_'+str(i)+'.png')
 count+=1
 #plt.close()
#print imgs.shape
u0 = np.zeros((imgs.shape[0],imgs.shape[1]))
v0 = np.zeros_like(u0)
#print filelist

for num in range(0,imgs.shape[0]-1):
 
 print "pair number ",num
 f1 = imgs[num,:,:,:]
 f2 = imgs[num+1,:,:,:]
 #fig1 = plt.figure()
 #plt.imshow(f1)
 #fig1.savefig('source'+str(num)+'.png')
 #fig2 = plt.figure()
 #plt.imshow(f1)
 #fig2.savefig('source'+str(num+1)+'.png')
 #plt.close(fig1)
 #plt.close(fig2)
 #del fig1
 #del fig2
 t1=time()
 u,v = deepflow(f1,f2,u0,v0)
 u0=u
 v0=v
 print time()-t1
 #u/=max(u.max(),v.max())
 #v/=max(u.max(),v.max())
 w_mag = np.sqrt((u**2.0 + v**2.0))
 #w_mag/=w_mag.mean()
 w_angle = np.arctan2(-v,-u)/np.pi
 
 w_angle[w_angle<0]+=360
 
 np.savetxt('angle_'+str(num)+'.txt',w_angle,delimiter=' ')
 np.savetxt('mag_'+str(num)+'.txt',w_mag,delimiter=' ')

def getcolorwheel():
 RY = 15
 YG = 6
 GC = 4
 CB = 11
 BM = 13
 MR = 6
 ncols = RY+YG+GC+CB+BM+MR
 colorwheel = np.zeros((ncols*10000,3))
 k = 0
 for i in range(0,RY): 
  setcols(colorwheel,255,255*i/RY,0,k)
  k+=1 
 for i in range(0,YG): 
  setcols(colorwheel,255-255*i/YG,255,0,k)
  k+=1 
 for i in range(0,GC): 
  setcols(colorwheel,0,255,255.0*i/GC,k)
  k+=1 
 for i in range(0,CB): 
  setcols(colorwheel,0,255-255*i/CB,255,k)
  k+=1 
 for i in range(0,BM): 
  setcols(colorwheel,255*i/BM,0,255,k)
  k+=1 
 for i in range(0,MR): 
  setcols(colorwheel,255,0,255-255*i/MR,k)
  k+=1 
 return colorwheel

def setcols(colorwheel,r,g,b,pos):
 colorwheel[pos][0]=r
 colorwheel[pos][1]=g
 colorwheel[pos][2]=b

def convert_to_color(colorwheel,w_mag,w_angle):
 rad = w_mag
 a = angle 
 fk = (a+1.0)/2.0 * (54.0)
 k0 = fk.astype(int)
 k1 = (k0+1)%55
 f = fk - k0.astype(float)
 pix = np.zeros((w_mag.shape[0],w_mag.shape[1],3))
 col0 = np.zeros_like(fk)
 col1 = np.zeros_like(fk)
 col = np.zeros_like(fk)
 for i in range(0,rad.shape[0]):
  for j in range(0,rad.shape[1]):

   for k in range(0,3):
    col0[i,j] = colorwheel[k0[i,j]][k]/255.0
    col1[i,j] = colorwheel[k1[i,j]][k]/255.0
    col[i,j] = (1-f[i,j])*col0[i,j] + f[i,j]*col1[i,j]
    if(rad[i,j] <=1.0):
     col[i,j] = 1.0 - rad[i,j]*(1.0-col[i,j])
    else:
     col[i,j] *=0.75
    pix[i,j,2-k] = np.int(col[i,j]*255.0)
 return pix

colorwheel = getcolorwheel()

for k in range(0,imgs.shape[0]-1):
 angle = np.genfromtxt('angle_'+str(k)+'.txt',delimiter=' ')
 mag = np.genfromtxt('mag_'+str(k)+'.txt',delimiter=' ')

 pix = convert_to_color(colorwheel,angle,mag)
 matrix = np.zeros((128,128,3))
 figure1 = plt.figure()
 plt.axis('off')
 plt.imshow(pix)
 figure1.savefig('pix_'+str(k)+'.png')
 plt.close() 
 testimg = im.open('pix_'+str(k)+'.png').resize((128,128),im.ANTIALIAS)

 im1 = im.open('source_'+str(k)+'.png').convert(testimg.mode)
 im2 = im.open('source_'+str(k+1)+'.png').convert(testimg.mode)
 overlay1 =im.blend(im1,im2,0.7)
 overlay4 = im.blend(overlay1,testimg,0.5)

 overlay4.save("pixtest_"+str(k)+".png")
 #print "here"

