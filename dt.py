import ctypes
import numpy
from numpy import ctypeslib
from ctypes import c_int,c_double,c_float

ctypes.cdll.LoadLibrary("./libdt.so")
lib= ctypes.CDLL("libdt.so")
lib.dtpy.argtypes=[
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=2,flags="C_CONTIGUOUS"),#image scores
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=2,flags="C_CONTIGUOUS"),#image dest
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=2,flags="C_CONTIGUOUS"),#image defy
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=2,flags="C_CONTIGUOUS"),#image defx
    c_int #dimy
    ,c_int #dimx
    ,c_float #ay
    ,c_float #ax
    ,c_float #by
    ,c_float #bx
    ]

#void dt2D(ftype *src, ftype *M,ftype *Iy,ftype *Ix, int dimy ,int dimx, ftype ayy, ftype axx, ftype axy, ftype by, ftype bx)
lib.dt2D.argtypes=[
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=2,flags="C_CONTIGUOUS"),#image scores
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=2,flags="C_CONTIGUOUS"),#image dest
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=2,flags="C_CONTIGUOUS"),#image defy
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=2,flags="C_CONTIGUOUS"),#image defx
    c_int #dimy
    ,c_int #dimx
    ,c_float #ayy
    ,c_float #axx
    ,c_float #axy
    ,c_float #by
    ,c_float #bx
    ]


def dt(img,ay,ax,by,bx):
    res=numpy.zeros(img.shape,dtype=numpy.float32)
    dy=numpy.zeros(img.shape,dtype=numpy.float32)
    dx=numpy.zeros(img.shape,dtype=numpy.float32)
    lib.dtpy(img,res,dy,dx,img.shape[0],img.shape[1],ay,ax,by,bx)
    return res,dy,dx

def dt2(img,ay,ax,axy,by,bx):
    res=-1000*numpy.ones(img.shape,dtype=numpy.float32)
    dy=numpy.zeros(img.shape,dtype=numpy.float32)
    dx=numpy.zeros(img.shape,dtype=numpy.float32)
    lib.dt2D(img,res,dy,dx,img.shape[0],img.shape[1],ay,ax,axy,by,bx)
    return res,dy,dx

def mydt(img,myay,myax,myby,mybx):
    ay=myay
    ax=myax
    by=-2*myby*myay
    bx=-2*mybx*myax
    cy=myby**2*myay
    cx=mybx**2*myax
    res=-1000*numpy.ones(img.shape,dtype=numpy.float32)
    dy=numpy.zeros(img.shape,dtype=numpy.float32)
    dx=numpy.zeros(img.shape,dtype=numpy.float32)
    lib.dtpy(img,res,dy,dx,img.shape[0],img.shape[1],ay,ax,by,bx)
    res=res-cy-cx
    return res,dy,dx

from scipy.ndimage.interpolation import rotate
from numpy.linalg import eig
from math import atan2,cos,sin

def dt2rot(img,ay,ax,axy,by,bx):
    szy=img.shape[0]
    szx=img.shape[1]
    axy=-axy
    val,vec=eig([[ax,axy],[axy,ay]])
    rad=atan2(vec[0,1],vec[0,0])
    ang=rad/numpy.pi*180
    img2=rotate(img,ang,mode='nearest')
    dtim,Iy,Ix=mydt(img2,ay/cos(rad),ax/cos(rad),by,bx)
    res=rotate(dtim,-ang,reshape=False,mode='nearest')
    dy=rotate(Iy,-ang,reshape=False,mode='nearest')
    dx=rotate(Ix,-ang,reshape=False,mode='nearest')
    cty=(res.shape[0]-szy)/2
    ctx=(res.shape[1]-szx)/2
    dy=dy[cty:cty+szy,ctx:ctx+szx]#.astype(numpy.int)
    dx=dx[cty:cty+szy,ctx:ctx+szx]#.astype(numpy.int)
    dx=numpy.round(dx*cos(-rad)+dy*sin(-rad)).astype(numpy.int)
    dy=numpy.round(dx*sin(-rad)-dy*cos(-rad)).astype(numpy.int)
    return res[cty:cty+szy,ctx:ctx+szx],dy,dx


if __name__ == "__main__":
    dimy=100
    dimx=100
    #a=numpy.random.random((dimy,dimx)).astype(numpy.float32)
    #im=numpy.zeros((dimy,dimx),numpy.float32)
    im=-numpy.ones((dimy,dimx),numpy.float32)
    im[50,50]=5
    #im[40,50]=1
    #dta=numpy.zeros((dimy,dimx),dtype=numpy.float32)
    #dy=numpy.zeros((dimy,dimx),dtype=numpy.float32)
    #dx=numpy.zeros((dimy,dimx),dtype=numpy.float32)
    #lib.dtpy(a,dta,dy,dx,dimy,dimy,1,1,0,0)
    a=0.01
    b=0#10
    #dtim,Iy,Ix=mydt(im,a,a,b,b)
    dtim,Iy,Ix=dt2(im,a,4*a,-0.00,b,b)
    dtimr,Iyr,Ixr=dt2rot(im,a,4*a,-0.00,b,b)
    import pylab
    pylab.figure(1)
    pylab.clf()
    pylab.imshow(im)#(numpy.concatenate((im,im,im,im,im,im,im,im,im),0))
    pylab.figure(2)
    pylab.clf()
    pylab.imshow(dtim)#(numpy.concatenate((dtim,dtim,dtim,dtim,dtim,dtim,dtim,dtim,dtim),0))
    pylab.figure(3)
    pylab.clf()
    pylab.imshow(Iy)#(numpy.concatenate((Iy,Iy,Iy,Iy,Iy,Iy,Iy,Iy,Iy,Iy,Iy,Iy),0))
    pylab.figure(4)
    pylab.clf()
    pylab.imshow(Ix)#(numpy.concatenate((Ix,Ix,Ix,Ix,Ix,Ix,Ix,Ix,Ix,Ix,Ix,Ix),0))
    pylab.draw()
    pylab.show()
