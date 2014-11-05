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

lib.fdtpy.argtypes=[
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

lib.ffdtpy.argtypes=[
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


#void fdt1D(float *f,float *d,int *p, int n) 
lib.fdt1D.argtypes=[
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=1,flags="C_CONTIGUOUS"),#image scores
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=1,flags="C_CONTIGUOUS"),#image dest
    numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS"),#image defy
    c_int, #n
    c_float ,#a
    c_float #b
    ]


#void dt2D(ftype *src, ftype *M,ftype *Iy,ftype *Ix, int dimy ,int dimx, ftype ayy, ftype axx, ftype axy, ftype by, ftype bx)
lib.dt2D.argtypes=[
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=2,flags="C_CONTIGUOUS"),#image scores
    numpy.ctypeslib.ndpointer(dtype=c_float,ndim=2,flags="C_CONTIGUOUS"),#image dest
    numpy.ctypeslib.ndpointer(dtype=c_int,ndim=2,flags="C_CONTIGUOUS"),#image defy
    numpy.ctypeslib.ndpointer(dtype=c_int,ndim=2,flags="C_CONTIGUOUS"),#image defx
    c_int #dimy
    ,c_int #dimx
    ,c_float #ayy
    ,c_float #axx
    ,c_float #axy
    ,c_float #by
    ,c_float #bx
    ]

#void rotate(ftype imgin,int x,int y,int ang,ftype imgout)
lib.rotatec.argtypes=[
    numpy.ctypeslib.ndpointer(dtype=c_float,flags="C_CONTIGUOUS"),#image source
    c_int, #dimx
    c_int, #dimy
    c_int, #nch
    c_float, #ang
    numpy.ctypeslib.ndpointer(dtype=c_float,flags="C_CONTIGUOUS"),#image dest
    c_int, #ndimx
    c_int, #ndimy
    ]

lib.rotatec_bi.argtypes=[
    numpy.ctypeslib.ndpointer(dtype=c_float,flags="C_CONTIGUOUS"),#image source
    c_int, #dimx
    c_int, #dimy
    c_int, #nch
    c_float, #ang
    numpy.ctypeslib.ndpointer(dtype=c_float,flags="C_CONTIGUOUS"),#image dest
    c_int, #ndimx
    c_int, #ndimy
    ]

#void refine(ftype *img,int sy,int sx,int* defy, int defx,ftype ay,ftype ax,ftype axy,ftype *dst)
lib.refine.argtypes=[
    numpy.ctypeslib.ndpointer(dtype=c_float,flags="C_CONTIGUOUS"),#image source
    c_int, #dimy
    c_int, #dimx
    numpy.ctypeslib.ndpointer(dtype=c_int,flags="C_CONTIGUOUS"),#defy
    numpy.ctypeslib.ndpointer(dtype=c_int,flags="C_CONTIGUOUS"),#defx
    c_float, #ay
    c_float, #ax
    c_float, #axy
    numpy.ctypeslib.ndpointer(dtype=c_float,flags="C_CONTIGUOUS"),#image DT
    ]


def test1d():
    import pylab
    sz=10
    dd=numpy.zeros(sz,dtype=numpy.float32)
    dd[sz/2]=sz
    out=numpy.zeros(sz,dtype=numpy.float32)
    I=numpy.zeros(sz,dtype=numpy.int32)
    lib.fdt1D(dd,out,I,sz)
    pylab.plot(dd)
    pylab.plot(out)
    return out,I

def rotate_fast(imgin,ang):
    dimy=imgin.shape[0]
    dimx=imgin.shape[1]
    if len(imgin.shape)>2:
        nch=imgin.shape[2]
    else:
        nch=1
    rad=ang/180.0*numpy.pi
    newx=int(round(abs(dimx*cos(rad))+abs(dimy*sin(rad))))
    newy=int(round(abs(dimx*sin(rad))+abs(dimy*cos(rad))))
    imgout=numpy.zeros((newy,newx,nch),dtype=numpy.float32)
    lib.rotatec_bi(imgin,dimx,dimy,nch,rad,imgout,newx,newy)
    if len(imgin.shape)<3:
        imgout=imgout.reshape((imgout.shape[0],imgout.shape[1]))
    return imgout

def rotate_dt(imgin,rad,order=1):
    dimy=imgin.shape[0]
    dimx=imgin.shape[1]
    newx=int(round(abs(dimx*cos(rad))+abs(dimy*sin(rad))))
    newy=int(round(abs(dimx*sin(rad))+abs(dimy*cos(rad))))
    imgout=numpy.zeros((newy,newx),dtype=numpy.float32)
    if order==0:
        lib.rotatec(imgin,dimx,dimy,1,rad,imgout,newx,newy)
    else:
        lib.rotatec_bi(imgin,dimx,dimy,1,rad,imgout,newx,newy)
    return imgout

def dt(img,ay,ax,by,bx):
    res=numpy.zeros(img.shape,dtype=numpy.float32)
    dy=numpy.zeros(img.shape,dtype=numpy.float32)
    dx=numpy.zeros(img.shape,dtype=numpy.float32)
    lib.dtpy(img,res,dy,dx,img.shape[0],img.shape[1],ay,ax,by,bx)
    return res,dy,dx

def fdt(img,ay,ax,by,bx):
    res=numpy.zeros(img.shape,dtype=numpy.float32)
    dy=numpy.zeros(img.shape,dtype=numpy.float32)
    dx=numpy.zeros(img.shape,dtype=numpy.float32)
    lib.fdtpy(img,res,dy,dx,img.shape[0],img.shape[1],ay,ax,by,bx)
    return res,dy,dx

def ffdt(img,ay,ax,by,bx):
    res=numpy.zeros(img.shape,dtype=numpy.float32)
    dy=numpy.zeros(img.shape,dtype=numpy.float32)
    dx=numpy.zeros(img.shape,dtype=numpy.float32)
    lib.ffdtpy(img,res,dy,dx,img.shape[0],img.shape[1],ay,ax,by,bx)
    return res,dy,dx

def dt2(img,ay,ax,axy,by,bx):
    res=-1000*numpy.ones(img.shape,dtype=numpy.float32)
    dy=numpy.zeros(img.shape,dtype=numpy.float32)
    dx=numpy.zeros(img.shape,dtype=numpy.float32)
    lib.dt2D(img,res,dy,dx,img.shape[0],img.shape[1],ay,ax,axy,by,bx)
    return res,dy,dx

#from prof import do_profile
#@do_profile()
def mydt(img,myay,myax,myby,mybx,fast=True):
    ay=myay
    ax=myax
    by=-2*myby*myay
    bx=-2*mybx*myax
    cy=myby**2*myay
    cx=mybx**2*myax
    res=-1000*numpy.ones(img.shape,dtype=numpy.float32)
    dy=numpy.zeros(img.shape,dtype=numpy.float32)
    dx=numpy.zeros(img.shape,dtype=numpy.float32)
    if fast:
        lib.ffdtpy(img,res,dy,dx,img.shape[0],img.shape[1],ay,ax,by,bx) #notice that b does not work for the moment
    else:
        lib.dtpy(img,res,dy,dx,img.shape[0],img.shape[1],ay,ax,by,bx)
    res=res-cy-cx
    return res,dy,dx

from scipy.ndimage.interpolation import rotate,map_coordinates
from numpy.linalg import eig
#from scipy.linalg import eig
from math import atan2,atan,cos,sin

def eig22(A):
    A=numpy.array(A)
    trA=numpy.trace(A)
    val1=trA+numpy.sqrt(trA**2-4*(A[0,0]*A[1,1]-A[1,0]*A[0,1]))
    val2=trA-numpy.sqrt(trA**2-4*(A[0,0]*A[1,1]-A[1,0]*A[0,1]))
    return (0.5*val1,.5*val2)

#from prof import do_profile
#@do_profile()
def dt2rot(img,ay,ax,axy,by,bx,fast=False):
    if abs(axy)<min(ax,ay)*0.01:
        #print "Fast DT",ax,ay,axy
        return mydt(img,ay,ax,0,0,fast)    
    intrp=0
    szy=img.shape[0]
    szx=img.shape[1]
    axy=-axy
    val,vec=eig([[ax,axy],[axy,ay]])
    if numpy.any(val)<0:
        print "Error, negative eigenvalues!"
        raw_input()
    rad=atan(vec[0,1]/vec[0,0])
    #rad=atan2(vec[0,1],vec[0,0])
    ang=rad/numpy.pi*180
    #print "Angle",ang
    #val=val/numpy.sqrt(numpy.sum(val**2))
    #dfsd
    #mm=img.min()
    #img=img+mm
    #print sin(rad)*szx
    if fast:
        img2=rotate_dt(img,rad,order=0)
    else:
        img2=rotate(img,ang,mode='nearest',order=0)    
    dtim,Iy,Ix=mydt(img2,val[1],val[0],0,0,fast)
    #dtim=numpy.zeros(img.shape,dtype=numpy.float32)
    #Iy=numpy.zeros(img.shape,dtype=numpy.float32)
    #Ix=numpy.zeros(img.shape,dtype=numpy.float32)
    #lib.ffdtpy(img,dtim,Iy,Ix,img.shape[0],img.shape[1],ay,ax,by,bx)
    #dtim[0,-int(sin(rad)*szx)]=100
    #res=rotate(dtim,-ang,reshape=False,mode='nearest',order=0)#-mm
#    res=rotate(dtim,-ang,mode='nearest',order=intrp)#-mm
    if fast:
        res=rotate_dt(dtim,-rad,order=intrp)#-mm
    else:
        res=rotate(dtim,-ang,mode='nearest',order=intrp)#-mm
    #idx=numpy.mgrid[:res.shape[0],:res.shape[1]]
    #res=map_coordinates(res,idx,order=1)
    #dy=rotate(Iy,-ang,reshape=False,mode='nearest',order=0)
    #dx=rotate(Ix,-ang,reshape=False,mode='nearest',order=0)
    if fast:
       dy=rotate_dt(Iy,-rad,order=intrp)
       dx=rotate_dt(Ix,-rad,order=intrp)
    else:
        dy=rotate(Iy,-ang,mode='nearest',order=intrp)
        dx=rotate(Ix,-ang,mode='nearest',order=intrp)
    #print sin(-rad)*cos(-rad)*szy,sin(-rad)*sin(-rad)*szx
    #adx=numpy.round(dx*cos(-rad)+dy*sin(-rad)-szx*cos(-rad)*sin(-rad))
    if rad>0:
        dtx=img2.shape[0]*sin(rad)
        dty=0
    else:
        dtx=0
        dty=-img2.shape[1]*sin(rad)
    adx=(dx*cos(-rad)+dy*sin(-rad))+dtx#-szx*cos(-rad)*sin(-rad))
    #ady=numpy.round(-dx*sin(-rad)+dy*cos(-rad)+szy*sin(-rad)*sin(-rad))
    ady=(-dx*sin(-rad)+dy*cos(-rad))+dty#+szy*sin(-rad)*sin(-rad))
    cty=(res.shape[0]-szy)/2
    ctx=(res.shape[1]-szx)/2
    if abs(bx)>ctx or abs(by)>cty:
        print "ERROR!!!!!!!!!"
        dsfsf       
        aby=abs(by);abx=abs(bx)
        res2=-10000*numpy.ones(res.shape+2*numpy.array([aby,abx]),dtype=res.dtype)
        res2[aby:-aby,abx:-abx]=res
        res=res2
        dy2=numpy.zeros(ady.shape+2*numpy.array([aby,abx]),dtype=dy.dtype)
        dy2[aby:-aby,abx:-abx]=ady
        ady=dy2
        dx2=numpy.zeros(adx.shape+2*numpy.array([aby,abx]),dtype=dx.dtype)
        dx2[aby:-aby,abx:-abx]=adx
        adx=dx2
        cty=cty+aby
        ctx=ctx+abx
    #fdy=ady-cty
    #fdx=adx-ctx#.astype(numpy.int)
    fdy=((ady[cty-by:cty+szy-by,ctx-bx:ctx+szx-bx]).round()-cty).astype(numpy.int32)
    fdx=((adx[cty-by:cty+szy-by,ctx-bx:ctx+szx-bx]).round()-ctx).astype(numpy.int32)
    #round refinement
    dst=-1000*numpy.ones(img.shape,dtype=numpy.float32)
    lib.refine(img,img.shape[0],img.shape[1],fdy,fdx,ay,ax,axy,dst)
    return dst,fdy,fdx
    if 0:#refinement
        dispy=[0,0,0,1,1,1,2,2,2]
        dispx=[0,1,2,0,1,2,0,1,2]
        mesh=numpy.mgrid[:fdy.shape[0],:fdy.shape[1]]
        defy=fdy.round()-mesh[0]
        defx=fdx.round()-mesh[1]
        ydf=numpy.array((defy[0:-2,0:-2],defy[0:-2,1:-1],defy[0:-2,2:],defy[1:-1,0:-2],defy[1:-1,1:-1],defy[1:-1,2:],defy[2:,0:-2],defy[2:,1:-1],defy[2:,2:]))
        xdf=numpy.array((defx[0:-2,0:-2],defx[0:-2,1:-1],defx[0:-2,2:],defx[1:-1,0:-2],defx[1:-1,1:-1],defx[1:-1,2:],defx[2:,0:-2],defx[2:,1:-1],defx[2:,2:]))
        scr=numpy.array((img[0:-2,0:-2],img[0:-2,1:-1],img[0:-2,2:],img[1:-1,0:-2],img[1:-1,1:-1],img[1:-1,2:],img[2:,0:-2],img[2:,1:-1],img[2:,2:]))
        #mesh1=numpy.mgrid[:9,:defy.shape[0]-2,:defy.shape[1]-2]
        mesh1=numpy.mgrid[:defy.shape[0]-2,:defy.shape[1]-2]
        #fscr=scr[mesh1[0],mesh1[1]+ydf.astype(numpy.int),mesh1[2]+xdf.astype(numpy.int)]-ydf**2*ay-xdf**2*ax+2*axy*ydf*xdf
        fscr=img[mesh1[0]+ydf.astype(numpy.int),mesh1[1]+xdf.astype(numpy.int)]-ydf**2*ay-xdf**2*ax+2*axy*ydf*xdf
        res2=fscr.max(0)
        sel=fscr.argmax(0)
        mesh2=numpy.mgrid[:defy.shape[0]-2,:defy.shape[1]-2]
        defy=ydf[sel,mesh2[0],mesh2[1]].round()+sel/3-1
        defx=xdf[sel,mesh2[0],mesh2[1]].round()+sel%3-1
        if 0:
            import pylab
            pylab.figure();pylab.imshow(res2);pylab.show()
            fsdf
        return res[cty-1:cty+szy+1,ctx-1:ctx+szx+1],fdy[cty-1:cty+szy+1,ctx-1:ctx+szx+1],fdx[cty-1:cty+szy+1,ctx-1:ctx+szx+1]
    #fdy=(fdy-(fdy-mesh[0]).mean()).round().astype(numpy.int)
    #fdx=(fdx-(fdx-mesh[1]).mean()).round().astype(numpy.int)
    return res[cty-by:cty+szy-by,ctx-bx:ctx+szx-bx],fdy,fdx

def checkdt(im,dtim,Iy,Ix,ay,ax,axy):
    for px in range(im.shape[1]):
        for py in range(im.shape[0]):
            dy=py-Iy[py,px];dx=px-Ix[py,px]
            df=(ay*dy**2+ax*dx**2+2*dx*dy*axy)
            app=im[Iy[py,px],Ix[py,px]]
            assert(abs(dtim[py,px]-(app-df))<0.0001)

if __name__ == "__main__":
    dimy=100
    dimx=150
    #a=numpy.random.random((dimy,dimx)).astype(numpy.float32)
    #im=numpy.zeros((dimy,dimx),numpy.float32)
    #im=-numpy.ones((dimy,dimx),numpy.float32)
    im=100*numpy.random.random((dimy,dimx)).astype(numpy.float32)
    im[40,50]=150
    im[25,65]=150
    #im[40,50]=1
    #dta=numpy.zeros((dimy,dimx),dtype=numpy.float32)
    #dy=numpy.zeros((dimy,dimx),dtype=numpy.float32)
    #dx=numpy.zeros((dimy,dimx),dtype=numpy.float32)
    #lib.dtpy(a,dta,dy,dx,dimy,dimy,1,1,0,0)
    ax=0.01;ay=0.5#0.01
    axy=0.1#0.1
    #dtim,Iy,Ix=mydt(im,a,a,b,b)
    #dtim,Iy,Ix=dt2(im,a,2*a,-0.00,by,bx)
    #dtim,Iy,Ix=dt(im,ay,ax,by,bx)
    #dtimr,Iyr,Ixr=ffdt(im,ay,ax,bx,by)
    dtim,Iy,Ix=dt2rot(im,ay,ax,axy,0,0,fast=True)
    for px in range(im.shape[1]):
        for py in range(im.shape[0]):
    #px=10;py=10
            dy=py-Iy[py,px];dx=px-Ix[py,px]
            df=(ay*dy**2+ax*dx**2+2*dx*dy*axy)
            app=im[Iy[py,px],Ix[py,px]]
            assert(abs(dtim[py,px]-(app-df))<0.0001)
    dtimf,Iyf,Ixf=dt2rot(im,ay,ax,axy,0,0,fast=False)
    if 0:
        import pylab
        pylab.figure(1)
        pylab.clf()
        pylab.imshow(dtim-dtimf)#(numpy.concatenate((im,im,im,im,im,im,im,im,im),0))
        pylab.figure(2)
        pylab.clf()
        pylab.imshow(dtimf)#(numpy.concatenate((dtim,dtim,dtim,dtim,dtim,dtim,dtim,dtim,dtim),0))
        pylab.figure(3)
        pylab.clf()
        pylab.imshow(Iy-Iyf)#(numpy.concatenate((Iy,Iy,Iy,Iy,Iy,Iy,Iy,Iy,Iy,Iy,Iy,Iy),0))
        pylab.figure(4)
        pylab.clf()
        pylab.imshow(Ix-Ixf)#(numpy.concatenate((Ix,Ix,Ix,Ix,Ix,Ix,Ix,Ix,Ix,Ix,Ix,Ix),0))
        pylab.draw()
        pylab.show()
    #print "Max Error",numpy.abs(dtimr-dtim).max(),numpy.abs(Iyr-Iy).max(),numpy.abs(Ixr-Ix).max()
    #print "Mean Error",numpy.abs(dtimr-dtim).mean(),numpy.abs(Iyr-Iy).mean(),numpy.abs(Ixr-Ix).mean()


