#project masks to different rotations
#from numba import autojit
import numpy
from math import sin,cos

#@autojit
def precompute(mask,hog):
    """
    precompute the correlation mask with hog
    """
    szy=mask.shape[0]
    szx=mask.shape[1]
    hy=hog.shape[0]
    hx=hog.shape[1]
    res=numpy.zeros((szy,szx,hy,hx),dtype=numpy.float32)
    for py in range(szy):
        for px in range(szx):
            res[py,px]=numpy.dot(hog,mask[py,px])
    return res

#@autojit
def pattern4(a):
    """
    score depends on the number of hog cells occupied in the image
    """
    a=abs(a%180)
    if a>90:
        a=90-(a-90)
    if a==0 or a==15 or a==30:
        pattern=numpy.array([[0,1,2,3]],dtype=numpy.int32)
    elif a==45:
        pattern=numpy.array([[0,1,2],[1,2,3]],dtype=numpy.int32)
    elif a==60:
        pattern=numpy.array([[0,3],[1,2]],dtype=numpy.int32)
    elif a==75:
        pattern=numpy.array([[0],[3]],dtype=numpy.int32)
    elif a==90:
        pattern=numpy.array([[]],dtype=numpy.int32)
    return pattern

def pattern4_cos(c):
    """
    score depends on the number of hog cells occupied in the image
    """
    c=abs(c)
    #pattern=numpy.array([[]],dtype=numpy.int32)
    if c<sin(37.5/180.0*numpy.pi):
        pattern=numpy.array([[0,1,2,3]],dtype=numpy.int32)
    elif c<sin(52.5/180.0*numpy.pi):
        pattern=numpy.array([[0,1,2],[1,2,3]],dtype=numpy.int32)
    elif c<sin(67.5/180.0*numpy.pi):
        pattern=numpy.array([[0,3],[1,2]],dtype=numpy.int32)
    elif c<sin(82.5/180.0*numpy.pi):
        pattern=numpy.array([[0],[3]],dtype=numpy.int32)
    else:
        pattern=numpy.array([[]],dtype=numpy.int32)
    #if c>cos(37.5/180.0*numpy.pi):
    #    pattern=numpy.array([[0,1,2,3]],dtype=numpy.int32)
    #elif c>cos(52.5/180.0*numpy.pi):
    #    pattern=numpy.array([[0,1,2],[1,2,3]],dtype=numpy.int32)
    #elif c>cos(67.5/180.0*numpy.pi):
    #    pattern=numpy.array([[0,3],[1,2]],dtype=numpy.int32)
    #elif c>cos(82.5/180.0*numpy.pi):
    #    pattern=numpy.array([[0],[3]],dtype=numpy.int32)
    #else:
    #    pattern=numpy.array([[]],dtype=numpy.int32)
    return pattern

def pattern5_cos(c):
    """
    score depends on the number of hog cells occupied in the image
    """
    c=abs(c)
    if c<sin(37.5/180.0*numpy.pi):
        pattern=numpy.array([[0,1,2,3,4]],dtype=numpy.int32)
    elif c<sin(52.5/180.0*numpy.pi):
        pattern=numpy.array([[0,1,2,3],[1,2,3,4]],dtype=numpy.int32)
    elif c<sin(67.5/180.0*numpy.pi):
        pattern=numpy.array([[0,2,4],[1,2,3]],dtype=numpy.int32)
    elif c<sin(82.5/180.0*numpy.pi):
        pattern=numpy.array([[0,4],[1,3]],dtype=numpy.int32)
        #pattern=numpy.array([[0],[3]],dtype=numpy.int32)
    else:
        pattern=numpy.array([[]],dtype=numpy.int32)
    return pattern


#import pyrHOG2
def pattern4_rot(c):
    c=c%360
    pattern0=numpy.mgrid[:4,:4].astype(numpy.int32)+(c/90)*4
    pattern1=numpy.mgrid[:4,:4].astype(numpy.int32)+(c/90)*4+4
#    if c>=0 and c<90:
#        #pattern1=pattern1.T[:,::-1]
#        #pattern1=pyrHOG2.hogrotate(pattern1,angle=90,obin=9)
#    if c>=90 and c<180:
#        pattern0=pattern0.T[:,::-1]
#        pattern1=pattern1.T[:,::-1]
#        #pattern0=pyrHOG2.hogrotate(pattern1,angle=90,obin=9)
#        #pattern1=pattern1[::-1,::-1]
#        #pattern1=pyrHOG2.hogrotate(pattern1,angle=180,obin=9)
#    if c>=180 and c<270:
#        pattern0=pattern0[::-1,::-1]
#        pattern1=pattern1[::-1,::-1]
#        #pattern0=pyrHOG2.hogrotate(pattern1,angle=180,obin=9)
#        #pattern1=pattern1.T[::-1]
#        #pattern1=pyrHOG2.hogrotate(pattern1,angle=270,obin=9)
#    if c>=270 and c<360:
#        pattern0=pattern0.T[::-1]
#        pattern1=pattern1.T[::-1]
#        #pattern0=pyrHOG2.hogrotate(pattern1,angle=270,obin=9)
    dt=c%90
#   |||||
#   |||||
#   |||||
#   |||||
    if dt<=(15+30)/2.0:
        pt=pattern0.reshape((2,4,4,1))
#    ||
#    ||||
#   || ||
#   ||||
#     ||
    if dt>(15+30)/2.0 and dt<=(30+45)/2.0:
        pt=-numpy.ones((2,5,5,1))
        pt[:,:2,1:3,0]=pattern0[:,:2,:2]
        pt[:,1:3,3:5,0]=pattern0[:,:2,2:4]
        pt[:,2:4,:2,0]=pattern0[:,2:4,:2]
        pt[:,3:5,2:4,0]=pattern0[:,2:4,2:4]
#      +
#     +++
#    ++ ++
#     +++
#      +
    if dt>(30+45)/2.0 and dt<=(45+60/2.0):
        pt=-numpy.ones((2,5,5,4))
        pt[:,0,2,0]=pattern0[:,0,0];pt[:,0,2,1]=pattern1[:,0,3]
        
        pt[:,1,1,0]=pattern0[:,1,0];pt[:,1,1,1]=pattern0[:,2,0];pt[:,1,1,2]=pattern1[:,0,1];pt[:,1,1,3]=pattern1[:,0,2]
        pt[:,1,2,0]=pattern0[:,1,1];pt[:,1,2,1]=pattern1[:,1,2]
        pt[:,1,3,0]=pattern0[:,0,1];pt[:,1,3,1]=pattern0[:,0,2];pt[:,1,3,2]=pattern1[:,1,3];pt[:,1,3,3]=pattern1[:,2,3]

        pt[:,2,0,0]=pattern0[:,3,0];pt[:,2,0,1]=pattern1[:,0,0]
        pt[:,2,1,0]=pattern0[:,2,1];pt[:,2,1,1]=pattern1[:,1,1]
        #pt[1,2,:]=0
        pt[:,2,3,0]=pattern0[:,1,2];pt[:,2,3,1]=pattern1[:,2,2]
        pt[:,2,4,0]=pattern0[:,0,3];pt[:,2,4,1]=pattern0[:,3,3]

        pt[:,3,1,0]=pattern0[:,3,1];pt[:,3,1,1]=pattern0[:,3,2];pt[:,3,1,2]=pattern1[:,1,0];pt[:,3,1,3]=pattern1[:,2,0]
        pt[:,3,2,0]=pattern0[:,2,2];pt[:,3,2,1]=pattern1[:,2,1]
        pt[:,3,3,0]=pattern0[:,1,3];pt[:,3,3,1]=pattern0[:,2,3];pt[:,3,3,2]=pattern1[:,3,1];pt[:,3,3,3]=pattern1[:,3,2]

        pt[:,4,2,0]=pattern0[:,3,3];pt[:,4,2,1]=pattern1[:,3,0]
#       --
#     ----
#     -- --
#      ----
#      --
    if dt>(45+60)/2.0 and dt<=(60+75)/2.0:
        pt=-numpy.ones((2,5,5,1))
        pt[:,1:3,:2,0]=pattern1[:,:2,:2]
        pt[:,3:5,1:3,0]=pattern1[:,2:4,:2]
        pt[:,:2,2:4,0]=pattern1[:,:2,2:4]
        pt[:,2:4,3:5,0]=pattern1[:,2:4,2:4]
#   ----
#   ----
#   ----
#   ----
    if dt>(60+75)/2.0:
        pt=pattern1.reshape((2,4,4,1))

    return pt

def mat_rot(m,angle=90):
    newm=numpy.zeros(m.shape,dtype=m.dtype)
    if angle==90:
        for l in range(m.shape[2]):
            newm[:,:,l]=m[:,:,l].T[:,::-1]
    if angle==180:
        newm=m[::-1,::-1]
    if angle==270:
        for l in range(m.shape[2]):
            newm[:,:,l]=m[:,:,l].T[::-1,:]
    return newm

def template_rot(t):
    import pyrHOG2
    tr=-numpy.ones((4,4,4,31),dtype=t.dtype)#4 rotations, positionsy, positionx, angle
    tr[0]=t
    tr[1]=pyrHOG2.hogrotate(mat_rot(t,90),angle=90,obin=9)
    tr[2]=pyrHOG2.hogrotate(mat_rot(t,180),angle=180,obin=9)
    tr[3]=pyrHOG2.hogrotate(mat_rot(t,270),angle=270,obin=9)
    return tr
#@autojit
def pattern4_bis(a):
    """
    score depends on the number of hog cells of the 0 degrees model
    """
    a=abs(a)
    if a==0 or a==15 or a==30:
        pattern=numpy.array([[0,1,2,3]])
    elif a==45:
        pattern=numpy.array([[0,1,3],[-1,2,-1]])
    elif a==60:
        pattern=numpy.array([[0,3],[1,2]])
    elif a>60:
        pattern=numpy.array([[]])    
    #elif a==75:
    #    pattern=numpy.array([[0],[1],[2],[3]])
    #elif a>=90:
    #    pattern=numpy.array([[]])
    return pattern

#@autojit
def prjhog(mask,pty,ptx):
    hog=numpy.zeros((pty.shape[1],ptx.shape[1],mask.shape[2]),dtype=mask.dtype)
    spty=float(pty.shape[0])
    sptx=float(ptx.shape[0])
    for py in range(pty.shape[1]):
        for pym in range(pty.shape[0]):
            for px in range(ptx.shape[1]):
                for pxm in range(ptx.shape[0]):
                    #if pty[pym,py]!=-1 and ptx[pxm,px]!=-1: 
                    hog[py,px]=hog[py,px]+mask[pty[pym,py],ptx[pxm,px]]/spty/sptx
    return hog

def prjhogrot(mask,pt):
    spty=(pt.shape[1])
    sptx=(pt.shape[2])
    spm=(pt.shape[3])
    hog=numpy.zeros((spty,sptx,mask.shape[3]),dtype=mask.dtype)
    for py in range(spty):
        for px in range(sptx):
            tot=float(numpy.sum(pt[0,py,px,:]!=-1))
            for pm in range(spm):
                if pt[0,py,px,pm]!=-1 and pt[1,py,px,pm]!=-1: 
                    hog[py,px]=hog[py,px]+mask[pt[0,py,px,pm]/4%4,pt[0,py,px,pm]%4,pt[1,py,px,pm]%4]/tot
    return hog

from ctypes import c_int,c_float,CDLL,cdll
cdll.LoadLibrary("./cproject.so")
pr=CDLL("cproject.so")
pr.cproject.argtypes=[c_int,c_int,c_int,c_int,numpy.ctypeslib.ndpointer(dtype=c_float,ndim=4,flags="C_CONTIGUOUS"),c_int,c_int,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=2,flags="C_CONTIGUOUS"),c_int,c_int,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=2,flags="C_CONTIGUOUS"),numpy.ctypeslib.ndpointer(dtype=c_float,ndim=2,flags="C_CONTIGUOUS")]

#from prof import *
#@do_profile()
def project(res,pty,ptx):
    szy=res.shape[0]
    szx=res.shape[1]
    hy=res.shape[2]
    hx=res.shape[3]
    res2=numpy.zeros((hy+szy,hx+szx),dtype=numpy.float32)
    pr.cproject(res.shape[0],res.shape[1],res.shape[2],res.shape[3],res,pty.shape[0],pty.shape[1],pty,ptx.shape[0],ptx.shape[1],ptx,res2)
    #if not(numpy.all(res2==0)):
    #    print "Stop!!!"
    #    sdfs
    return res2


#@autojit
def project_(res,pty,ptx):
    """
    compute the correlation with angles ax and ay
    assumes a part of 4x4 hogs
    """
    szy=res.shape[0]
    szx=res.shape[1]
    hy=res.shape[2]
    hx=res.shape[3]
    spty=float(pty.shape[0])
    sptx=float(ptx.shape[0])
    res2=numpy.zeros((hy+szy,hx+szx),dtype=numpy.float32)
    for py in range(pty.shape[1]):
        for pym in range(pty.shape[0]):
            for px in range(ptx.shape[1]):
                for pxm in range(ptx.shape[0]):
                    res2[szy-py:szy-py+hy,szx-px:szx-px+hx]+=res[pty[pym,py],ptx[pxm,px]]/spty/sptx
                    #res2[py:py+hy,px:px+hx]=res2[py:py+hy,px:px+hx]+res[pty[pym,py],ptx[pxm,px]]/float(pty.shape[0])/float(ptx.shape[0])
                    #res2[szy-py:szy-py+hy,szx-px:szx-px+hx]=res2[szy-py:szy-py+hy,szx-px:szx-px+hx]+res[pty[pym,py],ptx[pxm,px]]/spty/sptx
    return res2

#@autojit
def prjhog_bis(mask,pty,ptx):
    hog=numpy.zeros((pty.shape[1],ptx.shape[1],mask.shape[2]),dtype=mask.dtype)
    for py in range(pty.shape[1]):
        for pym in range(pty.shape[0]):
            for px in range(ptx.shape[1]):
                for pxm in range(ptx.shape[0]):
                    if pty[pym,py]!=-1 and ptx[pxm,px]!=-1: 
                        hog[py,px]=hog[py,px]+mask[pty[pym,py],ptx[pxm,px]]
    return hog

#def prjhog(mask,pty,ptx):
#    hog=numpy.zeros((pty.shape[1],ptx.shape[1],mask.shape[2]),dtype=mask.dtype)
#    for py in range(pty.shape[1]):
#        for pym in range(pty.shape[0]):
#            for px in range(ptx.shape[1]):
#                for pxm in range(ptx.shape[0]):
#                    hog[py,px]=hog[py,px]+mask[pty[pym,py],ptx[pxm,px]]/float(pty.shape[0])/float(ptx.shape[0])
#    return hog

#@autojit
def invprjhog(hog,pty,ptx):
    #hog=numpy.zeros((pty.shape[1],ptx.shape[1],mask.shape[2]),dtype=mask.dtype)
    mask=numpy.zeros((4,4,hog.shape[2]),dtype=hog.dtype)
    for py in range(pty.shape[1]):
        for pym in range(pty.shape[0]):
            for px in range(ptx.shape[1]):
                for pxm in range(ptx.shape[0]):
                    mask[pty[pym,py],ptx[pxm,px]]=mask[pty[pym,py],ptx[pxm,px]]+hog[py,px]/float(pty.shape[0])/float(ptx.shape[0])
                    #mask[pty[pym,py],ptx[pxm,px]]=hog[py,px]/float(numpy.sum(pty==pty[pym,py]))/float(numpy.sum(ptx==ptx[pxm,px]))
                        #hog[py,px]=hog[py,px]+mask[pty[pym,py],ptx[pxm,px]]
    return mask

def invprjhog_bis(hog,pty,ptx):
    #hog=numpy.zeros((pty.shape[1],ptx.shape[1],mask.shape[2]),dtype=mask.dtype)
    mask=numpy.zeros((4,4,hog.shape[2]),dtype=hog.dtype)
    for py in range(pty.shape[1]):
        for pym in range(pty.shape[0]):
            for px in range(ptx.shape[1]):
                for pxm in range(ptx.shape[0]):
                    if pty[pym,py]!=-1 and ptx[pxm,px]!=-1: 
                        mask[pty[pym,py],ptx[pxm,px]]=hog[py,px]
                        #hog[py,px]=hog[py,px]+mask[pty[pym,py],ptx[pxm,px]]
    return mask


#@autojit
def getproj(y,z,glangy,angy,hsize=4):
    return y*numpy.cos(glangy/180.0*numpy.pi)-hsize/2.0*(numpy.cos(angy/180.0*numpy.pi))-z*numpy.sin(angy/180.0*numpy.pi)

#@autojit
def project_bis(res,pty,ptx):
    """
    compute the correlation with angles ax and ay
    assumes a part of 4x4 hogs
    """
    szy=res.shape[0]
    szx=res.shape[1]
    hy=res.shape[2]
    hx=res.shape[3]
    res2=numpy.zeros((hy+szy,hx+szx),dtype=numpy.float32)
    for py in range(pty.shape[1]):
            for pym in range(pty.shape[0]):
                for px in range(ptx.shape[1]):
                    for pxm in range(ptx.shape[0]):
                        if pty[pym,py]!=-1 and ptx[pxm,px]!=-1:
                        #res2[py:py+hy,px:px+hx]=res2[py:py+hy,px:px+hx]+res[pty[pym,py],ptx[pxm,px]]/float(pty.shape[0])/float(ptx.shape[0])
                            res2[szy-py:szy-py+hy,szx-px:szx-px+hx]=res2[szy-py:szy-py+hy,szx-px:szx-px+hx]+res[pty[pym,py],ptx[pxm,px]]#/float(pty.shape[0])/float(ptx.shape[0])
    return res2

def test():
    import time
    t=time.time()
    mask=numpy.random.random((4,4,31)).astype(numpy.float32)
    pty=pattern4(-30)
    ptx=pattern4(45)
    pmask=prjhog(mask,pty,ptx)
    hog=numpy.random.random((1000,1000,31)).astype(numpy.float32)    
    res=precompute(mask,hog)
    res2=project(res,pty,ptx)
    print "Elapsed time:",time.time()-t







