#project masks to different rotations
from numba import autojit
import  numpy

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
    a=abs(a)
    if a==0 or a==15 or a==30:
        pattern=numpy.array([[0,1,2,3]],dtype=numpy.int32)
    elif a==45:
        pattern=numpy.array([[0,1,2],[1,2,3]],dtype=numpy.int32)
    elif a==60:
        pattern=numpy.array([[0,3],[1,2]],dtype=numpy.int32)
    elif a==75:
        pattern=numpy.array([[0],[3]],dtype=numpy.int32)
    elif a>=90:
        pattern=numpy.array([[]],dtype=numpy.int32)
    return pattern

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

from ctypes import c_int,c_float,CDLL,cdll
cdll.LoadLibrary("./cproject.so")
pr=CDLL("cproject.so")
pr.cproject.argtypes=[c_int,c_int,c_int,c_int,numpy.ctypeslib.ndpointer(dtype=c_float,ndim=4,flags="C_CONTIGUOUS"),c_int,c_int,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=2,flags="C_CONTIGUOUS"),c_int,c_int,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=2,flags="C_CONTIGUOUS"),numpy.ctypeslib.ndpointer(dtype=c_float,ndim=2,flags="C_CONTIGUOUS")]

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


@autojit
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







