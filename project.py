#project masks to different rotations
import  numpy

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

def pattern4(a):
    if a==0 or a==15 or a==30:
        pattern=numpy.array([[0,1,2,3]])
    elif a==45:
        pattern=numpy.array([[0,1,2],[1,2,3]])
    elif a==60:
        pattern=numpy.array([[0,3],[1,2]])
    elif a==75:
        pattern=numpy.array([[0],[3]])
    elif a==90:
        pattern=numpy.array([[]])
    return pattern

def project(mask,res,pty,ptx):
    """
    compute the correlation with angles ax and ay
    assumes a part of 4x4 hogs
    """
    szy=mask.shape[0]
    szx=mask.shape[1]
    hy=res.shape[2]
    hx=res.shape[3]
    res2=numpy.zeros((hy+szy,hx+szx),dtype=numpy.float32)
    for py in range(pty.shape[1]):
            for pym in range(pty.shape[0]):
                for px in range(ptx.shape[1]):
                    for pxm in range(ptx.shape[0]):
                        res2[py:py+hy,px:px+hx]=res2[py:py+hy,px:px+hx]+res[pty[pym,py],ptx[pxm,px]]/float(pty.shape[0])/float(ptx.shape[0])
    return res2
