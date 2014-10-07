#from numba import autojit
import util
import pyrHOG2
import drawHOG
import pylab
import numpy
import project
from math import sin,cos,floor,pi
#import project_fast as project
#from numba import autojit

class part3D(object):
    def __init__(self,mask,y,x,z,lz,ay,ax,dfay=0.0,dfax=0.0,dfaz=0.0,dfby=0,dfbx=0,dfbz=0):
    #def __init__(self,mask,y,x,z,lz,ay,ax,dfay=1.0,dfax=1.0,dfaz=1.0,dfby=0,dfbx=0,dfbz=0):
        self.mask=mask
        self.y=y
        self.x=x
        self.z=z
        self.lz=lz
        self.ay=ay
        self.ax=ax
        #quadratic cost
        self.dfay=dfay
        self.dfax=dfax
        self.dfaz=dfaz
        #translation
        #not needed because using onlt lz
        #self.dfby=dfby #set to 0?
        #self.dfbx=dfbx #set to 0?
        #self.dfbz=dfbz #allow only translation in z

def rotate(v,a):
    a=a/180.0*pi
    res=[0,0]#numpy.zeros(2,dtype=v.dtype)
    res[0] = v[0]*cos(a) - v[1]*sin(a)
    res[1] = v[0]*sin(a) + v[1]*cos(a)
    return res

def rotatex(v,a):
    res=rotate([v[1],v[2]],a)
    return [v[0],res[0],res[1]]

def rotatey(v,a):
    res=rotate([v[0],v[2]],a)
    return [res[0],v[1],res[1]]

def rotatez(v,a):
    res=rotate([v[0],v[1]],a)
    return [res[0],res[1],v[2]]

def Mrotx(a):
    rad=a/180.0*pi
    R=numpy.array([[1,0,0,0],
               [0,cos(rad),-sin(rad),0],
               [0,sin(rad),cos(rad),0],
               [0,0,0,1]])                    
    return R

def Mroty(a):
    rad=a/180.0*pi
    R=numpy.array([[cos(rad),0,sin(rad),0],
               [0,1,0,0],
               [-sin(rad),0,cos(rad),0],
               [0,0,0,1]])                    
    return R

def Mrotz(a):
    rad=a/180.0*pi
    R=numpy.array([[cos(rad),-sin(rad),0,0],
               [sin(rad),cos(rad),0,0],
               [0,0,1,0],
               [0,0,0,1]])                    
    return R




BIS=False
USEBIASES=False

from ctypes import cdll,CDLL,c_float,c_int,byref,POINTER
cdll.LoadLibrary("./cproject.so")
pr=CDLL("cproject.so")
pr.getproj.argtypes=[c_float,c_float,c_int,c_int,c_int,c_int,c_int,c_float,c_float,c_float,c_float,c_int,POINTER(c_float),POINTER(c_float)]
pr.getproj2.argtypes=[c_float,c_float,c_int,c_int,c_int,c_int,c_int,c_float,c_float,c_float,c_float,c_int,POINTER(c_float),POINTER(c_float)]
pr.interpolate.argtypes=[c_int,c_int,c_int,c_int,c_int,numpy.ctypeslib.ndpointer(dtype=c_float,ndim=5,flags="C_CONTIGUOUS"),c_int,c_int,c_int,c_int,c_int,c_int,c_int,c_int,c_int,c_int,c_float,c_float,numpy.ctypeslib.ndpointer(dtype=c_float,ndim=2,flags="C_CONTIGUOUS")]
pr.normal.argtypes=[c_int,c_int,c_int,c_int,POINTER(c_float),POINTER(c_float),POINTER(c_float)]

#def visible(angy,angx):
#    p=[0,0,1];
#    p=rotatey(p,angx)
#    p=rotatex(p,angy)
#    return p[2]>0

def normal(angy,angx,glangy,glangx,p=[0,0,1.0]):
    #p=[0,0,1.0];
    p=rotatey(p,angx)
    p=rotatex(p,angy)
    p=rotatey(p,glangx)
    p=rotatex(p,glangy)
    #p=rotatex(p,angx)
    #p=rotatey(p,angy)
    #p=rotatex(p,glangx)
    #p=rotatey(p,glangy)
    return p

def normal_fast(angy,angx,glangy,glangx,p=[0,0,1.0]):
    px=c_float(p[0]);py=c_float(p[1]);pz=c_float(p[2])
    pr.normal(angy,angx,glangy,glangx,byref(px),byref(py),byref(pz))
    return [px.value,py.value,pz.value]

def reproj(angy,angx,glangy,glangx,p=[0,1.0]):
    p=rotatex(p,-glangy)
    p=rotatey(p,-glangx)
    p=rotatex(p,-angy)
    p=rotatey(p,-angx) 
    #p=rotatey(p,-glangy)
    #p=rotatex(p,-glangx)
    #p=rotatey(p,-angy)
    #p=rotatex(p,-angx)
    return p

def drawParts(model,center,scale,def3D,glangy,glangx,glangz,hsize=4,pixh=15,border=2,nhog=30,bis=BIS,val=None):
    size=modelsize(model,[glangy],[glangx],[glangz],force=True)[0,0,0]
    cposy=c_float(0.0);cposx=c_float(0.0)
    cposy2=c_float(0.0);cposx2=c_float(0.0)
    for idw,w in enumerate(model["ww"]):
        n=normal(w.ay,w.ax,glangy,glangx)
        if n[2]<0.0001:#face not visible
            continue
        #print def3D[idw]
        #def3D[c][0]/2.0,def3D[c][1]/2.0,0
        pr.getproj2(-hsize/2.0,-hsize/2.0,glangx,glangy,glangz,w.ax,w.ay,w.x-def3D[idw][0],w.y-def3D[idw][1],w.z-def3D[idw][2],w.lz,hsize,byref(cposx),byref(cposy))
        pr.getproj2(hsize/2.0,hsize/2.0,glangx,glangy,glangz,w.ax,w.ay,w.x-def3D[idw][0],w.y-def3D[idw][1],w.z-def3D[idw][2],w.lz,hsize,byref(cposx2),byref(cposy2))
        nposy=(cposy.value-size[0])/scale*hsize+center[0];nposx=(cposx.value-size[1])/scale*hsize+center[1]
        nposy2=(cposy2.value-size[0])/scale*hsize+center[0];nposx2=(cposx2.value-size[1])/scale*hsize+center[1]
        util.box(nposy,nposx,nposy2,nposx2,lw=2,color=(0,1*n[2],0,1))        



#@autojit
def showModel(model,glangy,glangx,glangz,hsize=4,pixh=15,border=2,nhog=30,bis=BIS,val=None):
    size=modelsize(model,[glangy],[glangx],[glangz],force=True)[0,0,0]
    print size
    nhogy=size[2]-size[0]+1
    nhogx=size[3]-size[1]+1
    #nhog=100
    nnhog=0
    img=numpy.zeros((pixh*nhogy,pixh*nhogx))
    #img2=numpy.zeros((pixh*nhogy,pixh*nhogx))
    cposy=c_float(0.0);cposx=c_float(0.0)
    for w in model["ww"]:
        #angy=(w.ay+glangy)
        #angx=(w.ax+glangx)
        n=normal(w.ay,w.ax,glangy,glangx)
        #print "Normal",n
        #print "Angles",angy,angx,n,n[2]>0.0001
        if n[2]<0.0001:#face not visible
            continue
        if bis:
            part=drawHOG.drawHOG(project.prjhog_bis(w.mask,project.pattern4_bis(angy),project.pattern4_bis(angx)),hogpix=pixh,border=border,val=val)
        else:
            part=drawHOG.drawHOG(project.prjhog(w.mask,project.pattern4_cos(n[1]),project.pattern4_cos(n[0])),hogpix=pixh,border=border,val=val)
            #print part.shape
        #print "Outside",w.x,w.y,w.z,w.lz
        pr.getproj(size[1],size[0],glangx,glangy,glangz,w.ax,w.ay,w.x,w.y,w.z,w.lz,hsize,byref(cposx),byref(cposy))
        nposy=cposy.value;nposx=cposx.value
        #print "Pose",nposy,nposx
        #nposy=-size[0]+w.y*cos(glangy/180.0*numpy.pi)-hsize/2.0*(cos(angy/180.0*numpy.pi))-w.z*sin(angy/180.0*numpy.pi)
        #nposx=-size[1]+w.x*cos(glangx/180.0*numpy.pi)-hsize/2.0*(cos(angx/180.0*numpy.pi))-w.z*sin(angx/180.0*numpy.pi)
        #print nposy,nposx,glangx,angx,w.x*cos(glangx/180.0*numpy.pi),-w.z*sin(angx/180.0*numpy.pi),-hsize/2.0*(cos(angx/180.0*numpy.pi)),w.x*cos(glangx/180.0*numpy.pi)-hsize/2.0*(cos(angx/180.0*numpy.pi))-w.z*sin(angx/180.0*numpy.pi)
        img[nnhog/2*pixh+numpy.round(nposy*pixh):nnhog/2*pixh+numpy.round(nposy*pixh)+part.shape[0],nnhog/2*pixh+numpy.round(nposx*pixh):nnhog/2*pixh+numpy.round(nposx*pixh)+part.shape[1]]=part        
        #img2[nhog/2*pixh+nposy*pixh:nhog/2*pixh+nposy*pixh+part.shape[0],nhog/2*pixh+nposx*pixh:nhog/2*pixh+nposx*pixh+part.shape[1]]=part2        
        if 0:
            pylab.clf()
            pylab.imshow(img)    
            pylab.draw()
            pylab.show()
            raw_input()

    #pylab.figure()
    pylab.imshow(img)    
    #pylab.figure()
    #pylab.imshow(img2)    
    pylab.draw()
    pylab.show()
    #raw_input()

#@autojit
def showHOG(model,lhog,glangy,glangx,glangz,hsize=4,pixh=15,border=2,nhog=30,bis=BIS,val=None):
    size=modelsize(model,[glangy],[glangx],[glangz],force=True)[0,0,0]
    nhogy=size[2]-size[0]#+3
    nhogx=size[3]-size[1]#+3
    #nhog=100
    nnhog=0
    img=numpy.zeros((pixh*nhogy,pixh*nhogx))
    #img=numpy.zeros((pixh*nhog,pixh*nhog))
    #img2=numpy.zeros((pixh*nhog,pixh*nhog))
    cposy=c_float(0.0);cposx=c_float(0.0)
    for idh,h in enumerate(lhog):
        w=model["ww"][idh]
        #angy=(w.ay+glangy)
        #angx=(w.ax+glangx)
        n=normal(w.ay,w.ax,glangy,glangx)
        #print "Angles",angy,angx,n,n[2]>0.0001
        if n[2]<0.0001:#face not visible
            continue
        #angy=w.ay+glangy
        #angx=w.ax+glangx
        if bis:
            part=drawHOG.drawHOG(project.prjhog_bis(h,project.pattern4_bis(angy),project.pattern4_bis(angx)),hogpix=pixh,border=border,val=val)
        else:
            #part=drawHOG.drawHOG(project.prjhog(h,project.pattern4(angy),project.pattern4(angx)),hogpix=pixh,border=border,val=val)
            part=drawHOG.drawHOG(project.prjhog(h,project.pattern4_cos(n[1]),project.pattern4_cos(n[0])),hogpix=pixh,border=border,val=val)
            #part=drawHOG.drawHOG(project.prjhog(w.mask,project.pattern4_cos(n[0]),project.pattern4_cos(n[1])),hogpix=pixh,border=border,val=val)
        #nposy=w.y*cos(glangy/180.0*numpy.pi)-hsize/2.0*(cos(angy/180.0*numpy.pi))-w.z*sin(angy/180.0*numpy.pi)
        #nposx=w.x*cos(glangx/180.0*numpy.pi)-hsize/2.0*(cos(angx/180.0*numpy.pi))-w.z*sin(angx/180.0*numpy.pi)
        pr.getproj(size[1],size[0],glangx,glangy,glangz,w.ax,w.ay,w.x,w.y,w.z,w.lz,hsize,byref(cposx),byref(cposy))
        nposy=cposy.value;nposx=cposx.value
        img[nnhog/2*pixh+numpy.round(nposy*pixh):nnhog/2*pixh+numpy.round(nposy*pixh)+part.shape[0],nnhog/2*pixh+numpy.round(nposx*pixh):nnhog/2*pixh+numpy.round(nposx*pixh)+part.shape[1]]=part  
        #img[nhog/2*pixh+nposy*pixh:nhog/2*pixh+nposy*pixh+part.shape[0],nhog/2*pixh+nposx*pixh:nhog/2*pixh+nposx*pixh+part.shape[1]]=part        
        #img2[nhog/2*pixh+nposy*pixh:nhog/2*pixh+nposy*pixh+part.shape[0],nhog/2*pixh+nposx*pixh:nhog/2*pixh+nposx*pixh+part.shape[1]]=part2        
        if 0:
            pylab.clf()
            pylab.imshow(img)    
            pylab.draw()
            pylab.show()
            raw_input()

    #pylab.figure()
    pylab.imshow(img)    
    #pylab.figure()
    #pylab.imshow(img2)    
    pylab.draw()
    pylab.show()
    #raw_input()


def modelsize_old(model,pglangy,pglangx,force=False):
    hsize=model["ww"][0].mask.shape[0]
    if force or not(model.has_key("size")) or (model.has_key("size") and model["size"].shape!=(len(pglangy),len(pglangx),4)):#not(model.has_key("size"))::
        model["size"]=numpy.zeros((len(pglangy),len(pglangx),4))
        for gly,glangy in enumerate(pglangy):
            for glx,glangx in enumerate(pglangx):
                lminym=[]
                lminxm=[]
                #precompute max and min size
                for w in model["ww"]:
                    #mm=model["ww"][l]
                    angy=w.ay+glangy
                    angx=w.ax+glangx
                    nposy=w.y*cos(glangy/180.0*numpy.pi)-hsize/2.0*(cos(angy/180.0*numpy.pi))-w.z*sin(angy/180.0*numpy.pi)
                    nposx=w.x*cos(glangx/180.0*numpy.pi)-hsize/2.0*(cos(angx/180.0*numpy.pi))-w.z*sin(angx/180.0*numpy.pi)
                    lminym.append(nposy)
                    lminxm.append(nposx)
                #print angx,cos(angx/180.0*numpy.pi)
                minym=numpy.min(lminym)-hsize#*cos(angy/180.0*numpy.pi)
                minxm=numpy.min(lminxm)-hsize#*cos(angx/180.0*numpy.pi)
                maxym=numpy.max(lminym)+hsize#*cos(angy/180.0*numpy.pi)#project.pattern4_bis(angy).shape[1]
                maxxm=numpy.max(lminxm)+hsize#*cos(angx/180.0*numpy.pi)#project.pattern4_bis(angx).shape[1]
                model["size"][gly,glx]=(minym,minxm,maxym,maxxm)#numpy.round((minym,minxm,maxym,maxxm))
    return model["size"]

#@autojit
def modelsize_last(model,pglangy,pglangx,pglangz,force=False):
    hsize=model["ww"][0].mask.shape[0]
    if force or not(model.has_key("size")) or (model.has_key("size") and model["size"].shape!=(len(pglangy),len(pglangx),len(pglangz),4)):#not(model.has_key("size"))::
        model["size"]=numpy.zeros((len(pglangy),len(pglangx),len(pglangz),4))
        cposy=c_float(0.0);cposx=c_float(0.0)
        for gly,glangy in enumerate(pglangy):
            for glx,glangx in enumerate(pglangx):
                for glz,glangz in enumerate(pglangz):
                    lminym=[]
                    lminxm=[]
                    #laddy=[]
                    #laddx=[]
                    #precompute max and min size
                    for w in model["ww"]:
                        #mm=model["ww"][l]
                        angy=w.ay+glangy#(w.ay+glangy+180)%360-180
                        angx=w.ax+glangx
                        n=normal(w.ay,w.ax,glangy,glangx)
                        if n[2]<0.0001:#face not visible
                            continue
                        #laddy.append(-hsize/2.0*(cos(angy/180.0*numpy.pi)))
                        pr.getproj(0,0,glangx,glangy,glangz,w.ax,w.ay,w.x,w.y,w.z,w.lz,hsize,byref(cposx),byref(cposy))
                        #nposy=w.y*cos(glangy/180.0*numpy.pi)-hsize/2.0*(cos(angy/180.0*numpy.pi))-w.z*sin(angy/180.0*numpy.pi)
                        #nposx=w.x*cos(glangx/180.0*numpy.pi)-hsize/2.0*(cos(angx/180.0*numpy.pi))-w.z*sin(angx/180.0*numpy.pi)
                        #lminym.append(nposy-hsize/2.0*(cos(angy/180.0*numpy.pi)))
                        nposy=cposy.value
                        nposx=cposx.value
                        #print glangy,glangx,glangz,nposy,nposx
                        lminym.append(nposy)
                        #print "N=",n,hsize,hsize*(1+n[1])
                        lminym.append(nposy+hsize*(1+n[0]))#(cos(angy/180.0*numpy.pi)))
                        #lminym.append(nposy-hsize/2.0*(1-n[0]))#(cos(angy/180.0*numpy.pi)))
                        lminxm.append(nposx)
                        lminxm.append(nposx+hsize*(1+n[1]))#(cos(angx/180.0*numpy.pi)))
                        #lminxm.append(nposx-hsize/2.0*(1-n[1]))#(cos(angx/180.0*numpy.pi)))
                    #print angx,cos(angx/180.0*numpy.pi)
                    if lminym==[]:
                        (minym,minxm,maxym,maxxm)=(.0,.0,.0,.0)
                    else:
                        minym=numpy.min(lminym)#*cos(angy/180.0*numpy.pi)
                        minxm=numpy.min(lminxm)#*cos(angx/180.0*numpy.pi)
                        maxym=numpy.max(lminym)#*cos(angy/180.0*numpy.pi)#project.pattern4_bis(angy).shape[1]
                        maxxm=numpy.max(lminxm)#*cos(angx/180.0*numpy.pi)#project.pattern4_bis(angx).shape[1]
                    model["size"][gly,glx,glz]=(minym,minxm,maxym,maxxm)#numpy.round((minym,minxm,maxym,maxxm))
    return model["size"]

def modelsize(model,pglangy,pglangx,pglangz,force=False):
    hsize=model["ww"][0].mask.shape[0]
    if force or not(model.has_key("size")) or (model.has_key("size") and model["size"].shape!=(len(pglangy),len(pglangx),len(pglangz),4)):#not(model.has_key("size"))::
        model["size"]=numpy.zeros((len(pglangy),len(pglangx),len(pglangz),4))
        cposy=c_float(0.0);cposx=c_float(0.0)
        cposy2=c_float(0.0);cposx2=c_float(0.0)
        for gly,glangy in enumerate(pglangy):
            for glx,glangx in enumerate(pglangx):
                for glz,glangz in enumerate(pglangz):
                    lminym=[]
                    lminxm=[]
                    #laddy=[]
                    #laddx=[]
                    #precompute max and min size
                    for w in model["ww"]:
                        #mm=model["ww"][l]
                        #angy=w.ay+glangy#(w.ay+glangy+180)%360-180
                        #angx=w.ax+glangx
                        n=normal(w.ay,w.ax,glangy,glangx)
                        if n[2]<0.0001:#face not visible
                            continue
                        #laddy.append(-hsize/2.0*(cos(angy/180.0*numpy.pi)))
                        pr.getproj2(-hsize/2.0,-hsize/2.0,glangx,glangy,glangz,w.ax,w.ay,w.x,w.y,w.z,w.lz,hsize,byref(cposx),byref(cposy))
                        pr.getproj2(hsize/2.0,hsize/2.0,glangx,glangy,glangz,w.ax,w.ay,w.x,w.y,w.z,w.lz,hsize,byref(cposx2),byref(cposy2))
                        nposy=cposy.value
                        nposx=cposx.value
                        nposy2=cposy2.value
                        nposx2=cposx2.value
                        #print glangy,glangx,glangz,nposy,nposx
                        lminym.append(nposy)
                        #print "N=",n,hsize,hsize*(1+n[1])
                        lminym.append(nposy2)#(cos(angy/180.0*numpy.pi)))
                        #lminym.append(nposy-hsize/2.0*(1-n[0]))#(cos(angy/180.0*numpy.pi)))
                        lminxm.append(nposx)
                        lminxm.append(nposx2)#(cos(angx/180.0*numpy.pi)))
                        #lminxm.append(nposx-hsize/2.0*(1-n[1]))#(cos(angx/180.0*numpy.pi)))
                    #print angx,cos(angx/180.0*numpy.pi)
                    if lminym==[]:
                        (minym,minxm,maxym,maxxm)=(.0,.0,.0,.0)
                    else:
                        minym=numpy.min(lminym)#*cos(angy/180.0*numpy.pi)
                        minxm=numpy.min(lminxm)#*cos(angx/180.0*numpy.pi)
                        maxym=numpy.max(lminym)#*cos(angy/180.0*numpy.pi)#project.pattern4_bis(angy).shape[1]
                        maxxm=numpy.max(lminxm)#*cos(angx/180.0*numpy.pi)#project.pattern4_bis(angx).shape[1]
                    model["size"][gly,glx,glz]=(minym,minxm,maxym,maxxm)#numpy.round((minym,minxm,maxym,maxxm))
    return model["size"]


def det2(model,hog,ppglangy=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90],ppglangx=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90],ppglangz=[-20,-10,0,10,20],selangy=None,selangx=None,selangz=None,bis=BIS,k=1,usebiases=USEBIASES):
    if selangy==None:
        selangy=range(len(ppglangy))
    if selangx==None:
        selangx=range(len(ppglangx))
    if selangz==None:
        selangz=range(len(ppglangz))
    return det2_cache(model,hog,ppglangy,ppglangx,ppglangz,selangy,selangx,selangz,bis,k,usebiases)

def det2_def(model,hog,ppglangy=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90],ppglangx=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90],ppglangz=[-20,-10,0,10,20],selangy=None,selangx=None,selangz=None,bis=BIS,k=1,usebiases=USEBIASES):
    if selangy==None:
        selangy=range(len(ppglangy))
    if selangx==None:
        selangx=range(len(ppglangx))
    if selangz==None:
        selangz=range(len(ppglangz))
    return det2_cache_def(model,hog,ppglangy,ppglangx,ppglangz,selangy,selangx,selangz,bis,k,usebiases)

#@autojit
def det2_(model,hog,ppglangy,ppglangx,ppglangz,selangy,selangx,selangz,bis,k):
#def det2_(model,hog,ppglangy=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90],ppglangx=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90],selangy=None,selangx=None,bis=BIS,k=1):
    #if selangy==None:
    #    selangy=range(len(ppglangy))
    #if selangx==None:
    #    selangx=range(len(ppglangx))
    hsy=hog.shape[0]
    hsx=hog.shape[1]
    prec=[]
    for w in model["ww"]:
        prec.append(project.precompute(w.mask,hog))
    modelsize(model,ppglangy,ppglangx,ppglangz)
    maxym=numpy.max(model["size"][:,:,:,2])#+1#numpy.max([el.y for el in model["ww"]])+1
    maxxm=numpy.max(model["size"][:,:,:,3])#+1#numpy.max([el.x for el in model["ww"]])+1
    minym=numpy.min(model["size"][:,:,:,0])#-1#numpy.max([el.y for el in model["ww"]])+1
    minxm=numpy.min(model["size"][:,:,:,1])#-1#numpy.max([el.x for el in model["ww"]])+1
    deltay=maxym-minym
    deltax=maxxm-minxm
    maxmy = int(deltay+1)
    maxmx = int(deltax+1)
    #maxmy = numpy.round(deltay+1)
    #maxmx = numpy.round(deltax+1)
    hsize=model["ww"][0].mask.shape[0]
    #res=-1000*numpy.ones((len(ppglangy),len(ppglangx),hsy-minym+maxym+hsize+2,hsx-minxm+maxxm+hsize+2),dtype=numpy.float32)      
    res=numpy.ones((len(ppglangy),len(ppglangx),len(ppglangz),hsy-minym+maxym+hsize+2,hsx-minxm+maxxm+hsize+2),dtype=numpy.float32)*numpy.float32(-1000.0)
    #resc=res.copy()
    nposy=c_float(0.0);nposx=c_float(0.0)
    for gly in selangy:
        for glx in selangx:
            for glz in selangz:
                if usebiases:
                    res[gly,glx,glz]=model["biases"][gly,glx,glz]*k#0
                else:
                    res[gly,glx,glz]=0
                lminym=[]
                lminxm=[]
                minym=model["size"][gly,glx,glz,0];minxm=model["size"][gly,glx,glz,1]
                for l in range(len(model["ww"])):
                    mm=model["ww"][l]
                    #angy=(w.ay+glangy)
                    #angx=(w.ax+glangx)
                    n=normal(w.ay,w.ax,glangy,glangx)
                    #print "Angles",angy,angx,n,n[2]>0.0001
                    if n[2]<0.0001:#face not visible
                        continue
                    scr=project.project(prec[l],project.pattern4_cos(n[0]),project.pattern4_cos(n[1]))
                    pr.getproj(minxm,minym,ppglangx[glx],ppglangy[gly],ppglangz[glz],mm.ax,mm.ay,mm.x,mm.y,mm.z,mm.lz,hsize,byref(nposx),byref(nposy))
                    pposy=nposy.value
                    pposx=nposx.value
                    posy=int(floor(pposy))
                    posx=int(floor(pposx))
                    disty=pposy-posy
                    distx=pposx-posx
                    #print maxmy-posy,maxmx-posx,nposy,nposx,glangx,angx
                    res[gly,glx,glz,maxmy-posy:maxmy-posy+hsy+hsize,maxmx-posx:maxmx-posx+hsx+hsize]+=(1-disty)*(1-distx)*scr
                    res[gly,glx,glz,maxmy-posy:maxmy-posy+hsy+hsize,maxmx-(posx+1):maxmx-(posx+1)+hsx+hsize]+=(1-disty)*(distx)*scr
                    res[gly,glx,glz,maxmy-(posy+1):maxmy-(posy+1)+hsy+hsize,maxmx-(posx):maxmx-(posx)+hsx+hsize]+=(disty)*(1-distx)*scr
                    res[gly,glx,glz,maxmy-(posy+1):maxmy-(posy+1)+hsy+hsize,maxmx-(posx+1):maxmx-(posx+1)+hsx+hsize]+=(disty)*(distx)*scr
                    #pr.interpolate(res.shape[0],res.shape[1],res.shape[2],res.shape[3],res.shape[4],res,glx,gly,glz,maxmx,maxmy,posx,posy,hsx,hsy,hsize,distx,disty,scr)
    #if numpy.sum(numpy.abs(res-resc))>0.0001:
    #    gsdgdf
    return res                

#from prof import *
#@do_profile()
def det2_cache(model,hog,ppglangy,ppglangx,ppglangz,selangy,selangx,selangz,bis,k,usebiases):
#def det2_(model,hog,ppglangy=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90],ppglangx=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90],selangy=None,selangx=None,bis=BIS,k=1):
    #if selangy==None:
    #    selangy=range(len(ppglangy))
    #if selangx==None:
    #    selangx=range(len(ppglangx))
    hsy=hog.shape[0]
    hsx=hog.shape[1]
    prec=[]
    #print "Entering"
    for w in model["ww"]:
        prec.append(project.precompute(w.mask,hog))
    modelsize(model,ppglangy,ppglangx,ppglangz)
    maxym=numpy.max(model["size"][:,:,:,2])#+1#numpy.max([el.y for el in model["ww"]])+1
    maxxm=numpy.max(model["size"][:,:,:,3])#+1#numpy.max([el.x for el in model["ww"]])+1
    minym=numpy.min(model["size"][:,:,:,0])#-1#numpy.max([el.y for el in model["ww"]])+1
    minxm=numpy.min(model["size"][:,:,:,1])#-1#numpy.max([el.x for el in model["ww"]])+1
    deltay=maxym-minym
    deltax=maxxm-minxm
    maxmy = int(deltay+1)
    maxmx = int(deltax+1)
    #maxmy = numpy.round(deltay+1)
    #maxmx = numpy.round(deltax+1)
    hsize=model["ww"][0].mask.shape[0]
    #res=-1000*numpy.ones((len(ppglangy),len(ppglangx),hsy-minym+maxym+hsize+2,hsx-minxm+maxxm+hsize+2),dtype=numpy.float32)      
    res=numpy.ones((len(ppglangy),len(ppglangx),len(ppglangz),hsy-minym+maxym+hsize+2,hsx-minxm+maxxm+hsize+2),dtype=numpy.float32)*numpy.float32(-1000.0)
    #resc=res.copy()
    nposy=c_float(0.0);nposx=c_float(0.0)
    cache=numpy.zeros((len(model["ww"]),len(ppglangy),len(ppglangx)),dtype=object)
    for gly in selangy:
        for glx in selangx:
            for glz in selangz:
                if usebiases:
                    if glz>=model["biases"].shape[2] and gly>=model["biases"].shape[0]:
                        res[gly,glx,glz]=model["biases"][0,glx]*k#0
                    elif glz>=model["biases"].shape[2]:
                        res[gly,glx,glz]=model["biases"][gly,glx,0]*k#0
                    elif gly>=model["biases"].shape[0]:
                        res[gly,glx,glz]=model["biases"][0,glx,glz]*k#0
                    else:
                        res[gly,glx,glz]=model["biases"][gly,glx,glz]*k#0
                else:
                    res[gly,glx,glz]=0
                #resc[gly,glx,glz]=model["biases"][gly,glx]*k#0
                lminym=[]
                lminxm=[]
                minym=model["size"][gly,glx,glz,0];minxm=model["size"][gly,glx,glz,1]
                for l in range(len(model["ww"])):
                    mm=model["ww"][l]
                    angy=(mm.ay+ppglangy[gly])
                    angx=(mm.ax+ppglangx[glx])
                    n=normal(mm.ay,mm.ax,ppglangy[gly],ppglangx[glx])
                    #print "Angles",angy,angx,n,n[2]>0.0001
                    if n[2]<0.0001:#face not visible
                        continue
                    #angy=(mm.ay+ppglangy[gly]+180)%360-180#(w.ay+glangy+180)%360-180
                    #angx=(mm.ax+ppglangx[glx]+180)%360-180
                    #angz=(ppglangz[glz]+180)%360-180
                    #if abs(angx)>90 or abs(angy)>90:
                    #    continue               
                    #if bis:
                    #    scr=project.project_bis(prec[l],project.pattern4_bis(angy),project.pattern4_bis(angx))
                    #else:
                    #NOTICE that now bis does not work!!!!
                    if type(cache[l,gly,glx])==int:
                        #scr=project.project(prec[l],project.pattern4_cos(n[0]),project.pattern4_cos(n[1]))
                        scr=project.project(prec[l],project.pattern4_cos(n[1]),project.pattern4_cos(n[0]))
                        auxscr=scr.copy()
                        cache[l,gly,glx]=auxscr
                        if abs(n[0])<sin((30+45)/2.0/180.0*numpy.pi) and abs(n[1])<sin((30+45)/2.0/180.0*numpy.pi):
                        #should be considered for general angles
                        #if abs(angy)<(30+45)/2.0 and abs(angx)<(30+45)/2.0:
                            #cache[l,4:9,4:9]=auxscr
                            for lly in selangy:
                                for llx in selangx:
                                    auxn=normal(mm.ay,mm.ax,ppglangy[lly],ppglangx[llx])
                                    if abs(auxn[0])<sin((30+45)/2.0/180.0*numpy.pi) and abs(auxn[1])<sin((30+45)/2.0/180.0*numpy.pi):
                                    #if abs(mm.ay+ppglangy[lly])<(30+45)/2.0 and abs(mm.ax+ppglangx[llx])<(30+45)/2.0:
                                        cache[l,lly,llx]=auxscr
                    else:
                        scr=cache[l,gly,glx]
                    #if type(scr)!=numpy.ndarray:
                    #    print type(scr)
                    #    dsfsd
                    #print scr.shape
                    #nposy=-minym+mm.y*cos(ppglangy[gly]/180.0*numpy.pi)-hsize/2.0*(cos(angy/180.0*numpy.pi))-mm.z*sin(angy/180.0*numpy.pi)
                    #nposx=-minxm+mm.x*cos(ppglangx[glx]/180.0*numpy.pi)-hsize/2.0*(cos(angx/180.0*numpy.pi))-mm.z*sin(angx/180.0*numpy.pi)
                    pr.getproj(minxm,minym,ppglangx[glx],ppglangy[gly],ppglangz[glz],mm.ax,mm.ay,mm.x,mm.y,mm.z,mm.lz,hsize,byref(nposx),byref(nposy))
                    #print nposy,nposx
                    #nposy=-miny+project.getproj(mm.y,mm.z,glangy,angy)
                    #nposx=-minx+project.getproj(mm.x,mm.z,glangx,angx)
                    pposy=nposy.value
                    pposx=nposx.value
                    #print "Dense input",minxm,minym,ppglangx[glx],ppglangy[gly],ppglangz[glz],mm.ax,mm.ay,mm.x,mm.y,mm.z,mm.lz,hsize
                    #print "Dense",pposy,pposx
                    posy=int(floor(pposy))
                    posx=int(floor(pposx))
                    disty=pposy-posy
                    distx=pposx-posx
                    #print maxmy-posy,maxmx-posx,nposy,nposx,glangx,angx
                    res[gly,glx,glz,maxmy-posy:maxmy-posy+hsy+hsize,maxmx-posx:maxmx-posx+hsx+hsize]+=(1-disty)*(1-distx)*scr
                    res[gly,glx,glz,maxmy-posy:maxmy-posy+hsy+hsize,maxmx-(posx+1):maxmx-(posx+1)+hsx+hsize]+=(1-disty)*(distx)*scr
                    res[gly,glx,glz,maxmy-(posy+1):maxmy-(posy+1)+hsy+hsize,maxmx-(posx):maxmx-(posx)+hsx+hsize]+=(disty)*(1-distx)*scr
                    res[gly,glx,glz,maxmy-(posy+1):maxmy-(posy+1)+hsy+hsize,maxmx-(posx+1):maxmx-(posx+1)+hsx+hsize]+=(disty)*(distx)*scr
                    #pr.interpolate(res.shape[0],res.shape[1],res.shape[2],res.shape[3],res.shape[4],res,glx,gly,glz,maxmx,maxmy,posx,posy,hsx,hsy,hsize,distx,disty,scr)
    #if numpy.sum(numpy.abs(res-resc))>0.0001:
    #    gsdgdf
    return res                

def reduceQ(Q):
    """
    reduces Q form 4x4 to 3x3 in the sense: f=[x,y,z,1]'Q[x,y,z,1], min_z f such that df/dz=0 
    --> z=-(q_xz*x+q_yz*y+q_tz)/qz where
    Q=[[qx qxy qxz qxt],
       [qyx qy qyz qyt],
       [qzx qzy qz qzt],
       [qtx qty qtz qt]]
    """
    qx=Q[0,0];qy=Q[1,1];qz=Q[2,2];qt=Q[3,3]
    assert (Q-Q.T<0.00001).all()#symmtric
    qxy=Q[1,0];qxz=Q[2,0];qxt=Q[3,0]
    qyx=Q[0,1];qyz=Q[2,1];qyt=Q[3,1]
    qzx=Q[0,2];qzy=Q[1,2];qzt=Q[3,2]    
    qtx=Q[0,3];qty=Q[1,3];qtz=Q[2,3]    
    Qr=numpy.zeros((3,3),dtype=Q.dtype)
    Qr[0,0]=qx+qz*qxz*qzx;Qr[1,1]=qy+qz*qyz*qzy;Qr[2,2]=qt-2*qtz/qz*(qxz+qyz+qtz)
    Qr[1,0]=Qr[0,1]=qxy+qz*qyz*qxz
    Qr[2,0]=Qr[0,2]=qtx+(qz+qtz+qxz-2*qxz/qz*(qxz+qyz+qtz))/2.0
    Qr[2,1]=Qr[1,2]=qty+(qz+qtz+qyz-2*qyz/qz*(qxz+qyz+qtz))/2.0
    return Qr

def reduceQ2(Q):
    """
    reduces Q form 4x4 to 3x3 in the sense: f=[x,y,z,1]'Q[x,y,z,1], min_z f such that df/dz=0 
    --> z=-(q_xz*x+q_yz*y+q_tz)/qz where
    Q=[[qx qxy qxz qxt],
       [qyx qy qyz qyt],
       [qzx qzy qz qzt],
       [qtx qty qtz qt]]
    """
    qx=Q[0,0];qy=Q[1,1];qz=Q[2,2]
    assert (Q-Q.T<0.00001).all()#symmtric
    qxy=Q[1,0];qxz=Q[2,0];
    qyx=Q[0,1];qyz=Q[2,1];
    qzx=Q[0,2];qzy=Q[1,2];
    Qr=numpy.zeros((2,2),dtype=Q.dtype)
    Qr[0,0]=qx-qxz*qzx/qz;Qr[1,1]=qy-qyz*qzy/qz
    Qr[1,0]=Qr[0,1]=qxy-qyz*qxz/qz
    return Qr


import dt#,time
#tt=0;ta=0

#from scipy.ndimage.interpolation import map_coordinates
#from prof import *
#@do_profile()
def det2_cache_def(model,hog,ppglangy,ppglangx,ppglangz,selangy,selangx,selangz,bis,k,usebiases):
#def det2_(model,hog,ppglangy=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90],ppglangx=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90],selangy=None,selangx=None,bis=BIS,k=1):
    #if selangy==None:
    #    selangy=range(len(ppglangy))
    #if selangx==None:
    #    selangx=range(len(ppglangx))
    hsy=hog.shape[0]
    hsx=hog.shape[1]
    prec=[]
    #print "Entering"
    for w in model["ww"]:
        prec.append(project.precompute(w.mask,hog))
    modelsize(model,ppglangy,ppglangx,ppglangz,force=True)
    maxym=numpy.max(model["size"][:,:,:,2])#+1#numpy.max([el.y for el in model["ww"]])+1
    maxxm=numpy.max(model["size"][:,:,:,3])#+1#numpy.max([el.x for el in model["ww"]])+1
    minym=numpy.min(model["size"][:,:,:,0])#-1#numpy.max([el.y for el in model["ww"]])+1
    minxm=numpy.min(model["size"][:,:,:,1])#-1#numpy.max([el.x for el in model["ww"]])+1
    deltay=maxym-minym
    deltax=maxxm-minxm
    maxmy = int(deltay+1)
    maxmx = int(deltax+1)
    #maxmy = numpy.round(deltay+1)
    #maxmx = numpy.round(deltax+1)
    hsize=model["ww"][0].mask.shape[0]
    #res=-1000*numpy.ones((len(ppglangy),len(ppglangx),hsy-minym+maxym+hsize+2,hsx-minxm+maxxm+hsize+2),dtype=numpy.float32)      
    res=numpy.ones((len(ppglangy),len(ppglangx),len(ppglangz),hsy-minym+maxym+hsize+2,hsx-minxm+maxxm+hsize+2),dtype=numpy.float32)*numpy.float32(-1000.0)
    #res2=numpy.ones((len(ppglangy),len(ppglangx),len(ppglangz),hsy-minym+maxym+hsize+2,hsx-minxm+maxxm+hsize+2),dtype=numpy.float32)*numpy.float32(-1000.0)
    #resp=numpy.zeros((len(model["ww"]),len(ppglangy),len(ppglangx),len(ppglangz),hsy-minym+maxym+hsize+2,hsx-minxm+maxxm+hsize+2),dtype=numpy.float32)*numpy.float32(-1000.0)
    #ldy=numpy.zeros((len(model["ww"]),len(ppglangy),len(ppglangx),len(ppglangz),hsy-minym+maxym+hsize+2,hsx-minxm+maxxm+hsize+2),dtype=numpy.float32)*numpy.int32(-1)
    #ldx=numpy.zeros((len(model["ww"]),len(ppglangy),len(ppglangx),len(ppglangz),hsy-minym+maxym+hsize+2,hsx-minxm+maxxm+hsize+2),dtype=numpy.float32)*numpy.int32(-1)
    ldy=numpy.ones((len(model["ww"]),len(ppglangy),len(ppglangx),len(ppglangz),hsy-minym+maxym+hsize+2,hsx-minxm+maxxm+hsize+2),dtype=numpy.float32)#*numpy.int32(-1)
    ldy=ldy[:,:,:,:].cumsum(4)-1
    ldx=numpy.ones((len(model["ww"]),len(ppglangy),len(ppglangx),len(ppglangz),hsy-minym+maxym+hsize+2,hsx-minxm+maxxm+hsize+2),dtype=numpy.float32)#*numpy.int32(-1)
    ldx=ldx[:,:,:,:].cumsum(5)-1
    #resc=res.copy()
    nposy=c_float(0.0);nposx=c_float(0.0)
    defy=c_float(0.0);defx=c_float(0.0)
    cache=numpy.zeros((len(model["ww"]),len(ppglangy),len(ppglangx)),dtype=object)
    #ldy=-numpy.ones((len(ppglangy),len(ppglangx),len(ppglangz),len(model["ww"])),dtype=object)
    #ldx=-numpy.ones((len(ppglangy),len(ppglangx),len(ppglangz),len(model["ww"])),dtype=object)
    for gly in selangy:
        for glx in selangx:
            for glz in selangz:
                if usebiases:
                    if glz>=model["biases"].shape[2] and gly>=model["biases"].shape[0]:
                        res[gly,glx,glz]=model["biases"][0,glx]*k#0
                        #res2[gly,glx,glz]=model["biases"][0,glx]*k#0
                    elif glz>=model["biases"].shape[2]:
                        res[gly,glx,glz]=model["biases"][gly,glx,0]*k#0
                        #res2[gly,glx,glz]=model["biases"][gly,glx,0]*k#0
                    elif gly>=model["biases"].shape[0]:
                        res[gly,glx,glz]=model["biases"][0,glx,glz]*k#0
                        #res2[gly,glx,glz]=model["biases"][0,glx,glz]*k#0
                    else:
                        res[gly,glx,glz]=model["biases"][gly,glx,glz]*k#0
                        #res2[gly,glx,glz]=model["biases"][gly,glx,glz]*k#0
                else:
                    res[gly,glx,glz]=0
                    #res2[gly,glx,glz]=0
                #resc[gly,glx,glz]=model["biases"][gly,glx]*k#0
                lminym=[]
                lminxm=[]
                minym=model["size"][gly,glx,glz,0];minxm=model["size"][gly,glx,glz,1]
                for l in range(len(model["ww"])):
                    mm=model["ww"][l]
                    #angy=(mm.ay+ppglangy[gly])
                    #angx=(mm.ax+ppglangx[glx])
                    n=normal(mm.ay,mm.ax,ppglangy[gly],ppglangx[glx])
                    #print "Angles",angy,angx,n,n[2]>0.0001
                    if n[2]<0.0001:#face not visible
                        continue
                    if type(cache[l,gly,glx])==int:
                        scr=project.project(prec[l],project.pattern4_cos(n[1]),project.pattern4_cos(n[0]))
                        auxscr=scr.copy()
                        cache[l,gly,glx]=auxscr
                        #should be considered for general angles
                        #if abs(angy)<(30+45)/2.0 and abs(angx)<(30+45)/2.0:
                        if abs(n[0])<sin((30+45)/2.0/180.0*numpy.pi) and abs(n[1])<sin((30+45)/2.0/180.0*numpy.pi):
                            #cache[l,4:9,4:9]=auxscr
                            for lly in selangy:
                                for llx in selangx:
                                    auxn=normal(mm.ay,mm.ax,ppglangy[lly],ppglangx[llx])
                                    if abs(auxn[0])<sin((30+45)/2.0/180.0*numpy.pi) and abs(auxn[1])<sin((30+45)/2.0/180.0*numpy.pi):
                                    #if abs(mm.ay+ppglangy[lly])<(30+45)/2.0 and abs(mm.ax+ppglangx[llx])<(30+45)/2.0:
                                        cache[l,lly,llx]=auxscr
                    else:
                        scr=cache[l,gly,glx]
                    #if type(scr)!=numpy.ndarray:
                    #    print type(scr)
                    #    dsfsd
                    #print scr.shape
                    #nposy=-minym+mm.y*cos(ppglangy[gly]/180.0*numpy.pi)-hsize/2.0*(cos(angy/180.0*numpy.pi))-mm.z*sin(angy/180.0*numpy.pi)
                    #nposx=-minxm+mm.x*cos(ppglangx[glx]/180.0*numpy.pi)-hsize/2.0*(cos(angx/180.0*numpy.pi))-mm.z*sin(angx/180.0*numpy.pi)
                    pr.getproj(minxm,minym,ppglangx[glx],ppglangy[gly],ppglangz[glz],mm.ax,mm.ay,mm.x,mm.y,mm.z,mm.lz,hsize,byref(nposx),byref(nposy))
                    # set lz=0 because it is used in the dt
                    #pr.getproj(minxm,minym,ppglangx[glx],ppglangy[gly],ppglangz[glz],mm.ax,mm.ay,mm.x,mm.y,mm.z,0,hsize,byref(nposx),byref(nposy))
                    #print nposy,nposx
                    #nposy=-miny+project.getproj(mm.y,mm.z,glangy,angy)
                    #nposx=-minx+project.getproj(mm.x,mm.z,glangx,angx)
                    pposy=nposy.value
                    pposx=nposx.value
                    #print "Dense input",minxm,minym,ppglangx[glx],ppglangy[gly],ppglangz[glz],mm.ax,mm.ay,mm.x,mm.y,mm.z,mm.lz,hsize
                    #print "Dense",pposy,pposx
                    posy=int(floor(pposy))
                    posx=int(floor(pposx))
                    disty=pposy-posy
                    distx=pposx-posx
                    #print maxmy-posy,maxmx-posx,nposy,nposx,glangx,angx
                    #auxscr=numpy.zeros((scr.shape[0]+1,scr.shape[1]+1,4),dtype=scr.dtype)
                    #auxloc=numpy.zeros(numpy.array(scr.shape)+1,dtype=numpy.int32)
                    #auxscr[:-1,:-1,0]=scr
                    #auxscr[1:,:-1,1]= (disty)*(1-distx)*scr
                    #auxscr[:-1,1:,2]= (1-disty)*(distx)*scr
                    #auxscr[1:,1:,3]= (disty)*(distx)*scr 
                    #transform parameters
                    #for the moment is wrong because it should generate an elipsis with any possible rotation and not only aligned to the axis
                    #ay=mm.dfay;ax=mm.dfax,by=mm.dfby,bx=mm.dfbx
                    #pr.getproj(minxm,minym,ppglangx[glx],ppglangy[gly],ppglangz[glz],mm.ax,mm.ay,mm.dfax,mm.dfay,mm.dfaz,mm.lz,hsize,byref(defx),byref(defy))
                    #build Q
                    Q=numpy.array([[mm.dfax,0,0,0],
                                   [0,mm.dfay,0,0],
                                   [0,0,mm.dfaz,0],#[0,0,mm.dfaz,mm.lz/2],
                                   [0,0,0,1]])#[0,0,mm.lz/2,1]])                    
                    #transform local--> can be dene only once
                    Ry=Mroty(mm.ax)
                    Rx=Mrotx(mm.ay)
                    Q=numpy.dot(numpy.dot(Ry,Q),Ry.T)
                    Q=numpy.dot(numpy.dot(Rx,Q),Rx.T)
                    #transform global
                    Ry=Mroty(ppglangx[glx])
                    Rx=Mrotx(ppglangy[gly])
                    Q=numpy.dot(numpy.dot(Ry,Q),Ry.T)
                    Q=numpy.dot(numpy.dot(Rx,Q),Rx.T)
                    Qr=reduceQ2(Q[:3,:3])    
                    #modified dt
                    #auxscr,ddy,ddx=dt.mydt(auxscr,ay,ax,by,bx)
                    ay=Qr[1,1];ax=Qr[0,0];axy=Qr[1,0];by=0;bx=0
                    #by=2*Qr[2,1];bx=2*Qr[2,0]
                    #auxscr,ddy,ddx=dt.mydt(auxscr,ay,ax,by,bx)
                    #dfds
                    #auxscr,ddy,ddx=dt.dt2(auxscr,ay,ax,axy,by,bx)
                    #ay=0.001;ax=0.001;axy=0.0;by=0;bx=0#by=2*Qr[2,1];bx=2*Qr[2,0]
                    #mt=time.time()
                    auxscr,ddy,ddx=dt.dt2rot(scr,ay,ax,axy,by,bx)
                    #for t in range(1):
                        #smp=numpy.random.random(2)
                        #pt=(smp*(numpy.array(auxscr.shape)-5).round()).astype(numpy.int)
                        ##pt=[0,0]
                        #parts=[ddy[pt[0],pt[1]],ddx[pt[0],pt[1]]]
                        #auxhog=project.invprjhog(hog[floor(pt[0]-round(parts[0])+1):,floor(pt[1]-round(parts[1])+1):],project.pattern4_cos(n[0]),project.pattern4_cos(n[1]))
                        #myscr=numpy.sum(auxhog*mm.mask)
                        ##print myscr
                        ##for mdy in range(-5,6):
                        ##    for mdx in range(-5,6):
                        ##        #print (scr[pt[0]+mdy,pt[1]+mdx]-myscr),mdy,mdx
                        ##        if abs(scr[pt[0]+mdy+5,pt[1]+mdx+5]-myscr)<0.00000001:
                        ##            print (scr[pt[0]+mdy+5,pt[1]+mdx+5]-myscr),mdy,mdx,posy,posx
                        ##            print "Found right..."
                        ##            raw_input()
                        #assert(abs(scr[pt[0]+5,pt[1]+5]-myscr)<0.00001)
                    #print
                    #auxscr=scr
                    #auxscr,ddy,ddx=dt.dt2rot(scr[::2,::2],ay,ax,axy,by,bx)
                    #idx=numpy.mgrid[:scr.shape[0],:scr.shape[1]]/2.0
                    #auxscr=map_coordinates(auxscr,idx,order=1)
                    #ddy=map_coordinates(ddy,idx,order=1)
                    #ddx=map_coordinates(ddx,idx,order=1)
                    #tt+= time.time()-mt
                    #mt=time.time()    
                    ldy[l,gly,glx,glz,maxmy-posy:maxmy-posy+hsy+hsize,maxmx-(posx):maxmx-(posx)+hsx+hsize]=ddy+maxmy-posy
                    ldx[l,gly,glx,glz,maxmy-posy:maxmy-posy+hsy+hsize,maxmx-(posx):maxmx-(posx)+hsx+hsize]=ddx+maxmx-posx
                    #resp[l,gly,glx,glz,maxmy-posy:maxmy-posy+hsy+hsize,maxmx-(posx):maxmx-(posx)+hsx+hsize]=auxscr
                    res[gly,glx,glz,maxmy-posy:maxmy-posy+hsy+hsize,maxmx-(posx):maxmx-(posx)+hsx+hsize]=res[gly,glx,glz,maxmy-posy:maxmy-posy+hsy+hsize,maxmx-(posx):maxmx-(posx)+hsx+hsize]+auxscr
                    #res2[gly,glx,glz,maxmy-posy:maxmy-posy+hsy+hsize,maxmx-(posx):maxmx-(posx)+hsx+hsize]=res2[gly,glx,glz,maxmy-posy:maxmy-posy+hsy+hsize,maxmx-(posx):maxmx-(posx)+hsx+hsize]+scr
                    #ta+= time.time()-mt
                    #print tt,ta," ",
                    #asddsf
                    #print "Part in:",l,Qr
                    if 0: #l==0: #plot gaussian for each part
                        p=numpy.array((10,5)).reshape(2,1)
                        print "PART:",l
                        print "def",mm.dfax,mm.dfay,mm.dfaz
                        Q1=Q[:3,:3]
                        print "Q=",Q1
                        Q1r=reduceQ2(Q1)
                        print "Qr=",Q1r
                        print "p=",p
                        z=-(Q1[2,0]*p[0]+Q1[2,1]*p[1])/Q[2,2]
                        print "z=",z
                        z2=-(Q1[2,0]*p[1]+Q1[2,1]*p[0])/Q[2,2]
                        print "z2=",z2
                        p3D=numpy.array((p[0],p[1],z)).reshape(3,1)
                        s2d=numpy.dot(numpy.dot(p.T,Q1r),p)[0,0]
                        print "Score2D",s2d
                        s3d=numpy.dot(numpy.dot(p3D.T,Q1),p3D)[0,0]
                        print "Score3D",s3d
                        p3D[2]=z+1
                        print "Score3D z+1",numpy.dot(numpy.dot(p3D.T,Q1),p3D)[0,0]
                        p3D[2]=z-1
                        print "Score3D z-1",numpy.dot(numpy.dot(p3D.T,Q1),p3D)[0,0]
                        print "glx",ppglangx[glx],"gly",ppglangy[gly],"ax",mm.ax,ax,"ay",mm.ay,ay, "axy",axy
                        assert(s2d-s3d<0.0001)
                        im=numpy.zeros((100,100),dtype=numpy.float32)
                        im[50,50]=10
                        dtim,iddy,iddx=dt.dt2rot(im,ay,ax,axy,by,bx)
                        pylab.clf()
                        pylab.imshow(dtim)
                        pylab.draw()
                        pylab.show()
                        raw_input()

                    #res[gly,glx,glz,maxmy-posy:maxmy-posy+hsy+hsize,maxmx-(posx+1):maxmx-(posx+1)+hsx+hsize]+=(1-disty)*(distx)*scr
                    #res[gly,glx,glz,maxmy-(posy+1):maxmy-(posy+1)+hsy+hsize,maxmx-(posx):maxmx-(posx)+hsx+hsize]+=(disty)*(1-distx)*scr
                    #res[gly,glx,glz,maxmy-(posy+1):maxmy-(posy+1)+hsy+hsize,maxmx-(posx+1):maxmx-(posx+1)+hsx+hsize]+=(disty)*(distx)*scr
                    #res[gly,glx,glz,maxmy-posy:maxmy-posy+hsy+hsize,maxmx-posx:maxmx-posx+hsx+hsize]+=(1-disty)*(1-distx)*scr
                    #res[gly,glx,glz,maxmy-posy:maxmy-posy+hsy+hsize,maxmx-(posx+1):maxmx-(posx+1)+hsx+hsize]+=(1-disty)*(distx)*scr
                    #res[gly,glx,glz,maxmy-(posy+1):maxmy-(posy+1)+hsy+hsize,maxmx-(posx):maxmx-(posx)+hsx+hsize]+=(disty)*(1-distx)*scr
                    #res[gly,glx,glz,maxmy-(posy+1):maxmy-(posy+1)+hsy+hsize,maxmx-(posx+1):maxmx-(posx+1)+hsx+hsize]+=(disty)*(distx)*scr
                    #pr.interpolate(res.shape[0],res.shape[1],res.shape[2],res.shape[3],res.shape[4],res,glx,gly,glz,maxmx,maxmy,posx,posy,hsx,hsy,hsize,distx,disty,scr)
    #if numpy.sum(numpy.abs(res-resc))>0.0001:
    #    gsdgdf
    if 0:
        pylab.figure(100)
        pylab.clf()
        pylab.imshow(res[0,18,0])
        pylab.show()
        pylab.draw()
        raw_input()
    return res,ldy,ldx            

#@autojit
def drawdet(ldet):
    if type(ldet)!=list:
        ldet=[ldet]
    for idl,l in enumerate(ldet):
       util.box(l["bbox"][0],l["bbox"][1],l["bbox"][2],l["bbox"][3],lw=2)
       pylab.text(l["bbox"][1],l["bbox"][0],"(%d,%4f,%d,%d,%d)"%(idl,l["scr"],l["ang"][0],l["ang"][1],l["ang"][2]))
       #print res[dd],l,dd[0],dd[1]

#@autojit
def rundet(img,model,angy=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90],angx=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90],angz=[-10,0,10],interv=5,sbin=4,maxdet=1000,sort="scr",bbox=None,selangy=None,selangx=None,selangz=None,numhyp=1000,k=1,bis=BIS,usebiases=USEBIASES,usedef=True,skip=15):
    hog=pyrHOG2.pyrHOG(img,interv=interv,cformat=True,sbin=sbin)
    modelsize(model,angy,angx,angz,force=True)
    hsize=model["ww"][0].mask.shape[0]
    maxym=numpy.max(model["size"][:,:,:,2])#numpy.max([el.y for el in model["ww"]])+1
    maxxm=numpy.max(model["size"][:,:,:,3])#numpy.max([el.x for el in model["ww"]])+1
    minym=numpy.min(model["size"][:,:,:,0])#numpy.max([el.y for el in model["ww"]])+1
    minxm=numpy.min(model["size"][:,:,:,1])#numpy.max([el.x for el in model["ww"]])+1
    deltay=maxym-minym+hsize+1
    deltax=maxxm-minxm+hsize+1
    #maxmy=#numpy.max(model["size"][:,:,2])+hsize+1#numpy.max([el.y for el in model["ww"]])+hsize+1
    #maxmx=#numpy.max(model["size"][:,:,3])+hsize+1#numpy.max([el.x for el in model["ww"]])+hsize+1
    show=False
    ldet=[]
    modelsize(model,angy,angx,angz,force=True)
    cposx=c_float(0.0);cposy=c_float(0.0)
    for idr,r in enumerate(hog.hog[:-skip]):
        #print idr,"/",len(hog.hog)
        if show:
            pylab.figure()
            pylab.imshow(img)
        #print "USe DEF++++++++++++++",usedef
        import time
        t=time.time()
        if usedef:
            res,ldy,ldx=det2_def(model,hog.hog[idr],ppglangy=angy,ppglangx=angx,ppglangz=angz,selangy=selangy,selangx=selangx,selangz=selangz,k=k,bis=bis,usebiases=usebiases)
            #res=det2(model,hog.hog[idr],ppglangy=angy,ppglangx=angx,ppglangz=angz,selangy=selangy,selangx=selangx,selangz=selangz,k=k,bis=bis,usebiases=usebiases)
        else:
            #res,ldy,ldx=det2_def(model,hog.hog[idr],ppglangy=angy,ppglangx=angx,ppglangz=angz,selangy=selangy,selangx=selangx,selangz=selangz,k=k,bis=bis,usebiases=usebiases)
            res=det2(model,hog.hog[idr],ppglangy=angy,ppglangx=angx,ppglangz=angz,selangy=selangy,selangx=selangx,selangz=selangz,k=k,bis=bis,usebiases=usebiases)
        #dsfsd
        #print "Pure detecotion",time.time()-t
        order=(-res).argsort(None)
        for l in range(min(numhyp,len(order))):
            dd=numpy.unravel_index(order[l],res.shape)
            (minym,minxm,maxym,maxxm)=model["size"][dd[0],dd[1],dd[2]]
            pcy1=(dd[3]-deltay+hsize/2)*sbin/hog.scale[idr]
            pcx1=(dd[4]-deltax+hsize/2)*sbin/hog.scale[idr]
            pcy2=(dd[3]-deltay+(maxym-minym)+hsize/2)*sbin/hog.scale[idr]
            pcx2=(dd[4]-deltax+(maxxm-minxm)+hsize/2)*sbin/hog.scale[idr]
            pdfy=[];pdfx=[];pddy=[];pddx=[];scrp=[]
            if usedef:
                glangy=angy[dd[0]]
                glangx=angx[dd[1]]
                glangz=angz[dd[2]]
                for idp,p in enumerate(model["ww"]):
                    #minym=model["size"][dd[0],dd[1],dd[2],0];minxm=model["size"][dd[0],dd[1],dd[2],1]
                    #auxhog=numpy.zeros((hsize,hsize,model["ww"][0].mask.shape[2]),dtype=model["ww"][0].mask.dtype)
                    #langy=(p.ay+glangy)
                    #langx=(p.ax+glangx)

                    n=normal(p.ay,p.ax,glangy,glangx)#this is the bottle-nek, find a way to compute it faster...
                    if n[2]<0.0001:#face not visible
                        pdfy.append(-1);pddy.append(0)
                        pdfx.append(-1);pddx.append(0)
                        scrp.append(0)
                        continue

                    #pr.getproj(minxm,minym,glangx,glangy,glangz,p.ax,p.ay,p.x,p.y,p.z,p.lz,hsize,byref(cposx),byref(cposy))
                    #nposy=cposy.value#+deltay
                    #nposx=cposx.value#+deltax
                    #posy=int(numpy.floor((nposy)))#+deltay
                    #posx=int(numpy.floor((nposx)))#+deltax
                    #disty=nposy-posy
                    #distx=nposx-posx
                    pdfy.append(ldy[idp,dd[0],dd[1],dd[2],dd[3],dd[4]])
                    pdfx.append(ldx[idp,dd[0],dd[1],dd[2],dd[3],dd[4]])
                    pddy.append(dd[3]-pdfy[-1])
                    pddx.append(dd[4]-pdfx[-1])
                    #y=ldy[dd[0],dd[1],dd[2],idp]
                    #x=ldx[dd[0],dd[1],dd[2],idp]
                    #if type(y)==int:#invisible
                    #    pdfy.append(-1)
                    #    pddy.append(0)
                    #else:
                    #    ppy=min(max(0,dd[3]+posy-deltay),y.shape[0]-1)
                    #    ppx=min(max(0,dd[4]+posx-deltax),x.shape[1]-1)
                    #    pdfy.append(y[ppy,ppx]);pddy.append(pdfy[-1]-ppy)
                    #if type(x)==int:#invisible
                    #    pdfx.append(-1)
                    #    pddx.append(0)
                    #else:
                    #    ppy=min(max(0,dd[3]+posy-deltay),y.shape[0]-1)
                    #    ppx=min(max(0,dd[4]+posx-deltax),x.shape[1]-1)
                    #    pdfx.append(x[ppy,ppx]);pddx.append(pdfx[-1]-ppx)
                    #scrp.append(resp[idp,dd[0],dd[1],dd[2],dd[3],dd[4]])
                    #assert(scrp[-1]-resp[idp,dd[0],dd[1],dd[2],pdfy[-1],pdfx[-1]]<0.0001)
                    if 0:
                        print deltay,deltax
                        print "Res:",dd[3],dd[4]
                        print "Def:",pdfy[-1],pdfx[-1]
                        print "DD:",round(pddy[-1]),round(pddx[-1])
                        raw_input()
            ldet.append({"id":0,"bbox":[pcy1,pcx1,pcy2,pcx2],"ang":(dd[0],dd[1],dd[2]),"scr":res[dd]-model["rho"],"scl":hog.scale[idr],"hog":idr,"pos":(dd[3],dd[4]),"fpos":(dd[3]-deltay,dd[4]-deltax),"pdef":(pdfy,pdfx),"ddef":(pddy,pddx),"scrp":scrp})
            if bbox!=None:
                ldet[-1]["ovr"]=util.overlap(ldet[-1]["bbox"],bbox)
            if show:
                util.box(pcy1,pcx1,pcy2,pcx2,lw=2)
                pylab.text(pcy1,pcx1,"(%d,%d,%d)"%(l,dd[0],dd[1]))
                print res[dd],l,dd[0],dd[1]
        if show:
            pylab.draw()
            pylab.show()
            raw_input()
    if sort=="scr":
        ldet.sort(key=lambda by: -by["scr"])
    else:
        ldet.sort(key=lambda by: -by["ovr"])
    return hog,ldet[:maxdet]

#@autojit
def getfeat(model,hog,angy,angx,angz,ang,pos,k,bis=BIS,usebiases=USEBIASES):
    import project
    lhog=[]
    hsize=model["ww"][0].mask.shape[0]
    glangy=angy[ang[0]]
    glangx=angx[ang[1]]
    glangz=angz[ang[2]]
    modelsize(model,angy,angx,angz,force=True)
    maxym=numpy.max(model["size"][:,:,:,2])#numpy.max([el.y for el in model["ww"]])+1
    maxxm=numpy.max(model["size"][:,:,:,3])#numpy.max([el.x for el in model["ww"]])+1
    minym=numpy.min(model["size"][:,:,:,0])#numpy.max([el.y for el in model["ww"]])+1
    minxm=numpy.min(model["size"][:,:,:,1])#numpy.max([el.x for el in model["ww"]])+1
    deltay=maxym-minym+1#numpy.ceil(maxym-minym)
    deltax=maxxm-minxm+1#numpy.ceil(maxxm-minxm)
    m2=hog
    m2pad=numpy.zeros((m2.shape[0]+2*deltay,m2.shape[1]+2*deltax,m2.shape[2]),dtype=m2.dtype)
    m2pad[deltay:deltay+m2.shape[0],deltax:deltax+m2.shape[1]]=m2
    #hog=m2pad
    scr=0
    cposx=c_float(0.0);cposy=c_float(0.0)
    for idp,p in enumerate(model["ww"]):
        minym=model["size"][ang[0],ang[1],ang[2],0];minxm=model["size"][ang[0],ang[1],ang[2],1]
        #lhog.append(hog[pos[0]:pos[0]+hsize,pos[1]:pos[1]+hsize])
        auxhog=numpy.zeros((hsize,hsize,model["ww"][0].mask.shape[2]),dtype=model["ww"][0].mask.dtype)
        #langy=(glangy+p.ay-180)%360+180
        #langx=(glangx+p.ax-180)%360+180
        #langy=(p.ay+glangy)
        #langx=(p.ax+glangx)
        n=normal(p.ay,p.ax,glangy,glangx)
        #print "Normal",n
        #print "Angles Feat",langy,langx,n,n[2]>0.0001
        if n[2]<0.0001:#face not visible
            lhog.append(numpy.zeros(model["ww"][idp].mask.shape,dtype=model["ww"][idp].mask.dtype))
            continue
        pr.getproj(minxm,minym,glangx,glangy,glangz,p.ax,p.ay,p.x,p.y,p.z,p.lz,hsize,byref(cposx),byref(cposy))
        nposy=cposy.value#+deltay
        nposx=cposx.value#+deltax
        #nposy=-minym+project.getproj(p.y,p.z,glangy,langy)#+deltay
        #nposx=-minxm+project.getproj(p.x,p.z,glangx,langx)#+deltax
        #print "FEAT input",minxm,minym,glangx,glangy,glangz,p.ax,p.ay,p.x,p.y,p.z,p.lz,hsize
        #print "FEAT",nposy,nposx
        posy=int(numpy.floor((nposy)))#+deltay
        posx=int(numpy.floor((nposx)))#+deltax
        disty=nposy-posy
        distx=nposx-posx
        #print posy,posx
        if bis:
            auxhog=auxhog+project.invprjhog_bis(m2pad[deltay+pos[0]+posy:,deltax+pos[1]+posx:],project.pattern4_bis(langy),project.pattern4_bis(langx))*(1-disty)*(1-distx)
            auxhog=auxhog+project.invprjhog_bis(m2pad[deltay+pos[0]+posy:,deltax+pos[1]+posx+1:],project.pattern4_bis(langy),project.pattern4_bis(langx))*(1-disty)*(distx)
            auxhog=auxhog+project.invprjhog_bis(m2pad[deltay+pos[0]+posy+1:,deltax+pos[1]+posx:],project.pattern4_bis(langy),project.pattern4_bis(langx))*(disty)*(1-distx)
            auxhog=auxhog+project.invprjhog_bis(m2pad[deltay+pos[0]+posy+1:,deltax+pos[1]+posx+1:],project.pattern4_bis(langy),project.pattern4_bis(langx))*(disty)*(distx)
        else:
            auxhog=auxhog+project.invprjhog(m2pad[deltay+pos[0]+posy:,deltax+pos[1]+posx:],project.pattern4_cos(n[1]),project.pattern4_cos(n[0]))*(1-disty)*(1-distx)
            auxhog=auxhog+project.invprjhog(m2pad[deltay+pos[0]+posy:,deltax+pos[1]+posx+1:],project.pattern4_cos(n[1]),project.pattern4_cos(n[0]))*(1-disty)*(distx)
            auxhog=auxhog+project.invprjhog(m2pad[deltay+pos[0]+posy+1:,deltax+pos[1]+posx:],project.pattern4_cos(n[1]),project.pattern4_cos(n[0]))*(disty)*(1-distx)
            auxhog=auxhog+project.invprjhog(m2pad[deltay+pos[0]+posy+1:,deltax+pos[1]+posx+1:],project.pattern4_cos(n[1]),project.pattern4_cos(n[0]))*(disty)*(distx)
        lhog.append(auxhog)
        scr+=numpy.sum(model["ww"][idp].mask*lhog[idp])
    if usebiases:
        biases=numpy.zeros((model["biases"].shape[0],model["biases"].shape[1],model["biases"].shape[2]),dtype=numpy.float32)
        biases[ang[0],ang[1],ang[2]]=1.0*k#model["biases"][ang[0],ang[1]]    
        scr+=model["biases"][ang[0],ang[1],ang[2]]*k
        return lhog,biases,scr
    return lhog,numpy.array([]),scr

def getfeatDef(model,hog,angy,angx,angz,ang,pos,parts,k,mlz,bis=BIS,usebiases=USEBIASES):
    import project
    lhog=[]
    hsize=model["ww"][0].mask.shape[0]
    glangy=angy[ang[0]]
    glangx=angx[ang[1]]
    glangz=angz[ang[2]]
    modelsize(model,angy,angx,angz,force=True)
    maxym=numpy.max(model["size"][:,:,:,2])#numpy.max([el.y for el in model["ww"]])+1
    maxxm=numpy.max(model["size"][:,:,:,3])#numpy.max([el.x for el in model["ww"]])+1
    minym=numpy.min(model["size"][:,:,:,0])#numpy.max([el.y for el in model["ww"]])+1
    minxm=numpy.min(model["size"][:,:,:,1])#numpy.max([el.x for el in model["ww"]])+1
    deltay=maxym-minym+1#numpy.ceil(maxym-minym)
    deltax=maxxm-minxm+1#numpy.ceil(maxxm-minxm)
    m2=hog
    m2pad=numpy.zeros((m2.shape[0]+2*deltay,m2.shape[1]+2*deltax,m2.shape[2]),dtype=m2.dtype)
    m2pad[deltay:deltay+m2.shape[0],deltax:deltax+m2.shape[1]]=m2
    #hog=m2pad
    scr=0
    df2D=0
    df3D=0
    df3D1=0
    cposx=c_float(0.0);cposy=c_float(0.0)
    df=[]
    for idp,p in enumerate(model["ww"]):
        minym=model["size"][ang[0],ang[1],ang[2],0];minxm=model["size"][ang[0],ang[1],ang[2],1]
        #lhog.append(hog[pos[0]:pos[0]+hsize,pos[1]:pos[1]+hsize])
        auxhog=numpy.zeros((hsize,hsize,model["ww"][0].mask.shape[2]),dtype=model["ww"][0].mask.dtype)
        #langy=(glangy+p.ay-180)%360+180
        #langx=(glangx+p.ax-180)%360+180
        #langy=(p.ay+glangy)
        #langx=(p.ax+glangx)
        n=normal(p.ay,p.ax,glangy,glangx)
        #print "Normal",n
        #print "Angles Feat",langy,langx,n,n[2]>0.0001
        if n[2]<0.0001:#face not visible
            lhog.append(numpy.zeros(model["ww"][idp].mask.shape,dtype=model["ww"][idp].mask.dtype))
            df.append((0,0,0,0))
            continue
        pr.getproj(minxm,minym,glangx,glangy,glangz,p.ax,p.ay,p.x,p.y,p.z,p.lz,hsize,byref(cposx),byref(cposy))
        nposy=cposy.value#+deltay
        nposx=cposx.value#+deltax
        #nposy=-minym+project.getproj(p.y,p.z,glangy,langy)#+deltay
        #nposx=-minxm+project.getproj(p.x,p.z,glangx,langx)#+deltax
        #print "FEAT input",minxm,minym,glangx,glangy,glangz,p.ax,p.ay,p.x,p.y,p.z,p.lz,hsize
        #print "FEAT",nposy,nposx
        posy=int(numpy.floor((nposy)))#+deltay
        posx=int(numpy.floor((nposx)))#+deltax
        disty=nposy-posy
        distx=nposx-posx
        mm=p
        Q=numpy.array([[mm.dfax,0,0,0],
                       [0,mm.dfay,0,0],
                       [0,0,mm.dfaz,0],#[0,0,mm.dfaz,mm.lz/2],
                       [0,0,0,1]])#[0,0,mm.lz/2,1]])                    
        #transform local--> can be done only once
        Ry=Mroty(mm.ax)
        Rx=Mrotx(mm.ay)
        Q=numpy.dot(numpy.dot(Ry,Q),Ry.T)
        Q=numpy.dot(numpy.dot(Rx,Q),Rx.T)
        #transform global
        Ry=Mroty(glangx)
        Rx=Mrotx(glangy)
        Q=numpy.dot(numpy.dot(Ry,Q),Ry.T)
        Q=numpy.dot(numpy.dot(Rx,Q),Rx.T)
        Qr=reduceQ2(Q[:3,:3])    
        #modified dt
        #auxscr,ddy,ddx=dt.mydt(auxscr,ay,ax,by,bx)
        ay=Qr[1,1];ax=Qr[0,0];axy=Qr[1,0];by=0;bx=0
        #print "Part:",idp,Qr
        #print posy,posx
        #if parts[1][idp]==-1:#not visible
        #    auxhog=numpy.zeros((4,4,31),dtype=numpy.float32)
        #else:
        #print "def",round(parts[0][idp]),round(parts[1][idp])
        #auxhog=project.invprjhog(m2pad[parts[0][idp]:,parts[1][idp]:],project.pattern4_cos(n[0]),project.pattern4_cos(n[1]))
        auxhog=project.invprjhog(m2pad[floor(deltay+pos[0]-round(parts[0][idp])+posy):,floor(deltax+pos[1]-round(parts[1][idp])+posx):],project.pattern4_cos(n[1]),project.pattern4_cos(n[0]))
        #auxhog=project.invprjhog(m2pad[floor(deltay+pos[0]-round(parts[0][idp])+posy):,floor(deltax+pos[1]-round(parts[1][idp])+posx):],project.pattern4_cos(n[0]),project.pattern4_cos(0))
        #print floor(parts[0][idp]),floor(parts[1][idp])
        #auxhog=auxhog+project.invprjhog(m2pad[deltay+pos[0]+posy:,deltax+pos[1]+posx:],project.pattern4_cos(n[0]),project.pattern4_cos(n[1]))#*(1-disty)*(1-distx)
        #auxhog=auxhog+project.invprjhog(m2pad[deltay+pos[0]+posy:,deltax+pos[1]+posx+1:],project.pattern4_cos(n[0]),project.pattern4_cos(n[1]))*(1-disty)*(distx)
        #auxhog=auxhog+project.invprjhog(m2pad[deltay+pos[0]+posy+1:,deltax+pos[1]+posx:],project.pattern4_cos(n[0]),project.pattern4_cos(n[1]))*(disty)*(1-distx)
        #auxhog=auxhog+project.invprjhog(m2pad[deltay+pos[0]+posy+1:,deltax+pos[1]+posx+1:],project.pattern4_cos(n[0]),project.pattern4_cos(n[1]))*(disty)*(distx)
        #deff=parts[0][idp]
        lhog.append(auxhog)
        scr+=numpy.sum(model["ww"][idp].mask*lhog[idp])
        xx=numpy.round([parts[1][idp],parts[0][idp]]).reshape(2,1)
        #xx=numpy.round([parts[0][idp],parts[1][idp]]).reshape(2,1)
        #Qrr=Qr[:3,:3];Qrr[2,:2]=0;Qrr[:2,2]=0#not sure if it is correct to remove b
        #df2D+=parts[0][idp]**2*ax+parts[1][idp]**2*ay#+parts[0][idp]*parts[1][idp]*axy
        df2D+=numpy.dot(numpy.dot(xx.T,Qr),xx)[0,0]
        #print xx,df2D
        #if round(parts[0][idp])!=0 or round(parts[1][idp]!=0):
        #    print "problems"
        #    raw_input()
        #qxy=Q[1,0];qxz=Q[2,0];qxt=Q[3,0]
        #qyx=Q[0,1];qyz=Q[2,1];qyt=Q[3,1]
        #qzx=Q[0,2];qzy=Q[1,2];qzt=Q[3,2]    
        #qtx=Q[0,3];qty=Q[1,3];qtz=Q[2,3]    
        z=-(Q[2,0]*xx[0]+Q[2,1]*xx[1])/Q[2,2]
        x3d=numpy.array([xx[0],xx[1],z]).reshape(3,1)
        df3D1+=numpy.dot(numpy.dot(x3d.T,Q[:3,:3]),x3d)[0,0]
        assert(df2D-df3D1<0.00001)
        #back to initial coordinates
        Rx=Mrotx(-glangy)
        Ry=Mroty(-glangx)
        x3d=numpy.dot(Ry[:3,:3],numpy.dot(Rx[:3,:3],x3d))
        Rx=Mrotx(-mm.ay)
        Ry=Mroty(-mm.ax)
        x3d=numpy.dot(Ry[:3,:3],numpy.dot(Rx[:3,:3],x3d))
        #usez=True
        #if usez:
        #    x3d[-1]=x3d[-2]
        #    df3D+=x3d[0]**2*p.dfax+x3d[1]**2*p.dfay+x3d[2]**2*p.dfaz+x3d[3]*p.dfbz
        #else:
        #    x3d[-1]=0
        #    df3D+=numpy.dot(x3d**2,[p.dfax,p.dfay,p.dfaz,0])
        #x3d[-1]=x3d[-2]*mlz
        df3D+=numpy.dot(x3d.T**2,[p.dfax,p.dfay,p.dfaz])[0]
        assert(df2D-df3D<0.00001)
        df.append((x3d[0,0],x3d[1,0],x3d[2,0],0.0))    
        #print "2D:",xx,"3D:",x3d
    if usebiases:
        biases=numpy.zeros((model["biases"].shape[0],model["biases"].shape[1],model["biases"].shape[2]),dtype=numpy.float32)
        biases[ang[0],ang[1],ang[2]]=1.0*k#model["biases"][ang[0],ang[1]]    
        scr+=model["biases"][ang[0],ang[1],ang[2]]*k
        return lhog,biases,df,scr,df2D,df3D
    return lhog,numpy.array([]),df,scr,df2D,df3D



if __name__ == "__main__":

    im=util.myimread("./a.png")

    hog=pyrHOG2.pyrHOG(im,cformat=True,sbin=4)

    #myhog=hog.hog[8]
    #imh=drawHOG.drawHOG(myhog);pylab.figure();pylab.imshow(imh)
    #hmodel=myhog[0:8,12:20] # model

    myhog=hog.hog[0]
    hmodel=myhog[8:24,50:66] # model

    lmask=[]
    step=2
    size=4
    ang=numpy.array([-45,-30,-15,0,15,30,45])
    for py in range(len(ang)):
        for px in range(len(ang)):
            #lmask.append(part3D(hmodel[py*2:(py+1)*2+2,px*2:(px+1)*2+2],0,0,15,ang[py],ang[px]))#-15*cos(ang[py]/180.0*numpy.pi)*cos(ang[px]/180.0*numpy.pi),ang[py],ang[px]))
            #lmask.append(part3D(hmodel[py*2:(py+1)*2+2,px*2:(px+1)*2+2],0,0,0,25,-ang[py],-ang[px]))
            lmask.append(part3D(hmodel[py*2:(py+1)*2+2,px*2:(px+1)*2+2],py*2,px*2,0,15,0,0))
            #lmask.append(part3D(hmodel[py*2:(py+1)*2+2,px*2:(px+1)*2+2],0,px*2,py*2,0,0,0))
            #lmask.append(part3D(hmodel[py*2:(py+1)*2+2,px*2:(px+1)*2+2],py*2,px*2,-15*cos(ang[py]/180.0*numpy.pi)*cos(ang[px]/180.0*numpy.pi),0,0)),ang[py],ang[px]))
            #lmask.append(part3D(hmodel[py*2:(py+1)*2+2,px*2:(px+1)*2+2],py*2,px*2,0,ang[py],ang[px]))
    #lmask=[]
    #for py in range(4):
    #    for px in range(4):
    #        lmask.append(part3D(hmodel[py*4:(py+1)*4,px*4:(px+1)*4],py*4-6,px*4-6,20,0,0))
    #mask0=hmodel[0:4,0:4].copy()
    #mask1=hmodel[0:4,4:8].copy()
    #mask2=hmodel[4:8,0:4].copy()
    #mask3=hmodel[4:8,4:8].copy()

    #lmask[5].z=1;lmask[6].z=1;lmask[9].z=1;lmask[10].z=1;

    model={"ww":lmask}
    model["biases"]=numpy.zeros((13,13))
    model["rho"]=0
    models=[model]
    util.save("init.model",models)
    #model={"ww":[part3D(mask0,0,0,0,0,0),part3D(mask1,0,4,0,0,0),part3D(mask2,4,0,0,0,0),part3D(mask3,4,4,0,0,0)]}

    im2=util.myimread("./a.png",resize=1.0)

    glangy=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90]
    glangx=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90]
    glangz=[-30,-15,0,15,30]
    fhog,ldet=rundet(im2,model,angy=glangy,angx=glangx,angz=glangz,selangy=[6],selangx=[3],selangz=[2])
    #fhog,ldet=rundet(im2,model,angy=[0],angx=[-60,-45,-30,-15,0,15,30,45,60],bbox=[100,100,200,150])
    pylab.figure(100)
    pylab.imshow(im2)
    for ld in ldet[:1]:
        feat,biases,scr=getfeat(model,fhog.hog[ld["hog"]],glangy,glangx,glangz,ld["ang"],ld["fpos"],1)
        assert((abs(scr-ld["scr"])/(scr+0.0001))<0.0001)
        pylab.figure(110)
        pylab.clf()
        showHOG(model,feat,glangy[ld["ang"][0]],glangx[ld["ang"][1]],nhog=60,val=1)
        pylab.figure(120)
        pylab.clf()
        showHOG(model,feat,0,0,nhog=60,val=1)
        pylab.figure(100)
        drawdet(ld)
        pylab.draw()
        print ld
        #raw_input()
    #fhog,ldet=rundet(im2,model,angy=[0],angx=[0])
    #res=det(model,myhog)
    exit()
    #util.imshow(res[0,0])
    #dsgf

    myhog=hog.hog[0]
    hsy=myhog.shape[0]
    hsx=myhog.shape[1]
    prec=[]
    for w in model["ww"]:
        prec.append(project.precompute(w.mask,myhog))
       

    ang0=project.pattern4(0)


    maxmy=numpy.max([el.y for el in model["ww"]])+1
    maxmx=numpy.max([el.x for el in model["ww"]])+1

    glangx=-75
    glangy=-90
    bis=True
    hsize=4

    lminym=[]
    lminxm=[]
    for glangx in [-75]:#[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75]:
    #draw model
        if 1:
            pixh=15
            border=2
            nhog=30
            img=numpy.zeros((pixh*nhog,pixh*nhog))
            img2=numpy.zeros((pixh*nhog,pixh*nhog))
            for w in model["ww"]:
                angy=w.ay+glangy
                angx=w.ax+glangx
                part=drawHOG.drawHOG(project.prjhog_bis(w.mask,project.pattern4_bis(angy),project.pattern4_bis(angx)),hogpix=pixh,border=border)
                part2=drawHOG.drawHOG(project.prjhog(w.mask,project.pattern4(angy),project.pattern4(angx)),hogpix=pixh,border=border)
                nposy=w.y*cos(glangy/180.0*numpy.pi)-hsize/2.0*(cos(angy/180.0*numpy.pi))-w.z*sin(angy/180.0*numpy.pi)
                nposx=w.x*cos(glangx/180.0*numpy.pi)-hsize/2.0*(cos(angx/180.0*numpy.pi))-w.z*sin(angx/180.0*numpy.pi)
                img[nhog/2*pixh+nposy*pixh:nhog/2*pixh+nposy*pixh+part.shape[0],nhog/2*pixh+nposx*pixh:nhog/2*pixh+nposx*pixh+part.shape[1]]=part        
                img2[nhog/2*pixh+nposy*pixh:nhog/2*pixh+nposy*pixh+part.shape[0],nhog/2*pixh+nposx*pixh:nhog/2*pixh+nposx*pixh+part.shape[1]]=part2        
                lminym.append(nposy)
                lminxm.append(nposx)
            pylab.figure()
            pylab.imshow(img)    
            pylab.figure()
            pylab.imshow(img2)    
            pylab.draw()
            pylab.show()
            raw_input()
        minym=numpy.min(lminym)
        minxm=numpy.min(lminxm)


    #lres=[]
    res=numpy.zeros((hsy+maxmy+hsize,hsx+maxmx+hsize),dtype=numpy.float32)
    for l in range(len(model["ww"])):
        mm=model["ww"][l]
        #lres.append(project.project(prec[l],ang0,ang0))
        #res[:hsy+4,:hsx+4]=res[:hsy+4,:hsx+4]+project.project(prec[l],ang0,ang0)
        angy=mm.ay+glangy
        angx=mm.ax+glangx
        if bis:
            scr=project.project_bis(prec[l],project.pattern4_bis(angy),project.pattern4_bis(angx))
        else:
            scr=project.project(prec[l],project.pattern4(angy),project.pattern4(angx))
        nposy=-minym+mm.y*cos(glangy/180.0*numpy.pi)-hsize/2.0*(cos(angy/180.0*numpy.pi))-mm.z*sin(angy/180.0*numpy.pi)
        nposx=-minxm+mm.x*cos(glangx/180.0*numpy.pi)-hsize/2.0*(cos(angx/180.0*numpy.pi))-mm.z*sin(angx/180.0*numpy.pi)
        posy=int(numpy.floor((nposy)))
        posx=int(numpy.floor((nposx)))
        disty=nposy-posy
        distx=nposx-posx
        #print maxmy-posy,maxmx-posx,nposy,nposx,glangx,angx
        res[maxmy-posy:maxmy-posy+hsy+hsize,maxmx-posx:maxmx-posx+hsx+hsize]=res[maxmy-posy:maxmy-posy+hsy+hsize,maxmx-posx:maxmx-posx+hsx+hsize]+(1-disty)*(1-distx)*scr
        res[maxmy-posy:maxmy-posy+hsy+hsize,maxmx-(posx+1):maxmx-(posx+1)+hsx+hsize]=res[maxmy-posy:maxmy-posy+hsy+hsize,maxmx-(posx+1):maxmx-(posx+1)+hsx+hsize]+(1-disty)*(distx)*scr
        res[maxmy-(posy+1):maxmy-(posy+1)+hsy+hsize,maxmx-(posx):maxmx-(posx)+hsx+hsize]=res[maxmy-(posy+1):maxmy-(posy+1)+hsy+hsize,maxmx-(posx):maxmx-(posx)+hsx+hsize]+(disty)*(1-distx)*scr
        res[maxmy-(posy+1):maxmy-(posy+1)+hsy+hsize,maxmx-(posx+1):maxmx-(posx+1)+hsx+hsize]=res[maxmy-(posy+1):maxmy-(posy+1)+hsy+hsize,maxmx-(posx+1):maxmx-(posx+1)+hsx+hsize]+(disty)*(distx)*scr
        #res[4-w.y:4-w.y+hsy+4,4-w.x:4-w.x+hsx+4]=project.project(prec[l],ang0,ang0)

######no longer used

#@autojit
def det(model,hog,pglangy=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90],pglangx=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90],bis=BIS):
    hsy=hog.shape[0]
    hsx=hog.shape[1]
    prec=[]
    for w in model["ww"]:
        prec.append(project.precompute(w.mask,hog))
    modelsize(model,pglangy,pglangx)
    maxym=numpy.max(model["size"][:,:,2])+1#numpy.max([el.y for el in model["ww"]])+1
    maxxm=numpy.max(model["size"][:,:,3])+1#numpy.max([el.x for el in model["ww"]])+1
    minym=numpy.min(model["size"][:,:,0])+1#numpy.max([el.y for el in model["ww"]])+1
    minxm=numpy.min(model["size"][:,:,1])+1#numpy.max([el.x for el in model["ww"]])+1
    maxmy=maxym
    maxmx=maxxm
    hsize=model["ww"][0].mask.shape[0]
    #minym=hsize/2
    #minxm=hsize/2
    #res=numpy.zeros((len(pglangy),len(pglangx),hsy+-maxmy+hsize,hsx+maxmx+hsize),dtype=numpy.float32)
    res=numpy.zeros((len(pglangy),len(pglangx),hsy-minym+maxym+hsize,hsx-minxm+maxxm+hsize),dtype=numpy.float32)
    for gly,glangy in enumerate(pglangy):
        for glx,glangx in enumerate(pglangx):
            lminym=[]
            lminxm=[]
            #precompute max and min size
            #for w in model["ww"]:
            #    #mm=model["ww"][l]
            #    angy=w.ay+glangy
            #    angx=w.ax+glangx
            #    nposy=w.y*cos(glangy/180.0*numpy.pi)-hsize/2.0*(cos(angy/180.0*numpy.pi))-w.z*sin(angy/180.0*numpy.pi)
            #    nposx=w.x*cos(glangx/180.0*numpy.pi)-hsize/2.0*(cos(angx/180.0*numpy.pi))-w.z*sin(angx/180.0*numpy.pi)
            #    lminym.append(nposy)
            #    lminxm.append(nposx)
            #minym=numpy.min(lminym)
            #minxm=numpy.min(lminxm)
            #maxym=numpy.max(lminym)+project.pattern4_bis(angy).shape[1]
            #maxxm=numpy.max(lminxm)+project.pattern4_bis(angx).shape[1]
            #model["size"][gly,glx]=(minym,minxm,maxym,maxxm)
            minym=model["size"][gly,glx,0];minxm=model["size"][gly,glx,1]
            maxym=model["size"][gly,glx,2];maxxm=model["size"][gly,glx,3]
            for l in range(len(model["ww"])):
                mm=model["ww"][l]
                #lres.append(project.project(prec[l],ang0,ang0))
                #res[:hsy+4,:hsx+4]=res[:hsy+4,:hsx+4]+project.project(prec[l],ang0,ang0)
                angy=mm.ay+glangy
                angx=mm.ax+glangx
                if abs(angx)>90 or abs(angy)>90:
                    continue
                if bis:
                    scr=project.project_bis(prec[l],project.pattern4_bis(angy),project.pattern4_bis(angx))
                else:
                    scr=project.project(prec[l],project.pattern4(angy),project.pattern4(angx))
                nposy=-minym+mm.y*cos(glangy/180.0*numpy.pi)-hsize/2.0*(cos(angy/180.0*numpy.pi))-mm.z*sin(angy/180.0*numpy.pi)
                nposx=-minxm+mm.x*cos(glangx/180.0*numpy.pi)-hsize/2.0*(cos(angx/180.0*numpy.pi))-mm.z*sin(angx/180.0*numpy.pi)
                posy=int(floor((nposy)))
                posx=int(floor((nposx)))
                disty=nposy-posy
                distx=nposx-posx
                #print maxmy-posy,maxmx-posx,nposy,nposx,glangx,angx
                res[gly,glx][maxmy-posy:maxmy-posy+hsy+hsize,maxmx-posx:maxmx-posx+hsx+hsize]=res[gly,glx][maxmy-posy:maxmy-posy+hsy+hsize,maxmx-posx:maxmx-posx+hsx+hsize]+(1-disty)*(1-distx)*scr
                res[gly,glx][maxmy-posy:maxmy-posy+hsy+hsize,maxmx-(posx+1):maxmx-(posx+1)+hsx+hsize]=res[gly,glx][maxmy-posy:maxmy-posy+hsy+hsize,maxmx-(posx+1):maxmx-(posx+1)+hsx+hsize]+(1-disty)*(distx)*scr
                res[gly,glx][maxmy-(posy+1):maxmy-(posy+1)+hsy+hsize,maxmx-(posx):maxmx-(posx)+hsx+hsize]=res[gly,glx][maxmy-(posy+1):maxmy-(posy+1)+hsy+hsize,maxmx-(posx):maxmx-(posx)+hsx+hsize]+(disty)*(1-distx)*scr
                res[gly,glx][maxmy-(posy+1):maxmy-(posy+1)+hsy+hsize,maxmx-(posx+1):maxmx-(posx+1)+hsx+hsize]=res[gly,glx][maxmy-(posy+1):maxmy-(posy+1)+hsy+hsize,maxmx-(posx+1):maxmx-(posx+1)+hsx+hsize]+(disty)*(distx)*scr
                #res[gly,glx][maxmy-posy:maxmy-posy+hsy+hsize,maxmx-posx:maxmx-posx+hsx+hsize]=res[gly,glx][maxmy-posy:maxmy-posy+hsy+hsize,maxmx-posx:maxmx-posx+hsx+hsize]+(1-disty)*(1-distx)*scr
                #res[gly,glx][maxmy-posy:maxmy-posy+hsy+hsize,maxmx-(posx+1):maxmx-(posx+1)+hsx+hsize]=res[gly,glx][maxmy-posy:maxmy-posy+hsy+hsize,maxmx-(posx+1):maxmx-(posx+1)+hsx+hsize]+(1-disty)*(distx)*scr
                #res[gly,glx][maxmy-(posy+1):maxmy-(posy+1)+hsy+hsize,maxmx-(posx):maxmx-(posx)+hsx+hsize]=res[gly,glx][maxmy-(posy+1):maxmy-(posy+1)+hsy+hsize,maxmx-(posx):maxmx-(posx)+hsx+hsize]+(disty)*(1-distx)*scr
                #res[gly,glx][maxmy-(posy+1):maxmy-(posy+1)+hsy+hsize,maxmx-(posx+1):maxmx-(posx+1)+hsx+hsize]=res[gly,glx][maxmy-(posy+1):maxmy-(posy+1)+hsy+hsize,maxmx-(posx+1):maxmx-(posx+1)+hsx+hsize]+(disty)*(distx)*scr
                #res[4-w.y:4-w.y+hsy+4,4-w.x:4-w.x+hsx+4]=project.project(prec[l],ang0,ang0)
    return res                
            




