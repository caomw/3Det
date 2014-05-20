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
    def __init__(self,mask,y,x,z,lz,ay,ax,dfay=1.0,dfax=1.0,dfaz=1.0,dfby=0,dfbx=0,dfbz=0):
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

BIS=False
USEBIASES=False

from ctypes import cdll,CDLL,c_float,c_int,byref,POINTER
cdll.LoadLibrary("./cproject.so")
pr=CDLL("cproject.so")
pr.getproj.argtypes=[c_float,c_float,c_int,c_int,c_int,c_int,c_int,c_float,c_float,c_float,c_float,c_int,POINTER(c_float),POINTER(c_float)]
pr.interpolate.argtypes=[c_int,c_int,c_int,c_int,c_int,numpy.ctypeslib.ndpointer(dtype=c_float,ndim=5,flags="C_CONTIGUOUS"),c_int,c_int,c_int,c_int,c_int,c_int,c_int,c_int,c_int,c_int,c_float,c_float,numpy.ctypeslib.ndpointer(dtype=c_float,ndim=2,flags="C_CONTIGUOUS")]

#def visible(angy,angx):
#    p=[0,0,1];
#    p=rotatey(p,angx)
#    p=rotatex(p,angy)
#    return p[2]>0

def normal(angy,angx,glangy,glangx,p=[0,0,1.0]):
    #p=[0,0,1.0];
    p=rotatex(p,angx)
    p=rotatey(p,angy)
    p=rotatex(p,glangx)
    p=rotatey(p,glangy)
    return p

def reproj(angy,angx,glangy,glangx,p=[0,1.0]):
    p=rotatey(p,-glangy)
    p=rotatex(p,-glangx)
    p=rotatey(p,-angy)
    p=rotatex(p,-angx)
    return p

#@autojit
def showModel(model,glangy,glangx,glangz,hsize=4,pixh=15,border=2,nhog=30,bis=BIS,val=None):
    size=modelsize(model,[glangy],[glangx],[glangz],force=True)[0,0,0]
    nhogy=size[2]-size[0]+3
    nhogx=size[3]-size[1]+3
    #nhog=100
    nnhog=0
    img=numpy.zeros((pixh*nhogy,pixh*nhogx))
    #img2=numpy.zeros((pixh*nhogy,pixh*nhogx))
    cposy=c_float(0.0);cposx=c_float(0.0)
    for w in model["ww"]:
        angy=(w.ay+glangy)
        angx=(w.ax+glangx)
        n=normal(w.ay,w.ax,glangy,glangx)
        #print "Normal",n
        #print "Angles",angy,angx,n,n[2]>0.0001
        if n[2]<0.0001:#face not visible
            continue
        if bis:
            part=drawHOG.drawHOG(project.prjhog_bis(w.mask,project.pattern4_bis(angy),project.pattern4_bis(angx)),hogpix=pixh,border=border,val=val)
        else:
            part=drawHOG.drawHOG(project.prjhog(w.mask,project.pattern4_cos(n[0]),project.pattern4_cos(n[1])),hogpix=pixh,border=border,val=val)
        #print "Outside",w.x,w.y,w.z
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
        angy=(w.ay+glangy)
        angx=(w.ax+glangx)
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
            part=drawHOG.drawHOG(project.prjhog(w.mask,project.pattern4_cos(n[0]),project.pattern4_cos(n[1])),hogpix=pixh,border=border,val=val)
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
def modelsize(model,pglangy,pglangx,pglangz,force=False):
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
                    angy=(w.ay+glangy)
                    angx=(w.ax+glangx)
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
                if usebiases:#NOTE: it works properly only if the model did not use gly and glz
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
                        scr=project.project(prec[l],project.pattern4_cos(n[0]),project.pattern4_cos(n[1]))
                        auxscr=scr.copy()
                        cache[l,gly,glx]=auxscr
                        #should be considered for general angles
                        if abs(angy)<45 and abs(angx)<45:
                            #cache[l,4:9,4:9]=auxscr
                            for lly in selangy:
                                for llx in selangx:
                                    if abs(mm.ay+ppglangy[lly])<45 and abs(mm.ax+ppglangx[llx])<45:
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

import dt

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
    defy=c_float(0.0);defx=c_float(0.0)
    cache=numpy.zeros((len(model["ww"]),len(ppglangy),len(ppglangx)),dtype=object)
    for gly in selangy:
        for glx in selangx:
            for glz in selangz:
                if usebiases:#NOTE: it works properly only if the model did not use gly and glz
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
                        scr=project.project(prec[l],project.pattern4_cos(n[0]),project.pattern4_cos(n[1]))
                        auxscr=scr.copy()
                        cache[l,gly,glx]=auxscr
                        #should be considered for general angles
                        if abs(angy)<(30+45)/2.0 and abs(angx)<(30+45)/2.0:
                            #cache[l,4:9,4:9]=auxscr
                            for lly in selangy:
                                for llx in selangx:
                                    if abs(mm.ay+ppglangy[lly])<(30+45)/2.0 and abs(mm.ax+ppglangx[llx])<(30+45)/2.0:
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
                    auxscr=numpy.zeros(numpy.array(scr.shape)+1,dtype=scr.dtype)
                    auxscr[:-1,:-1]+=(1-disty)*(1-distx)*scr
                    auxscr[1:,:-1]+= (disty)*(1-distx)*scr
                    auxscr[:-1,1:]+= (1-disty)*(distx)*scr
                    auxscr[1:,1:]+= (disty)*(distx)*scr 
                    #transform parameters
                    #for the moment is wrong because it should generate an elipsis with any possible rotation and not only aligned to the axis
                    #ay=mm.dfay;ax=mm.dfax,by=mm.dfby,bx=mm.dfbx
                    #pr.getproj(minxm,minym,ppglangx[glx],ppglangy[gly],ppglangz[glz],mm.ax,mm.ay,mm.dfax,mm.dfay,mm.dfaz,mm.lz,hsize,byref(defx),byref(defy))
                    
                    #modified dt
                    #auxscr,ddy,ddx=dt.mydt(auxscr,ay,ax,by,bx)
                    ay=0.001;ax=0.001;axy=0;by=0;bx=0
                    auxscr,ddy,ddx=dt.mydt(auxscr,ay,ax,by,bx)
                    #dfds
                    #auxscr,ddy,ddx=dt.dt2(auxscr,ay,ax,axy,by,bx)
                    res[gly,glx,glz,maxmy-posy:maxmy-posy+hsy+hsize,maxmx-(posx):maxmx-(posx)+hsx+hsize]+=(1-disty)*(1-distx)*auxscr[:-1,:-1]
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
    return res                

#@autojit
def drawdet(ldet):
    if type(ldet)!=list:
        ldet=[ldet]
    for idl,l in enumerate(ldet):
       util.box(l["bbox"][0],l["bbox"][1],l["bbox"][2],l["bbox"][3],lw=2)
       pylab.text(l["bbox"][1],l["bbox"][0],"(%d,%4f,%d,%d,%d)"%(idl,l["scr"],l["ang"][0],l["ang"][1],l["ang"][2]))
       #print res[dd],l,dd[0],dd[1]

#@autojit
def rundet(img,model,angy=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90],angx=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90],angz=[-10,0,10],interv=5,sbin=4,maxdet=1000,sort="scr",bbox=None,selangy=None,selangx=None,selangz=None,numhyp=1000,k=1,bis=BIS,usebiases=USEBIASES,usedef=True):
    hog=pyrHOG2.pyrHOG(img,interv=interv,cformat=True,sbin=sbin)
    modelsize(model,angy,angx,angz)
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
    for idr,r in enumerate(hog.hog):
        #print idr,"/",len(hog.hog)
        if show:
            pylab.figure()
            pylab.imshow(img)
        if usedef:
            res=det2_def(model,hog.hog[idr],ppglangy=angy,ppglangx=angx,ppglangz=angz,selangy=selangy,selangx=selangx,selangz=selangz,k=k,bis=bis,usebiases=usebiases)
        else:
            res=det2(model,hog.hog[idr],ppglangy=angy,ppglangx=angx,ppglangz=angz,selangy=selangy,selangx=selangx,selangz=selangz,k=k,bis=bis,usebiases=usebiases)
        #dsfsd
        order=(-res).argsort(None)
        for l in range(min(numhyp,len(order))):
            dd=numpy.unravel_index(order[l],res.shape)
            (minym,minxm,maxym,maxxm)=model["size"][dd[0],dd[1],dd[2]]
            pcy1=(dd[3]-deltay+hsize/2)*sbin/hog.scale[idr]
            pcx1=(dd[4]-deltax+hsize/2)*sbin/hog.scale[idr]
            pcy2=(dd[3]-deltay+(maxym-minym)+hsize/2)*sbin/hog.scale[idr]
            pcx2=(dd[4]-deltax+(maxxm-minxm)+hsize/2)*sbin/hog.scale[idr]
            ldet.append({"id":0,"bbox":[pcy1,pcx1,pcy2,pcx2],"ang":(dd[0],dd[1],dd[2]),"scr":res[dd]-model["rho"],"scl":hog.scale[idr],"hog":idr,"pos":(dd[3],dd[4]),"fpos":(dd[3]-deltay,dd[4]-deltax)})
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
    modelsize(model,angy,angx,angz)#,force=True)
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
        langy=(p.ay+glangy)
        langx=(p.ax+glangx)
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
            auxhog=auxhog+project.invprjhog(m2pad[deltay+pos[0]+posy:,deltax+pos[1]+posx:],project.pattern4_cos(n[0]),project.pattern4_cos(n[1]))*(1-disty)*(1-distx)
            auxhog=auxhog+project.invprjhog(m2pad[deltay+pos[0]+posy:,deltax+pos[1]+posx+1:],project.pattern4_cos(n[0]),project.pattern4_cos(n[1]))*(1-disty)*(distx)
            auxhog=auxhog+project.invprjhog(m2pad[deltay+pos[0]+posy+1:,deltax+pos[1]+posx:],project.pattern4_cos(n[0]),project.pattern4_cos(n[1]))*(disty)*(1-distx)
            auxhog=auxhog+project.invprjhog(m2pad[deltay+pos[0]+posy+1:,deltax+pos[1]+posx+1:],project.pattern4_cos(n[0]),project.pattern4_cos(n[1]))*(disty)*(distx)
        lhog.append(auxhog)
        scr+=numpy.sum(model["ww"][idp].mask*lhog[idp])
    if usebiases:
        biases=numpy.zeros((model["biases"].shape[0],model["biases"].shape[1],model["biases"].shape[2]),dtype=numpy.float32)
        biases[ang[0],ang[1],ang[2]]=1.0*k#model["biases"][ang[0],ang[1]]    
        scr+=model["biases"][ang[0],ang[1],ang[2]]*k
        return lhog,biases,scr
    return lhog,numpy.array([]),scr


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
            




