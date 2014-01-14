
class part3D(object):
    def __init__(self,mask,y,x,z,ay,ax):
        self.mask=mask
        self.y=y
        self.x=x
        self.z=z
        self.ay=ay
        self.ax=ax

import util
import pyrHOG2
import drawHOG
import pylab
import numpy
import project

def showModel(model,glangy,glangx,hsize=4,pixh=15,border=2,nhog=30,bis=True,val=None):
    size=modelsize(model,[glangy],[glangx],force=True)[0,0]
    nhogy=size[2]-size[0]+3
    nhogx=size[3]-size[1]+3
    #nhog=100
    nnhog=0
    img=numpy.zeros((pixh*nhogy,pixh*nhogx))
    img2=numpy.zeros((pixh*nhogy,pixh*nhogx))
    for w in model["ww"]:
        angy=w.ay+glangy
        angx=w.ax+glangx
        if bis:
            part=drawHOG.drawHOG(project.prjhog_bis(w.mask,project.pattern4_bis(angy),project.pattern4_bis(angx)),hogpix=pixh,border=border,val=val)
        else:
            part=drawHOG.drawHOG(project.prjhog(w.mask,project.pattern4(angy),project.pattern4(angx)),hogpix=pixh,border=border,val=val)
        nposy=-size[0]+w.y*numpy.cos(glangy/180.0*numpy.pi)-hsize/2.0*(numpy.cos(angy/180.0*numpy.pi))-w.z*numpy.sin(angy/180.0*numpy.pi)
        nposx=-size[1]+w.x*numpy.cos(glangx/180.0*numpy.pi)-hsize/2.0*(numpy.cos(angx/180.0*numpy.pi))-w.z*numpy.sin(angx/180.0*numpy.pi)
        #print nposy,nposx,glangx,angx,w.x*numpy.cos(glangx/180.0*numpy.pi),-w.z*numpy.sin(angx/180.0*numpy.pi),-hsize/2.0*(numpy.cos(angx/180.0*numpy.pi)),w.x*numpy.cos(glangx/180.0*numpy.pi)-hsize/2.0*(numpy.cos(angx/180.0*numpy.pi))-w.z*numpy.sin(angx/180.0*numpy.pi)
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

def showHOG(model,lhog,glangy,glangx,hsize=4,pixh=15,border=2,nhog=30,bis=True,val=None):
    img=numpy.zeros((pixh*nhog,pixh*nhog))
    img2=numpy.zeros((pixh*nhog,pixh*nhog))
    for idh,h in enumerate(lhog):
        w=model["ww"][idh]
        angy=w.ay+glangy
        angx=w.ax+glangx
        if bis:
            part=drawHOG.drawHOG(project.prjhog_bis(h,project.pattern4_bis(angy),project.pattern4_bis(angx)),hogpix=pixh,border=border,val=val)
        else:
            part=drawHOG.drawHOG(project.prjhog(h,project.pattern4(angy),project.pattern4(angx)),hogpix=pixh,border=border,val=val)
        nposy=w.y*numpy.cos(glangy/180.0*numpy.pi)-hsize/2.0*(numpy.cos(angy/180.0*numpy.pi))-w.z*numpy.sin(angy/180.0*numpy.pi)
        nposx=w.x*numpy.cos(glangx/180.0*numpy.pi)-hsize/2.0*(numpy.cos(angx/180.0*numpy.pi))-w.z*numpy.sin(angx/180.0*numpy.pi)
        img[nhog/2*pixh+nposy*pixh:nhog/2*pixh+nposy*pixh+part.shape[0],nhog/2*pixh+nposx*pixh:nhog/2*pixh+nposx*pixh+part.shape[1]]=part        
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
                    nposy=w.y*numpy.cos(glangy/180.0*numpy.pi)-hsize/2.0*(numpy.cos(angy/180.0*numpy.pi))-w.z*numpy.sin(angy/180.0*numpy.pi)
                    nposx=w.x*numpy.cos(glangx/180.0*numpy.pi)-hsize/2.0*(numpy.cos(angx/180.0*numpy.pi))-w.z*numpy.sin(angx/180.0*numpy.pi)
                    lminym.append(nposy)
                    lminxm.append(nposx)
                #print angx,numpy.cos(angx/180.0*numpy.pi)
                minym=numpy.min(lminym)-hsize#*numpy.cos(angy/180.0*numpy.pi)
                minxm=numpy.min(lminxm)-hsize#*numpy.cos(angx/180.0*numpy.pi)
                maxym=numpy.max(lminym)+hsize#*numpy.cos(angy/180.0*numpy.pi)#project.pattern4_bis(angy).shape[1]
                maxxm=numpy.max(lminxm)+hsize#*numpy.cos(angx/180.0*numpy.pi)#project.pattern4_bis(angx).shape[1]
                model["size"][gly,glx]=(minym,minxm,maxym,maxxm)#numpy.round((minym,minxm,maxym,maxxm))
    return model["size"]

def modelsize(model,pglangy,pglangx,force=False):
    hsize=model["ww"][0].mask.shape[0]
    if force or not(model.has_key("size")) or (model.has_key("size") and model["size"].shape!=(len(pglangy),len(pglangx),4)):#not(model.has_key("size"))::
        model["size"]=numpy.zeros((len(pglangy),len(pglangx),4))
        for gly,glangy in enumerate(pglangy):
            for glx,glangx in enumerate(pglangx):
                lminym=[]
                lminxm=[]
                #laddy=[]
                #laddx=[]
                #precompute max and min size
                for w in model["ww"]:
                    #mm=model["ww"][l]
                    angy=w.ay+glangy
                    angx=w.ax+glangx
                    #laddy.append(-hsize/2.0*(numpy.cos(angy/180.0*numpy.pi)))
                    nposy=w.y*numpy.cos(glangy/180.0*numpy.pi)-hsize/2.0*(numpy.cos(angy/180.0*numpy.pi))-w.z*numpy.sin(angy/180.0*numpy.pi)
                    nposx=w.x*numpy.cos(glangx/180.0*numpy.pi)-hsize/2.0*(numpy.cos(angx/180.0*numpy.pi))-w.z*numpy.sin(angx/180.0*numpy.pi)
                    #lminym.append(nposy-hsize/2.0*(numpy.cos(angy/180.0*numpy.pi)))
                    lminym.append(nposy)
                    lminym.append(nposy+hsize/2.0*(numpy.cos(angy/180.0*numpy.pi)))
                    #lminxm.append(nposx-hsize/2.0*(numpy.cos(angx/180.0*numpy.pi)))
                    lminxm.append(nposx)
                    lminxm.append(nposx+hsize/2.0*(numpy.cos(angx/180.0*numpy.pi)))
                #print angx,numpy.cos(angx/180.0*numpy.pi)
                minym=numpy.min(lminym)#*numpy.cos(angy/180.0*numpy.pi)
                minxm=numpy.min(lminxm)#*numpy.cos(angx/180.0*numpy.pi)
                maxym=numpy.max(lminym)#*numpy.cos(angy/180.0*numpy.pi)#project.pattern4_bis(angy).shape[1]
                maxxm=numpy.max(lminxm)#*numpy.cos(angx/180.0*numpy.pi)#project.pattern4_bis(angx).shape[1]
                model["size"][gly,glx]=(minym,minxm,maxym,maxxm)#numpy.round((minym,minxm,maxym,maxxm))
    return model["size"]

def det(model,hog,pglangy=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90],pglangx=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90],bis=True):
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
            #    nposy=w.y*numpy.cos(glangy/180.0*numpy.pi)-hsize/2.0*(numpy.cos(angy/180.0*numpy.pi))-w.z*numpy.sin(angy/180.0*numpy.pi)
            #    nposx=w.x*numpy.cos(glangx/180.0*numpy.pi)-hsize/2.0*(numpy.cos(angx/180.0*numpy.pi))-w.z*numpy.sin(angx/180.0*numpy.pi)
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
                if bis:
                    scr=project.project_bis(prec[l],project.pattern4_bis(angy),project.pattern4_bis(angx))
                else:
                    scr=project.project(prec[l],project.pattern4(angy),project.pattern4(angx))
                nposy=-minym+mm.y*numpy.cos(glangy/180.0*numpy.pi)-hsize/2.0*(numpy.cos(angy/180.0*numpy.pi))-mm.z*numpy.sin(angy/180.0*numpy.pi)
                nposx=-minxm+mm.x*numpy.cos(glangx/180.0*numpy.pi)-hsize/2.0*(numpy.cos(angx/180.0*numpy.pi))-mm.z*numpy.sin(angx/180.0*numpy.pi)
                posy=int(numpy.floor((nposy)))
                posx=int(numpy.floor((nposx)))
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

def det2(model,hog,ppglangy=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90],ppglangx=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90],selangy=None,selangx=None,bis=True,k=1):
    if selangy==None:
        selangy=range(len(ppglangy))
    if selangx==None:
        selangx=range(len(ppglangx))
    hsy=hog.shape[0]
    hsx=hog.shape[1]
    prec=[]
    for w in model["ww"]:
        prec.append(project.precompute(w.mask,hog))
    modelsize(model,ppglangy,ppglangx)
    maxym=numpy.max(model["size"][:,:,2])#+1#numpy.max([el.y for el in model["ww"]])+1
    maxxm=numpy.max(model["size"][:,:,3])#+1#numpy.max([el.x for el in model["ww"]])+1
    minym=numpy.min(model["size"][:,:,0])#-1#numpy.max([el.y for el in model["ww"]])+1
    minxm=numpy.min(model["size"][:,:,1])#-1#numpy.max([el.x for el in model["ww"]])+1
    deltay=maxym-minym
    deltax=maxxm-minxm
    maxmy = (deltay+1)
    maxmx = (deltax+1)
    #maxmy = numpy.round(deltay+1)
    #maxmx = numpy.round(deltax+1)
    hsize=model["ww"][0].mask.shape[0]
    #minym=hsize/2
    #minxm=hsize/2
    #res=numpy.zeros((len(pglangy),len(pglangx),hsy+-maxmy+hsize,hsx+maxmx+hsize),dtype=numpy.float32)
    res=-1000*numpy.ones((len(ppglangy),len(ppglangx),hsy-minym+maxym+hsize+2,hsx-minxm+maxxm+hsize+2),dtype=numpy.float32)      
    for gly in selangy:
        for glx in selangx:
            res[gly,glx]=model["biases"][gly,glx]*k#0
            lminym=[]
            lminxm=[]
            minym=model["size"][gly,glx,0];minxm=model["size"][gly,glx,1]
            for l in range(len(model["ww"])):
                mm=model["ww"][l]
                angy=mm.ay+ppglangy[gly]
                angx=mm.ax+ppglangx[glx]
                if bis:
                    scr=project.project_bis(prec[l],project.pattern4_bis(angy),project.pattern4_bis(angx))
                else:
                    scr=project.project(prec[l],project.pattern4(angy),project.pattern4(angx))
                nposy=-minym+mm.y*numpy.cos(ppglangy[gly]/180.0*numpy.pi)-hsize/2.0*(numpy.cos(angy/180.0*numpy.pi))-mm.z*numpy.sin(angy/180.0*numpy.pi)
                nposx=-minxm+mm.x*numpy.cos(ppglangx[glx]/180.0*numpy.pi)-hsize/2.0*(numpy.cos(angx/180.0*numpy.pi))-mm.z*numpy.sin(angx/180.0*numpy.pi)
                #nposy=-miny+project.getproj(mm.y,mm.z,glangy,angy)
                #nposx=-minx+project.getproj(mm.x,mm.z,glangx,angx)
                posy=int(numpy.floor((nposy)))
                posx=int(numpy.floor((nposx)))
                disty=nposy-posy
                distx=nposx-posx
                #print maxmy-posy,maxmx-posx,nposy,nposx,glangx,angx
                res[gly,glx][maxmy-posy:maxmy-posy+hsy+hsize,maxmx-posx:maxmx-posx+hsx+hsize]=res[gly,glx][maxmy-posy:maxmy-posy+hsy+hsize,maxmx-posx:maxmx-posx+hsx+hsize]+(1-disty)*(1-distx)*scr
                res[gly,glx][maxmy-posy:maxmy-posy+hsy+hsize,maxmx-(posx+1):maxmx-(posx+1)+hsx+hsize]=res[gly,glx][maxmy-posy:maxmy-posy+hsy+hsize,maxmx-(posx+1):maxmx-(posx+1)+hsx+hsize]+(1-disty)*(distx)*scr
                res[gly,glx][maxmy-(posy+1):maxmy-(posy+1)+hsy+hsize,maxmx-(posx):maxmx-(posx)+hsx+hsize]=res[gly,glx][maxmy-(posy+1):maxmy-(posy+1)+hsy+hsize,maxmx-(posx):maxmx-(posx)+hsx+hsize]+(disty)*(1-distx)*scr
                res[gly,glx][maxmy-(posy+1):maxmy-(posy+1)+hsy+hsize,maxmx-(posx+1):maxmx-(posx+1)+hsx+hsize]=res[gly,glx][maxmy-(posy+1):maxmy-(posy+1)+hsy+hsize,maxmx-(posx+1):maxmx-(posx+1)+hsx+hsize]+(disty)*(distx)*scr
                #res[gly,glx]=res[gly,glx]+model["biases"][gly,glx]
                #res[gly,glx][maxmy-posy:maxmy-posy+hsy+hsize,maxmx-posx:maxmx-posx+hsx+hsize]=res[gly,glx][maxmy-posy:maxmy-posy+hsy+hsize,maxmx-posx:maxmx-posx+hsx+hsize]+(1-disty)*(1-distx)*scr
                #res[gly,glx][maxmy-posy:maxmy-posy+hsy+hsize,maxmx-(posx+1):maxmx-(posx+1)+hsx+hsize]=res[gly,glx][maxmy-posy:maxmy-posy+hsy+hsize,maxmx-(posx+1):maxmx-(posx+1)+hsx+hsize]+(1-disty)*(distx)*scr
                #res[gly,glx][maxmy-(posy+1):maxmy-(posy+1)+hsy+hsize,maxmx-(posx):maxmx-(posx)+hsx+hsize]=res[gly,glx][maxmy-(posy+1):maxmy-(posy+1)+hsy+hsize,maxmx-(posx):maxmx-(posx)+hsx+hsize]+(disty)*(1-distx)*scr
                #res[gly,glx][maxmy-(posy+1):maxmy-(posy+1)+hsy+hsize,maxmx-(posx+1):maxmx-(posx+1)+hsx+hsize]=res[gly,glx][maxmy-(posy+1):maxmy-(posy+1)+hsy+hsize,maxmx-(posx+1):maxmx-(posx+1)+hsx+hsize]+(disty)*(distx)*scr
                #res[4-w.y:4-w.y+hsy+4,4-w.x:4-w.x+hsx+4]=project.project(prec[l],ang0,ang0)
    return res                



def drawdet(ldet):
    if type(ldet)!=list:
        ldet=[ldet]
    for idl,l in enumerate(ldet):
       util.box(l["bbox"][0],l["bbox"][1],l["bbox"][2],l["bbox"][3],lw=2)
       pylab.text(l["bbox"][1],l["bbox"][0],"(%d,%d,%d,%d)"%(idl,l["scr"],l["ang"][0],l["ang"][1]))
       #print res[dd],l,dd[0],dd[1]

def rundet(img,model,angy=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90],angx=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90],interv=5,sbin=4,maxdet=1000,sort="scr",bbox=None,selangy=None,selangx=None,numhyp=1000,k=1):
    hog=pyrHOG2.pyrHOG(img,interv=interv,cformat=True,sbin=sbin)
    modelsize(model,angy,angx)
    hsize=model["ww"][0].mask.shape[0]
    maxym=numpy.max(model["size"][:,:,2])#numpy.max([el.y for el in model["ww"]])+1
    maxxm=numpy.max(model["size"][:,:,3])#numpy.max([el.x for el in model["ww"]])+1
    minym=numpy.min(model["size"][:,:,0])#numpy.max([el.y for el in model["ww"]])+1
    minxm=numpy.min(model["size"][:,:,1])#numpy.max([el.x for el in model["ww"]])+1
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
        res=det2(model,hog.hog[idr],ppglangy=angy,ppglangx=angx,selangy=selangy,selangx=selangx,k=k)
        order=(-res).argsort(None)
        for l in range(min(numhyp,len(order))):
            dd=numpy.unravel_index(order[l],res.shape)
            (minym,minxm,maxym,maxxm)=model["size"][dd[0],dd[1]]
            pcy1=(dd[2]-deltay+hsize/2)*sbin/hog.scale[idr]
            pcx1=(dd[3]-deltax+hsize/2)*sbin/hog.scale[idr]
            pcy2=(dd[2]-deltay+(maxym-minym)+hsize/2)*sbin/hog.scale[idr]
            pcx2=(dd[3]-deltax+(maxxm-minxm)+hsize/2)*sbin/hog.scale[idr]
            ldet.append({"id":0,"bbox":[pcy1,pcx1,pcy2,pcx2],"ang":(dd[0],dd[1]),"scr":res[dd]-model["rho"],"scl":hog.scale[idr],"hog":idr,"pos":(dd[2],dd[3]),"fpos":(dd[2]-deltay,dd[3]-deltax)})
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

def getfeat(model,hog,angy,angx,ang,pos,k):
    import project
    lhog=[]
    hsize=model["ww"][0].mask.shape[0]
    glangy=angy[ang[0]]
    glangx=angx[ang[1]]
    modelsize(model,angy,angx)
    maxym=numpy.max(model["size"][:,:,2])#numpy.max([el.y for el in model["ww"]])+1
    maxxm=numpy.max(model["size"][:,:,3])#numpy.max([el.x for el in model["ww"]])+1
    minym=numpy.min(model["size"][:,:,0])#numpy.max([el.y for el in model["ww"]])+1
    minxm=numpy.min(model["size"][:,:,1])#numpy.max([el.x for el in model["ww"]])+1
    deltay=maxym-minym+1#numpy.ceil(maxym-minym)
    deltax=maxxm-minxm+1#numpy.ceil(maxxm-minxm)
    m2=hog
    m2pad=numpy.zeros((m2.shape[0]+2*deltay,m2.shape[1]+2*deltax,m2.shape[2]),dtype=m2.dtype)
    m2pad[deltay:deltay+m2.shape[0],deltax:deltax+m2.shape[1]]=m2
    #hog=m2pad
    scr=0
    for idp,p in enumerate(model["ww"]):
        minym=model["size"][ang[0],ang[1],0];minxm=model["size"][ang[0],ang[1],1]
        #lhog.append(hog[pos[0]:pos[0]+hsize,pos[1]:pos[1]+hsize])
        auxhog=numpy.zeros((hsize,hsize,model["ww"][0].mask.shape[2]),dtype=model["ww"][0].mask.dtype)
        langy=glangy+p.ay
        langx=glangx+p.ax
        nposy=-minym+project.getproj(p.y,p.z,glangy,langy)#+deltay
        nposx=-minxm+project.getproj(p.x,p.z,glangx,langx)#+deltax
        posy=int(numpy.floor((nposy)))#+deltay
        posx=int(numpy.floor((nposx)))#+deltax
        disty=nposy-posy
        distx=nposx-posx
        #print posy,posx
        auxhog=auxhog+project.invprjhog_bis(m2pad[deltay+pos[0]+posy:,deltax+pos[1]+posx:],project.pattern4_bis(langy),project.pattern4_bis(langx))*(1-disty)*(1-distx)
        auxhog=auxhog+project.invprjhog_bis(m2pad[deltay+pos[0]+posy:,deltax+pos[1]+posx+1:],project.pattern4_bis(langy),project.pattern4_bis(langx))*(1-disty)*(distx)
        auxhog=auxhog+project.invprjhog_bis(m2pad[deltay+pos[0]+posy+1:,deltax+pos[1]+posx:],project.pattern4_bis(langy),project.pattern4_bis(langx))*(disty)*(1-distx)
        auxhog=auxhog+project.invprjhog_bis(m2pad[deltay+pos[0]+posy+1:,deltax+pos[1]+posx+1:],project.pattern4_bis(langy),project.pattern4_bis(langx))*(disty)*(distx)
        lhog.append(auxhog)
        scr+=numpy.sum(model["ww"][idp].mask*lhog[idp])
    biases=numpy.zeros((13,13),dtype=numpy.float32)
    biases[ang[0],ang[1]]=1.0#model["biases"][ang[0],ang[1]]
    scr+=model["biases"][ang[0],ang[1]]*k
    return lhog,biases,scr


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
    for py in range(7):
        for px in range(7):
            #lmask.append(part3D(hmodel[py*2:(py+1)*2+2,px*2:(px+1)*2+2],0,0,15,ang[py],ang[px]))#-15*numpy.cos(ang[py]/180.0*numpy.pi)*numpy.cos(ang[px]/180.0*numpy.pi),ang[py],ang[px]))
            lmask.append(part3D(hmodel[py*2:(py+1)*2+2,px*2:(px+1)*2+2],0,0,-15,ang[py],ang[px]))
            #lmask.append(part3D(hmodel[py*2:(py+1)*2+2,px*2:(px+1)*2+2],py*2,px*2,-15*numpy.cos(ang[py]/180.0*numpy.pi)*numpy.cos(ang[px]/180.0*numpy.pi),0,0)),ang[py],ang[px]))
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
    #model={"ww":[part3D(mask0,0,0,0,0,0),part3D(mask1,0,4,0,0,0),part3D(mask2,4,0,0,0,0),part3D(mask3,4,4,0,0,0)]}

    im2=util.myimread("./a.png",resize=2.0)

    glangy=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90]
    glangx=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90]
    fhog,ldet=rundet(im2,model,angy=glangy,angx=glangx,selangy=[6],selangx=[1])
    #fhog,ldet=rundet(im2,model,angy=[0],angx=[-60,-45,-30,-15,0,15,30,45,60],bbox=[100,100,200,150])
    pylab.figure(100)
    pylab.imshow(im2)
    for ld in ldet:
        feat,scr=getfeat(model,fhog.hog[ld["hog"]],glangy,glangx,ld["ang"],ld["fpos"])
        assert((abs(scr-ld["scr"])/(scr+0.0001))<0.0001)
        pylab.figure(110)
        pylab.clf()
        showHOG(model,feat,glangy[ld["ang"][0]],glangx[ld["ang"][1]],nhog=60,bis=False,val=1)
        pylab.figure(120)
        pylab.clf()
        showHOG(model,feat,0,0,nhog=60,bis=False,val=1)
        pylab.figure(100)
        drawdet(ld)
        pylab.draw()
        print ld
        raw_input()
    #fhog,ldet=rundet(im2,model,angy=[0],angx=[0])
    #res=det(model,myhog)
    fsfs
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
                nposy=w.y*numpy.cos(glangy/180.0*numpy.pi)-hsize/2.0*(numpy.cos(angy/180.0*numpy.pi))-w.z*numpy.sin(angy/180.0*numpy.pi)
                nposx=w.x*numpy.cos(glangx/180.0*numpy.pi)-hsize/2.0*(numpy.cos(angx/180.0*numpy.pi))-w.z*numpy.sin(angx/180.0*numpy.pi)
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
        nposy=-minym+mm.y*numpy.cos(glangy/180.0*numpy.pi)-hsize/2.0*(numpy.cos(angy/180.0*numpy.pi))-mm.z*numpy.sin(angy/180.0*numpy.pi)
        nposx=-minxm+mm.x*numpy.cos(glangx/180.0*numpy.pi)-hsize/2.0*(numpy.cos(angx/180.0*numpy.pi))-mm.z*numpy.sin(angx/180.0*numpy.pi)
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

            




