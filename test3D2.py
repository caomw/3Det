
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

def show(model,glangy,glangx,hsize=4,pixh=15,border=2,nhog=30):
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
    pylab.figure()
    pylab.imshow(img)    
    #pylab.figure()
    #pylab.imshow(img2)    
    pylab.draw()
    pylab.show()
    #raw_input()

def det(model,hog,pglangx=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90],pglangy=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90],bis=True):
    hsy=hog.shape[0]
    hsx=hog.shape[1]
    prec=[]
    for w in model["ww"]:
        prec.append(project.precompute(w.mask,hog))
    maxmy=numpy.max([el.y for el in model["ww"]])+1
    maxmx=numpy.max([el.x for el in model["ww"]])+1
    hsize=4
    #minym=hsize/2
    #minxm=hsize/2
    res=numpy.zeros((len(pglangy),len(pglangx),hsy+maxmy+hsize,hsx+maxmx+hsize),dtype=numpy.float32)
    for gly,glangy in enumerate(pglangy):
        for glx,glangx in enumerate(pglangx):
            lminym=[]
            lminxm=[]
            for w in model["ww"]:
                #mm=model["ww"][l]
                angy=w.ay+glangy
                angx=w.ax+glangx
                nposy=w.y*numpy.cos(glangy/180.0*numpy.pi)-hsize/2.0*(numpy.cos(angy/180.0*numpy.pi))-w.z*numpy.sin(angy/180.0*numpy.pi)
                nposx=w.x*numpy.cos(glangx/180.0*numpy.pi)-hsize/2.0*(numpy.cos(angx/180.0*numpy.pi))-w.z*numpy.sin(angx/180.0*numpy.pi)
                lminym.append(nposy)
                lminxm.append(nposx)
            minym=numpy.min(lminym)
            minxm=numpy.min(lminxm)
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

def rundet(img,model,angy=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75],angx=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75]):
    hog=pyrHOG2.pyrHOG(im,cformat=True,sbin=4)
    #for idr,r in enumerate(hog.hog):


im=util.myimread("./a.png")

hog=pyrHOG2.pyrHOG(im,cformat=True,sbin=4)

#myhog=hog.hog[8]
#imh=drawHOG.drawHOG(myhog);pylab.figure();pylab.imshow(imh)
#hmodel=myhog[0:8,12:20] # model

myhog=hog.hog[0]
hmodel=myhog[8:24,50:66] # model

lmask=[]
for py in range(4):
    for px in range(4):
        lmask.append(part3D(hmodel[py*4:(py+1)*4,px*4:(px+1)*4],py*4,px*4,0,0,0))
#mask0=hmodel[0:4,0:4].copy()
#mask1=hmodel[0:4,4:8].copy()
#mask2=hmodel[4:8,0:4].copy()
#mask3=hmodel[4:8,4:8].copy()
lmask[5].z=1;lmask[6].z=1;lmask[9].z=1;lmask[10].z=1;

model={"ww":lmask}
#model={"ww":[part3D(mask0,0,0,0,0,0),part3D(mask1,0,4,0,0,0),part3D(mask2,4,0,0,0,0),part3D(mask3,4,4,0,0,0)]}

res=det(model,myhog)
show(model,0,0)
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

        




