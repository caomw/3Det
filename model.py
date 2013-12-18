
import numpy
#danger: code dupicated in pyrHOG2.py: find a solution
import test3D2

def initmodel3D():

    lmask=[]
    step=2
    size=4
    if 0:
        profile=[0.8,0.9,1,0.9,0.8,0.7,0.65,0.7,0.8,0.9,1,0.9,0.8]
        ang=numpy.array([-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90])
        #ang=numpy.array([-60,-45,-30,-15,0,15,30,45,60])
        #ang=numpy.array([-30,-15,0,15,30])
        for py in range(len(ang)):
            for px in range(len(ang)):
                #lmask.append(test3D2.part3D(0.1*numpy.ones((4,4,31),dtype=numpy.float32),0,0,-25*numpy.cos(ang[py]/180.0*numpy.pi)*numpy.cos(ang[px]/180.0*numpy.pi),ang[py],ang[px]))
                #lmask.append(test3D2.part3D(0.1*numpy.ones((4,4,31),dtype=numpy.float32),0,0,-17,ang[py],ang[px]))
                lmask.append(test3D2.part3D(0.001*numpy.ones((4,4,31),dtype=numpy.float32),0,0,-17*profile[px],ang[py],ang[px]))
    else:
        #profile=[0.8,0.9,1,0.9,0.8,0.7,0.65,0.7,0.8,0.9,1,0.9,0.8]
        #ang=numpy.array([-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90])
        ang=numpy.array([-45,-15,0,0,0,15,45])
        z=numpy.array([0,1,2,2,2,1,0])
        #ang=numpy.array([-30,-15,0,15,30])
        for py in range(1,len(ang)):
            for px in range(len(ang)):
                #lmask.append(test3D2.part3D(0.1*numpy.ones((4,4,31),dtype=numpy.float32),0,0,-25*numpy.cos(ang[py]/180.0*numpy.pi)*numpy.cos(ang[px]/180.0*numpy.pi),ang[py],ang[px]))
                #lmask.append(test3D2.part3D(0.1*numpy.ones((4,4,31),dtype=numpy.float32),0,0,-17,ang[py],ang[px]))
                lmask.append(test3D2.part3D(0.00001*numpy.ones((4,4,31),dtype=numpy.float32),(py-4)*4,(px-4)*4,-2*z[px],ang[py],ang[px]))

    biases=numpy.zeros((13,13),dtype=numpy.float32)
    models=[{"ww":lmask,"biases":biases,"rho":0}]
    return models

def initmodel(fy,fx,N,useRL,lenf,CRF=False,small2=False):
    ww=[]
    hww=[]
    voc=[]
    dd=[]    
    lev=1
    for l in range(lev):
        if useRL:
            lowf=numpy.zeros((fy*2**l,fx*2**l,lenf)).astype(numpy.float32)
            lowf[:(fy*2**l)/2,:,2]=0.1/lenf
            lowf[(fy*2**l)/2:,:,7]=0.1/lenf
            lowf[:(fy*2**l)/2,:,11]=0.1/lenf
            lowf[(fy*2**l)/2:,:,16]=0.1/lenf
            lowf[:(fy*2**l)/2,:,18+2]=0.1/lenf
            lowf[(fy*2**l)/2:,:,18+7]=0.1/lenf
        else:
            lowf=numpy.ones((fy*2**l,fx*2**l,lenf)).astype(numpy.float32)
        ww.append(lowf)
        rho=0
    mynorm=0
    for wc in ww:
        mynorm+=numpy.sum(wc**2)
    for idw,wc in enumerate(ww):
        ww[idw]=wc*0.1/numpy.sqrt(mynorm)
    model={"ww":ww,"rho":rho,"fy":ww[0].shape[0],"fx":ww[0].shape[1],"N":N}
    if CRF:
        #cost=0.01*numpy.ones((2,fy*2,fx*2),dtype=numpy.float32)
        cost=0.01*numpy.ones((8,fy*2,fx*2),dtype=numpy.float32)
        model["cost"]=cost
    if small2:
        model["small2"]=numpy.array([0.0,0.0,0.0])#2x2,4x4,not used
    model["norm"]=(fy*fx)
    return model

def convert(simple,N,E):
    import extra
    extended=[]
    for l in simple:
        extended.append(extra.flip(extra.flip(l)))#copy.copy(simple)
    NE=N+2*E
    for idl,l in enumerate(simple):#for each component
        sy=l["ww"][0].shape[0]/N
        sx=l["ww"][0].shape[1]/N
        extended[idl]["ww"][0]=numpy.zeros((sy*NE,sx*NE,l["ww"][0].shape[2]),dtype=numpy.float32)
        padww=numpy.zeros((sy*N+2*E,sx*N+2*E,l["ww"][0].shape[2]),dtype=numpy.float32)
        padww[E:padww.shape[0]-E,E:padww.shape[1]-E]=l["ww"][0]
        for py in range(sy):
            for px in range(sx):
                extended[idl]["ww"][0][py*NE:(py+1)*NE,px*NE:(px+1)*NE]=padww[py*N:(py+1)*N+2*E,px*N:(px+1)*N+2*E]
    return extended

def convert2(wext,N,E):
    NE=N+2*E
    numy=wext.shape[0]/NE
    numx=wext.shape[1]/NE
    wsimple=numpy.zeros((numy*N,numx*N,wext.shape[2]),dtype=wext.dtype)
    for py in range(numy):
        for px in range(numx):
            wsimple[py*N:(py+1)*N,px*N:(px+1)*N]=wsimple[py*N:(py+1)*N,px*N:(px+1)*N]+wext[E+py*NE:E+py*NE+N,E+px*NE:E+px*NE+N]
    return wsimple

def mreduce(wext,N,E,nthr=0.01):
    NE=N+2*E
    wred=wext.copy()#numpy.array(wext.shape,dtpye=wext.dtype)
    for py in range(wext.shape[0]/NE):
        for px in range(wext.shape[1]/NE):
            if numpy.sqrt(numpy.sum(wred[py*NE:(py+1)*NE,px*NE:(px+1)*NE]**2))<nthr:
                wred[py*NE:(py+1)*NE,px*NE:(px+1)*NE]=0
    return wred

def dreduce(cost,N,E,nthr=0.01):
    NE=N+2*E
    ncost=cost.copy()#numpy.array(wext.shape,dtpye=wext.dtype)
    for py in range(cost.shape[0]):
        for px in range(cost.shape[1]):
            #print numpy.sqrt(numpy.sum(cost[py:(py+1),px:(px+1)]**2))
            if numpy.sqrt(numpy.sum(cost[py:(py+1),px:(px+1)]**2))<nthr:
                #print "IN"
                ncost[py:(py+1),px:(px+1)]=0
                #print ncost[py:(py+1),px:(px+1)]
    #print ncost
    return ncost

def drawParts(m,N,E,hpix=15):
    import drawHOG
    import util
    import pylab
    NE=N+2*E
    pylab.clf()
    bigim=numpy.zeros(((m.shape[0]/NE*N+N)*hpix,(m.shape[1]/NE*N+N)*hpix))
    for py in range(m.shape[0]/NE):
        for px in range(m.shape[1]/NE):
            im=drawHOG.drawHOG(m[py*NE:(py+1)*NE,px*NE:(px+1)*NE])
            bigim[py*N*hpix:((py+1)*N+2*E)*hpix,px*N*hpix:((px+1)*N+2*E)*hpix]=bigim[py*N*hpix:((py+1)*N+2*E)*hpix,px*N*hpix:((px+1)*N+2*E)*hpix]+im
    for py in range(m.shape[0]/NE-2):
        for px in range(m.shape[1]/NE-2):
            util.box(py*N*hpix,px*N*hpix,((py+1)*N+2*E)*hpix,((px+1)*N+2*E)*hpix,lw=3,col="r-")
    px=m.shape[0]/NE-1
    for py in range(m.shape[0]/NE-2):
        #for repx in range(m.shape[1]/NE)
        util.box(py*N*hpix,px*N*hpix,((py+1)*N+2*E)*hpix,((px+1)*N+2*E)*hpix,lw=3,col="r-")
    py=m.shape[1]/NE-1
    for px in range(m.shape[1]/NE-2):
        util.box(py*N*hpix,px*N*hpix,((py+1)*N+2*E)*hpix,((px+1)*N+2*E)*hpix,lw=3,col="r-")
    py=m.shape[1]/NE-1
    px=m.shape[0]/NE-1
    util.box(py*N*hpix,px*N*hpix,((py+1)*N+2*E)*hpix,((px+1)*N+2*E)*hpix,lw=3,col="r-")
    pylab.imshow(bigim)
            #raw_input()
    return bigim

def drawPartsSeparate(m,N,E,hpix=15,graph=True):
    import drawHOG
    import util
    import pylab
    pylab.clf()
    NE=N+2*E
    if graph:
        AD=0
    else:
        AD=2*E
    bigim=(numpy.min(m)+numpy.max(m))/2.0*numpy.ones(((m.shape[0]/NE*(NE+1))*hpix,(m.shape[1]/NE*(NE+1))*hpix))
    for py in range(m.shape[0]/NE):
        for px in range(m.shape[1]/NE):
            im=drawHOG.drawHOG(m[py*NE:(py+1)*NE,px*NE:(px+1)*NE])
            bigim[(E/2.0+py*(NE+1))*hpix:(E/2.0+(py+1)*(NE+1)-E)*hpix,(E/2.0+px*(NE+1))*hpix:(E/2.0+(px+1)*(NE+1)-E)*hpix]=im
            util.box((E/2.0+py*(NE+1))*hpix,(E/2.0+px*(NE+1))*hpix,(E/2.0+(py+1)*(NE+1)-E)*hpix,(E/2.0+(px+1)*(NE+1)-E)*hpix,lw=2,col="r-")
    for py in range(m.shape[0]/NE-1):
        for px in range(m.shape[1]/NE):
            pylab.plot([(E/2.0+NE/2+px*(NE+1))*hpix,(E/2.0+NE/2+(px)*(NE+1))*hpix],[(AD+E/2.0+NE/2+py*(NE+1))*hpix,(E/2.0+NE/2+(py+1)*(NE+1)-AD)*hpix],"w-",lw=2,alpha=0.5)
    for py in range(m.shape[0]/NE):
        for px in range(m.shape[1]/NE-1):
            pylab.plot([(AD+E/2.0+NE/2+px*(NE+1))*hpix,(E/2.0+NE/2+(px+1)*(NE+1)-AD)*hpix],[(E/2.0+NE/2+py*(NE+1))*hpix,(E/2.0+NE/2+(py)*(NE+1))*hpix],"w-",lw=2,alpha=0.5)
    pylab.imshow(bigim)
    pylab.axis("off")
    return bigim

def model2w(model,deform,usemrf,usefather,k=1,lastlev=0,usebow=False,useCRF=False,small2=False):
    w=numpy.zeros(0,dtype=numpy.float32)
    if model.has_key("norm"):
        norm=model["norm"]
    else:
        norm=1
    for l in range(len(model["ww"])-lastlev):
        #print "here"#,item
        w=numpy.concatenate((w,model["ww"][l].flatten()/float(norm)))
    if usebow:
        for l in range(len(model["hist"])-lastlev):
            w=numpy.concatenate((w,model["hist"][l].flatten()))
    if useCRF:
        w=numpy.concatenate((w,(model["cost"]/float(k)).flatten()))
    if small2:
        w=numpy.concatenate((w,model["small2"].flatten()))
    return w

def feat2flatten(feat):
    hsize=feat[0].size
    flat=numpy.zeros(hsize*len(feat),dtype=feat[0].dtype)
    for idl,l in enumerate(feat):
        flat[idl*hsize:(idl+1)*hsize]=l.flatten()
    return flat

def model2w3D(model):
    w=numpy.zeros(0,dtype=numpy.float32)
    for l in range(len(model["ww"])):
        #print "here"#,item
        w=numpy.concatenate((w,model["ww"][l].mask.flatten()))
    w=numpy.concatenate((w,model["biases"].flatten()))
    return w

def w2model(descr,N,E,rho,lev,fsz,fy=[],fx=[],bin=5,siftsize=2,deform=False,usemrf=False,usefather=False,k=1,norm=1,mindef=0.001,useoccl=False,usebow=False,useCRF=False,small2=False):
        #does not work with occlusions
        """
        build a new model from the weights of the SVM
        """     
        ww=[]  
        p=0
        occl=[0]*lev
        d=descr
        NE=N+2*E
        for l in range(lev):
            dp=(fy*fx)*4**l*fsz
            ww.append((norm*d[p:p+dp].reshape((fy*2**l,fx*2**l,fsz))).astype(numpy.float32))
            p+=dp
            if useoccl:
                occl[l]=d[p]
                p+=1
        hist=[]
        if usebow:
            for l in range(lev):
                hist.append(d[p:p+bin**(siftsize**2)].astype(numpy.float32))
                #hist.append(numpy.zeros(625,dtype=numpy.float32))
                p=p+bin**(siftsize**2)
        m={"ww":ww,"rho":rho,"fy":fy,"fx":fx,"occl":occl,"N":N,"E":E}
        if useCRF:
            m["cost"]=((d[p:p+8*(fy/NE)*(fx/NE)].reshape((8,fy/NE,fx/NE))*(k)))#.clip(mindef,10))
            p=p+8*(fy/NE)*(fx/NE)
            #m["cost"]=((d[p:p+4*(2*fy)*(2*fx)].reshape((4,2*fy,2*fx))/float(k)).clip(mindef,10))
            #p=p+4*(2*fy)*(2*fx)
        if small2:
            m["small2"]=d[p:]
        #m["facial"]=numpy.array([0.05,0.05 ,0.05,0.95 ,0.95,0.05 ,0.95,0.95])
        m["facial"]=numpy.array([7.5,5, 7.5,7.5, 13.5,6, 13.5,11.5, 15.0,9, 13,9, 7.5,10, 7.5,13, 11.5,7, 11.5,11 ])/20.0
        #m["facial"]=numpy.array([6,5])/18.0 #,6,8])/18.0
        m["facial"][::2]=m["facial"][::2]*(ww[0].shape[0]/(NE/N))
        m["facial"][1::2]=m["facial"][1::2]*(ww[0].shape[1]/(NE/N))
        return m


def w2model3D(oldmodel,descr,rho):
    hsize=oldmodel["ww"][0].mask.size
    hshape=oldmodel["ww"][0].mask.shape
    for idl,l in enumerate(oldmodel["ww"]):
        oldmodel["ww"][idl].mask=descr[idl*hsize:(idl+1)*hsize].reshape(hshape)
    oldmodel["biases"]=descr[(idl+1)*hsize:(idl+1)*hsize+13*13].reshape((13,13))
    oldmodel["rho"]=rho
    return oldmodel


