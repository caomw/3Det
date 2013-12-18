# training of the new CRF model
# denseCRF [category] [configuration]

##################some import
import matplotlib
matplotlib.use("Agg") #if run outside ipython do not show any figure
from database import *
from multiprocessing import Pool
import util
import pyrHOG2
#import pyrHOG2RL
import extra
import VOCpr
import model
import time
import copy
import itertools
import sys
import crf3
import logging as lg
import os
import pegasos2 as pegasos
import denseCRFtest

########################## load configuration parametes

print "Loading default configuration config.py"
from config import * #default configuration      

import_name=""
if len(sys.argv)>2: #specific configuration
    print "Loading configuration from %s"%sys.argv[2]
    import_name=sys.argv[2]
    exec "from config_%s import *"%import_name  

cfg.cls=sys.argv[1]
#save a local configuration copy 
import shutil
shutil.copyfile("config_"+import_name+".py",cfg.testpath+cfg.cls+"%d"%cfg.numcl+"_"+cfg.testspec+".cfg.py")
testname=cfg.testpath+cfg.cls+("%d"%cfg.numcl)+"_"+cfg.testspec
if cfg.checkpoint:
    import os
    if not os.path.exists(cfg.localdata):
        os.makedirs(cfg.localdata)
    localsave=cfg.localdata+cfg.cls+("%d"%cfg.numcl)+"_"+cfg.testspec
#cfg.useRL=False#for the moment
cfg.show=False
cfg.auxdir=""
cfg.numhyp=5
#cfg.numneg= 10
bias=cfg.bias
#cfg.bias=bias
#cfg.posovr= 0.75
#cfg.perc=0.25
#just for a fast test
#cfg.maxpos = 50
#cfg.maxneg = 20
#cfg.maxexamples = 10000
#cfg.maxtest = 20#100
parallel=True
cfg.show=False
#cfg.neginpos=False
localshow=cfg.localshow
numcore=cfg.multipr
if cfg.multipr==False:
    parallel=False
    numcore=1
notreg=0
if cfg.trunc:
    lenf=32
else:
    lenf=31
#cfg.numcl=3
#cfg.valreg=0.01#set in configuration
#cfg.useRL=True

######################### setup log file 
import os
lg.basicConfig(filename=testname+".log",format='%(asctime)s %(message)s',datefmt='%I:%M:%S %p',level=lg.DEBUG)
lg.info("#################################################################################")
lg.info("############## Starting the training on %s on %s dataset ################"%(os.uname()[1],cfg.db))
lg.info("Software Version:%s"%cfg.version)
#################### wrappers

import detectCRF
from multiprocessing import Manager

manager=Manager()
d=manager.dict()       

def hardNegCache(x):
    if x["control"]["cache_full"]:
        return [],[],[]
    else:
        return detectCRF.hardNeg(x)

def hardNegPosCache(x):
    if x["control"]["cache_full"]:
        return [],[],[]
    else:
        return detectCRF.hardNegPos(x)

mypool = Pool(numcore) #maxtasksperchild=10 #keep the child processes as small as possible 

########################load training and test samples
if cfg.db=="VOC":
    if cfg.year=="2007":
        trPosImages=getRecord(VOC07Data(select="pos",cl="%s_trainval.txt"%cfg.cls,
                        basepath=cfg.dbpath,#"/home/databases/",
                        usetr=True,usedf=False),cfg.maxpos)
        trPosImagesNoTrunc=getRecord(VOC07Data(select="pos",cl="%s_trainval.txt"%cfg.cls,
                        basepath=cfg.dbpath,#"/home/databases/",
                        usetr=False,usedf=False),cfg.maxpos)
        trNegImages=getRecord(VOC07Data(select="neg",cl="%s_trainval.txt"%cfg.cls,
                        basepath=cfg.dbpath,#"/home/databases/",#"/share/ISE/marcopede/database/",
                        usetr=True,usedf=False),cfg.maxneg)
        trNegImagesFull=getRecord(VOC07Data(select="neg",cl="%s_trainval.txt"%cfg.cls,
                        basepath=cfg.dbpath,usetr=True,usedf=False),cfg.maxnegfull)
        #test
        tsPosImages=getRecord(VOC07Data(select="pos",cl="%s_test.txt"%cfg.cls,
                        basepath=cfg.dbpath,#"/home/databases/",#"/share/ISE/marcopede/database/",
                        usetr=True,usedf=False),cfg.maxtest)
        tsNegImages=getRecord(VOC07Data(select="neg",cl="%s_test.txt"%cfg.cls,
                        basepath=cfg.dbpath,#"/home/databases/",#"/share/ISE/marcopede/database/",
                        usetr=True,usedf=False),cfg.maxneg)
        tsImages=numpy.concatenate((tsPosImages,tsNegImages),0)
        tsImagesFull=getRecord(VOC07Data(select="all",cl="%s_test.txt"%cfg.cls,
                        basepath=cfg.dbpath,
                        usetr=True,usedf=False),cfg.maxtestfull)

elif cfg.db=="buffy":
    trPosImages=getRecord(Buffy(select="all",cl="trainval.txt",
                    basepath=cfg.dbpath,
                    trainfile="buffy/",
                    imagepath="buffy/images/",
                    annpath="buffy/",
                    usetr=True,usedf=False),cfg.maxpos)
    trPosImagesNoTrunc=trPosImages
    trNegImages=getRecord(DirImages(imagepath=cfg.dbpath+"INRIAPerson/train_64x128_H96/neg/"),cfg.maxneg)
    trNegImagesFull=getRecord(DirImages(imagepath=cfg.dbpath+"INRIAPerson/train_64x128_H96/neg/"),cfg.maxnegfull)
    #test
    tsPosImages=getRecord(Buffy(select="all",cl="test.txt",
                    basepath=cfg.dbpath,
                    trainfile="buffy/",
                    imagepath="buffy/images/",
                    annpath="buffy/",
                    usetr=True,usedf=False),cfg.maxtest)
    tsImages=tsPosImages#numpy.concatenate((tsPosImages,tsNegImages),0)
    tsImagesFull=tsPosImages=getRecord(Buffy(select="all",cl="test.txt",
                    basepath=cfg.dbpath,
                    trainfile="buffy/",
                    imagepath="buffy/images/",
                    annpath="buffy/",
                    usetr=True,usedf=False),cfg.maxtestfull)

elif cfg.db=="inria":
    trPosImages=getRecord(InriaPosData(basepath=cfg.dbpath),cfg.maxpos)
    trPosImagesNoTrunc=trPosImages
    trNegImages=getRecord(InriaNegData(basepath=cfg.dbpath),cfg.maxneg)#check if it works better like this
    trNegImagesFull=getRecord(InriaNegData(basepath=cfg.dbpath),cfg.maxnegfull)
    #test
    tsImages=getRecord(InriaTestFullData(basepath=cfg.dbpath),cfg.maxtest)
    tsImagesFull=tsImages
elif cfg.db=="imagenet":
    #training
    trPosImages1=getRecord(imageNet(select="all",cl="%s_trainval.txt"%cfg.cls,
                    basepath=cfg.dbpath,
                    trainfile="/tandem/",
                    imagepath="/tandem/images/",
                    annpath="/tandem/Annotation/n02835271/",
                    usetr=True,usedf=False),cfg.maxpos/2)
    trPosImages2=getRecord(VOC07Data(select="pos",cl="%s_trainval.txt"%"bicycle",
                    basepath=cfg.dbpath,#"/home/databases/",
                    usetr=True,usedf=False),cfg.maxpos/2)
    trPosImagesNoTrunc2=getRecord(VOC07Data(select="pos",cl="%s_trainval.txt"%"bicycle",
                    basepath=cfg.dbpath,#"/home/databases/",
                    usetr=False,usedf=False),cfg.maxpos)
    trPosImages=numpy.concatenate((trPosImages1,trPosImages2),0)
    trPosImagesNoTrunc=numpy.concatenate((trPosImages1,trPosImagesNoTrunc2),0)
    trNegImages=getRecord(VOC07Data(select="neg",cl="%s_trainval.txt"%"bicycle",
                    basepath=cfg.dbpath,#"/home/databases/",#"/share/ISE/marcopede/database/",
                    usetr=True,usedf=False),cfg.maxneg)
    trNegImagesFull=getRecord(VOC07Data(select="neg",cl="%s_trainval.txt"%"bicycle",
                    basepath=cfg.dbpath,usetr=True,usedf=False),cfg.maxnegfull)
    #test  
    tsPosImages1=getRecord(imageNet(select="all",cl="%s_test.txt"%cfg.cls,
                    basepath=cfg.dbpath,
                    trainfile="/tandem/",
                    imagepath="/tandem/images/",
                    annpath="/tandem/Annotation/n02835271/",
                    usetr=True,usedf=False),cfg.maxtest/2)
    tsPosImages2=getRecord(VOC07Data(select="pos",cl="%s_test.txt"%"bicycle",
                    basepath=cfg.dbpath,#"/home/databases/",#"/share/ISE/marcopede/database/",
                    usetr=True,usedf=False),cfg.maxtest/2)        
    tsNegImages=getRecord(VOC07Data(select="neg",cl="%s_test.txt"%"bicycle",
                    basepath=cfg.dbpath,#"/home/databases/",#"/share/ISE/marcopede/database/",
                    usetr=True,usedf=False),cfg.maxneg)
    tsImages=numpy.concatenate((tsPosImages1,tsPosImages2,tsNegImages),0)
    tsImagesFull=getRecord(VOC07Data(select="all",cl="%s_test.txt"%"bicycle",
                    basepath=cfg.dbpath,
                    usetr=True,usedf=False),cfg.maxtestfull)
elif cfg.db=="LFW":
    tfold=0 #test fold 0 other 9 for training
    aux=getRecord(LFW(basepath=cfg.dbpath,fold=0),cfg.maxpos,facial=True,pose=True)
    trPosImages=numpy.array([],dtype=aux.dtype)
    for l in range(10):
        aux=getRecord(LFW(basepath=cfg.dbpath,fold=l,fake=cfg.nobbox),cfg.maxpos,facial=True,pose=True)
        if l==tfold:
            tsImages=getRecord(LFW(basepath=cfg.dbpath,fold=l),cfg.maxtest,facial=True,pose=True)
        else:
            trPosImages=numpy.concatenate((trPosImages,aux),0)
    trPosImagesNoTrunc=trPosImages
    trNegImages=getRecord(InriaNegData(basepath=cfg.dbpath),cfg.maxneg)#check if it works better like this
    trNegImagesFull=getRecord(InriaNegData(basepath=cfg.dbpath),cfg.maxnegfull)
    #test
    #tsImages=getRecord(InriaTestFullData(basepath=cfg.dbpath),cfg.maxtest)
    tsImagesFull=tsImages
elif cfg.db=="AFLW":
    trPosImages=getRecord(AFLW(basepath=cfg.dbpath,fold=0),cfg.maxpos,facial=True,pose=True)#cfg.useFacial)
    trPosImagesNoTrunc=trPosImages[:900]
    trNegImages=getRecord(InriaNegData(basepath=cfg.dbpath),cfg.maxneg)#check if it works better like this
    trNegImagesFull=getRecord(InriaNegData(basepath=cfg.dbpath),cfg.maxnegfull)
    #test
    tsImages=getRecord(AFLW(basepath=cfg.dbpath,fold=0),cfg.maxtest,facial=True,pose=True)#cfg.useFacial)
    tsImagesFull=tsImages
elif cfg.db=="MultiPIEunif":
    cameras=["11_0","12_0","09_0","06_0","13_0","14_0","05_1","05_0","04_1","19_0","20_0","01_0","24_0"]
    conditions=5
    subjects=14
    aux=getRecord(MultiPIE(basepath=cfg.dbpath),cfg.maxpos,facial=True,pose=True)
    #total samples= conditions * cameras * subjects
    #5*13*14=910
    trPosImages=numpy.array([],dtype=aux.dtype)
    for ss in range(subjects):
        for cc in cameras: 
            trPosImages=numpy.concatenate((trPosImages,getRecord(MultiPIE(basepath=cfg.dbpath,camera=cc),conditions,facial=True,pose=True)))
    trPosImagesNoTrunc=trPosImages
    trNegImages=getRecord(InriaNegData(basepath=cfg.dbpath),cfg.maxneg)#check if it works better like this
    trNegImagesFull=getRecord(InriaNegData(basepath=cfg.dbpath),cfg.maxnegfull)
    #test
    tsImages=getRecord(MultiPIE(basepath=cfg.dbpath,session="session02"),cfg.maxtest,facial=True,pose=True)#cfg.useFacial)
    tsImagesFull=tsImages

elif cfg.db=="MultiPIE2":
    #cameras=["11_0","12_0","09_0","08_0","13_0","14_0","05_1","05_0","04_1","19_0","20_0","01_0","24_0"]
    #cameras=["110","120","090","080","130","140","051","050","041","190","200","010","240"]
    cameras=["080","130","140","051","050","041","190"]
    #conditions=2
    #subjects=1#25
    aux=getRecord(MultiPIE2(basepath=cfg.dbpath),cfg.maxpos,facial=True,pose=True)
    trPosImages=numpy.array([],dtype=aux.dtype)
    #for ss in range(subjects):
    #    trPosImages=numpy.concatenate((trPosImages,getRecord(MultiPIE(basepath=cfg.dbpath,camera=cameras[6],subject="%03d"%ss),conditions,facial=True,pose=True)))
    #conditions=5
    #subjects=50 #to reach 600
    #13*50=650   
    for cc in cameras: 
        #for ss in range(subjects):
        if cc=="051":
            conditions=cfg.maxpos/3#150#300
        else:
            conditions=cfg.maxpos/7#50
        trPosImages=numpy.concatenate((trPosImages,getRecord(MultiPIE2(basepath=cfg.dbpath,camera=cc),conditions,facial=True,pose=True)))
        #print conditions,"LEN",len(trPosImages)

    trPosImagesNoTrunc=trPosImages[:len(trPosImages)/2]
    trNegImages=getRecord(InriaNegData(basepath=cfg.dbpath),cfg.maxneg)#check if it works better like this
    trNegImagesFull=getRecord(InriaNegData(basepath=cfg.dbpath),cfg.maxnegfull)
    #test
    #subjects=10
    conditions=5
    tsImages=numpy.array([],dtype=aux.dtype)
    for cc in cameras:#range(subjects):
        tsImages=numpy.concatenate((tsImages,getRecord(MultiPIE2(basepath=cfg.dbpath,camera=cc),conditions,facial=True,pose=True)))
    #tsImages=getRecord(MultiPIE(basepath=cfg.dbpath,session="session02"),cfg.maxtest,facial=True,pose=True)#cfg.useFacial)
    tsImagesFull=tsImages


elif cfg.db=="MultiPIEramanan":
    cameras=["11_0","12_0","09_0","08_0","13_0","14_0","05_1","05_0","04_1","19_0","20_0","01_0","24_0"]
    cameras2=["110","120","090","080","130","140","051","050","041","190","200","010","240"]
    #conditions=2
    #subjects=1#25
    aux=getRecord(MultiPIE(basepath=cfg.dbpath),cfg.maxpos,facial=True,pose=True)
    trPosImages=numpy.array([],dtype=aux.dtype)
    #for ss in range(subjects):
    #    trPosImages=numpy.concatenate((trPosImages,getRecord(MultiPIE(basepath=cfg.dbpath,camera=cameras[6],subject="%03d"%ss),conditions,facial=True,pose=True)))
    conditions=1
    subjects=5#0 #to reach 600
    #13*50=650   
    for cc in cameras: 
        for ss in range(subjects):
            trPosImages=numpy.concatenate((trPosImages,getRecord(MultiPIE(basepath=cfg.dbpath,camera=cc,subject="%03d"%ss),conditions,facial=True,pose=True)))

    trPosImagesNoTrunc=trPosImages
    trNegImages=getRecord(InriaNegData(basepath=cfg.dbpath),cfg.maxneg)#check if it works better like this
    trNegImagesFull=getRecord(InriaNegData(basepath=cfg.dbpath),cfg.maxnegfull)
    #test
    subjects=10
    conditions=2
    tsImages=numpy.array([],dtype=aux.dtype)
    for ss in range(subjects):
        tsImages=numpy.concatenate((tsImages,getRecord(MultiPIE(basepath=cfg.dbpath,session="session02",camera=cameras[6],subject="%03d"%ss),conditions,facial=True,pose=True)))
    #tsImages=getRecord(MultiPIE(basepath=cfg.dbpath,session="session02"),cfg.maxtest,facial=True,pose=True)#cfg.useFacial)
    tsImagesFull=tsImages


elif cfg.db=="MultiPIEfron":
    cameras=["11_0","12_0","09_0","06_0","13_0","14_0","05_1","05_0","04_1","19_0","20_0","01_0","24_0"]
    conditions=2
    subjects=150
    aux=getRecord(MultiPIE(basepath=cfg.dbpath),cfg.maxpos,facial=True,pose=True)
    trPosImages=numpy.array([],dtype=aux.dtype)
    for ss in range(subjects):
        trPosImages=numpy.concatenate((trPosImages,getRecord(MultiPIE(basepath=cfg.dbpath,camera=cameras[6],subject="%03d"%ss),conditions,facial=True,pose=True)))
    trPosImagesNoTrunc=trPosImages
    trNegImages=getRecord(InriaNegData(basepath=cfg.dbpath),cfg.maxneg)#check if it works better like this
    trNegImagesFull=getRecord(InriaNegData(basepath=cfg.dbpath),cfg.maxnegfull)
    #test
    subjects=10
    conditions=2
    tsImages=numpy.array([],dtype=aux.dtype)
    for ss in range(subjects):
        tsImages=numpy.concatenate((tsImages,getRecord(MultiPIE(basepath=cfg.dbpath,session="session02",camera=cameras[6],subject="%03d"%ss),conditions,facial=True,pose=True)))
    #tsImages=getRecord(MultiPIE(basepath=cfg.dbpath,session="session02"),cfg.maxtest,facial=True,pose=True)#cfg.useFacial)
    tsImagesFull=tsImages


########################compute aspect ratio and dector size 
import stats
#lfy,lfx=stats.build_components(trPosImages,cfg)
lfy,lfx=stats.build_components_pose(trPosImages,cfg)#should be better, but not tested yet!

cfg.fy=lfy#[7,10]#lfy
cfg.fx=lfx#[11,7]#lfx
# the real detector size would be (cfg.fy,cfg.fx)*2 hog cells
initial=True
loadedchk=False
last_round=False
if cfg.checkpoint and not cfg.forcescratch:

    #check if the last AP is already there stop because everything has been done
    if os.path.exists("%s_final.png"%(testname)):
        print "Model already completed, nothing to do!!!"
        lg.info("Model already completed and evaluated, nothing to do!")    
        sys.exit()

    #load last model
    for l in range(cfg.posit):
        try:
            models=util.load(testname+"%d.model"%l)
            print "Loaded model %d"%(l)
            lg.info("Loaded model %d"%(l))    
        except:
            if l>0:
                print "Model %d does not exist"%(l)
                lg.info("Model %d does not exist"%(l))    
                #break
            else:
                print "No model found"
                break
        #lg.info("Loaded model")    
    try:
        print "Begin loading old status..."
        #os.path.exists(localsave+".pos.valid")
        fd=open(localsave+".pos.valid","r")
        fd.close()
        dpos=util.load(localsave+".pos.chk")
        lpdet=dpos["lpdet"]
        lpfeat=dpos["lpfeat"]
        lpedge=dpos["lpedge"]
        cpit=dpos["cpit"]
        last_round=dpos["last_round"]
        initial=False
        loadedchk=True
        lg.info("""Loaded old positive checkpoint:
Number Positive SV:%d                        
        """%(len(lpdet)))
        lndet=[]
        cnit=0
        #if at this point is already enough for the checkpoint
        #os.path.exists(localsave+".neg.valid")
        fd=open(localsave+".neg.valid","r")
        fd.close()
        dneg=util.load(localsave+".neg.chk")
        lndet=dneg["lndet"]
        lnfeat=dneg["lnfeat"]
        lnedge=dneg["lnedge"]
        cnit=dneg["cnit"]
        lg.info("""Loaded negative checkpoint:
Number Negative SV:%d                                
        """%(len(lndet)))
        print "Loaded old status..."
    except:
        pass

    try: #load the final model and test 
        models=util.load(testname+"_final.model")
        print "Loaded final model"
        lg.info("Loaded final model")    
        #last_round=True
        cpit=cfg.posit
        initial=False
        loadedchk=True
    except:
        pass
    

import pylab as pl

if initial:
    cpit=0
    cnit=0
    
    models=model.initmodel3D()

    #########add thresholds
    for m in models:
        m["thr"]=-2

    lndet=[] #save negative detections
    lnfeat=[] #
    lnedge=[] #
    lndetnew=[]

    lpdet=[] #save positive detections
    lpfeat=[] #
    lpedge=[] #

###################### rebuild w
waux=[]
rr=[]
w1=numpy.array([])
sizereg=numpy.zeros(cfg.numcl,dtype=numpy.int32)
#from model m to w
for idm,m in enumerate(models[:cfg.numcl]):
    if cfg.use3D:
        waux.append(model.model2w3D(models[idm]))        
    else:    
        waux.append(model.model2w(models[idm],False,False,False,useCRF=True,k=cfg.k))
    rr.append(models[idm]["rho"])
    w1=numpy.concatenate((w1,waux[-1],-numpy.array([models[idm]["rho"]])/bias))
    sizereg[idm]=13*13
#w2=w #old w
w=w1

#add ids clsize and cumsize for each model
clsize=[]
cumsize=numpy.zeros(cfg.numcl+1,dtype=numpy.int)
for l in range(cfg.numcl):
    models[l]["id"]=l
    clsize.append(len(waux[l])+1)
    cumsize[l+1]=cumsize[l]+len(waux[l])+1
clsize=numpy.array(clsize)

util.save("%s%d.model"%(testname,0),models)
lg.info("Built first model")    

total=[]
posratio=[]
cache_full=False

#from scipy.ndimage import zoom
import detectCRF
from multiprocessing import Pool
import itertools

#just to compute totPosEx when using check points
arg=[]
for idl,l in enumerate(trPosImages):
    bb=l["bbox"]
    for idb,b in enumerate(bb):
        if cfg.useRL:
            arg.append({"idim":idl,"file":l["name"],"idbb":idb,"bbox":b,"models":models,"facial":l["facial"],"pose":l["pose"],"cfg":cfg,"flip":False})    
            arg.append({"idim":idl,"file":l["name"],"idbb":idb,"bbox":b,"models":models,"facial":l["facial"],"pose":l["pose"],"cfg":cfg,"flip":True})    
        else:
            arg.append({"idim":idl,"file":l["name"],"idbb":idb,"bbox":b,"models":models,"facial":l["facial"],"pose":l["pose"],"cfg":cfg,"flip":False})    
totPosEx=len(arg)

lg.info("Starting Main loop!")
####################### repeat scan positives
for it in range(cpit,cfg.posit):
    lg.info("############# Positive iteration %d ################"%it)
    #mypool = Pool(numcore)
    #counters
    padd=0
    pbetter=0
    pworst=0
    pold=0
    skipos=False
    if it==0:
        cfg.mysort="ovr"
    else:
        cfg.mysort="scr"

    ########## rescore old positive detections
    lg.info("Rescoring %d Positive detections"%len(lpdet))
    for idl,l in enumerate(lpdet):
        idm=l["id"]
        ang=l["ang"]
        scr=0
        for idp,p in enumerate(models[idm]["ww"]):
            scr=scr+numpy.sum(p.mask*lpfeat[idl][idp])
        #scr+=numpy.sum(lpedge[idl])
        scr+=models[idm]["biases"][ang[0],ang[1]]*cfg.k
        lpdet[idl]["scr"]=scr-models[idm]["rho"]#numpy.sum(models[idm]["ww"][0]*lpfeat[idl])+numpy.sum(models[idm]["cost"]*lpedge[idl])-models[idm]["rho"]#-rr[idm]/bias

    if not cfg.checkpoint or not loadedchk:
        arg=[]
        for idl,l in enumerate(trPosImages):
            bb=l["bbox"]
            for idb,b in enumerate(bb):
                #if b[4]==1:#only for truncated
                if cfg.useRL:
                    arg.append({"idim":idl,"file":l["name"],"idbb":idb,"bbox":b,"models":models,"facial":l["facial"],"pose":l["pose"],"cfg":cfg,"flip":False})    
                    arg.append({"idim":idl,"file":l["name"],"idbb":idb,"bbox":b,"models":models,"facial":l["facial"],"pose":l["pose"],"cfg":cfg,"flip":True})    
                else:
                    arg.append({"idim":idl,"file":l["name"],"idbb":idb,"bbox":b,"models":models,"facial":l["facial"],"pose":l["pose"],"cfg":cfg,"flip":False})    

        totPosEx=len(arg)
        #lpdet=[];lpfeat=[];lpedge=[]
        if not(parallel):
            itr=itertools.imap(detectCRF.refinePos,arg)        
        else:
            itr=mypool.imap(detectCRF.refinePos,arg)

        lg.info("############## Staritng Scan of %d Positives BBoxs ###############"%totPosEx)
        for ii,res in enumerate(itr):
            found=False
            if res[0]!=[]:
                #compare new score with old
                newdet=res[0]
                newfeat=res[1]
                newedge=res[2]
                for idl,l in enumerate(lpdet):
                    #print "Newdet",newdet["idim"],"Olddet",l["idim"]
                    if (newdet["idim"]==l["idim"]): #same image
                        if (newdet["idbb"]==l["idbb"]): #same bbox
                            if (newdet["scr"]>l["scr"]):#compare score
                                print "New detection has better score"
                                lpdet[idl]=newdet
                                lpfeat[idl]=newfeat
                                lpedge[idl]=newedge
                                found=True
                                pbetter+=1
                            else:
                                print "New detection has worse score"
                                found=True
                                pworst+=1
                if not(found):
                    print "Added a new sample"
                    lpdet.append(res[0])
                    lpfeat.append(res[1])
                    lpedge.append(res[2])
                    padd+=1
            else: #not found any detection with enough overlap
                print "Example not found!"
                for idl,l in enumerate(lpdet):
                    iname=arg[ii]["file"].split("/")[-1]
                    if cfg.useRL:
                        if arg[ii]["flip"]:
                            iname=iname+".flip"
                    if (iname==l["idim"]): #same image
                        if (arg[ii]["idbb"]==l["idbb"]): #same bbox
                            print "Keep old detection"                        
                            pold+=1
                            found=True
            if localshow:
                im=util.myimread(arg[ii]["file"],arg[ii]["flip"])
                rescale,y1,x1,y2,x2=res[3]
                if res[0]!=[]:
                    if found:
                        text="Already detected example"
                    else:
                        text="Added a new example"
                else:
                    if found:
                        text="Keep old detection"
                    else:
                        text="No detection"
                cbb=arg[ii]["bbox"]
                if arg[ii]["flip"]:
                    cbb = (util.flipBBox(im,[cbb])[0])
                cbb=numpy.array(cbb)[:4].astype(numpy.int)
                cbb[0]=(cbb[0]-y1)*rescale
                cbb[1]=(cbb[1]-x1)*rescale
                cbb[2]=(cbb[2]-y1)*rescale
                cbb[3]=(cbb[3]-x1)*rescale
                im=extra.myzoom(im[y1:y2,x1:x2],(rescale,rescale,1),1)
                pylab.figure(300)
                pylab.clf()
                iname=arg[ii]["file"].split("/")[-1]
                if res[0]!=[]:
                    detectCRF.visualize3D([res[0]],cfg.N,im,cbb,iname+"\n"+text,line=True,nograph=True)
                else:
                    detectCRF.visualize3D([],cfg.N,im,cbb,iname+"\n"+text,nograph=True)
                if cfg.useFacial:
                    from extra import locatePoints,locatePointsInter
                    gtfp=arg[ii]["facial"]
                    if arg[ii]["flip"]:
                        gtfp[0::2]=(cbb[3]-cbb[1])-gtfp[::2]
                    pylab.plot(cbb[1]+gtfp[0::2],cbb[0]+gtfp[1::2],"or", markersize=11)
                    if res[0]!=[]:
                        anchor=models[res[0]["id"]]["facial"]
                        if res[0]["id"]==1:
                            inv=[14,15, 12,13, 6,7, 4,5, 8,9, 10,11, 2,3, 0,1, 18,19, 16,17]
                            anchor=anchor[inv]
                        #grid=[]
                        #for lx in range(1,models[0]["ww"][0].shape[1]*cfg.N/(cfg.N+2*cfg.E)-1):
                        #    for ly in range(1,models[0]["ww"][0].shape[0]*cfg.N/(cfg.N+2*cfg.E)-1):
                        #        grid+=[ly,lx]
                        #efp=numpy.array(locatePointsInter(res[:1],cfg.N,numpy.array(grid))[0])        
                        #pylab.plot(efp[1::2],efp[0:-1:2],"xb",markersize=5)
                        #pylab.draw()
                        #pylab.show()
                        #raw_input()
                        efp=numpy.array(locatePoints(res[:1],cfg.N,anchor)[0])
                        pylab.plot(efp[1::2],efp[0:-1:2],"ob",markersize=7)
                        #auxan=anchor.copy()
                        #auxan[::2]=auxan[::2]+1
                        #efp=numpy.array(locatePoints(res[:1],cfg.N,auxan)[0])
                        #pylab.plot(efp[1::2],efp[0:-1:2],"og",markersize=7)
                        #auxan=anchor.copy()
                        #auxan[1::2]=auxan[1::2]+1                                               
                        #efp=numpy.array(locatePoints(res[:1],cfg.N,auxan)[0])
                        #pylab.plot(efp[1::2],efp[0:-1:2],"om",markersize=7)
                        #auxan=anchor.copy()
                        #auxan=auxan+1
                        #efp=numpy.array(locatePoints(res[:1],cfg.N,auxan)[0])
                        #pylab.plot(efp[1::2],efp[0:-1:2],"oc",markersize=7)
                    pylab.draw()
                    pylab.show()
                    #raw_input()

                #raw_input()
        print "Added examples",padd
        print "Improved examples",pbetter
        print "Old examples score",pworst
        print "Old examples bbox",pold
        total.append(padd+pbetter+pworst+pold)
        print "Total",total,"/",len(arg)
        lg.info("############## End Scan of Positives BBoxs ###############")
        lg.info("""Added examples %d
        Improved examples %d
        Old examples score %d
        Old examples bbox %d
        Total %d/%d
        """%(padd,pbetter,pworst,pold,total[-1],len(arg)))
        #be sure that total is counted correctly
        assert(total[-1]==len(lpdet))
    else:
        loadedchk=False
        total.append(len(lpdet))
        skipos=True
    if cfg.statrain:
        #save on a file and evaluate with annotations
        detVOC=[]
        for l in lpdet:
            detVOC.append([l["idim"].split("/")[-1].split(".")[0],l["scr"],l["bbox"][1],l["bbox"][0],l["bbox"][3],l["bbox"][2]])

        #plot AP
        tp,fp,scr,tot=VOCpr.VOCprRecord(trPosImages,detVOC,show=False,ovr=0.5)
        pylab.figure(15,figsize=(4,4))
        pylab.clf()
        rc,pr,ap=VOCpr.drawPrfast(tp,fp,tot)
        pylab.draw()
        pylab.show()
        print "AP=",ap
        #save in different formats
        util.savedetVOC(detVOC,testname+"_trpos.txt")
        util.save(testname+"_trpos.det",{"det":lpdet[:500]})#takes a lot of space use only first 500
        util.savemat(testname+"_trpos.mat",{"tp":tp,"fp":fp,"scr":scr,"tot":tot,"rc":rc,"pr":pr,"ap":ap})
        pylab.savefig(testname+"_trpos.png")

    
    if it>cpit:
        oldposl,negl,reg,nobj,hpos,hneg=pegasos.objective(trpos,trneg,trposcl,trnegcl,clsize,w,cfg.svmc,sizereg=sizereg,valreg=cfg.valreg)              

    #build training data for positives
    trpos=[]
    trposcl=[]
    lg.info("Building Training data from positive detections")
    for idl,l in enumerate(lpdet):#enumerate(lndet):
        efeat=lpfeat[idl]#.flatten()
        eedge=lpedge[idl]#.flatten()
        if lpdet[idl]["id"]>=cfg.numcl:#flipped version
            efeat=pyrHOG2.hogflip(efeat)
            #eedge=pyrHOG2.crfflip(eedge)
        trpos.append(numpy.concatenate((model.feat2flatten(efeat),eedge.flatten()*cfg.k,[bias])))
        trposcl.append(l["id"]%cfg.numcl)
        dscr=numpy.sum(trpos[-1]*w[cumsize[trposcl[-1]]:cumsize[trposcl[-1]+1]])#-models[0]["rho"]
        #print "Error:",abs(dscr-l["scr"])
        if (abs(dscr-l["scr"])/dscr>0.0002):
            print "Error in checking the score function"
            print "Feature score",dscr,"CRF score",l["scr"]
            lg.info("Error in checking the score function")
            lg.info("Feature score %f CRF score %f"%(dscr,l["scr"]))
            raw_input()

    ########### check positive convergence    
    if it>cpit:
        lg.info("################# Checking Positive Convergence ##############")
        newposl,negl,reg,nobj,hpos,hneg=pegasos.objective(trpos,trneg,trposcl,trnegcl,clsize,w,cfg.svmc,sizereg=sizereg,valreg=cfg.valreg)
        #lposl.append(newposl)
        #add a bound on not found examples
        boldposl=oldposl/float(totPosEx)+(totPosEx-total[-2])*(1-cfg.posthr)
        bnewposl=newposl/float(totPosEx)+(totPosEx-total[-1])*(1-cfg.posthr)
        if cfg.posconvbound:
            posratio.append((boldposl-bnewposl)/boldposl)
        else:
            posratio.append((boldposl-bnewposl)/(newposl/float(totPosEx)))#divide without bound to be more strict!
        print "Old pos loss:",oldposl,boldposl
        print "New pos loss:",newposl,bnewposl
        print "Ratio Pos loss",posratio
        lg.info("Old pos loss:%f Bounded:%f"%(oldposl,boldposl))
        lg.info("New pos loss:%f Bounded:%f"%(newposl,bnewposl))
        lg.info("Ratio Pos loss:%f"%posratio[-1])
        if bnewposl>boldposl:
            print "Warning increasing positive loss\n"
            lg.error("Warning increasing positive loss")
            raw_input()
        if (posratio[-1]<cfg.convPos):
            lg.info("Very small positive improvement: convergence at iteration %d!"%it)
            print "Very small positive improvement: convergence at iteration %d!"%it
            last_round=True 
            #trNegImages=trNegImagesFull
            #tsImages=tsImagesFull

    if it==cfg.posit-1 or last_round:#even not converging compute the full dataset
        last_round=True        
        trNegImages=trNegImagesFull

    #save positives
    if cfg.checkpoint:
        lg.info("Begin Positive check point it:%d (%d positive examples)"%(it,len(lpdet)))
        try:
            os.remove(localsave+".pos.valid")
        except:
            pass
        util.save(localsave+".pos.chk",{"lpdet":lpdet,"lpedge":lpedge,'lpfeat':lpfeat,"cpit":it,"last_round":last_round})
        open(localsave+".pos.valid","w").close()
        lg.info("End Positive check point")

 
    ########### repeat scan negatives
    lastcount=0
    negratio=[]
    negratio2=[]
    for nit in range(cfg.negit):
        
        lg.info("############### Negative Scan iteration %d ##############"%nit)
        ########### from detections build training data
        trneg=[]
        trnegcl=[]
        lg.info("Building Training data from negative detections")
        for idl,l in enumerate(lndet):
            efeat=lnfeat[idl]#.flatten()
            eedge=lnedge[idl]#.flatten()
            if lndet[idl]["id"]>=cfg.numcl:#flipped version
                efeat=pyrHOG2.hogflip(efeat)
            #    eedge=pyrHOG2.crfflip(eedge)
            #trneg.append(numpy.concatenate((efeat.flatten(),cfg.k*eedge.flatten(),[bias])))
            trneg.append(numpy.concatenate((model.feat2flatten(efeat),eedge.flatten()*cfg.k,[bias])))
            trnegcl.append(l["id"]%cfg.numcl)
            dscr=numpy.sum(trneg[-1]*w[cumsize[trnegcl[-1]]:cumsize[trnegcl[-1]+1]])
            #trnegcl.append(lndet[idl]["id"]%cfg.numcl)
            #dscr=numpy.sum(trneg[-1]*w[cumsize[trnegcl[-1]]:cumsize[trnegcl[-1]+1]])
            #print "Error:",abs(dscr-l["scr"])
            if not(skipos):#do not check if loaded trneg from checkpoint
                if (abs((dscr-l["scr"])/dscr)>0.0002):
                    print "Error in checking the score function"
                    print "Feature score",dscr,"CRF score",l["scr"]
                    lg.info("Error in checking the score function")
                    lg.info("Feature score %f CRF score %f"%(dscr,l["scr"]))
                    #raw_input()

        #if no negative sample add empty negatives
        for l in range(cfg.numcl):
            if numpy.sum(numpy.array(trnegcl)==l)==0:
                trneg.append(numpy.concatenate((numpy.zeros(models[l]["ww"][0].mask.size*len(models[l]["ww"]),dtype=models[l]["ww"][0].mask.dtype),numpy.zeros(13*13,dtype=numpy.float32),[bias])))
                trnegcl.append(l)
                lg.info("No negative samples; add empty negatives")

        ############ check negative convergency
        if nit>0: # and not(limit):
            lg.info("################ Checking Negative Convergence ##############")
            posl,negl,reg,nobj,hpos,hneg=pegasos.objective(trpos,trneg,trposcl,trnegcl,clsize,w,cfg.svmc,sizereg=sizereg,valreg=cfg.valreg)#,notreg)
            print "NIT:",nit,"OLDLOSS",old_nobj,"NEWLOSS:",nobj
            negratio.append(nobj/(old_nobj+0.000001))
            negratio2.append((posl+negl)/(old_posl+old_negl+0.000001))
            print "RATIO: newobj/oldobj:",negratio,negratio2
            lg.info("OldLoss:%f NewLoss:%f"%(old_nobj,nobj))
            lg.info("Ratio %f"%(negratio[-1]))
            lg.info("Ratio without reg %f"%(negratio2[-1]))
            #if (negratio[-1]<1.05):
            if (negratio[-1]<cfg.convNeg) and not(cache_full):
                lg.info("Very small loss increment: negative convergence at iteration %d!"%nit)
                print "Very small loss increment: negative convergence at iteration %d!"%nit
                break

        ############train a new detector with new positive and all negatives
        lg.info("############# Building a new Model ####################")
        #print elements per model
        for l in range(cfg.numcl):
            print "Model",l
            print "Positive Examples:",numpy.sum(numpy.array(trposcl)==l)
            print "Negative Examples:",numpy.sum(numpy.array(trnegcl)==l)
            lg.info("Before training Model %d"%l)
            lg.info("Positive Examples:%d"%(numpy.sum(numpy.array(trposcl)==l)))
            lg.info("Negative Examples:%d"%(numpy.sum(numpy.array(trnegcl)==l)))    

        #import pegasos   
        if cfg.useSGD:
            w,r,prloss=pegasos.trainCompSGD(trpos,trneg,"",trposcl,trnegcl,oldw=w,pc=cfg.svmc,k=numcore*2,numthr=numcore,eps=0.001,sizereg=sizereg,valreg=cfg.valreg,lb=cfg.lb)#,notreg=notreg)
        else:
            w,r,prloss=pegasos.trainCompBFG(trpos,trneg,"",trposcl,trnegcl,oldw=w,pc=cfg.svmc,k=numcore*2,numthr=numcore,eps=0.001,sizereg=sizereg,valreg=cfg.valreg,lb=cfg.lb)#,notreg=notreg)

        pylab.figure(300,figsize=(4,4))
        pylab.clf()
        pylab.plot(w)
        pylab.title("dimensions of W")
        pylab.draw()
        pylab.show()
        #raw_input()

        old_posl,old_negl,old_reg,old_nobj,old_hpos,old_hneg=pegasos.objective(trpos,trneg,trposcl,trnegcl,clsize,w,cfg.svmc,sizereg=sizereg,valreg=cfg.valreg)#,notreg) 
        waux=[]
        rr=[]
        w1=numpy.array([])
        #from w to model m1
        for idm,m in enumerate(models[:cfg.numcl]):
            #models[idm]=model.w2model(w[cumsize[idm]:cumsize[idm+1]-1],cfg.N,cfg.E,-w[cumsize[idm+1]-1]*bias,len(m["ww"]),lenf,m["ww"][0].shape[0],m["ww"][0].shape[1],useCRF=True,k=cfg.k)
            models[idm]=model.w2model3D(models[idm],w[:-1],-w[-1]*bias)
            models[idm]["id"]=idm
            #models[idm]["ra"]=w[cumsize[idm+1]-1]
            #from model to w #changing the clip...
            if cfg.use3D:
                waux.append(model.model2w3D(models[idm]))
            else:
                waux.append(model.model2w(models[idm],False,False,False,useCRF=True,k=cfg.k))
            #rr.append(models[idm]["rho"])
            w1=numpy.concatenate((w1,waux[-1],-numpy.array([models[idm]["rho"]])/bias))
        assert(numpy.sum(numpy.abs(w1-w))<0.0002)
        w2=w
        w=w1

        if cfg.useRL:
            #add flipped models
            for idm in range(cfg.numcl):
                models[cfg.numcl+idm]=(extra.flip(models[idm]))
                models[cfg.numcl+idm]["id"]=idm+cfg.numcl


        util.save("%s%d.model"%(testname,it),models)
        lg.info("Saved model it:%d nit:%d"%(it,nit))

        #visualize models
        atrposcl=numpy.array(trposcl)
        import test3D2
        pylab.figure(400,figsize=(10,10))
        pylab.clf()
        test3D2.showModel(models[0],0,0,nhog=50,bis=False)
        pylab.draw()
        pylab.show()
        pylab.savefig("%s_3D%dq.png"%(testname,it))
        if 0:
            for idm,m in enumerate(models[:cfg.numcl]):   
                import drawHOG
                imm=drawHOG.drawHOG(model.convert2(m["ww"][0],cfg.N,cfg.E))
                pl.figure(100+idm,figsize=(3,3))
                pl.clf()
                pl.imshow(imm)
                pl.title("b:%.3f h:%.4f d:%.4f"%(m["rho"],numpy.sum(m["ww"][0]**2),numpy.sum(m["cost"]**2)))
                pl.xlabel("#%d"%(numpy.sum(atrposcl==idm)))
                lg.info("Model %d Samples:%d bias:%f |hog|:%f |def|:%f"%(idm,numpy.sum(atrposcl==idm),m["rho"],numpy.sum(m["ww"][0]**2),numpy.sum(m["cost"]**2)))
                pl.draw()
                pl.show()
                pylab.savefig("%s_hog%d_cl%d.png"%(testname,it,idm))
                #CRF
                pl.figure(110+idm,figsize=(5,5))
                pl.clf()
                extra.showDef(m["cost"][:4])
                pl.draw()
                pl.show()
                pylab.savefig("%s_def%dl_cl%d.png"%(testname,it,idm))
                lg.info("Deformation Min:%f Max:%f"%(m["cost"].min(),m["cost"].max()))
                pl.figure(120+idm,figsize=(5,5))
                pl.clf()
                extra.showDef(m["cost"][4:])
                pl.draw()
                pl.show()
                pylab.savefig("%s_def%dq_cl%d.png"%(testname,it,idm))

        ########## rescore old negative detections
        lg.info("Rescoring %d Negative detections"%len(lndet))
        for idl,l in enumerate(lndet):
            idm=l["id"]
            ang=l["ang"]
            scr=0
            for idp,p in enumerate(models[idm]["ww"]):
                scr=scr+numpy.sum(p.mask*lnfeat[idl][idp])
            scr+=models[idm]["biases"][ang[0],ang[1]]*cfg.k#numpy.sum(lnedge[idl])
            lndet[idl]["scr"]=scr-models[idm]["rho"]#numpy.sum(models[idm]["ww"][0]*lpfeat[idl])+numpy.sum(models[idm
            #lndet[idl]["scr"]=numpy.sum(models[idm]["ww"][0]*lnfeat[idl])+numpy.sum(models[idm]["cost"]*lnedge[idl])-models[idm]["rho"]#-rr[idm]/bias

        ######### filter negatives
        lg.info("############### Filtering Negative Detections ###########")
        ltosort=[-x["scr"] for x in lndet]
        lord=numpy.argsort(ltosort)
        #remove dense data
        trneg=[]
        trnegcl=[]

        #filter and build negative vectors
        auxdet=[]
        auxfeat=[]
        auxedge=[]
        nsv=numpy.sum(-numpy.array(ltosort)>-1)
        limit=max(cfg.maxexamples/2,nsv) #at least half of the cache
        if (nsv>cfg.maxexamples):
            lg.error("Negative SVs(%d) don't fit in cache %d"%(nsv,cfg.maxexamples))
            print "Warning SVs don't fit in cache"
            raw_input()
        #limit=min(cfg.maxexamples,limit) #as maximum full cache
        for idl in lord[:limit]:#to maintain space for new samples
            auxdet.append(lndet[idl])
            auxfeat.append(lnfeat[idl])
            auxedge.append(lnedge[idl])
            #efeat=lnfeat[idl]#.flatten()
            #eedge=lnedge[idl]#.flatten()
            #if lndet[idl]["id"]>=cfg.numcl:#flipped version
            #    efeat=pyrHOG2.hogflip(efeat)
            #    eedge=pyrHOG2.crfflip(eedge)
            #trneg.append(numpy.concatenate((efeat.flatten(),eedge.flatten())))
            #trnegcl.append(lndet[idl]["id"]%cfg.numcl)
            
        lndet=auxdet
        lnfeat=auxfeat
        lnedge=auxedge

        print "Negative Samples before filtering:",len(ltosort)
        #print "New Extracted Negatives",len(lndetnew)
        print "Negative Support Vectors:",nsv
        print "Negative Cache Vectors:",len(lndet)
        print "Maximum cache vectors:",cfg.maxexamples
        lg.info("""Negative samples before filtering:%d
Negative Support Vectors %d
Negative in cache vectors %d
        """%(len(ltosort),nsv,len(lndet)))
        #if len(lndetnew)+numpy.sum(-numpy.array(ltosort)>-1)>cfg.maxexamples:
        #    print "Warning support vectors do not fit in cache!!!!"
        #    raw_input()


        ########### scan negatives
        #if last_round:
        #    trNegImages=trNegImagesFull
        from multiprocessing import Manager
        d["cache_full"]=False
        cache_full=False
        lndetnew=[];lnfeatnew=[];lnedgenew=[]
        arg=[]
        #for idl,l in enumerate(trNegImages):
        totn=len(trNegImages)
        for idl1 in range(totn):
            idl=(idl1+lastcount)%totn
            l=trNegImages[idl]
            #bb=l["bbox"]
            #for idb,b in enumerate(bb):
            arg.append({"idim":idl,"file":l["name"],"idbb":0,"bbox":[],"models":models,"cfg":cfg,"flip":False,"control":d}) 
        lg.info("############### Starting Scan of %d negative images #############"%len(arg))
        if not(parallel):
            itr=itertools.imap(hardNegCache,arg)        
        else:
            itr=mypool.imap(hardNegCache,arg)

        for ii,res in enumerate(itr):
            print "Total negatives:",len(lndetnew)
            if localshow and res[0]!=[]:
                im=myimread(arg[ii]["file"])
                detectCRF.visualize3D(res[0][:5],cfg.N,im)
            lndetnew+=res[0]
            lnfeatnew+=res[1]
            lnedgenew+=res[2]
            if len(lndetnew)+len(lndet)>cfg.maxexamples and not(cache_full):
                #if not cache_full:
                lastcount=arg[ii]["idim"]
                print "Examples exceeding the cache limit at image %d!"%lastcount
                print "So far I have done %d/%d!"%(ii,len(arg))
                lg.info("Examples exceeding the cache limit at image %d!"%lastcount)
                lg.info("So far I have done %d/%d!"%(ii,len(arg)))
                #raw_input()
                #mypool.terminate()
                #mypool.join()
                cache_full=True
                d["cache_full"]=True
        if cache_full:
            lg.info("Cache is full!!!")
        lg.info("############### End Scan negatives #############")
        lg.info("Found %d hard negatives"%len(lndetnew))
        ########### scan negatives in positives
        
        if cfg.neginpos:
            arg=[]
            for idl,l in enumerate(trPosImages[:len(trNegImages)/2]):#only first 100
                #bb=l["bbox"]
                #for idb,b in enumerate(bb):
                arg.append({"idim":idl,"file":l["name"],"idbb":0,"bbox":l["bbox"],"models":models,"cfg":cfg,"flip":False,"control":d})    

            lg.info("############### Starting Scan negatives in %d positves images #############"%len(arg))
            #lndetnew=[];lnfeatnew=[];lnedgenew=[]
            if not(parallel):
                itr=itertools.imap(detectCRF.hardNegPos,arg)        
            else:
                itr=mypool.imap(hardNegPosCache,arg)

            for ii,res in enumerate(itr):
                print "Total Negatives:",len(lndetnew)
                if localshow and res[0]!=[]:
                    im=myimread(arg[ii]["file"])
                    detectCRF.visualize2(res[0][:5],cfg.N,im)
                lndetnew+=res[0]
                lnfeatnew+=res[1]
                lnedgenew+=res[2]
                if len(lndetnew)+len(lndet)>cfg.maxexamples:
                    print "Examples exeding the cache limit!"
                    #raw_input()
                    #mypool.terminate()
                    #mypool.join()
                    cache_full=True
                    d["cache_full"]=True
            if cache_full:
                lg.info("Cache is full!!!")    
            lg.info("############### End Scan neg in positives #############")
            lg.info("Found %d hard negatives"%len(lndetnew))

        ########### include new detections in the old pool discarding doubles
        #auxdet=[]
        #auxfeat=[]
        #aux=[]
        lg.info("############# Insert new detections in the pool #################")
        oldpool=len(lndet)
        lg.info("Old pool size:%d"%len(lndet))
        imid=numpy.array([x["idim"] for x in lndet])
        for newid,newdet in enumerate(lndetnew): # for each newdet
            #newdet=ldetnew[newid]
            remove=False
            #for oldid,olddet in enumerate(lndet): # check with the old
            for oldid in numpy.where(imid==newdet["idim"])[0]:
                olddet=lndet[oldid]
                if (newdet["idim"]==olddet["idim"]): #same image
                    if (newdet["scl"]==olddet["scl"]): #same scale
                        if (newdet["id"]==olddet["id"]): #same model
                            if numpy.all(newdet["ang"]==olddet["ang"]): #same orientation
                                if numpy.all(newdet["fpos"]==olddet["fpos"]):
                                    #same features
                                    print "diff:",abs(newdet["scr"]-olddet["scr"]),
                                    assert(abs(newdet["scr"]-olddet["scr"])<0.0005)
                                    scr=0
                                    for idp,p in enumerate(lnfeatnew[newid]):
                                        scr+=numpy.sum(numpy.abs(p-lnfeat[oldid][idp]))
                                    assert(scr<0.0005)
                                    #assert(numpy.all(lnedgenew[newid]==lnedge[oldid]))
                                    print "Detection",newdet["idim"],newdet["scr"],newdet["scl"],newdet["id"],"is double --> removed!"
                                    remove=True
            if not(remove):
                lndet.append(lndetnew[newid])
                lnfeat.append(lnfeatnew[newid])
                lnedge.append(lnedgenew[newid])
        lg.info("New pool size:%d"%(len(lndet)))
        lg.info("Dobles removed:%d"%(oldpool+len(lndetnew)-len(lndet)))
        #save negatives
        if cfg.checkpoint:
            lg.info("Begin checkpoint Negative iteration %d (%d negative examples)"%(nit,len(lndet)))
            try:
                os.remove(localsave+".neg.valid")
            except:
                pass
            util.save(localsave+".neg.chk",{"lndet":lndet,"lnedge":lnedge,'lnfeat':lnfeat,"cnit":nit})
            open(localsave+".neg.valid","w").close()
            #touch a file to be sure you have finished
            lg.info("End saving negative detections")
        #raw_input()
                
    #mypool.close()
    #mypool.join()
    ##############test
    #import denseCRFtest
    #denseCRFtest.runtest(models,tsImages,cfg,parallel=True,numcore=numcore,save="%s%d"%(testname,it),detfun=lambda x :detectCRF.test(x,numhyp=1,show=False),show=localshow)

    #compute thresholds positives
    lg.info("Computing positive thresholds")
    for m in models:
        m["thr"]=0
    for idl,l in enumerate(lpdet):
        idm=l["id"]
        scr=0
        for idp,p in enumerate(models[idm]["ww"]):
            scr=scr+numpy.sum(p.mask*lpfeat[idl][idp])
        lpdet[idl]["scr"]=scr-models[idm]["rho"]
        #lpdet[idl]["scr"]=numpy.sum(models[idm]["ww"][0]*lpfeat[idl])+numpy.sum(models[idm]["cost"]*lpedge[idl])-models[idm]["rho"]#-rr[idm]/bias
        mid=lpdet[idl]["id"]%cfg.numcl
        if lpdet[idl]["scr"]<models[mid]["thr"]:
            models[mid]["thr"]=lpdet[idl]["scr"]
            #models[mid+cfg.numcl]["thr"]=lpdet[idl]["scr"]
    #lg.info("Minimum thresholds for positives:",)
    for idm,m in enumerate(models):
        print "Minimum thresholds",m["thr"]
        lg.info("Model %d:%f"%(idm,m["thr"]))

    lg.info("############# Run test on %d positive examples #################"%len(tsImages))
    ap=denseCRFtest.runtest(models,tsImages,cfg,parallel=parallel,numcore=numcore,save="%s%d"%(testname,it),show=localshow,pool=mypool,detfun=denseCRFtest.testINC)
    lg.info("Ap is:%f"%ap)
    if last_round:
        break

lg.info("############# Run test on all (%d) examples #################"%len(tsImagesFull))
util.save("%s_final.model"%(testname),models)
ap=denseCRFtest.runtest(models,tsImagesFull,cfg,parallel=parallel,numcore=numcore,save="%s_final"%(testname),show=localshow,pool=mypool,detfun=denseCRFtest.testINC)
lg.info("Ap is:%f"%ap)
print "Training Finished!!!"
lg.info("End of the training!!!!")
#delete cache files if there
try:
    os.remove(localsave+".pos.valid")
    os.remove(localsave+".neg.valid")
    os.remove(localsave+".pos.chk")
    os.remove(localsave+".neg.chk")
except:
    pass

