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
import detectCRF

def runtest(models,tsImages,cfg,parallel=True,numcore=4,detfun=detectCRF.test,save=False,show=False,pool=None,returndet=False):

    #parallel=True
    #cfg.show=not(parallel)
    #numcore=4
    #mycfg=
    if parallel:
        if pool!=None:
            mypool=pool #use already created pool
        else:
            mypool = Pool(numcore)
    arg=[]

    for idl,l in enumerate(tsImages):
        #bb=l["bbox"]
        #for idb,b in enumerate(bb):
        arg.append({"idim":idl,"file":l["name"],"idbb":0,"bbox":[],"models":models,"cfg":cfg,"flip":False})    

    print "----------Test-----------"
    ltdet=[];
    if not(parallel):
        #itr=itertools.imap(detectCRF.test,arg)        
        #itr=itertools.imap(lambda x:detectCRF.test(x,numhyp=1),arg) #this can also be used       
        itr=itertools.imap(detfun,arg)
    else:
        #itr=mypool.map(detectCRF.test,arg)
        itr=mypool.imap(detfun,arg) #for parallle lambda does not work

    for ii,res in enumerate(itr):
        if show:
            im=myimread(arg[ii]["file"],resize=cfg.resize)
            if tsImages[ii]["bbox"]!=[]:
                if 0:#cfg.db=="AFLW":#reduce bounding box for training with AFLW
                    for l in res:
                        dd=(l["ang"][1]-6)/6.0
                        w=l["bbox"][3]-l["bbox"][1]
                        #print l["bbox"]
                        newb=(l["bbox"][0],l["bbox"][1]+0.5*w*max(dd,0),l["bbox"][2],l["bbox"][3]-0.5*w*max(-dd,0))
                        l["bbox"]=newb
                        #print l["bbox"]
                        #raw_input()
                #detectCRF.visualize2(res[:3],cfg.N,im,bb=tsImages[ii]["bbox"][0])
                detectCRF.visualize3D(models,res[:1],cfg.N,im,bb=tsImages[ii]["bbox"][0],npart=cfg.npart,cangy=cfg.cangy,cangx=cfg.cangx,cangz=cfg.cangz)
            else:
                detectCRF.visualize3D(models,res[:1],cfg.N,im,cangy=cfg.cangy,cangx=cfg.cangx,cangz=cfg.cangz)
            print [x["scr"] for x in res[:5]]
            #lfeat,biases=getfeature3D(det,f,model,angy,angx,angz,k,trunc=0,usebiases=False,usedef=False):
            #raw_input()
        ltdet+=res

    if parallel:
        if pool==None:
            mypool.close() 
            mypool.join() 

    #sort detections
    ltosort=[-x["scr"] for x in ltdet]
    lord=numpy.argsort(ltosort)
    aux=[]
    for l in lord:
        aux.append(ltdet[l])
    ltdet=aux

    #save on a file and evaluate with annotations
    detVOC=[]
    for l in ltdet:
        #detVOC.append([l["idim"].split("/")[-1].split(".")[0],l["scr"],l["bbox"][1],l["bbox"][0],l["bbox"][3],l["bbox"][2],(l["ang"][1]-12)*15])
        detVOC.append([l["idim"].split("/")[-1].split(".")[0],l["scr"],l["bbox"][1]/cfg.resize,l["bbox"][0]/cfg.resize,l["bbox"][3]/cfg.resize,l["bbox"][2]/cfg.resize,(l["ang"][1]-len(cfg.cangx)/2)*15])

    #plot AP
    tp,fp,scr,tot=VOCpr.VOCprRecord(tsImages,detVOC,show=False,ovr=0.5)
    pylab.figure(15,figsize=(4,4))
    pylab.clf()
    rc,pr,ap=VOCpr.drawPrfast(tp,fp,tot)
    pylab.draw()
    pylab.show()
    print "AP=",ap
    if type(save)==str:
        testname=save
        pylab.savefig(testname+".png")
    if not(returndet):
        #ptp,pfp,pscr,ptot=VOCpr.VOCprPose(tsImages,detVOC,show=False,ovr=0.5,posethr=22.5)
        #ptp,pfp,pscr,ptot=VOCpr.VOCprPose(tsImages,detVOC,show=False,ovr=0.5,posethr=16)
        ptp,pfp,pscr,ptot=VOCpr.VOCprPose(tsImages,detVOC,show=False,ovr=0.5,posethr=16)
        pylab.figure(16,figsize=(4,4))
        pylab.clf()
        prc,ppr,pap=VOCpr.drawPrfast(ptp,pfp,ptot)
        pylab.draw()
        pylab.show()
        print "AP=",ap
        print "Total",ptot,tot
        print "PEAP=",pap
        print "% at 15",sum(ptp)/float(sum(tp))
        print "Right % at 15",sum(ptp)/float(ptot)
        #sdfsd
    #save in different formats
    if type(save)==str:
        testname=save
        util.savedetVOC(detVOC,testname+".txt")
        util.save(testname+".det",{"det":ltdet[:1500]})#takes a lot of space use only first 500
        util.savemat(testname+".mat",{"tp":tp,"fp":fp,"scr":scr,"tot":tot,"rc":rc,"pr":pr,"ap":ap})
        pylab.savefig(testname+"_pos.png")
    if returndet:
        return ap,ltdet
    return ap


#use a different number of hypotheses
def test(x):
    return detectCRF.test(x,show=False,inclusion=False,onlybest=False) #in bicycles is 

def testINC(x):
    return detectCRF.test(x,show=False,inclusion=True,onlybest=True) #in bicycles is better and faster with 1 hypotheses

def testINC03(x):
    return detectCRF.test(x,show=False,inclusion=True,onlybest=True,ovr=0.3) #in bicycles is better and 

########################## load configuration parametes
if __name__ == '__main__':

    if 1: #use the configuration file
        print "Loading defautl configuration config.py"
        from config import * #default configuration      

        if len(sys.argv)>2: #specific configuration
            print "Loading configuration from %s"%sys.argv[2]
            import_name=sys.argv[2]
            exec "from config_%s import *"%import_name
            
        cfg.cls=sys.argv[1]
        #cfg.useRL=False#for the moment
        #cfg.show=False
        #cfg.auxdir=""
        #cfg.numhyp=5
        #cfg.rescale=True
        #cfg.numneg= 10
        #bias=100
        #cfg.bias=bias
        #just for a fast test
        #cfg.maxpos = 50
        #cfg.maxneg = 20
        #cfg.maxexamples = 10000
    else: #or set here the parameters
        print "Loading defautl configuration config.py"
        from config import * #default configuration      
    cfg.cls=sys.argv[1]
    cfg.numcl=2
    #cfg.dbpath="/home/owner/databases/"
    cfg.dbpath="/users/visics/mpederso/databases/"
    cfg.testpath="./data/test3/"#"./data/CRF/12_09_19/"
    cfg.testspec="3Dortogonal6"#"full2"
    cfg.db="AFW"#"3DVOC"#"AFW"#"MultiPIE2"#"images"#"AFW"#"MultiPIE2"#"VOC"
    cfg.maxtest=10000#before it was a 2000...
    cfg.maxneg=200
    cfg.use3D=True
    cfg.nobbox=False
    #cfg.db="imagenet"
    #cfg.cls="tandem"
    #cfg.N=
        

    testname=cfg.testpath+cfg.cls+("%d"%cfg.numcl)+"_"+cfg.testspec
    ########################load training and test samples
    if cfg.db=="VOC":
        if cfg.year=="2007":
            #test
            tsPosImages=getRecord(VOC07Data(select="pos",cl="%s_test.txt"%cfg.cls,
                            basepath=cfg.dbpath,#"/home/databases/",#"/share/ISE/marcopede/database/",
                            usetr=True,usedf=False),cfg.maxtest)
            tsNegImages=getRecord(VOC07Data(select="neg",cl="%s_test.txt"%cfg.cls,
                            basepath=cfg.dbpath,#"/home/databases/",#"/share/ISE/marcopede/database/",
                            usetr=True,usedf=False),cfg.maxneg)
            #tsImages=numpy.concatenate((tsPosImages,tsNegImages),0)
            tsImages=numpy.concatenate((tsPosImages,tsNegImages),0)
            tsImagesFull=getRecord(VOC07Data(select="all",cl="%s_test.txt"%cfg.cls,
                            basepath=cfg.dbpath,
                            usetr=True,usedf=False),10000)
    elif cfg.db=="buffy":
        trPosImages=getRecord(Buffy(select="all",cl="trainval.txt",
                        basepath=cfg.dbpath,
                        trainfile="buffy/",
                        imagepath="buffy/images/",
                        annpath="buffy/",
                        usetr=True,usedf=False),cfg.maxpos)
        trPosImagesNoTrunc=trPosImages
        trNegImages=getRecord(DirImages(imagepath=cfg.dbpath+"INRIAPerson/train_64x128_H96/neg/"),cfg.maxneg)
        trNegImagesFull=trNegImages
        #test
        tsPosImages=getRecord(Buffy(select="all",cl="test.txt",
                        basepath=cfg.dbpath,
                        trainfile="buffy/",
                        imagepath="buffy/images/",
                        annpath="buffy/",
                        usetr=True,usedf=False),cfg.maxtest)
        tsImages=tsPosImages#numpy.concatenate((tsPosImages,tsNegImages),0)
        tsImagesFull=tsPosImages
    elif cfg.db=="inria":
        trPosImages=getRecord(InriaPosData(basepath=cfg.dbpath),cfg.maxpos)
        trPosImagesNoTrunc=trPosImages
        trNegImages=getRecord(InriaNegData(basepath=cfg.dbpath),cfg.maxneg)#check if it works better like this
        trNegImagesFull=getRecord(InriaNegData(basepath=cfg.dbpath),cfg.maxnegfull)
        #test
        tsImages=getRecord(InriaTestFullData(basepath=cfg.dbpath),cfg.maxtest)
        tsImagesFull=tsImages
    elif cfg.db=="imagenet":
        tsPosImages=getRecord(imageNet(select="all",cl="%s_test.txt"%cfg.cls,
                        basepath=cfg.dbpath,
                        trainfile="/tandem/",
                        imagepath="/tandem/images/",
                        annpath="/tandem/Annotation/n02835271/",
                        usetr=True,usedf=False),cfg.maxtest)
        tsImages=tsPosImages#numpy.concatenate((tsPosImages,tsNegImages),0)
        tsImagesFull=tsPosImages
    elif cfg.db=="MultiPIE2":
        #cameras=["11_0","12_0","09_0","08_0","13_0","14_0","05_1","05_0","04_1","19_0","20_0","01_0","24_0"]
        cameras=["110","120","090","080","130","140","051","050","041","190","200","010","240"]
        #cameras=["080","130","140","051","050","041","190"]
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
                conditions=150#300
            else:
                conditions=50#50
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
    elif cfg.db=="AFLW":
        trPosImages=getRecord(AFLW(basepath=cfg.dbpath,fold=0),cfg.maxpos,facial=True,pose=True)#cfg.useFacial)
        trPosImagesNoTrunc=trPosImages[:900]
        trNegImages=getRecord(InriaNegData(basepath=cfg.dbpath),cfg.maxneg)#check if it works better like this
        trNegImagesFull=getRecord(InriaNegData(basepath=cfg.dbpath),cfg.maxnegfull)
        #test
        tsImages=getRecord(AFLW(basepath=cfg.dbpath,fold=0),cfg.maxtest,facial=True,pose=True)#cfg.useFacial)
        tsImagesFull=tsImages
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
    elif cfg.db=="AFW":
        tsImages=getRecord(AFW(basepath=cfg.dbpath),cfg.maxpos,facial=True,pose=True)
        tsImagesFull=tsImages
    elif cfg.db=="epfl":
        aux=getRecord(epfl(select="pos",cl="01",basepath=cfg.dbpath),cfg.maxpos,pose=True)
        trPosImages=numpy.array([],dtype=aux.dtype)
        numtrcars=10
        numtscars=10
        trcars=range(1,10)
        tscars=range(11,20)
        for car in trcars[:numtrcars]:
            trPosImages=numpy.concatenate((trPosImages,getRecord(epfl(select="pos",cl="%02d"%car,
                            basepath=cfg.dbpath,#"/home/databases/",
                            usetr=True,usedf=False,initimg=0,double=0),10000,pose=True)))
        trPosImagesInit=trPosImages
        trPosImagesNoTrunc=trPosImages
        trNegImages=getRecord(VOC07Data(select="neg",cl="%s_trainval.txt"%"car",
                            basepath=cfg.dbpath,#"/home/databases/",#"/share/ISE/marcopede/database/",
                            usetr=True,usedf=False),cfg.maxneg)
        trNegImagesFull=getRecord(VOC07Data(select="neg",cl="%s_trainval.txt"%"car",
                            basepath=cfg.dbpath,usetr=True,usedf=False),cfg.maxnegfull)
        tsPosImages=numpy.array([],dtype=aux.dtype)
        for car in tscars[:numtscars]:
            tsPosImages=numpy.concatenate((tsPosImages,getRecord(epfl(select="pos",cl="%02d"%car,
                            basepath=cfg.dbpath,#"/home/databases/",
                            usetr=True,usedf=False,initimg=0,double=0),10000,pose=True)))#[:20]
        tsImages=tsPosImages
        tsImagesFull=tsImages

    elif cfg.db=="3DVOC":
        trPosImages=getRecord(VOC3D(select="pos",cl="%s_train.txt"%cfg.cls,
                            basepath=cfg.dbpath,#"/home/databases/",
                            usetr=True,usedf=False),cfg.maxpos,pose=True)
        trPosImagesNoTrunc=getRecord(VOC3D(select="pos",cl="%s_train.txt"%cfg.cls,
                        basepath=cfg.dbpath,#"/home/databases/",
                        usetr=False,usedf=False),cfg.maxpos,pose=True)
        trNegImages=getRecord(VOC3D(select="neg",cl="%s_train.txt"%cfg.cls,
                        basepath=cfg.dbpath,#"/home/databases/",#"/share/ISE/marcopede/database/",
                        usetr=True,usedf=False),cfg.maxneg,pose=True)
        trNegImagesFull=getRecord(VOC3D(select="neg",cl="%s_train.txt"%cfg.cls,
                        basepath=cfg.dbpath,usetr=True,usedf=False),cfg.maxnegfull,pose=True)
        #test
        tsPosImages=getRecord(VOC3D(select="pos",cl="%s_val.txt"%cfg.cls,
                        basepath=cfg.dbpath,#"/home/databases/",#"/share/ISE/marcopede/database/",
                        usetr=True,usedf=False),cfg.maxtest,pose=True)
        tsNegImages=getRecord(VOC3D(select="neg",cl="%s_val.txt"%cfg.cls,
                        basepath=cfg.dbpath,#"/home/databases/",#"/share/ISE/marcopede/database/",
                        usetr=True,usedf=False),cfg.maxneg,pose=True)
        tsImages=numpy.concatenate((tsPosImages,tsNegImages),0)
        tsImagesFull=getRecord(VOC3D(select="all",cl="%s_val.txt"%cfg.cls,
                        basepath=cfg.dbpath,
                        usetr=True,usedf=False),10000,pose=True)
    elif cfg.db=="images":
        #tsImages=getRecord(DirImages(imagepath="/users/visics/mpederso/code/face-release1.0-basic/images/",ext="jpg"))
        #tsImages=getRecord(DirImages(imagepath="/esat/unuk/mpederso/images/",ext="jpg"))
        #tsImages=getRecord(DirImages(imagepath="/users/visics/mpederso/no_backup/buffy/images/buffy_s5e2/",ext="jpg"))et
        tsImages=getRecord(DirImages(imagepath="/users/visics/mpederso/code/git/3Def/3Det/",ext="jpg"))
        #tsImages=getRecord(DirImages(imagepath="/users/visics/mpederso/dwhelper/",ext="jpg"))
        tsImagesFull=tsImages


    ##############load model
    for l in range(cfg.posit):
        try:
            models=util.load("%s%d.model"%(testname,l))
            print "Loading Model %d"%l
        except:
            break
    #it=l-1
    #models=util.load("%s%d.model"%(testname,it))
    ######to comment down
    #it=6;testname="./data/person3_right"
    #it=12;testname="./data/CRF/12_09_23/bicycle3_fixed"
    #it=2;testname="./data/bicycle2_test"

    if 0: #standard configuration
        cfg.usebbTEST=False
        cfg.numhypTEST=1
        cfg.aiterTEST=3
        cfg.restartTEST=0
        cfg.intervTEST=10

    if 0: #used during training
        cfg.usebbTEST=True
        cfg.numhypTEST=50
        cfg.aiterTEST=1
        cfg.restartTEST=0
        cfg.intervTEST=5

    if 1: #personalized
        cfg.usebbTEST=True
        cfg.numhypTEST=50
        cfg.aiterTEST=3
        cfg.restartTEST=0
        cfg.intervTEST=5

    #cfg.numcl=2
    cfg.N=4
    cfg.useclip=True
    cfg.useFastDP=True
    #angles available to reduce memory
    #cfg.cangy=[-30,-15,0,15,30]#[-30,0,+30]
    #cfg.cangx=[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90]
    #cfg.cangz=[-20,-10,0,10,20]#[-10,0,10]
    #cfg.cangy=[-15,0,15]#[-15,0,15]#[0,5]#[-30,-15,0,15,30]#[-30,0,+30]
    #cfg.cangx=[-180,-165,-150,-135,-120,-105,-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90,105,120,135,150,165,180]
    #cfg.cangx=[-180,-135,-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90,135]#[-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90]#for faces
    #cfg.cangx=[-90,-80,-70,-60,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90]#for faces
    #cfg.cangz=[-5,0,+5]
    #selected
    cfg.angx=range(len(cfg.cangx))#[1,3,5,6,7,9,11]
    cfg.angy=range(len(cfg.cangy))#[4,6,8]
    cfg.angz=range(len(cfg.cangz))#[1,2,3]

    cfg.resize=1.0#2.0
    #testname="./data/CRF/12_10_02_parts_full/bicycle2_testN2_final"
    #testname="./data/person1_testN2best0"#inria1_inria3"bicycle2_testN4aiter3_final
    #testname="./data/bicycle2_testN4aiter3_final"
    #testname="./data/bicycle2_testN4aiter38"
    #testname="./data/bicycle2_testN36"
    #testname="./data/resultsN2/bicycle2_N2C2_final"
    #testname="./data/afterCVPR/bicycle2_force-bb_final"
    #testname="../../CRFdet/data/afterCVPR/12_01_10/cat2_force-bb_final"
    #testname="data/condor2/person3_full_condor219"
    #testname="data/condor_lowres/person2_morerigid_final"
    #testname="data/test2/face1_3Dfullright_final"
    #testname="data/test4/face1_test3Dperfect5"
    #testname="data/test4/face1_test3Donlyfrontal_final"
    #testname="data/test6/face1_3Drot2_final"
    #testname="data/test2/car1_3DVOCk20plus_final"
    #testname="data/test/car1_3DVOC_final"
    #testname="data/faces/face1_3Dafwfull2"
    #testname="data/faces/face1_3DmutliPIEfullRot_final"
    #testname="data/faces/face1_3DmutliPIEfullRotQuarter_final"
    #testname="data/faces/data/face1_3D_final"
    #testname="data/faces/data/face1_3DMPfix2Half_final"
    #testname="/users/visics/mpederso/code/git/3Def/3Det/data/3Deform/face1_Full14"
    #testname="/users/visics/mpederso/code/git/3Def/3Det/data/3Deform/face1_FastFull_final"
    #testname="/users/visics/mpederso/code/git/3Def/3Det/data/AFW/face1_AFLWCorr3sides_final"
    #testname="/users/visics/mpederso/code/git/3Def/3Det/data/AFW/face1_AFLW_slow14"
    #testname="/users/visics/mpederso/code/git/3Def/3Det/data/AFW/face1_AFLW20007"
    #testname="/users/visics/mpederso/code/git/3Def/3Det/data/VOC3Def/bicycle1_Deep2_final"
    #testname="/users/visics/mpederso/code/git/3Def/3Det/data/VOC3Def/bicycle1_New2Debug_final"
    #testname="/users/visics/mpederso/code/git/3Def/3Det/data/NEW/VOC3Def/car1_Fixed0"
    #testname="/users/visics/mpederso/code/git/3Def/3Det/data/NEW/VOC3Def/bicycle1_Good3"
    #testname="/users/visics/mpederso/code/git/3Def/3Det/data/NEW/VOC3Def/bicycle1_DEFixed_final"
    #testname="/users/visics/mpederso/code/git/3Def/3Det/data/NEW/AFLW/face1_AFLW2000Fixed_final"
    #testname="/users/visics/mpederso/code/git/3Def/3Det/data/NEW/AFLW/face1_Right_final"
    #testname="/users/visics/mpederso/code/git/3Def/3Det/data/NEW/AFLW/face1_Test3HOG_final"
    #testname="/users/visics/mpederso/code/git/3Def/3Det/data/NEW/AFLW/face1_MultiPIEstim20Z_final"
    testname="/users/visics/mpederso/code/git/3Def/3Det/data/NEW/AFLW/face1_MorePos5"
    #testname="/users/visics/mpederso/code/git/3Def/3Det/data/NEW/AFLW/face1_Double26"
    #testname="/users/visics/mpederso/code/git/3Def/3Det/data/NEW/AFLW/face1_Test2HOG_final"
    #testname="/users/visics/mpederso/code/git/3Def/3Det/data/NEW/VOC3Def/bicycle1_Estim20_final"
    #testname="/users/visics/mpederso/code/git/3Def/3Det/data/NEW/AFLW/face1_AFLWEstim20Z_final"
    #testname="/users/visics/mpederso/code/git/3Def/3Det/data/VOC3Def/bicycle1_Flat_Full_Fixed_final"
    #testname="/users/visics/mpederso/code/git/3Def/3Det/data/flat/face1_Full13view_2Def_final"
    #testname="/users/visics/mpederso/code/git/3Def/3Det/data/3Deform/face1_HigherRes_final"
    #testname="/users/visics/mpederso/code/git/3Def/3Det/data/flat/face1_Full2Def11"
    #testname="/users/visics/mpederso/code/git/3Def/3Det/data/unsupervised/face1_3DMPfix2Initial4"
    #testname="data/faces/ReducedViews/face1_3DMPfix2RVfull_final"
    #testname="data/unsupervised/face1_3DMPfix2Unsupervised256"
    #cfg.usebiases=True
    #cfg.usedef=True
    #testname="./data/unsupervised/face1_3Ddebug12_final"
    #testname="./data/VOC3D/bicycle1_fullVOC3D_final"
    #testname="./data/VOC3D/bicycle1_fullVOC3Dmoreneg2_final"
    #testname="data/faces/face1_3Dafwshort_final"
    #testname="data/test3/face1_3Dnewfull3"
    #cfg.trunc=1
    #cfg.flat=False
    #cfg.usebiases=True
    #cfg.k=20.0
    #cfg.resize=0.5
    models=util.load("%s.model"%(testname))
    #cfg.usedef=False
    #cfg.angx=[8]
    #cfg.cangy=[-30,-15,0,15,30]#[-30,0,+30]
    #cfg.cangz=[-20,-10,0,10,20]#[-10,0,10]
    #cfg.resize=0.5
    #cfg.cangy=[5]
    #cfg.skip=20
    #cfg.angx=[17]
    #cfg.usebiases=False
    #cfg.usedef=False
    #for mm in models[0]["ww"]:
    #    mm.lz=0.0
        #mm.dfay=1.0;mm.dfax=1.0;mm.dfaz=1.0#1
        #mm.dfay=.001;mm.dfax=0.001;mm.dfaz=.0001#1
        #mm.dfay=1;mm.dfax=1;mm.dfaz=1#1
        #mm.dfay=0.01;mm.dfax=0.01;mm.dfaz=0.0000000001#1
    #    #mm.lz=5 #does not work with different lz
    #models[0]["biases"]=numpy.concatenate((models[0]["biases"],models[0]["biases"],models[0]["biases"]),0)
    #models[0]["biases"]=0#numpy.zeros((1,25))
    #del models[0]
    #cfg.numcl=1
    #cfg.E=1
    #cfg.N=3
    #cfg.N=models[0]["N"]
    #models=util.load("%s%d.model"%(testname,it))
    #just for the new
    #for idm,m in enumerate(models):
    #    models[idm]["cost"]=models[idm]["cost"]*0.2
#        newc=numpy.zeros((8,aux.shape[1],aux.shape[2]),dtype=aux.dtype)
#        newc[:4]=aux
#        models[idm]["cost"]=newc
    ##############test
    #import itertools
    #runtest(models,tsImages,cfg,parallel=False,numcore=4,detfun=lambda x :detectCRF.test(x,numhyp=1,show=False),show=True)#,save="%s%d"%(testname,it))[196] is the many faces
    runtest(models,tsImagesFull,cfg,parallel=True,numcore=24,show=True,detfun=testINC03,save="./results/face_MorePos5")

