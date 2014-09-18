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

if __name__ == '__main__':

    if 0: #use the configuration file
        print "Loading defautl configuration config.py"
        from config import * #default configuration      

        if len(sys.argv)>2: #specific configuration
            print "Loading configuration from %s"%sys.argv[2]
            import_name=sys.argv[2]
            exec "from config_%s import *"%import_name
            
        #cfg.cls=sys.argv[1]
        cfg.useRL=False#for the moment
        cfg.show=False
        cfg.auxdir=""
        cfg.numhyp=5
        cfg.rescale=True
        cfg.numneg= 10
        bias=100
        cfg.bias=bias
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
        cfg.testpath="./data/"#"./data/CRF/12_09_19/"
        cfg.testspec="right"#"full2"
        cfg.db="3DVOC"#"AFLW"
        #cfg.cls="diningtable"
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
    elif cfg.db=="AFW":
        tsImages=getRecord(AFW(basepath=cfg.dbpath),cfg.maxpos)
        tsImagesFull=tsImages
    elif cfg.db=="AFLW":
        trPosImages=getRecord(AFLW(basepath=cfg.dbpath,fold=0),cfg.maxpos,facial=True,pose=True)#cfg.useFacial)
        trPosImagesNoTrunc=trPosImages[:900]
        trNegImages=getRecord(InriaNegData(basepath=cfg.dbpath),cfg.maxneg)#check if it works better like this
        trNegImagesFull=getRecord(InriaNegData(basepath=cfg.dbpath),cfg.maxnegfull)
        #test
        tsImages=getRecord(AFLW(basepath=cfg.dbpath,fold=0),cfg.maxtest,facial=True,pose=True)#cfg.useFacial)
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


    #build a dictionary with images as key to speed-up image based search
    gt={}
    for l in tsImagesFull:
        im=l[1].split("/")[-1]
        gt[im]=l[2]

    import pylab as pl
    import util
    import detectCRF
    #det=util.load("./data/CRF/12_10_02_parts_full/bicycle2_testN1_final.det")["det"]
    #det=util.load("./data/CRF/12_10_02_parts_full/bicycle2_testN2_final.det")["det"]
    #det=util.load("./data/resultsN2/%s2_N2C2_final.det"%(cfg.cls))["det"]
    #det=util.load("/users/visics/mpederso/code/git/condor-run/N2C2_posthr105/CRFdet/data/CRF/12_10_21/%s2_N2C2k015.det"%(cfg.cls))["det"]
    #det=util.load("/users/visics/mpederso/code/git/condor-run/N2C2_posthr105/CRFdet/data/CRF/12_10_21/%s2_N2C2k01_final.det"%(cfg.cls))["det"]
    #det=util.load("/users/visics/mpederso/code/git/condor-run/N2C2_posthr105/CRFdet/data/CRF/12_10_20/%s2_N2C2_final.det"%(cfg.cls))["det"]
    #det=util.load("/users/visics/mpederso/code/git/condor-run/N2C2_highres/CRFdet/data/CRF/12_11_01/bicycle2_N2C2_highres3.det")["det"]
    #det=util.load("/users/visics/mpederso/code/git/condor-run/N2C2_highres/CRFdet/data/CRF/12_11_01/%s2_N2C2_highres_final.det"%(cfg.cls))["det"]
    #det=util.load("/users/visics/mpederso/code/git/condor-run/N2C2_highres/CRFdet/data/CRF/12_10_29/%s2_N4C2highres_final.det"%(cfg.cls))["det"]
    #det=util.load("/users/visics/mpederso/code/git/condor-run/buffy/CRFdet/data/CRF/12_10_26/person3_buffyN2new_final.det")["det"]
    #det=util.load("/users/visics/mpederso/code/git/condor-run/N2C2_posthr105/CRFdet/data/CRF/12_10_20/%s2_N2C215.det"%(cfg.cls))["det"]
    #det=util.load("./data/resultsN2/%s2_N2C2_final.det"%(cfg.cls))["det"]
    #det=util.load("/users/visics/mpederso/code/git/condor-run/N2C2_posthr105/CRFdet/data/CRF/12_10_21/%s2_N2C2k01_final.det"%(cfg.cls))["det"]
    #det=util.load("/users/visics/mpederso/code/git/condor-run/N2C2_posthr105/CRFdet/data/CRF/12_10_20/%s2_N2C2_final.det"%(cfg.cls))["det"]
    #det=util.load("/users/visics/mpederso/code/git/condor-run/N2C2_posthr105/CRFdet/data/CRF/12_10_20/%s2_N2C26.det"%(cfg.cls))["det"]
    #cfg.N=4
    #det=util.load("/users/visics/mpederso/code/git/condor-run/N3C2_highres/CRFdet/data/CRF/12_11_10/%s2_N2C2highres2_final.det"%cfg.cls)["det"]
    #det=util.load("/users/visics/mpederso/code/git/condor-run/N3C2_highres/CRFdet/data/CRF/12_11_10/%s2_N2C2highres26.det"%cfg.cls)["det"]
    #det=util.load("/users/visics/mpederso/code/git/condor-run/N2C2_highres/CRFdet/data/CRF/12_11_01/bicycle2_N1C2highres2.det")["det"];cfg.N=1
    #det=util.load("/users/visics/mpederso/code/git/CRFdet/data/afterCVPR/12_01_10/%s2_force-bb3.det"%cfg.cls)["det"]
    #det=util.load("./data/condor/%s2_condor10.det"%cfg.cls)["det"]sftp://mpederso@ssh.esat.kuleuven.be/users/visics/mpederso/code/git/fastDP/CRFdet/data/debug2/car2_FULLsmall4.det
    #det=util.load("./data/debug2/%s2_higherlimit2.det"%cfg.cls)["det"]
    #det=util.load("./data/condor_lowres/%s2_morerigid_final.det"%cfg.cls)["det"]
    #det=util.load("data/AFW/AWF4.det")["det"]
    #det=util.load("AWFpose.det")["det"]    
    #det=util.load("/users/visics/mpederso/code/git/facial/CRFdet/data/MultiPIE/face2_PIE600_trpos.det")["det"]
    #det=util.load("PIEfull4.det")["det"]    
    #det=util.load("face1_flat.det")["det"]    
    #det=util.load("/users/visics/mpederso/code/git/3Def/3Det/data/faces/car1_3Dafwright_final.det")["det"]
    #det=util.load("/users/visics/mpederso/code/git/3Def/3Det/faces_Def14.det")["det"]
    #det=util.load("/users/visics/mpederso/code/git/3Def/3Det/faces_DeepFace3.det")["det"]
    det=util.load("/users/visics/mpederso/code/git/3Def/3Det/data/VOC3Def/bicycle1_Deep25Fixed5.det")["det"]
    #imgpath=cfg.dbpath+"multiPIE//"
    imgpath=cfg.dbpath+"PASCAL3D+_release1.0/JPEGImages/"
    #imgpath=cfg.dbpath+"afw/testimages/"
    #imgpath=cfg.dbpath+"aflw/data/flickr/"
    #imgpath=cfg.dbpath+"VOC2007/VOCdevkit/VOC2007/JPEGImages/"
    #imgpath=cfg.dbpath+"/buffy/images/"
    line=True
    cfg.N=1
    for idl,l in enumerate(det):
        #try:
        #    img=util.myimread(imgpath+"/0/"+l["idim"])
        #except:
        #    try:
        #        img=util.myimread(imgpath+"/2/"+l["idim"])
        #    except:
        #        img=util.myimread(imgpath+"/3/"+l["idim"])

        img=util.myimread(imgpath+l["idim"])
#just for buffy
#        try:
#           img=util.myimread(imgpath+"buffy_s5e2/"+l["idim"])
#        except:
#            try:    
#                img=util.myimread(imgpath+"buffy_s5e3/"+l["idim"])
#            except:
#                try:
#                    img=util.myimread(imgpath+"buffy_s5e4/"+l["idim"])
#                except:
#            try:
#                img=util.myimread(imgpath+"buffy_s5e5/"+l["idim"])
#            except:
#                try:
#                    img=util.myimread(imgpath+"buffy_s5e6/"+l["idim"])
#                except:
#                    pass

        #gooddet=-1
        ovr=[0]
        for idb,b in enumerate(gt[l["idim"]]):#for each bb gt
            ovr.append(util.overlap(b,l["bbox"]))
        if len(ovr)>0:
            #print "Best ovr",max(ovr)
            if max(ovr)>=0.5:
                #detectCRF.visualize2([l],cfg.N,img,text="rank:%d ovr:%.3f scl:%d"%(idl,max(ovr),l["hog"]),bb=gt[l["idim"]],color="w",line=line)
                detectCRF.visualize3D([l],cfg.N,img,bb=gt[l["idim"]],color="w")
            else:
                detectCRF.visualize3D([l],cfg.N,img,bb=gt[l["idim"]],color="r")
                #detectCRF.visualize2([l],cfg.N,img,text="rank:%d ovr:%.3f scl:%d"%(idl,max(ovr),l["hog"]),bb=gt[l["idim"]],color="r",line=line)
        else:
            detectCRF.visualize2([l],cfg.N,img,text="rank:%d"%(idl),color="r",line=line)
        #pl.figure(100)        
        #pl.clf()
        #pl.imshow(img)
        raw_input()







