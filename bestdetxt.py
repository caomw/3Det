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

def cmp(a,b):
    return a[1]<b[1]

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
        #cfg.cls=sys.argv[1]
        cfg.numcl=3
        #cfg.dbpath="/home/owner/databases/"
        cfg.dbpath="/users/visics/mpederso/databases/"
        cfg.testpath="./data/"#"./data/CRF/12_09_19/"
        cfg.testspec="right"#"full2"
        cfg.db="VOC"
        #cfg.N=
       

    import pylab as pl
    import util
    import detectCRF
    #det=util.load("./data/CRF/12_10_02_parts_full/bicycle2_testN1_final.det")["det"]
    #det=util.load("./data/CRF/12_10_02_parts_full/bicycle2_testN2_final.txt")
    #fl=open("./data/CRF/12_10_02_parts_full/bicycle2_testN2_final.txt")
    #fl=open("./data/inria1_inria3.txt")
    fl=open("/users/visics/mmathias/faces_marco7.txt")
    #fl=open("/users/visics/mmathias/faces_marco_multiscale.txt")
    det=fl.readlines()
    imgpath=cfg.dbpath+"/afw/testimages/"#VOC
    #imgpath=cfg.dbpath+"VOC2007/VOCdevkit/VOC2007/JPEGImages/"#VOC
    #imgpath=cfg.dbpath+"INRIAPerson/Test/pos/"#inria
    for idl,l in enumerate(det):
        imname,scr,b0,b1,b2,b3=l.split()
        if float(scr)<0:
            continue
        try:
            img=util.myimread(imgpath+imname+".png")
        except:
            img=util.myimread(imgpath+imname+".jpg")
        pl.figure(100)        
        pl.clf()
        pl.imshow(img)
        util.box(int(b1),int(b0),int(b3),int(b2),"w",lw=2)
        pl.title("Rank:%d Scr:%.3f"%(idl,float(scr)))
        pl.axis([0,img.shape[1],img.shape[0],0])
        #detectCRF.visualize2([l],2,img,text="rank:%d"%(idl))
        pl.draw()
        pl.show()
        raw_input()







