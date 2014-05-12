#from hog import *
import numpy
import pylab
from database import *
from util import box,overlap
from util2 import overlapx
import time

def VOCpr(gtImages,detfile,show=False,cl="person",usetr=True,usedf=False):
    """
        calculate the precision recall curve
    """
    detf=open(detfile,"r")
    detect=detf.readlines()
    imname=[]
    cnt=0
    ovr=0.49
    #print trPosImages.getTotal()
    tp=[]
    fp=[]
    tot=0
    for idx in range(gtImages.getTotal()):
        print gtImages.getImageName(idx)
        if show:
            img=gtImages.getImage(idx)
            pylab.figure(1)
            pylab.clf()
            pylab.imshow(img)
        #pyr=HOGcompute.HOGcrop(img,interv=interv)
        #pyr.pad()
        #pyr.pad()
        #pyr.contrast()
        rect=gtImages.getBBox(idx,cl=cl,usetr=usetr,usedf=usedf)
        print rect
        if show:
            for r in rect:
                pylab.figure(1)
                pylab.ioff()
                box(r[0],r[1],r[2],r[3],'b',lw=1.5)
                #raw_input()
        tot=tot+len(rect)
        #print len(rect),rect
        #print rect
        for l in detect:
            data=l.split(" ")
            if data[0]==gtImages.getImageName(idx).split("/")[-1].split(".")[0]:
                notfound=True
                rb=[float(data[3]),float(data[2]),float(data[5]),float(data[4])]
                if show:
                    pylab.ioff()
                    pylab.text(rb[1],rb[0],data[1])
                for id,r in enumerate(rect):
                    #pylab.figure(1)
                    #box(r[0],r[1],r[2],r[3],'b',lw=1.5)
                    #print "entered",data
                    #rb=[float(data[3]),float(data[2]),float(data[5]),float(data[4])]
                    #print rb,r,overlap(rb,r)
                    #pylab.text(rb[1],rb[0],data[1])
                    if overlap(rb,r)>=ovr:
                        if show:
                            pylab.ioff()
                            box(rb[0],rb[1],rb[2],rb[3],'g',lw=1.5)
                        del rect[id]
                        tp.append(float(data[1]))
                        notfound=False
                        break
                if notfound==True:
                    if show:
                        pylab.ioff()
                        box(rb[0],rb[1],rb[2],rb[3],'r',lw=1)                        
                    fp.append(float(data[1]))
                #print len(tp),len(fp),tot            
            #break
        if show:
            pylab.figure(1)
            pylab.show()
            pylab.draw()
        #raw_input()
    return tp,fp,tot

def VOCprlist(gtImages,detlist,show=False,usetr=True,usedf=False,ovr=0.5):
    """
        calculate the precision recall curve
    """
    #detf=open(detfile,"r")
    #detect=detf.readlines()
    imname=[]
    cnt=0
    #ovr=0.49
    #print trPosImages.getTotal()
    tp=[]
    fp=[]
    tot=0
    for idx in range(gtImages.getTotal()):
        print gtImages.getImageName(idx)
        if show:
            img=gtImages.getImage(idx)
            pylab.figure(1)
            pylab.clf()
            pylab.imshow(img)
        #pyr=HOGcompute.HOGcrop(img,interv=interv)
        #pyr.pad()
        #pyr.pad()
        #pyr.contrast()
        rect=gtImages.getBBox(idx,usetr=usetr,usedf=usedf)
        print rect
        if show:
            for r in rect:
                pylab.figure(1)
                pylab.ioff()
                box(r[0],r[1],r[2],r[3],'b',lw=1.5)
                #raw_input()
        tot=tot+len(rect)
        #print len(rect),rect
        #print rect
        for l in detlist:
            data=l#.split(" ")
            if data[0]==gtImages.getImageName(idx).split("/")[-1].split(".")[0]:
                notfound=True
                rb=[float(data[3]),float(data[2]),float(data[5]),float(data[4])]
                if show:
                    pylab.ioff()
                    pylab.text(rb[1],rb[0],data[1])
                for id,r in enumerate(rect):
                    #pylab.figure(1)
                    #box(r[0],r[1],r[2],r[3],'b',lw=1.5)
                    #print "entered",data
                    #rb=[float(data[3]),float(data[2]),float(data[5]),float(data[4])]
                    #print rb,r,overlap(rb,r)
                    #pylab.text(rb[1],rb[0],data[1])
                    if overlap(rb,r)>=ovr:
                        if show:
                            pylab.ioff()
                            box(rb[0],rb[1],rb[2],rb[3],'g',lw=1.5)
                        del rect[id]
                        tp.append(float(data[1]))
                        notfound=False
                        break
                if notfound==True:
                    if show:
                        pylab.ioff()
                        box(rb[0],rb[1],rb[2],rb[3],'r',lw=1)                        
                    fp.append(float(data[1]))
                #print len(tp),len(fp),tot            
            #break
        if show:
            pylab.figure(1)
            pylab.show()
            pylab.draw()
        #raw_input()
    return tp,fp,tot

def cmpscore(a,b):
    return -cmp(a[1],b[1])


def VOCprlistfast(gtImages,detlist,show=False,usetr=True,usedf=False,ovr=0.5):
    """
        calculate the precision recall curve
    """
    dimg={}
    tot=0
    for idx in range(gtImages.getTotal()):
        rect=gtImages.getBBox(idx)
        if rect!=[]:
            dimg[gtImages.getImageName(idx).split("/")[-1].split(".")[0]]=rect
        tot=tot+len(rect)
        #print tot
    imname=[]
    cnt=0
    tp=numpy.zeros(len(detlist))
    fp=numpy.zeros(len(detlist))
    detlist.sort(cmpscore)
    for idx,detbb in enumerate(detlist):#detlist[sortlist]):#gtImages.getTotal()):
        found=False
        if dimg.has_key(detbb[0]):
            rect=dimg[detbb[0]]#gtImages.getBBox(idx,usetr=usetr,usedf=usedf)
            #print rect
            found=False
            for r in rect:
                rb=(float(detbb[3]),float(detbb[2]),float(detbb[5]),float(detbb[4]))
                if overlap(rb,r)>=ovr:
                    dimg[detbb[0]].remove(r)
                    found=True
                    break
        if found:  
            tp[idx]=1#.append(float(detbb[1]))
        else:
            fp[idx]=1#.append(float(detbb[1]))
        if show:
            pylab.ioff()
            img=gtImages.getImageByName2(detbb[0])
            pylab.figure(1)
            pylab.clf()
            pylab.imshow(img)
            rb=(float(detbb[3]),float(detbb[2]),float(detbb[5]),float(detbb[4]))
            for r in rect:
                pylab.figure(1)
                pylab.ioff()
                box(r[0],r[1],r[2],r[3],'b',lw=1.5)
            if found:
                box(rb[0],rb[1],rb[2],rb[3],'g',lw=1.5)
            else:
                box(rb[0],rb[1],rb[2],rb[3],'r',lw=1.5)
            pylab.draw()
            pylab.show()
            rect=[]
            raw_input()

    return tp,fp,tot

def VOCprlistfastscore(gtImages,detlist,numim=numpy.inf,show=False,usetr=True,usedf=False,ovr=0.5):
    """
        calculate the precision recall curve
    """
    dimg={}
    tot=0
    for idx in range(min(gtImages.getTotal(),numim)):
        rect=gtImages.getBBox(idx)
        #if idx>288:
        #    print idx,rect
        if rect!=[]:
            #print gtImages.getImageName(idx).split("/")[-1].split(".")[0]
            dimg[gtImages.getImageName(idx).split("/")[-1].split(".")[0]]=rect
        tot=tot+len(rect)
    imname=[]
    cnt=0
    tp=numpy.zeros(len(detlist))
    fp=numpy.zeros(len(detlist))
    thr=numpy.zeros(len(detlist))
    detlist.sort(cmpscore)
    for idx,detbb in enumerate(detlist):
        #print detbb[1]
        found=False
        if dimg.has_key(detbb[0]):
            rect=dimg[detbb[0]]
            found=False
            for r in rect:
                rb=(float(detbb[3]),float(detbb[2]),float(detbb[5]),float(detbb[4]))
                #print "GT:",r
                #print "DET:",rb
                if overlap(rb,r)>=ovr:
                    dimg[detbb[0]].remove(r)
                    found=True
                    break
        if found:  
            tp[idx]=1
        else:
            fp[idx]=1
        thr[idx]=detbb[1]
        if show:
            prec=numpy.sum(tp)/float(numpy.sum(tp)+numpy.sum(fp))
            rec=numpy.sum(tp)/tot
            print "Scr:",detbb[1],"Prec:%.3f"%prec,"Rec:%.3f"%rec
            ss=raw_input()
            if ss=="s" or not(found):
                pylab.ioff()
                img=gtImages.getImageByName2(detbb[0])
                pylab.figure(1)
                pylab.clf()
                pylab.imshow(img)
                rb=(float(detbb[3]),float(detbb[2]),float(detbb[5]),float(detbb[4]))
                for r in rect:
                    pylab.figure(1)
                    pylab.ioff()
                    box(r[0],r[1],r[2],r[3],'b',lw=1.5)
                if found:
                    box(rb[0],rb[1],rb[2],rb[3],'g',lw=1.5)
                else:
                    box(rb[0],rb[1],rb[2],rb[3],'r',lw=1.5)
                pylab.draw()
                pylab.show()
                rect=[]

    return tp,fp,thr,tot

def VOCprRecord_old(gtImages,detlist,show=False,usetr=True,usedf=False,ovr=0.5):
    """
        calculate the precision recall curve
    """
    dimg={}
    tot=0
    for idx in range(len(gtImages)):
        rect=gtImages[idx]["bbox"][:]
        #if idx>288:
        #    print idx,rect
        if rect!=[]:
            #print gtImages.getImageName(idx).split("/")[-1].split(".")[0]
            dimg[gtImages[idx]["name"].split("/")[-1].split(".")[0]]=rect
        tot=tot+len(rect)
    imname=[]
    cnt=0
    tp=numpy.zeros(len(detlist))
    fp=numpy.zeros(len(detlist))
    thr=numpy.zeros(len(detlist))
    detlist.sort(cmpscore)
    for idx,detbb in enumerate(detlist):
        #print detbb[1]
        found=False
        if dimg.has_key(detbb[0]):
            rect=dimg[detbb[0]]
            found=False
            for r in rect:
                rb=(float(detbb[3]),float(detbb[2]),float(detbb[5]),float(detbb[4]))
                #print "GT:",r
                #print "DET:",rb
                if overlap(rb,r)>=ovr:
                    dimg[detbb[0]].remove(r)
                    found=True
                    break
        if found:  
            tp[idx]=1
        else:
            fp[idx]=1
        thr[idx]=detbb[1]
        if show:
            prec=numpy.sum(tp)/float(numpy.sum(tp)+numpy.sum(fp))
            rec=numpy.sum(tp)/tot
            print "Scr:",detbb[1],"Prec:%.3f"%prec,"Rec:%.3f"%rec
            ss=raw_input()
            if ss=="s" or not(found):
                pylab.ioff()
                img=gtImages.getImageByName2(detbb[0])
                pylab.figure(1)
                pylab.clf()
                pylab.imshow(img)
                rb=(float(detbb[3]),float(detbb[2]),float(detbb[5]),float(detbb[4]))
                for r in rect:
                    pylab.figure(1)
                    pylab.ioff()
                    box(r[0],r[1],r[2],r[3],'b',lw=1.5)
                if found:
                    box(rb[0],rb[1],rb[2],rb[3],'g',lw=1.5)
                else:
                    box(rb[0],rb[1],rb[2],rb[3],'r',lw=1.5)
                pylab.draw()
                pylab.show()
                rect=[]

    return tp,fp,thr,tot

def VOCprRecord(gtImages,detlist,show=False,ovr=0.5,pixels=None):
    """
        calculate the precision recall curve
    """
    dimg={}
    tot=0
    for idx in range(len(gtImages)):
        rect=gtImages[idx]["bbox"][:]
        #if idx>288:
        #    print idx,rect
        if rect!=[]:
            #print gtImages.getImageName(idx).split("/")[-1].split(".")[0]
            dimg[gtImages[idx]["name"].split("/")[-1].split(".")[0]]={"bbox":rect,"det":[False]*len(rect)}
            for i, recti in enumerate(rect):
                if recti[5] == 0:
                    tot=tot+1

    imname=[]
    cnt=0
    tp=numpy.zeros(len(detlist))
    fp=numpy.zeros(len(detlist))
    thr=numpy.zeros(len(detlist))
    detlist.sort(cmpscore)
    for idx,detbb in enumerate(detlist):
        #print detbb[1]
        found=False
        maxovr=0
        #gtdet=[False]
        gt=0
        if dimg.has_key(detbb[0]):
            rect=dimg[detbb[0]]["bbox"]
            found=False
            for ir,r in enumerate(rect):
                #gtdet.append(False)
                rb=(float(detbb[3]),float(detbb[2]),float(detbb[5]),float(detbb[4]))
                #print "GT:",r
                #print "DET:",rb
                if pixels==None:
                    covr=overlap(rb,r)
                else:
                    covr=overlapx(rb,r,pixels)
                if covr>=maxovr:
                    maxovr=covr
                    gt=ir
                    #dimg[detbb[0]].remove(r)
                    #found=True
                    #break

        if maxovr>ovr:
            if dimg[detbb[0]]["bbox"][gt][5] == 0:
                if not(dimg[detbb[0]]["det"][gt]):
                    tp[idx]=1
                    dimg[detbb[0]]["det"][gt]=True
                else:
                    fp[idx]=1
        else:
            fp[idx]=1

########### PASCAL 2010
#    if ovmax>=VOCopts.minoverlap
#        if ~gt(i).diff(jmax)
#            if ~gt(i).det(jmax)
#                tp(d)=1;            % true positive
#		        gt(i).det(jmax)=true;
#            else
#                fp(d)=1;            % false positive (multiple detection)
#            end
#        end
#    else
#        fp(d)=1;                    % false positive
#    end
########################



        thr[idx]=detbb[1]
        if show:
            prec=numpy.sum(tp)/float(numpy.sum(tp)+numpy.sum(fp))
            rec=numpy.sum(tp)/tot
            print "Scr:",detbb[1],"Prec:%.3f"%prec,"Rec:%.3f"%rec
            ss=raw_input()
            if ss=="s" or not(found):
                pylab.ioff()
                img=gtImages.getImageByName2(detbb[0])
                pylab.figure(1)
                pylab.clf()
                pylab.imshow(img)
                rb=(float(detbb[3]),float(detbb[2]),float(detbb[5]),float(detbb[4]))
                for r in rect:
                    pylab.figure(1)
                    pylab.ioff()
                    box(r[0],r[1],r[2],r[3],'b',lw=1.5)
                if found:
                    box(rb[0],rb[1],rb[2],rb[3],'g',lw=1.5)
                else:
                    box(rb[0],rb[1],rb[2],rb[3],'r',lw=1.5)
                pylab.draw()
                pylab.show()
                rect=[]

    return tp,fp,thr,tot

def VOCprPose(gtImages,detlist,show=False,ovr=0.5,pixels=None,posethr=15):
    """
        calculate the precision recall curve
    """
    dimg={}
    tot=0
    for idx in range(len(gtImages)):
        rect=gtImages[idx]["bbox"][:]
        #if idx>288:
        #    print idx,rect
        if rect!=[]:
            #print gtImages.getImageName(idx).split("/")[-1].split(".")[0]
            dimg[gtImages[idx]["name"].split("/")[-1].split(".")[0]]={"bbox":rect,"det":[False]*len(rect),"pose":gtImages[idx]["pose"]}
            for i, recti in enumerate(rect):
                if recti[5] == 0:
                    tot=tot+1

    imname=[]
    cnt=0
    tp=numpy.zeros(len(detlist))
    fp=numpy.zeros(len(detlist))
    thr=numpy.zeros(len(detlist))
    detlist.sort(cmpscore)
    for idx,detbb in enumerate(detlist):
        #print detbb[1]
        found=False
        maxovr=0
        #gtdet=[False]
        gt=0
        if dimg.has_key(detbb[0]):
            rect=dimg[detbb[0]]["bbox"]
            found=False
            for ir,r in enumerate(rect):
                #gtdet.append(False)
                rb=(float(detbb[3]),float(detbb[2]),float(detbb[5]),float(detbb[4]))
                #print "GT:",r
                #print "DET:",rb
                if pixels==None:
                    covr=overlap(rb,r)
                else:
                    covr=overlapx(rb,r,pixels)
                if covr>=maxovr:
                    maxovr=covr
                    gt=ir
                    #dimg[detbb[0]].remove(r)
                    #found=True
                    #break

        if maxovr>ovr:
            if dimg[detbb[0]]["bbox"][gt][5] == 0:
                print "GT",(dimg[detbb[0]]["pose"][gt][0][0]+180)%360-180,"pose",detbb[6]
                #if not(dimg[detbb[0]]["det"][gt]) and abs((dimg[detbb[0]]["pose"][gt]+180)%360-180-detbb[6])<posethr:                    #check also pose
                if not(dimg[detbb[0]]["det"][gt]) and abs((dimg[detbb[0]]["pose"][gt][0][0]+180)%360-180-detbb[6])<posethr:                    #check also pose
                    tp[idx]=1
                    dimg[detbb[0]]["det"][gt]=True
                else:
                    fp[idx]=1
        else:
            fp[idx]=1

########### PASCAL 2010
#    if ovmax>=VOCopts.minoverlap
#        if ~gt(i).diff(jmax)
#            if ~gt(i).det(jmax)
#                tp(d)=1;            % true positive
#		        gt(i).det(jmax)=true;
#            else
#                fp(d)=1;            % false positive (multiple detection)
#            end
#        end
#    else
#        fp(d)=1;                    % false positive
#    end
########################



        thr[idx]=detbb[1]
        if show:
            prec=numpy.sum(tp)/float(numpy.sum(tp)+numpy.sum(fp))
            rec=numpy.sum(tp)/tot
            print "Scr:",detbb[1],"Prec:%.3f"%prec,"Rec:%.3f"%rec
            ss=raw_input()
            if ss=="s" or not(found):
                pylab.ioff()
                img=gtImages.getImageByName2(detbb[0])
                pylab.figure(1)
                pylab.clf()
                pylab.imshow(img)
                rb=(float(detbb[3]),float(detbb[2]),float(detbb[5]),float(detbb[4]))
                for r in rect:
                    pylab.figure(1)
                    pylab.ioff()
                    box(r[0],r[1],r[2],r[3],'b',lw=1.5)
                if found:
                    box(rb[0],rb[1],rb[2],rb[3],'g',lw=1.5)
                else:
                    box(rb[0],rb[1],rb[2],rb[3],'r',lw=1.5)
                pylab.draw()
                pylab.show()
                rect=[]

    return tp,fp,thr,tot


#det should have imid and facial
def VOCprFacial(gtImages,det,show=False,ovr=0.5,pixels=None,mina=150,thr=0.05,delta=None):
    """
        calculate the precision recall curve
    """
    ddet={}
    tot=0
    for idx,dd in enumerate(det):
        if dd.has_key("facial"):
            facial=dd["facial"][:]
            mixid=dd["id"]
            #if idx>288:
            #    print idx,rect
            if facial!=[]:
                #print gtImages.getImageName(idx).split("/")[-1].split(".")[0]
                if ddet.has_key(dd["idim"]):
                    ddet[dd["idim"]].append((facial,mixid))
                else:
                    ddet[dd["idim"]]=[(facial,mixid)]
                for i, faciali in enumerate(facial):
                    #py1,px1,py2,px2,no,no=gtImages[idx]["bbox"][i]
                    #if (py2-py1)*(px2-px1)>mina
                    tot=tot+1
        
    ldist=[]
    lmaxdist=[]
    ldelta=[]
    for l in range(8):
        ldelta.append([])
    for face in gtImages:#for each images
        fpgt=face["facial"]
        for idfp,fp in enumerate(fpgt):#for each face in image
            py1,px1,py2,px2,no,no=face["bbox"][idfp]
            if float(py2-py1)>=mina+4:
                if ddet.has_key(face["name"].split("/")[-1]):
                    ddim=ddet[face["name"].split("/")[-1]]
                    #mixid=ddet[face["name"].split("/")[-1]][1]
                    mindist=10000
                    for iddd,ddd in enumerate(ddim):#for each detection in the same image    
                        dd=ddd[0]
                        mixid=ddd[1]
                        if delta==None or len(delta[mixid])==0 or numpy.all(not(numpy.isfinite(delta))):
                            dist=numpy.sqrt((fp[:,1]-dd[::2])**2+(fp[:,0]-dd[1::2])**2)
                            dy=fp[:,1]-dd[::2];dx=fp[:,0]-dd[1::2]
                        else:
                            dist=numpy.sqrt((fp[:,1]-dd[::2]-delta[mixid[iddd]][:delta.shape[0]/2])**2+(fp[:,0]-dd[1::2]-delta[mixid[iddd]][delta.shape[0]/2:])**2)
                            dy=fp[:,1]-dd[::2]-delta[mixid[iddd]][:delta[mixid[iddd]].shape[0]/2];dx=fp[:,0]-dd[1::2]-delta[mixid[iddd]][delta[mixid[iddd]].shape[0]/2:]
                        #dist=dist[numpy.isfinite(dist)]
                        dist[numpy.isinf(dist)]=0
                        #print sum(dist),fp[:,1],fp[:,0]
                        #raw_input()
                        #print dy,dx
                        #raw_input()
                        dist=dist/float(((py2-py1)+(px2-px1))/2)
                        if numpy.sum(dist)<mindist:
                            mindist=numpy.sum(dist)
                            fulldist=dist
                            bestid=iddd
                            bestdy=dy
                            bestdx=dx
                            bestid=mixid
                    ldist.append(fulldist)       
                    lmaxdist.append(numpy.mean(fulldist))
                    ldelta[bestid].append(numpy.concatenate((bestdy,bestdx)))
                else:
                    ldist.append([numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf])
                    lmaxdist.append(numpy.inf)
    #numpy.where(numpy.array(lf).sum(1)!=numpy.inf)[0]
    lmaxdist.sort()
    ald=[]
    if 0:
        for l in range(8):
            ald.append(numpy.array(ldelta[l]))
            ald[-1]=ald[-1][numpy.isfinite(numpy.sum(ald[-1],1)),:]#.mean(0)
            if len(ald[-1])>0:
                sald=numpy.argsort(ald[-1].sum(1))
                ald[-1]=ald[-1][sald[:len(ald[-1])/2]].mean(0)

    return ald,ldist,lmaxdist,numpy.sum(numpy.array(lmaxdist)<thr)/float(len(lmaxdist))

#det should have imid and facial
def VOCprFacial2(gtImages,det,show=False,ovr=0.5,pixels=None,mina=150,thr=0.05,delta=None,costoccl=numpy.inf):
    """
        calculate the precision recall curve
    """
    ddet={}
    tot=0
    for idx,dd in enumerate(det):
        if dd.has_key("facial"):
            facial=dd["facial"][:]
            mixid=dd["id"]
            #if idx>288:
            #    print idx,rect
            if facial!=[]:
                #print gtImages.getImageName(idx).split("/")[-1].split(".")[0]
                if ddet.has_key(dd["idim"]):
                    ddet[dd["idim"]].append((facial,mixid))
                else:
                    ddet[dd["idim"]]=[(facial,mixid)]
                for i, faciali in enumerate(facial):
                    #py1,px1,py2,px2,no,no=gtImages[idx]["bbox"][i]
                    #if (py2-py1)*(px2-px1)>mina
                    tot=tot+1
        
    ldist=[]
    lmaxdist=[]
    ldelta=[]
    for l in range(8):
        ldelta.append([])
    for face in gtImages:#for each images
        fpgt=face["facial"]
        for idfp,fp in enumerate(fpgt):#for each face in image
            py1,px1,py2,px2,no,no=face["bbox"][idfp]
            mindist=numpy.inf
            fulldist=numpy.inf
            if float(py2-py1)>=mina+4:
                #print face["name"].split("/")[-1]
                if ddet.has_key(face["name"].split("/")[-1]):
                    ddim=ddet[face["name"].split("/")[-1]]
                    #mixid=ddet[face["name"].split("/")[-1]][1]
                    mindist=numpy.inf
                    for iddd,ddd in enumerate(ddim):#for each detection in the same image    
                        dd=ddd[0]
                        mixid=ddd[1]
                        if numpy.any(fp-dd.reshape(fp.shape)==numpy.nan) and costocc==numpy.inf:
                            dist=[numpy.inf]*6
                        else:
                            #if delta==None or len(delta[mixid])==0 or numpy.all(not(numpy.isfinite(delta))):
                            dist=numpy.sqrt((fp[:,1]-dd[::2])**2+(fp[:,0]-dd[1::2])**2)
                            dy=fp[:,1]-dd[::2];dx=fp[:,0]-dd[1::2]
                            #dist=dist[numpy.isfinite(dist)]
                            #print dist
                            dist[numpy.isnan(dist)]=costoccl#numpy.inf
                            #raw_input()
                            #print sum(dist),fp[:,1],fp[:,0]
                            #raw_input()
                            #print dy,dx
                            #raw_input()
                            dist=dist/float(((py2-py1)+(px2-px1))/2)
                            #print numpy.mean(dist)
                        if numpy.sum(dist)<mindist:
                            mindist=numpy.sum(dist)
                            fulldist=dist
                            bestid=iddd
                            bestdy=dy
                            bestdx=dx
                            bestid=mixid
                    ldist.append(fulldist)       
                    lmaxdist.append(numpy.mean(fulldist))
                    ldelta[bestid].append(numpy.concatenate((bestdy,bestdx)))
                else:
                    ldist.append([numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf])
                    lmaxdist.append(numpy.inf)
                #print "best",lmaxdist[-1]
                #raw_input()
    #numpy.where(numpy.array(lf).sum(1)!=numpy.inf)[0]
    lmaxdist.sort()
    ald=[]
    if 0:
        for l in range(8):
            ald.append(numpy.array(ldelta[l]))
            ald[-1]=ald[-1][numpy.isfinite(numpy.sum(ald[-1],1)),:]#.mean(0)
            if len(ald[-1])>0:
                sald=numpy.argsort(ald[-1].sum(1))
                ald[-1]=ald[-1][sald[:len(ald[-1])/2]].mean(0)

    return ald,ldist,lmaxdist,numpy.sum(numpy.array(lmaxdist)<thr)/float(len(lmaxdist))



def VOCprFacialPlot(tsImages,det,part=None,mina=150,costoccl=numpy.inf):
    res=50
    x=numpy.linspace(0.0,0.15,res)
    dd=numpy.zeros(res)
    for idt,t in enumerate(x):
        aa,bb,cc,dd[idt]=VOCprFacial2(tsImages,det,thr=t,mina=mina,costoccl=costoccl)
        if part!=None:
            dd[idt]=numpy.sum(numpy.array(bb)[:,part]<t)/float(len(bb))
    #pylab.figure()
    pylab.xlabel("Average Localization Error as fraction of face size")
    pylab.ylabel("Fraction of testing faces")
    pylab.plot(x,dd,"-",lw=3)
    return x,dd

def VOCprProject(tsImages,det,models,N,E,part=None,show=False,fig=100):
    numcl=len(models)/2
    color=["r","g","b","w","y","m"]
    color2=["g","r","b","m","y","w"]
    ppts=[]
    for idl,l in enumerate(models[:numcl]):
        pylab.figure(fig+idl)
        import drawHOG
        import model
        im = drawHOG.drawHOG(model.convert2(l["ww"][0],N,E))
        pylab.clf()
        pylab.imshow(im)
        ppts.append([])#for each mmodel
    #dictionary of detections
    ddet={}
    for idx,dd in enumerate(det):
        if ddet.has_key(dd["idim"]):
            ddet[dd["idim"]].append(dd)
        else:
            ddet[dd["idim"]]=[dd]

    import extra
    reload(extra)
    for im in tsImages:
        gtann={"bbox":im["bbox"],"facial":im["facial"]}
        imn=im["name"].split("/")[-1]
        if ddet.has_key(imn):
            fpts,ids=extra.project(ddet[imn],gtann,N)
            for idl,l in enumerate(ids):
                pylab.figure(fig+l%numcl)
                if l>=numcl:
                    #continue
                    inv=[1,0,2,5,4,3]
                    #inv=[2,3, 0,1, 4,5, 10,11, 8,9, 6,7]
                    fpts[idl][:,1]=models[l%numcl]["ww"][0].shape[1]*(N/float(N+2*E))-fpts[idl][:,1]
                    fpts[idl]=fpts[idl][inv]
                #pylab.plot(fpts[idl][:,1]*15,fpts[idl][:,0]*15,"+",markersize=5)  
                ppts[l%numcl].append(fpts[idl])
                if show:
                    for p in range(len(fpts[idl])):
                        if l%numcl==2:
                            col=color2
                        else:
                            col=color
                        pylab.plot(fpts[idl][p,1]*15,fpts[idl][p,0]*15,"o"+col[p],markersize=5)  
    if show:
        pylab.draw()
        pylab.show()
    stdfp=[]
    means=[]
    ff=float(N)/(N+2*E)
    for idm,m in enumerate(ppts):
        stdfp.append([])
        means.append([])
        m1=numpy.array(m)
        print "Model",idm
        fpname=["Eyel","Eyer","Nose","Lipl","Lipc","Lipr"]
        for fp in range(m1.shape[1]):
            sel=numpy.isfinite(m1[:,fp,0])
            stdfp[-1].append(numpy.std(numpy.sqrt((m1[sel,fp,0]/models[idm]["ww"][0].shape[0]*ff*100)**2+(m1[sel,fp,1]/models[idm]["ww"][0].shape[1]*ff*100)**2)))
            means[-1].append(numpy.mean((m1[sel,fp,:]),0))
            print fpname[fp],"Std",stdfp[-1][-1],"Mean",means[-1][-1]
        print "Average std",numpy.mean(stdfp[-1])
    #print "Global Std,",numpy.mean(numpy.mean(numpy.array(stdfp)[numpy.isfinite(stdfp)],1),0)
    print "Global Std",numpy.mean(numpy.array(stdfp)[numpy.isfinite(stdfp)])
    return ppts,stdfp,means
    #raw_input()


def VOCprRecordthr(gtImages,detlist,show=False,ovr=0.5,pixels=None):
    """
        calculate the precision recall curve
    """
    dimg={}
    tot=0
    posd=[]
    for idx in range(len(gtImages)):
        rect=gtImages[idx]["bbox"][:]
        #if idx>288:
        #    print idx,rect
        if rect!=[]:
            #print gtImages.getImageName(idx).split("/")[-1].split(".")[0]
            dimg[gtImages[idx]["name"].split("/")[-1].split(".")[0]]={"bbox":rect,"det":[False]*len(rect)}
            for i, recti in enumerate(rect):
                if recti[5] == 0:
                    tot=tot+1

    imname=[]
    cnt=0
    tp=numpy.zeros(len(detlist))
    fp=numpy.zeros(len(detlist))
    thr=numpy.zeros(len(detlist))
    detlist.sort(cmpscore)
    for idx,detbb in enumerate(detlist):
        #print detbb[1]
        found=False
        maxovr=0
        #gtdet=[False]
        gt=0
        if dimg.has_key(detbb[0]):
            rect=dimg[detbb[0]]["bbox"]
            found=False
            for ir,r in enumerate(rect):
                #gtdet.append(False)
                rb=(float(detbb[3]),float(detbb[2]),float(detbb[5]),float(detbb[4]))
                #print "GT:",r
                #print "DET:",rb
                if pixels==None:
                    covr=overlap(rb,r)
                else:
                    covr=overlapx(rb,r,pixels)
                if covr>=maxovr:
                    maxovr=covr
                    gt=ir
                    #dimg[detbb[0]].remove(r)
                    #found=True
                    #break

        if maxovr>ovr:
            if dimg[detbb[0]]["bbox"][gt][5] == 0:
                if not(dimg[detbb[0]]["det"][gt]):
                    tp[idx]=1
                    dimg[detbb[0]]["det"][gt]=True
                    posd.append(detbb[1])
                else:
                    fp[idx]=1
        else:
            fp[idx]=1

########### PASCAL 2010
#    if ovmax>=VOCopts.minoverlap
#        if ~gt(i).diff(jmax)
#            if ~gt(i).det(jmax)
#                tp(d)=1;            % true positive
#		        gt(i).det(jmax)=true;
#            else
#                fp(d)=1;            % false positive (multiple detection)
#            end
#        end
#    else
#        fp(d)=1;                    % false positive
#    end
########################



        thr[idx]=detbb[1]
        if show:
            prec=numpy.sum(tp)/float(numpy.sum(tp)+numpy.sum(fp))
            rec=numpy.sum(tp)/tot
            print "Scr:",detbb[1],"Prec:%.3f"%prec,"Rec:%.3f"%rec
            ss=raw_input()
            if ss=="s" or not(found):
                pylab.ioff()
                img=gtImages.getImageByName2(detbb[0])
                pylab.figure(1)
                pylab.clf()
                pylab.imshow(img)
                rb=(float(detbb[3]),float(detbb[2]),float(detbb[5]),float(detbb[4]))
                for r in rect:
                    pylab.figure(1)
                    pylab.ioff()
                    box(r[0],r[1],r[2],r[3],'b',lw=1.5)
                if found:
                    box(rb[0],rb[1],rb[2],rb[3],'g',lw=1.5)
                else:
                    box(rb[0],rb[1],rb[2],rb[3],'r',lw=1.5)
                pylab.draw()
                pylab.show()
                rect=[]

    return tp,fp,thr,tot,posd


def VOCanalysis(gtImages,detlist,show=False,usetr=True,usedf=False,ovr=0.5):
    """
        calculate the precision recall curve
    """
    dimg={}
    tot=0
    for idx in range(len(gtImages)):
        rect=gtImages[idx]["bbox"][:]
        #if idx>288:
        #    print idx,rect
        if rect!=[]:
            #print gtImages.getImageName(idx).split("/")[-1].split(".")[0]
            dimg[gtImages[idx]["name"].split("/")[-1].split(".")[0]]={"bbox":rect,"det":[False]*len(rect)}
        tot=tot+len(rect)
    imname=[]
    cnt=0
    tp=numpy.zeros(len(detlist))
    fp=numpy.zeros(len(detlist))
    thr=numpy.zeros(len(detlist))

    tplist=[]
    fplist=[]
    fp2list=[]
    fnlist=[]

    detlist.sort(cmpscore)
    for idx,detbb in enumerate(detlist):
        #print detbb[1]
        found=False
        maxovr=0
        #gtdet=[False]
        gt=0
        if dimg.has_key(detbb[0]):
            rect=dimg[detbb[0]]["bbox"]
            found=False
            for ir,r in enumerate(rect):
                #gtdet.append(False)
                rb=(float(detbb[3]),float(detbb[2]),float(detbb[5]),float(detbb[4]))
                #print "GT:",r
                #print "DET:",rb
                covr=overlap(rb,r)
                if covr>=maxovr:
                    maxovr=covr
                    gt=ir
                    #dimg[detbb[0]].remove(r)
                    #found=True
                    #break
        if maxovr>ovr:
            if not(dimg[detbb[0]]["det"][gt]):
                tp[idx]=1
                dimg[detbb[0]]["det"][gt]=True
                tplist.append(detbb)
            else:
                fp[idx]=1
                fplist.append(detbb)
        else:
            fp[idx]=1
            fp2list.append(detbb)

    totalDetected  =0
    totalnoDetected=0

    for idx in range(len(gtImages)):
        rect=gtImages[idx]["bbox"][:]
        if rect!=[]:
            name = gtImages[idx]["name"].split("/")[-1].split(".")[0]
            bboxgt = dimg[name]
            for i in range(len(bboxgt["det"])):
                if bboxgt["det"][i]:
                    #bbox FOUND, it's ok
                    totalDetected += 1
                else:
                    #bbox not FOUND, add to FN
                    gtbb = [name,0,bboxgt["bbox"][i][0:4]]
                    fnlist.append(gtbb)
                    totalnoDetected += 1


    print "total Detected %d, total no Detected %d"%(totalDetected,totalnoDetected)

    #tplist.sort(key=lambda det: -det[1])
    #fplist.sort(key=lambda det: -det[1])
    #fnlist.sort(key=lambda det: -det[1])

    return tplist,fplist,fp2list,fnlist

def VOCprRecord_wrong(gtImages,detlist,show=False,usetr=True,usedf=False,ovr=0.5):
    """
        calculate the precision recall curve
    """
    dimg={}
    tot=0
    for idx in range(len(gtImages)):
        rect=gtImages[idx]["bbox"][:]
        #if idx>288:
        #    print idx,rect
        if rect!=[]:
            #print gtImages.getImageName(idx).split("/")[-1].split(".")[0]
            dimg[gtImages[idx]["name"].split("/")[-1].split(".")[0]]={"bbox":rect,"det":[False]*len(rect)}
        tot=tot+len(rect)
    imname=[]
    cnt=0
    tp=numpy.zeros(len(detlist))
    fp=numpy.zeros(len(detlist))
    thr=numpy.zeros(len(detlist))
    detlist.sort(cmpscore)
    for idx,detbb in enumerate(detlist):
        #print detbb[1]
        found=False
        maxovr=0
        #gtdet=[False]
        gt=0
        if dimg.has_key(detbb[0]):
            rect=dimg[detbb[0]]["bbox"]
            found=False
            for ir,r in enumerate(rect):
                #gtdet.append(False)
                rb=(float(detbb[3]),float(detbb[2]),float(detbb[5]),float(detbb[4]))
                #print "GT:",r
                #print "DET:",rb
                covr=overlap(rb,r)
                if covr>=maxovr:
                    maxovr=covr
                    gt=ir
                    #dimg[detbb[0]].remove(r)
                    #found=True
                    #break
        if maxovr>ovr:
            #if not(dimg[detbb[0]]["det"][gt]):
            tp[idx]=1
            #dimg[detbb[0]]["det"][gt]=True
            #else:
            #    fp[idx]=1
        else:
            fp[idx]=1
        thr[idx]=detbb[1]
        if show:
            prec=numpy.sum(tp)/float(numpy.sum(tp)+numpy.sum(fp))
            rec=numpy.sum(tp)/tot
            print "Scr:",detbb[1],"Prec:%.3f"%prec,"Rec:%.3f"%rec
            ss=raw_input()
            if ss=="s" or not(found):
                pylab.ioff()
                img=gtImages.getImageByName2(detbb[0])
                pylab.figure(1)
                pylab.clf()
                pylab.imshow(img)
                rb=(float(detbb[3]),float(detbb[2]),float(detbb[5]),float(detbb[4]))
                for r in rect:
                    pylab.figure(1)
                    pylab.ioff()
                    box(r[0],r[1],r[2],r[3],'b',lw=1.5)
                if found:
                    box(rb[0],rb[1],rb[2],rb[3],'g',lw=1.5)
                else:
                    box(rb[0],rb[1],rb[2],rb[3],'r',lw=1.5)
                pylab.draw()
                pylab.show()
                rect=[]

    return tp,fp,thr,tot

def viewSortDet(gtImages,detlist,numim=numpy.inf,opt="all",usetr=True,usedf=False,ovr=0.5):
    dimg={}
    tot=0
    for idx in range(min(gtImages.getTotal(),numim)):
        rect=gtImages.getBBox(idx)
        if rect!=[]:
            #print gtImages.getImageName(idx).split("/")[-1].split(".")[0]
            dimg[gtImages.getImageName(idx).split("/")[-1].split(".")[0]]=rect
        tot=tot+len(rect)
    imname=[]
    cnt=0
    tp=numpy.zeros(len(detlist))
    fp=numpy.zeros(len(detlist))
    thr=numpy.zeros(len(detlist))
    detlist.sort(cmpscore)
    for idx,detbb in enumerate(detlist):
        #print detbb[1]
        found=False
        if dimg.has_key(detbb[0]):
            rect=dimg[detbb[0]]
            found=False
            for r in rect:
                rb=(float(detbb[3]),float(detbb[2]),float(detbb[5]),float(detbb[4]))
                #print "GT:",r
                #print "DET:",rb
                if overlap(rb,r)>=ovr:
                    dimg[detbb[0]].remove(r)
                    found=True
                    break
        if found:  
            tp[idx]=1
        else:
            fp[idx]=1
        thr[idx]=detbb[1]
        if show:
            pylab.ioff()
            prec=numpy.sum(tp)/float(numpy.sum(tp)+numpy.sum(fp))
            rec=numpy.sum(tp)/tot
            print "Scr:",detbb[1],"Prec:",prec,"Rec:",rec
            img=gtImages.getImageByName2(detbb[0])
            pylab.figure(1)
            pylab.clf()
            pylab.imshow(img)
            rb=(float(detbb[3]),float(detbb[2]),float(detbb[5]),float(detbb[4]))
            for r in rect:
                pylab.figure(1)
                pylab.ioff()
                box(r[0],r[1],r[2],r[3],'b',lw=1.5)
            if found:
                box(rb[0],rb[1],rb[2],rb[3],'g',lw=1.5)
            else:
                box(rb[0],rb[1],rb[2],rb[3],'r',lw=1.5)
            pylab.draw()
            pylab.show()
            rect=[]
            raw_input()

    return tp,fp,thr,tot
    


def viewDet(gtImages,detfile,opt="all",usetr=True,usedf=False,stop=True,t=0.5):
    detf=open(detfile,"r")
    detect=detf.readlines()
    detlst=numpy.zeros((len(detect),5))
    namelst=[]
    pylab.ioff()
    for id,el in enumerate(detect):
        aux=el.split()
        namelst.append(aux[0])
        detlst[id,:]=aux[1:]
    srt=numpy.argsort(-detlst[:,0])
    imname=[]
    cnt=0
    ovr=0.49
    #print trPosImages.getTotal()
    tp=[]
    fp=[]
    tot=0
    pylab.figure()
    bb=numpy.zeros((4))
    for id in range(detlst.shape[0]):
        pylab.ioff()
        abb=detlst[srt[id]]
        conf=abb[0]
        bb[0]=abb[2];bb[1]=abb[1];bb[2]=abb[4];bb[3]=abb[3]
        pylab.clf()
        img=gtImages.getImageByName2(namelst[srt[id]])
        gtbb=gtImages.getBBoxByName(namelst[srt[id]],usetr=usetr,usedf=usedf)
        found=False
        for l in range(len(gtbb)):
            pylab.imshow(img)
            pylab.title("%s Confidence: %f"%(namelst[srt[id]],float(conf)))
            #box(gtbb[l][0],gtbb[l][1],gtbb[l][2],gtbb[l][3],col='b',lw="2")
            print overlap(bb[:],gtbb[l][:4])
            if overlap(bb[:],gtbb[l][:4])>0:
                if overlap(bb[:],gtbb[l][:4])>ovr:
                    box(gtbb[l][0],gtbb[l][1],gtbb[l][2],gtbb[l][3],col='y',lw="2")
                    box(bb[0],bb[1],bb[2],bb[3],col='g',lw="2")
                    pylab.show()
                    pylab.draw()
                    if stop:
                        raw_input()
                    else:
                        time.sleep(t)
                    found=True
                else:
                    box(gtbb[l][0],gtbb[l][1],gtbb[l][2],gtbb[l][3],col='y',lw="1")
                    #box(bb[0],bb[1],bb[2],bb[3],col='g',lw="2")
                    #raw_input()
            else:
                pass
                #pylab.imshow(img)
                #box(bb[0],bb[1],bb[2],bb[3],col='r',lw="2")
        if not(found):
            pylab.imshow(img)
            box(bb[0],bb[1],bb[2],bb[3],col='r',lw="2")
            pylab.show()
            pylab.draw()
            if stop:
                raw_input()
            else:
                time.sleep(t)

def VOCap(rec,prec):
    mrec=numpy.concatenate(([0],rec,[1]))
    mpre=numpy.concatenate(([0],prec,[0]))
    for i in range(len(mpre)-2,0,-1):
        mpre[i]=max(mpre[i],mpre[i+1]);
    i=numpy.where(mrec[1:]!=mrec[0:-1])[0]+1;
    ap=numpy.sum((mrec[i]-mrec[i-1])*mpre[i]);
    return ap

def VOColdap(rec,prec):
    rec=numpy.array(rec)
    prec=numpy.array(prec)
    ap=0.0
    for t in numpy.linspace(0,1,11):
        pr=prec[rec>=t]
        if pr.size==0:
            pr=0
        p=numpy.max(pr);
        print pr,p,t
        ap=ap+p/11.0;
    return ap

def drawPrfast(tp,fp,tot,show=True,col="g"):
    tp=numpy.cumsum(tp)
    fp=numpy.cumsum(fp)
    rec=tp/tot
    prec=tp/(fp+tp)
    ap=0
    for t in numpy.linspace(0,1,11):
        pr=prec[rec>=t]
        if pr.size==0:
            pr=0
        p=numpy.max(pr);
        ap=ap+p/11;
    ap1=VOCap(rec,prec)
    if show:    
        pylab.plot(rec,prec,'-%s'%col)
        pylab.title("AP=%.1f 11pt(%.1f)"%(ap1*100,ap*100))
        pylab.xlabel("Recall")
        pylab.ylabel("Precision")
        pylab.grid()
        pylab.gca().set_xlim((0,1))
        pylab.gca().set_ylim((0,1))
        pylab.show()
        pylab.draw()
    return rec,prec,ap

def drawPrfastscore(tp,fp,scr,tot,show=True):
    tp=numpy.cumsum(tp)
    fp=numpy.cumsum(fp)
    rec=tp/tot
    prec=tp/(fp+tp)
    #dif=numpy.abs(prec[1:]-rec[1:])
    dif=numpy.abs(prec[::-1]-rec[::-1])
    pos=dif.argmin()
    pos=len(dif)-pos-1
    ap=0
    for t in numpy.linspace(0,1,11):
        pr=prec[rec>=t]
        if pr.size==0:
            pr=0
        p=numpy.max(pr);
        ap=ap+p/11;
    if show:    
        pylab.plot(rec,prec,'-g')
        pylab.title("AP=%.3f EPRthr=%.3f"%(ap,scr[pos]))
        pylab.xlabel("Recall")
        pylab.ylabel("Precision")
        pylab.grid()
        pylab.show()
        pylab.draw()
    return rec,prec,scr,ap,scr[pos]


def drawMissRatePerImage(tp,fp,tot,nimg,show=True):
    tp=numpy.cumsum(tp)
    fp=numpy.cumsum(fp)
    miss=1-tp/tot
    fppi=fp/nimg
    ap=0
#    for t in numpy.linspace(0,1,11):
#        pr=prec[rec>=t]
#        if pr.size==0:
#            pr=0
#        p=numpy.max(pr);
#        ap=ap+p/11;
    if show:    
        pylab.plot(fppi,miss,'-g')
        pylab.title("AP=%.3f"%(ap))
        pylab.xlabel("False Positive Per Image")
        pylab.ylabel("Miss Rate")
        #pylab.loglog()
        pylab.semilogx()
        pylab.grid()
        pylab.show()
        pylab.draw()
    return fppi,miss,ap


def drawPr(tp,fp,tot,show=True,col="g"):
    """
        draw the precision recall curve
    """
    det=numpy.array(sorted(tp+fp))
    atp=numpy.array(tp)
    afp=numpy.array(fp)
    #pylab.figure()
    #pylab.clf()
    rc=numpy.zeros(len(det))
    pr=numpy.zeros(len(det))
    #prc=0
    #ppr=1
    for i,p in enumerate(det):
        pr[i]=float(numpy.sum(atp>=p))/numpy.sum(det>=p)
        rc[i]=float(numpy.sum(atp>=p))/tot
        #print pr,rc,p
    ap=0
    for c in numpy.linspace(0,1,num=11):
        if len(pr[rc>=c])>0:
            p=numpy.max(pr[rc>=c])
        else:
            p=0
        ap=ap+p/11
    if show:
        pylab.plot(rc,pr,'-%s'%col)
        pylab.title("AP=%.3f"%(ap))
        pylab.xlabel("Recall")
        pylab.ylabel("Precision")
        pylab.grid()
        pylab.show()
        pylab.draw()
    return rc,pr,ap
        

if __name__ == '__main__':
    gtImages=VOC06Data(select="all",trainfile="/media/DADES-2/VOC2006/VOCdevkit/VOC2006/ImageSets/person_val2.txt",
                        imagepath="/media/DADES-2/VOC2006/VOCdevkit/VOC2006/PNGImages/",
                        annpath="/media/DADES-2/VOC2006/VOCdevkit/VOC2006/Annotations/")
    interv=5
    dimy=10
    dimx=4
    trpos=1000
    trneg=10000
    #tname="./save/fast2_noflip_10x4_p1000_n10000_VOC06"%(dimy,dimx,trpos,trneg) #good score 0.1 in validation
    tname='./save/fast2_noflip_%dx%d_p%d_n%d_VOC06'%(dimy,dimx,trpos,trneg)
    #train1=tname+".svm"
    #w,rho=MPFutil.loadSvm(train1,dir="")
    #ww=w.reshape((dimy,dimx,20))
    #drawer=doitDraw(interv,dimy,dimx)
    #detect=doitDetections()
    detfile=tname+"_fast.txt"
    #fd=open(,"w")
    tp,fp,tot=VOCpr(gtImages,detfile)
    pylab.figure()
    rc,pr,ap=drawPr(tp,fp,tot)
    pylab.title(tname.split("/")[-1]+" ap=%f"%(ap))
    pylab.show()
    pylab.draw()
                       
##for idx in range(PosImages.getTotal()):
##    detect.reset()
##    img=PosImages.getImage(idx)
##    feat=HOGdetect(img,version="normal",interv=interv)
##    #feat.pad()
##    feat.contrast()
##    res=Detections(feat.scan(ww,rho,minlev=1))
##    pylab.figure(1)
##    pylab.clf()
##    pylab.imshow(img)
##    #res.foreach(drawer,thr=1.0)
##    res.foreachImage((dimy,dimx),detect,thr=0.0)
##    ldet=detect.clusterize()
##    for i in ldet:
##        #ii=i[0]+0.2*abs(i[0]-i[2]),i[1]+0.2*abs(i[1]-i[3]),i[2]-0.2*abs(i[0]-i[2]),i[3]-0.2*abs(i[1]-i[3])
##        ii=i
##        fd.write("%s %f %f %f %f %f\n"%(PosImages.getImageName(idx).split("/")[-1].split(".")[0],i[4],ii[1],ii[0],ii[3],ii[2]))
##    #print ldet
##    detect.show()
##    pylab.show()
##    pylab.draw()
##    print idx,"/",PosImages.getTotal()
##    #raw_input()
##fd.close()
