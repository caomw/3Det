import numpy
import ctypes
from ctypes import c_float,c_double,c_int,c_void_p,POINTER,pointer
import pylab

#ctypes.cdll.LoadLibrary("./libfastpegasos.so")
lpeg= ctypes.CDLL("./libfastpegasos2.so")

#void fast_objective_new(ftype *w,int numcomp,int *compx,int *compy,ftype **ptrsamplescomp,int totsamples,int *label,int *comp,ftype C,int iter,int part,int k,int numthr,ftype *reg,ftype *zero,ftype *ret)
#lpeg.fast_grad_new.argtypes=[
#    numpy.ctypeslib.ndpointer(dtype=c_double,ndim=1,flags="C_CONTIGUOUS")#w
#    ,c_int #numcomp
#    ,POINTER(c_int) #compx
#    ,POINTER(c_int) #compy
#    ,POINTER(c_void_p) #ptrsamples
#    ,c_int #numsamples
#    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS") #label
#    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS") #components
#    ,c_double #lambda
#    ,c_int #k
#    ,c_int #numthr
#    ,numpy.ctypeslib.ndpointer(dtype=c_double,ndim=1,flags="C_CONTIGUOUS") #components #reg
#    ,numpy.ctypeslib.ndpointer(dtype=c_double,ndim=1,flags="C_CONTIGUOUS") #components #zero
#    ,numpy.ctypeslib.ndpointer(dtype=c_double,ndim=1,flags="C_CONTIGUOUS") #return
#    ]


#void fast_objective_new(ftype *w,int numcomp,int *compx,int *compy,ftype **ptrsamplescomp,int totsamples,int *label,int *comp,ftype C,int iter,int part,int k,int numthr,ftype *reg,ftype *zero,ftype *ret)
#lpeg.fast_objective_new.argtypes=[
#    numpy.ctypeslib.ndpointer(dtype=c_double,ndim=1,flags="C_CONTIGUOUS")#w
#    ,c_int #numcomp
#    ,POINTER(c_int) #compx
#    ,POINTER(c_int) #compy
#    ,POINTER(c_void_p) #ptrsamples
#    ,c_int #numsamples
#    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS") #label
#    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS") #components
#    ,c_double #lambda
#    ,c_int #k
#    ,c_int #numthr
#    ,numpy.ctypeslib.ndpointer(dtype=c_double,ndim=1,flags="C_CONTIGUOUS") #components #reg
#    ,numpy.ctypeslib.ndpointer(dtype=c_double,ndim=1,flags="C_CONTIGUOUS") #components #zero
#    ,numpy.ctypeslib.ndpointer(dtype=c_double,ndim=1,flags="C_CONTIGUOUS") #return
#    ]

#lpeg.fast_objgrad_new.argtypes=[
#    numpy.ctypeslib.ndpointer(dtype=c_double,ndim=1,flags="C_CONTIGUOUS")#w
#    ,c_int #numcomp
#    ,POINTER(c_int) #compx
#    ,POINTER(c_int) #compy
#    ,POINTER(c_void_p) #ptrsamples
#    ,c_int #numsamples
#    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS") #label
#    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS") #components
#    ,c_double #lambda
#    ,c_int #k
#    ,c_int #numthr
#    ,numpy.ctypeslib.ndpointer(dtype=c_double,ndim=1,flags="C_CONTIGUOUS") #components #reg
#    ,numpy.ctypeslib.ndpointer(dtype=c_double,ndim=1,flags="C_CONTIGUOUS") #components #zero
#    ,numpy.ctypeslib.ndpointer(dtype=c_double,ndim=1,flags="C_CONTIGUOUS") #return
#    ,numpy.ctypeslib.ndpointer(dtype=c_double,ndim=1,flags="C_CONTIGUOUS") #grad
#    ]


#lpeg.fast_objgrad_parall_new.argtypes=[
#    numpy.ctypeslib.ndpointer(dtype=c_double,ndim=1,flags="C_CONTIGUOUS")#w
#    ,c_int #numcomp
#    ,POINTER(c_int) #compx
#    ,POINTER(c_int) #compy
#    ,POINTER(c_void_p) #ptrsamples
#    ,c_int #numsamples
#    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS") #label
#    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS") #components
#    ,c_double #lambda
#    ,c_int #k
#    ,c_int #numthr
#    ,numpy.ctypeslib.ndpointer(dtype=c_double,ndim=1,flags="C_CONTIGUOUS") #components #reg
#    ,numpy.ctypeslib.ndpointer(dtype=c_double,ndim=1,flags="C_CONTIGUOUS") #components #zero
#    ,numpy.ctypeslib.ndpointer(dtype=c_double,ndim=1,flags="C_CONTIGUOUS") #return
#    ,numpy.ctypeslib.ndpointer(dtype=c_double,ndim=1,flags="C_CONTIGUOUS") #grad
#    ]

lpeg.fast_objgrad_parall_float.argtypes=[
    numpy.ctypeslib.ndpointer(dtype=c_double,ndim=1,flags="C_CONTIGUOUS")#w
    ,c_int #numcomp
    ,POINTER(c_int) #compx
    ,POINTER(c_int) #compy
    ,POINTER(c_void_p) #ptrsamples
    ,c_int #numsamples
    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS") #label
    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS") #components
    ,c_double #lambda
    ,c_int #k
    ,c_int #numthr
    ,numpy.ctypeslib.ndpointer(dtype=c_double,ndim=1,flags="C_CONTIGUOUS") #components #reg
    ,numpy.ctypeslib.ndpointer(dtype=c_double,ndim=1,flags="C_CONTIGUOUS") #components #zero
    ,numpy.ctypeslib.ndpointer(dtype=c_double,ndim=1,flags="C_CONTIGUOUS") #return
    ,numpy.ctypeslib.ndpointer(dtype=c_double,ndim=1,flags="C_CONTIGUOUS") #grad
    ]

lpeg.fast_objgrad_float.argtypes=[
    numpy.ctypeslib.ndpointer(dtype=c_double,ndim=1,flags="C_CONTIGUOUS")#w
    ,c_int #numcomp
    ,POINTER(c_int) #compx
    ,POINTER(c_int) #compy
    ,POINTER(c_void_p) #ptrsamples
    ,c_int #numsamples
    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS") #label
    ,numpy.ctypeslib.ndpointer(dtype=c_int,ndim=1,flags="C_CONTIGUOUS") #components
    ,c_double #lambda
    ,c_int #k
    ,c_int #numthr
    ,numpy.ctypeslib.ndpointer(dtype=c_double,ndim=1,flags="C_CONTIGUOUS") #components #reg
    ,numpy.ctypeslib.ndpointer(dtype=c_double,ndim=1,flags="C_CONTIGUOUS") #components #zero
    ,numpy.ctypeslib.ndpointer(dtype=c_double,ndim=1,flags="C_CONTIGUOUS") #return
    ,numpy.ctypeslib.ndpointer(dtype=c_double,ndim=1,flags="C_CONTIGUOUS") #grad
    ]

#def obj_new(w,numcomp,compx,compy,ptrsamples,ntimes,label,components,c,k,numthr,reg,zero,grad):
#    ret=numpy.zeros((3),dtype=numpy.float64)
#    lpeg.fast_objective_new(w,numcomp,compx,compy,ptrsamples,ntimes,label,components,c,k,numthr,reg,zero,ret)
#    if 0:
#        pylab.figure(100)
#        pylab.plot(w)
#        pylab.show()
#        pylab.draw()
#        raw_input()
#    return ret.sum()

#def grad_new(w,numcomp,compx,compy,ptrsamples,ntimes,label,components,c,k,numthr,reg,zero,grad):
#    lpeg.fast_grad_new(w,numcomp,compx,compy,ptrsamples,ntimes,label,components,c,k,numthr,reg,zero,grad)
#    #grad=grad*10000000000
#    #print grad
#    if 0:
#        pylab.figure(100)
#        pylab.plot(grad)
#        pylab.show()
#        pylab.draw()
#        raw_input()
#    grad=grad
#    return grad

def debug_objgrad_float(w,numcomp,compx,compy,ptrsamples,ntimes,label,components,c,k,numthr,reg,zero,grad):
    import time
    ret=numpy.zeros((3),dtype=numpy.float64)
    t0=time.time()
    lpeg.fast_objgrad_float(w,numcomp,compx,compy,ptrsamples,ntimes,label,components,c,k,numthr,reg,zero,ret,grad)
    t1=time.time()-t0
    ret2=numpy.zeros((3),dtype=numpy.float64)
    grad2=numpy.zeros(grad.shape,grad.dtype)
    t0=time.time()
    lpeg.fast_objgrad_parall_float(w,numcomp,compx,compy,ptrsamples,ntimes,label,components,c,k,numthr,reg,zero,ret2,grad2)
    t2=time.time()-t0
    print "Normal Obj:%s Time:%f"%(ret.sum(),t1)
    print "Parallel Obj:%s Time:%f"%(ret2.sum(),t2)
    if abs(ret.sum()-ret2.sum())>0.00001:
        print "Error, different value"
        raw_input()
    if 0:
        pylab.figure(100)
        pylab.plot(grad)
        pylab.show()
        pylab.draw()
        raw_input()
    return ret.sum(),grad

#def objgrad_new(w,numcomp,compx,compy,ptrsamples,ntimes,label,components,c,k,numthr,reg,zero,grad):
#    ret=numpy.zeros((3),dtype=numpy.float64)
#    #lpeg.fast_objgrad_new(w,numcomp,compx,compy,ptrsamples,ntimes,label,components,c,k,numthr,reg,zero,ret,grad)
#    lpeg.fast_objgrad_parall_new(w,numcomp,compx,compy,ptrsamples,ntimes,label,components,c,k,numthr,reg,zero,ret,grad)
#    if 0:
#        pylab.figure(100)
#        pylab.plot(grad)
#        pylab.show()
#        pylab.draw()
#        raw_input()
#    return ret.sum(),grad

def objgrad_float(w,numcomp,compx,compy,ptrsamples,ntimes,label,components,c,k,numthr,reg,zero,grad):
    ret=numpy.zeros((3),dtype=numpy.float64)
    #lpeg.fast_objgrad_new(w,numcomp,compx,compy,ptrsamples,ntimes,label,components,c,k,numthr,reg,zero,ret,grad)
    lpeg.fast_objgrad_parall_float(w,numcomp,compx,compy,ptrsamples,ntimes,label,components,c,k,numthr,reg,zero,ret,grad)
    if 0:
        pylab.figure(100)
        pylab.plot(grad)
        pylab.show()
        pylab.draw()
        raw_input()
    return ret.sum(),grad

from scipy.optimize import fmin_l_bfgs_b

def trainCompLBGS_new(trpos,trneg,fname="",trposcl=None,trnegcl=None,oldw=None,dir="./save/",pc=0.017,path="/home/marcopede/code/c/liblinear-1.7",mintimes=30,maxtimes=200,eps=0.001,num_stop_count=5,numthr=1,k=1,regvec=[],zerovec=[],mulvec=[],limitvec=[]):
    """
        The same as trainSVMRaw but it does use files instad of lists:
        it is slower but it needs less memory.
    """
    #ff=open(fname,"a")
    if trposcl==None:
        trposcl=numpy.zeros(len(trpos),dtype=numpy.int)
    if trnegcl==None:
        trnegcl=numpy.zeros(len(trneg),dtype=numpy.int)
    numcomp=numpy.array(trposcl).max()+1
    trcomp=[]
    newtrcomp=[]
    cregvec=[]
    czerovec=[]
    cmulvec=[]
    climitvec=[]
    trcompcl=[]
    alabel=[]
    label=[]
    for l in range(numcomp):
        trcomp.append([])#*numcomp
        label.append([])
    #trcomp=[trcomp]
    #trcompcl=[]
    #label=[[]]*numcomp
    compx=[0]*numcomp
    compy=[0]*numcomp
    for l in range(numcomp):
        compx[l]=len(trpos[numpy.where(numpy.array(trposcl)==l)[0][0]])#+1
    for p,val in enumerate(trposcl):
        trcomp[val].append(trpos[p].astype(c_double))
        #trcomp[val].append(numpy.concatenate((trpos[p],[bias])).astype(c_float))
        #trcompcl.append(val)
        label[val].append(1)
        compy[val]+=1
    for p,val in enumerate(trnegcl):
        trcomp[val].append(trneg[p].astype(c_double))        
        #trcomp[val].append(numpy.concatenate((trneg[p],[bias])).astype(c_float))        
        #trcompcl.append(val)
        label[val].append(-1)
        compy[val]+=1
    ntimes=len(trpos)+len(trneg)
    fdim=numpy.sum(compx)#len(posnfeat[0])+1
    #bigm=numpy.zeros((ntimes,fdim),dtype=numpy.float32)
    w=numpy.zeros(fdim,dtype=c_double)
    if oldw!=None:
        w=oldw.astype(c_double)
        #w[:-1]=oldw
    #for l in range(posntimes):
    #    bigm[l,:-1]=posnfeat[l]
    #    bigm[l,-1]=bias
    #for l in range(negntimes):
    #    bigm[posntimes+l,:-1]=negnfeat[l]
    #    bigm[posntimes+l,-1]=bias
    #labels=numpy.concatenate((numpy.ones(posntimes),-numpy.ones(negntimes))).astype(numpy.float32)
    for l in range(numcomp):
        trcompcl=numpy.concatenate((trcompcl,numpy.ones(compy[l],dtype=numpy.int32)*l))
        alabel=numpy.concatenate((alabel,numpy.array(label[l])))
    trcompcl=trcompcl.astype(numpy.int32)
    alabel=alabel.astype(numpy.int32)
    arrint=(c_int*numcomp)
    arrfloat=(c_void_p*numcomp)
    #trcomp1=[list()]*numcomp
    for l in range(numcomp):#convert to array
        trcomp[l]=numpy.array(trcomp[l])
        newtrcomp.append(trcomp[l].ctypes.data_as(c_void_p))
        cregvec=numpy.concatenate((cregvec,regvec[l])).astype(numpy.float64)
        czerovec=numpy.concatenate((czerovec,zerovec[l])).astype(numpy.float64)
        cmulvec=numpy.concatenate((cmulvec,mulvec[l])).astype(numpy.float64)
        climitvec=numpy.concatenate((climitvec,limitvec[l])).astype(numpy.float64)
        #cregvec.append(regvec[l].ctypes.data_as(c_void_p))
        #czerovec.append(zerovec[l].ctypes.data_as(c_void_p))
        #cmulvec.append(mulvec[l].ctypes.data_as(c_void_p))
        #climitvec.append(limitvec[l].ctypes.data_as(c_void_p))
    print "Clusters size:",compx
    print "Clusters elements:",compy
    print "Starting Pegasos SVM training"
    obj=0.0
    ncomp=c_int(numcomp)
    
    bounds=numpy.zeros((2,len(w)),dtype=w.dtype)
    bounds[0]=limitvec[0]
    bounds[1]=1000
    bounds=list(bounds.T)    
    
    grad=numpy.zeros(w.size,w.dtype)
    w,fmin,dd=fmin_l_bfgs_b(objgrad_new,w,None,(ncomp,arrint(*compx),arrint(*compy),arrfloat(*newtrcomp),ntimes,alabel,trcompcl,pc,k,numthr,cregvec,czerovec,grad),iprint=1,factr=100000000,maxfun=1000,maxiter=1000,bounds=bounds)
    #w,fmin,dd=fmin_l_bfgs_b(debug_objgrad_new,w,None,(ncomp,arrint(*compx),arrint(*compy),arrfloat(*newtrcomp),ntimes,alabel,trcompcl,pc,k,numthr,cregvec,czerovec,grad),iprint=1,factr=100000000,maxfun=1000,maxiter=1000,bounds=bounds)
    #w,fmin,dd=fmin_l_bfgs_b(obj_new,w,grad_new,(ncomp,arrint(*compx),arrint(*compy),arrfloat(*newtrcomp),ntimes,alabel,trcompcl,pc,k,numthr,cregvec,czerovec,grad),iprint=1,factr=100000000,maxfun=1000,maxiter=1000,bounds=bounds)
    return w,0,fmin

def trainCompLBGS_float(trpos,trneg,fname="",trposcl=None,trnegcl=None,oldw=None,dir="./save/",pc=0.017,path="/home/marcopede/code/c/liblinear-1.7",mintimes=30,maxtimes=200,eps=0.001,num_stop_count=5,numthr=1,k=1,regvec=[],zerovec=[],mulvec=[],limitvec=[]):
    """
        The same as trainSVMRaw but it does use files instad of lists:
        it is slower but it needs less memory.
    """
    #ff=open(fname,"a")
    if trposcl==None:
        trposcl=numpy.zeros(len(trpos),dtype=numpy.int)
    if trnegcl==None:
        trnegcl=numpy.zeros(len(trneg),dtype=numpy.int)
    numcomp=numpy.array(trposcl).max()+1
    trcomp=[]
    newtrcomp=[]
    cregvec=[]
    czerovec=[]
    cmulvec=[]
    climitvec=[]
    trcompcl=[]
    alabel=[]
    label=[]
    for l in range(numcomp):
        trcomp.append([])#*numcomp
        label.append([])
    #trcomp=[trcomp]
    #trcompcl=[]
    #label=[[]]*numcomp
    compx=[0]*numcomp
    compy=[0]*numcomp
    for l in range(numcomp):
        compx[l]=len(trpos[numpy.where(numpy.array(trposcl)==l)[0][0]])#+1
    for p,val in enumerate(trposcl):
        trcomp[val].append(trpos[p].astype(c_float))
        #trcomp[val].append(numpy.concatenate((trpos[p],[bias])).astype(c_float))
        #trcompcl.append(val)
        label[val].append(1)
        compy[val]+=1
    for p,val in enumerate(trnegcl):
        trcomp[val].append(trneg[p].astype(c_float))        
        #trcomp[val].append(numpy.concatenate((trneg[p],[bias])).astype(c_float))        
        #trcompcl.append(val)
        label[val].append(-1)
        compy[val]+=1
    ntimes=len(trpos)+len(trneg)
    fdim=numpy.sum(compx)#len(posnfeat[0])+1
    #bigm=numpy.zeros((ntimes,fdim),dtype=numpy.float32)
    w=numpy.zeros(fdim,dtype=c_double)
    if oldw!=None:
        w=oldw.astype(c_double)
        #w[:-1]=oldw
    #for l in range(posntimes):
    #    bigm[l,:-1]=posnfeat[l]
    #    bigm[l,-1]=bias
    #for l in range(negntimes):
    #    bigm[posntimes+l,:-1]=negnfeat[l]
    #    bigm[posntimes+l,-1]=bias
    #labels=numpy.concatenate((numpy.ones(posntimes),-numpy.ones(negntimes))).astype(numpy.float32)
    for l in range(numcomp):
        trcompcl=numpy.concatenate((trcompcl,numpy.ones(compy[l],dtype=numpy.int32)*l))
        alabel=numpy.concatenate((alabel,numpy.array(label[l])))
    trcompcl=trcompcl.astype(numpy.int32)
    alabel=alabel.astype(numpy.int32)
    arrint=(c_int*numcomp)
    arrfloat=(c_void_p*numcomp)
    #trcomp1=[list()]*numcomp
    for l in range(numcomp):#convert to array
        trcomp[l]=numpy.array(trcomp[l])
        newtrcomp.append(trcomp[l].ctypes.data_as(c_void_p))
        cregvec=numpy.concatenate((cregvec,regvec[l])).astype(numpy.float64)
        czerovec=numpy.concatenate((czerovec,zerovec[l])).astype(numpy.float64)
        #cmulvec=numpy.concatenate((cmulvec,mulvec[l])).astype(numpy.float64)
        #climitvec=numpy.concatenate((climitvec,limitvec[l])).astype(numpy.float64)
        #cregvec.append(regvec[l].ctypes.data_as(c_void_p))
        #czerovec.append(zerovec[l].ctypes.data_as(c_void_p))
        #cmulvec.append(mulvec[l].ctypes.data_as(c_void_p))
        #climitvec.append(limitvec[l].ctypes.data_as(c_void_p))
    print "Clusters size:",compx
    print "Clusters elements:",compy
    print "Starting Pegasos SVM training"
    obj=0.0
    ncomp=c_int(numcomp)
    
    bounds=numpy.zeros((2,len(w)),dtype=w.dtype)
    bounds[0]=limitvec[0]
    bounds[1]=1000
    bounds=list(bounds.T)    
    
    grad=numpy.zeros(w.size,w.dtype)
    #w,fmin,dd=fmin_l_bfgs_b(debug_objgrad_float,w,None,(ncomp,arrint(*compx),arrint(*compy),arrfloat(*newtrcomp),ntimes,alabel,trcompcl,pc,k,numthr,cregvec,czerovec,grad),iprint=1,factr=100000000,maxfun=1000,maxiter=1000,bounds=bounds)
    w,fmin,dd=fmin_l_bfgs_b(objgrad_float,w,None,(ncomp,arrint(*compx),arrint(*compy),arrfloat(*newtrcomp),ntimes,alabel,trcompcl,pc,k,numthr,cregvec,czerovec,grad),iprint=1,factr=100000000,maxfun=3000,maxiter=3000,bounds=bounds)
    #w,fmin,dd=fmin_l_bfgs_b(debug_objgrad_new,w,None,(ncomp,arrint(*compx),arrint(*compy),arrfloat(*newtrcomp),ntimes,alabel,trcompcl,pc,k,numthr,cregvec,czerovec,grad),iprint=1,factr=100000000,maxfun=1000,maxiter=1000,bounds=bounds)
    #w,fmin,dd=fmin_l_bfgs_b(obj_new,w,grad_new,(ncomp,arrint(*compx),arrint(*compy),arrfloat(*newtrcomp),ntimes,alabel,trcompcl,pc,k,numthr,cregvec,czerovec,grad),iprint=1,factr=100000000,maxfun=1000,maxiter=1000,bounds=bounds)
    return w,0,fmin


def objective_new(trpos,trneg,trposcl,trnegcl,w,pc,numthr,regvec,zerovec):
    """
        The same as trainSVMRaw but it does use files instad of lists:
        it is slower but it needs less memory.
    """
    #ff=open(fname,"a")
    if trposcl==None:
        trposcl=numpy.zeros(len(trpos),dtype=numpy.int)
    if trnegcl==None:
        trnegcl=numpy.zeros(len(trneg),dtype=numpy.int)
    numcomp=numpy.array(trposcl).max()+1
    trcomp=[]
    newtrcomp=[]
    cregvec=[]
    czerovec=[]
    cmulvec=[]
    climitvec=[]
    trcompcl=[]
    alabel=[]
    label=[]
    for l in range(numcomp):
        trcomp.append([])#*numcomp
        label.append([])
    #trcomp=[trcomp]
    #trcompcl=[]
    #label=[[]]*numcomp
    compx=[0]*numcomp
    compy=[0]*numcomp
    for l in range(numcomp):
        compx[l]=len(trpos[numpy.where(numpy.array(trposcl)==l)[0][0]])#+1
    for p,val in enumerate(trposcl):
        trcomp[val].append(trpos[p].astype(c_double))
        #trcomp[val].append(numpy.concatenate((trpos[p],[bias])).astype(c_float))
        #trcompcl.append(val)
        label[val].append(1)
        compy[val]+=1
    for p,val in enumerate(trnegcl):
        trcomp[val].append(trneg[p].astype(c_double))        
        #trcomp[val].append(numpy.concatenate((trneg[p],[bias])).astype(c_float))        
        #trcompcl.append(val)
        label[val].append(-1)
        compy[val]+=1
    ntimes=len(trpos)+len(trneg)
    fdim=numpy.sum(compx)#len(posnfeat[0])+1
    #bigm=numpy.zeros((ntimes,fdim),dtype=numpy.float32)
    w=w.astype(c_double)
    for l in range(numcomp):
        trcompcl=numpy.concatenate((trcompcl,numpy.ones(compy[l],dtype=numpy.int32)*l))
        alabel=numpy.concatenate((alabel,numpy.array(label[l])))
    trcompcl=trcompcl.astype(numpy.int32)
    alabel=alabel.astype(numpy.int32)
    arrint=(c_int*numcomp)
    arrfloat=(c_void_p*numcomp)
    #trcomp1=[list()]*numcomp
    for l in range(numcomp):#convert to array
        trcomp[l]=numpy.array(trcomp[l])
        newtrcomp.append(trcomp[l].ctypes.data_as(c_void_p))
        cregvec=numpy.concatenate((cregvec,regvec[l])).astype(numpy.float64)
        czerovec=numpy.concatenate((czerovec,zerovec[l])).astype(numpy.float64)
    print "Clusters size:",compx
    print "Clusters elements:",compy
    obj=0.0
    ncomp=c_int(numcomp)
    ret=numpy.zeros((3),dtype=numpy.float64)    
    grad=numpy.zeros(w.size,w.dtype)
    lpeg.fast_objgrad_parall_new(w,ncomp,arrint(*compx),arrint(*compy),arrfloat(*newtrcomp),ntimes,alabel,trcompcl,pc,0,numthr,cregvec,czerovec,ret,grad)#added tt+10 to not restart form scratch
    #lpeg.fast_objgrad_new(w,ncomp,arrint(*compx),arrint(*compy),arrfloat(*newtrcomp),ntimes,alabel,trcompcl,pc,0,numthr,cregvec,czerovec,ret,grad)#added tt+10 to not restart form scratch
    #lpeg.fast_objective_new(w,ncomp,arrint(*compx),arrint(*compy),arrfloat(*newtrcomp),ntimes,alabel,trcompcl,pc,0,numthr,cregvec,czerovec,ret)#added tt+10 to not restart form scratch
    posl=ret[1];negl=ret[2];reg=ret[0];nobj=ret.sum();hpos=0;hneg=0;
    return posl,negl,reg,(posl+negl)+reg,0,0

def objective_float(trpos,trneg,trposcl,trnegcl,w,pc,numthr,regvec,zerovec):
    """
        The same as trainSVMRaw but it does use files instad of lists:
        it is slower but it needs less memory.
    """
    #ff=open(fname,"a")
    if trposcl==None:
        trposcl=numpy.zeros(len(trpos),dtype=numpy.int)
    if trnegcl==None:
        trnegcl=numpy.zeros(len(trneg),dtype=numpy.int)
    numcomp=numpy.array(trposcl).max()+1
    trcomp=[]
    newtrcomp=[]
    cregvec=[]
    czerovec=[]
    cmulvec=[]
    climitvec=[]
    trcompcl=[]
    alabel=[]
    label=[]
    for l in range(numcomp):
        trcomp.append([])#*numcomp
        label.append([])
    #trcomp=[trcomp]
    #trcompcl=[]
    #label=[[]]*numcomp
    compx=[0]*numcomp
    compy=[0]*numcomp
    for l in range(numcomp):
        compx[l]=len(trpos[numpy.where(numpy.array(trposcl)==l)[0][0]])#+1
    for p,val in enumerate(trposcl):
        trcomp[val].append(trpos[p].astype(c_float))
        #trcomp[val].append(numpy.concatenate((trpos[p],[bias])).astype(c_float))
        #trcompcl.append(val)
        label[val].append(1)
        compy[val]+=1
    for p,val in enumerate(trnegcl):
        trcomp[val].append(trneg[p].astype(c_float))        
        #trcomp[val].append(numpy.concatenate((trneg[p],[bias])).astype(c_float))        
        #trcompcl.append(val)
        label[val].append(-1)
        compy[val]+=1
    ntimes=len(trpos)+len(trneg)
    fdim=numpy.sum(compx)#len(posnfeat[0])+1
    #bigm=numpy.zeros((ntimes,fdim),dtype=numpy.float32)
    w=w.astype(c_double)
    for l in range(numcomp):
        trcompcl=numpy.concatenate((trcompcl,numpy.ones(compy[l],dtype=numpy.int32)*l))
        alabel=numpy.concatenate((alabel,numpy.array(label[l])))
    trcompcl=trcompcl.astype(numpy.int32)
    alabel=alabel.astype(numpy.int32)
    arrint=(c_int*numcomp)
    arrfloat=(c_void_p*numcomp)
    #trcomp1=[list()]*numcomp
    for l in range(numcomp):#convert to array
        trcomp[l]=numpy.array(trcomp[l])
        newtrcomp.append(trcomp[l].ctypes.data_as(c_void_p))
        cregvec=numpy.concatenate((cregvec,regvec[l])).astype(numpy.float64)
        czerovec=numpy.concatenate((czerovec,zerovec[l])).astype(numpy.float64)
    print "Clusters size:",compx
    print "Clusters elements:",compy
    obj=0.0
    ncomp=c_int(numcomp)
    ret=numpy.zeros((3),dtype=numpy.float64)    
    grad=numpy.zeros(w.size,w.dtype)
    lpeg.fast_objgrad_parall_float(w,ncomp,arrint(*compx),arrint(*compy),arrfloat(*newtrcomp),ntimes,alabel,trcompcl,pc,0,numthr,cregvec,czerovec,ret,grad)#added tt+10 to not restart form scratch
    #lpeg.fast_objgrad_new(w,ncomp,arrint(*compx),arrint(*compy),arrfloat(*newtrcomp),ntimes,alabel,trcompcl,pc,0,numthr,cregvec,czerovec,ret,grad)#added tt+10 to not restart form scratch
    #lpeg.fast_objective_new(w,ncomp,arrint(*compx),arrint(*compy),arrfloat(*newtrcomp),ntimes,alabel,trcompcl,pc,0,numthr,cregvec,czerovec,ret)#added tt+10 to not restart form scratch
    posl=ret[1];negl=ret[2];reg=ret[0];nobj=ret.sum();hpos=0;hneg=0;
    return posl,negl,reg,(posl+negl)+reg,0,0

