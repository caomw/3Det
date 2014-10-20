#include <stdio.h>
#include <stdlib.h>
#include <math.h> //for sqrt

//w is the weight vector to estimate with size wx
//ex is the example matrix of size exy x wx

// gcc -fPIC -c fast_pegasos.c -O3
//gcc -shared -Wl,-soname,libfast_pegasos.so -o libfast_pegasos.so  fast_pegasos.o

#define ftype float
//#define ftype double

#define maxcomp 100
#define beta 1000.0
static double inv_beta=1.0/beta;

static inline ftype add(ftype *a,ftype *b,int len)
{
    int c;
    for (c=0;c<len;c++)
    {
        a[c]=a[c]+b[c];
    }
}

static inline ftype mul(ftype *a,ftype b,int len)
{
    int c;
    for (c=0;c<len;c++)
    {
        a[c]=a[c]*b;
    }
}

static inline ftype add_d(double *a,ftype *b,int len)
{
    int c;
    for (c=0;c<len;c++)
    {
        a[c]=a[c]+b[c];
    }
}

static inline ftype add_d2(double *a,double *b,int len)
{
    int c;
    for (c=0;c<len;c++)
    {
        a[c]=a[c]+b[c];
    }
}

static inline ftype mul_d(double *a,ftype b,int len)
{
    int c;
    for (c=0;c<len;c++)
    {
        a[c]=a[c]*b;
    }
}

static void reg(ftype *a,ftype b,ftype d,int len,int sizereg)
{
    int c;
    for (c=0;c<len-sizereg;c++)//normal part
    {
        a[c]=a[c]-a[c]*b;
    }
    for (c=len-sizereg;c<len;c++)//regularize at d instead of 0
    {
        a[c]=a[c]-(a[c]-d)*b;
        //a[c]= (a[c]<0.1*d) ? 0.1*d : a[c];//limit the minimum pairwise cost
    }
    //printf("Val:%f ",a[len-2]);
/*    for (c=0;c<len-1;c++)//not regularize bias
    {
        if (len-c-2<sizereg)
        a[c]=(a[c]-d)*b;
        else
        a[c]=a[c]*b;
    }*/
}


static void limit(ftype *a,ftype d,int len,int sizereg)
{
    int c;
    for (c=len-sizereg;c<len;c++)//regularize at d instead of 0
    {
        //a[c]=a[c]-(a[c]-d)*b;
        //if (a[c]<0.1*d)
        //    a[c]=0.1*d;
        a[c]= (a[c]<d) ? d : a[c];//limit the minimum pairwise cost
    }
    //printf("Val:%f ",a[len-2]);
/*    for (c=0;c<len-1;c++)//not regularize bias
    {
        if (len-c-2<sizereg)
        a[c]=(a[c]-d)*b;
        else
        a[c]=a[c]*b;
    }*/
}


static inline ftype addmul(ftype *a,ftype *b,ftype c,int len)
{
    int cn;
    for (cn=0;cn<len;cn++)
    {
        a[cn]=a[cn]+b[cn]*c;
    }
}

static inline ftype addmulslow(ftype *a,ftype *b,ftype c,ftype smul,int len,int sizeslow)
{
    int cn;
    for (cn=0;cn<len-sizeslow;cn++)
    {
        a[cn]=a[cn]+b[cn]*c;
    }
    for (cn=len-sizeslow;cn<len;cn++)//regularize at d instead of 0
    {
        a[cn]=a[cn]+b[cn]*c*smul;
        //a[c]= (a[c]<0.1*d) ? 0.1*d : a[c];//limit the minimum pairwise cost
    }
}

static inline ftype addmul_d(double *a,ftype *b,double c,int len)
{
    int cn;
    for (cn=0;cn<len;cn++)
    {
        a[cn]=a[cn]+b[cn]*c;
    }
}

static inline ftype addmul_d2(double *a,double *b,double c,int len)
{
    int cn;
    for (cn=0;cn<len;cn++)
    {
        a[cn]=a[cn]+b[cn]*c;
    }
}


static inline ftype score(ftype *x,ftype *w,int len)
{
    int c;
    ftype scr=0;
    for (c=0;c<len;c++)
    {
        scr+=x[c]*w[c];
    }
    return scr;
}

static inline double score_d(ftype *x,double *w,int len)
{
    int c;
    double scr=0;
    for (c=0;c<len;c++)
    {
        scr+=x[c]*w[c];
    }
    return scr;
}

static inline ftype score2(ftype *x,ftype *w,ftype w0,int len,int sizereg)
{
    int c;
    ftype scr=0;
    for (c=0;c<len-sizereg;c++)//normal part
    {
        scr+=x[c]*w[c];
    }
    for (c=len-sizereg;c<len;c++)//regularize at d instead of 0
    {
        scr+=(x[c]-w0)*(w[c]-w0);
        //a[c]= (a[c]<0.1*d) ? 0.1*d : a[c];//limit the minimum pairwise cost
    }
    return scr;
}


static inline ftype score2_d(double *x,double *w,double w0,int len,int sizereg)
{
    int c;
    double scr=0;
    for (c=0;c<len-sizereg;c++)//normal part
    {
        scr+=x[c]*w[c];
    }
    for (c=len-sizereg;c<len;c++)//regularize at d instead of 0
    {
        scr+=(x[c]-w0)*(w[c]-w0);
        //a[c]= (a[c]<0.1*d) ? 0.1*d : a[c];//limit the minimum pairwise cost
    }
    return scr;
}

void fast_pegasos(ftype *w,int wx,ftype *ex,int exy,ftype *label,ftype lambda,int iter,int part)
{
    srand48(3);
    int c,y,t,pex;
    ftype *x,n,scr,norm,val;
    printf("Parts:%d \n Lambda:%g\n",part,lambda);
    //#pragma omp parallel for
    for (c=0;c<iter;c++)
    {
        t=c+part*iter+1;
        pex=(int)(drand48()*(exy-0.5));
        x=ex+pex*wx;
        y=label[pex];
        n=1.0/(lambda*t);
        scr=score(x,w,wx);
        //printf("rnd:%d y=%d scr=%g eta=%g\n",pex,y,scr,n);
        mul(w,1-n*lambda,wx);
        if (scr*y<1.0)
        {
            //mul(x,y*n,wx)
            addmul(w,x,y*n,wx);            
        }
        //printf("W0:%g",w[0]);
        norm=sqrt(score(w,w,wx));
        val=1/(sqrt(lambda)*(norm+0.0001));
        if (val<1.0)
            mul(w,val,wx);
    }
    printf("N:%g t:%d\n",n,t);
}

void fast_pegasos_comp(ftype *w,int numcomp,int *compx,int *compy,ftype **ptrsamplescomp,int totsamples,int *label,int *comp,ftype C,int iter,int part)
{
    int wx=0,wxtot=0,wcx;
    srand48(3);
    int c,cp,bcp,d,y,t,pex,pexcomp,totsz,sumszx[maxcomp],sumszy[maxcomp];//max 10 components
    ftype *x,n,scr,norm,val,ptrc,wscr,bwscr=-1.0;
    totsz=0;
    sumszx[0]=0;
    sumszy[0]=0;
    for (c=0;c<numcomp;c++)
    {
        wxtot+=compx[c];
        totsz+=compy[c];
        sumszx[c+1]=wxtot;
        sumszy[c+1]=totsz;
    }
    for (c=0;c<iter;c++)
    {
        t=c+part*iter+1;
        pex=(int)(drand48()*(totsamples-0.5));
        //printf("S: %d\n",pex);
        wx=compx[comp[pex]];
        x=ptrsamplescomp[comp[pex]]+(pex-sumszy[comp[pex]])*wx;
        y=label[pex];
        //printf("Y %d ",y);
        n=1.0/(t);
        //printf("C %d ",comp[pex]);
        scr=score(x,w+sumszx[comp[pex]],wx);
        //only the component l2_max
        bwscr=-1.0;
        for (cp=0;cp<numcomp;cp++)
        {   
            wscr=score(w+sumszx[cp],w+sumszx[cp],compx[cp]);
            //printf("Wscore(%d)=%f\n",cp,wscr);
            if (wscr>bwscr)
            {
                bwscr=wscr;
                bcp=cp;
            }
        }
        //printf("Regularize Component %d \n",bcp);
        mul(w+sumszx[bcp],1-n,compx[bcp]);    
        //all the vector
        //mul(w,1-n*lambda,wxtot);
        if (scr*y<1.0)
        {
            addmul(w+sumszx[comp[pex]],x,C*y*n*totsamples,wx);            
        }
    }
    printf("N:%g t:%d\n",n,t);
}

void fast_pegasos_comp_parall(ftype *w,int numcomp,int *compx,int *compy,ftype **ptrsamplescomp,int totsamples,int *label,int *comp,ftype C,int iter,int part,int k,int numthr,int *sizereg,ftype valreg,int *sizesmul,ftype valsmul,ftype lb)
{
    int wx=0,wxtot=0,wcx;
    #ifdef _OPENMP
    omp_set_num_threads(numthr);
    #endif
    printf("k=%d\n",k);
    srand48(3+part);
    int c,cp,bcp,d,y,t,pex,pexcomp,totsz,sumszx[maxcomp],sumszy[maxcomp];//max 10 components
    ftype *x,n,scr,norm,val,ptrc,wscr,bwscr=-1.0;
    totsz=0;
    sumszx[0]=0;
    sumszy[0]=0;
    int *pares,*pexarray,kk;
    pares   =malloc(sizeof(int)*k);
    pexarray=malloc(sizeof(int)*k);
    for (c=0;c<numcomp;c++)
    {
        wxtot+=compx[c];
        totsz+=compy[c];
        sumszx[c+1]=wxtot;
        sumszy[c+1]=totsz;
    }
    for (c=0;c<iter;c++)
    {
        t=c+part*iter+1;
        n=1.0/(t);
        //only the component l2_max*/
        bwscr=-1.0;
        for (cp=0;cp<numcomp;cp++)
        {   
            //wscr=score(w+sumszx[cp],w+sumszx[cp],compx[cp]);
            //just a test
            //wscr=score(w+sumszx[cp],w+sumszx[cp],compx[cp]-sizereg[cp]);
            //printf("Wscore(%d)=%f\n",cp,wscr);
            wscr=score2(w+sumszx[cp],w+sumszx[cp],valreg,compx[cp]-1,sizereg[cp]);
            if (wscr>bwscr)
            {
                bwscr=wscr;
                bcp=cp;
            }
        }
        //printf("Regularize Component %d Valreg:%f Sizereg:%d \n",bcp,valreg,sizereg[bcp]);
        //not regularize pairwise
        //reg(w+sumszx[bcp],n,valreg,compx[bcp]-sizereg[bcp],0);//0.01    
        //|w-w_0|
        reg(w+sumszx[bcp],n,valreg,compx[bcp]-1,sizereg[bcp]);//0.01    
        //|w|
        //reg(w+sumszx[bcp],n,valreg,compx[bcp],0);//0.01    
        //mul22(w+sumszx[bcp],1-n,valreg,compx[bcp],sizereg[bcp]);//0.01    
        //all the vector
        //mul(w,1-n*lambda,wxtot);
        for (kk=0;kk<k;kk++)
        {
            pexarray[kk]=(int)(drand48()*(totsamples-0.5));
        }
        //printf("here2!!!\n");
        #pragma omp parallel for private(scr,pex,x,y,wx)
        for (kk=0;kk<k;kk++)
        {          
            pex=pexarray[kk];
            wx=compx[comp[pex]];
            x=ptrsamplescomp[comp[pex]]+(pex-sumszy[comp[pex]])*wx;
            //printf("here2.3!!!\n");
            y=label[pex];
            //printf("Y %d ",y);
            //printf("C %d ",comp[pex]);
            scr=score(x,w+sumszx[comp[pex]],wx);
            //printf("here2.5!!!\n");
            if (scr*y<1.0)
            {
                pares[kk]=pex;
            }
            else
            {
                pares[kk]=-1;
            }
        }
        //printf("here3!!!\n");
        for (kk=0;kk<k;kk++)
        {
            if (pares[kk]!=-1)
            {
                //addmul(w,ex+pares[kk]*wx,(float)(label[pares[kk]])*n/(float)k,wx);            
                pex=pares[kk];
                wx=compx[comp[pex]];
                x=ptrsamplescomp[comp[pex]]+(pex-sumszy[comp[pex]])*wx;
                //addmulslow(ftype *a,ftype *b,ftype c,ftype smul,int len,int sizeslow)
                //printf("Val:%f,Size:%d\n",valsmul,sizesmul[comp[pex]]);
                addmulslow(w+sumszx[comp[pex]],x,(float)(label[pex])*n*C*totsamples/(float)k,valsmul,wx,sizesmul[comp[pex]]);   
                //addmul(w+sumszx[comp[pex]],x,(float)(label[pex])*n*C*totsamples/(float)k,wx);            
            }
        }
        for (cp=0;cp<numcomp;cp++)
            limit(w+sumszx[cp],lb,compx[cp]-1,sizereg[cp]);//0.01    
        /*if (scr*y<1.0)
        {
            addmul(w+sumszx[comp[pex]],x,C*y*n*totsamples,wx);            
        }*/
    }
    printf("N:%g t:%d\n",n,t);
    free(pares);
    free(pexarray);
}

double fast_obj(double *w,int numcomp,int *compx,int *compy,ftype **ptrsamplescomp,int totsamples,int *label,int *comp,ftype C,int numthr,int *sizereg,double valreg)
{
    int wx=0,wxtot=0,wcx;
    #ifdef _OPENMP
    omp_set_num_threads(numthr);
    #endif
    int c,cp,bcp,d,y,t,pex,pexcomp,totsz,sumszx[maxcomp],sumszy[maxcomp];//max 10 components
    ftype *x;
    double n,scr,norm,val,ptrc,wscr,bwscr=-1.0;
    totsz=0;
    sumszx[0]=0;
    sumszy[0]=0;
    int *pares,*pexarray,kk;
    //pares   =malloc(sizeof(int)*k);
    //pexarray=malloc(sizeof(int)*k);
    for (c=0;c<numcomp;c++)
    {
        wxtot+=compx[c];
        totsz+=compy[c];
        sumszx[c+1]=wxtot;
        sumszy[c+1]=totsz;
    }
    bwscr=-1.0;
    double scrs[10],Z=0;
    for (cp=0;cp<numcomp;cp++)
    {   
    //softmax
        wscr=score2_d(w+sumszx[cp],w+sumszx[cp],0,compx[cp]-1,sizereg[cp]);
        //printf("Wscore(%d)=%f\n",cp,wscr);
        scrs[cp]=wscr;
        if (wscr>bwscr)
        {
            bwscr=wscr;
            bcp=cp;
        }
    /*
        wscr=score2_d(w+sumszx[cp],w+sumszx[cp],0,compx[cp]-1,sizereg[cp]);
        //printf("Wscore(%d)=%f\n",cp,wscr);
        if (wscr>bwscr)
        {
            bwscr=wscr;
            bcp=cp;
        }
    */
    }
    for (cp=0;cp<numcomp;cp++)
    {
        double a=exp(beta*(scrs[cp]-bwscr));   
        Z+=a;
    }
    bwscr=bwscr+log(Z)*inv_beta;
    //regularization=bwscr
    double loss=0.0;
    for (c=0;c<totsz;c++)
    {
       // #pragma omp parallel for private(scr,pex,x,y,wx)
        //for (kk=0;kk<k;kk++)
        //{          
        pex=c;//pexarray[kk];
        wx=compx[comp[pex]];
        x=ptrsamplescomp[comp[pex]]+(pex-sumszy[comp[pex]])*wx;
        y=label[pex];
        //printf("Y %d ",y);
        //printf("C %d ",comp[pex]);
        scr=score_d(x,w+sumszx[comp[pex]],wx);
        if (scr*y<1.0)
            loss+=-(y*scr)+1.0; 
    }
    //printf("N:%g t:%d\n",n,t);
    //free(pares);
    //free(pexarray);
    return 0.5*bwscr+C*loss;
}

int fast_grad2(ftype *gr,ftype *w,int numcomp,int *compx,int *compy,ftype **ptrsamplescomp,int totsamples,int *label,int *comp,ftype C,int iter,int part,int k,int numthr,int *sizereg,ftype valreg,ftype lb)
{
    int wx=0,wxtot=0,wcx;
    #ifdef _OPENMP
    omp_set_num_threads(numthr);
    #endif
    printf("Gradient Fast\n");
    //srand48(3+part);
    int c,cp,bcp,d,y,t,pex,pexcomp,totsz,sumszx[maxcomp],sumszy[maxcomp];//max 10 components
    ftype *x,n,scr,norm,val,ptrc,wscr,bwscr=-1.0;
    totsz=0;
    sumszx[0]=0;
    sumszy[0]=0;
    int *pares,*pexarray,kk;
    //pares   =malloc(sizeof(int)*k);
    //pexarray=malloc(sizeof(int)*k);
    for (c=0;c<numcomp;c++)
    {
        wxtot+=compx[c];
        totsz+=compy[c];
        sumszx[c+1]=wxtot;
        sumszy[c+1]=totsz;
    }
    //regularization=bwscr
    //float loss=0.0;
    for (c=0;c<totsz;c++)
    {
       // #pragma omp parallel for private(scr,pex,x,y,wx)
        //for (kk=0;kk<k;kk++)
        //{          
        pex=c;//pexarray[kk];
        wx=compx[comp[pex]];
        x=ptrsamplescomp[comp[pex]]+(pex-sumszy[comp[pex]])*wx;
        y=label[pex];
        printf("c %d ",c);
        printf("C %d ",comp[pex]);
        scr=score(x,w+sumszx[comp[pex]],wx);
        if (scr*y<1.0)
            add(gr+sumszx[comp[pex]],x,wx);
    }
    mul(gr,C,wxtot);
    bwscr=-1.0;
    for (cp=0;cp<numcomp;cp++)
    {   
        wscr=score2(w+sumszx[cp],w+sumszx[cp],valreg,compx[cp]-1,sizereg[cp]);
        if (wscr>bwscr)
        {
            bwscr=wscr;
            bcp=cp;
        }
    }
    add(gr+sumszx[bcp],w+sumszx[bcp],compx[bcp]-1);
    //printf("N:%g t:%d\n",n,t);
    //free(pares);
    //free(pexarray);
}

void fast_grad3(double *gr,double *w,int numcomp,int *compx,int *compy,ftype **ptrsamplescomp,int totsamples,int *label,int *comp,ftype C,int numthr,int *sizereg,double valreg)
{
    int wx=0,wxtot=0,wcx;
    #ifdef _OPENMP
    omp_set_num_threads(numthr);
    #endif
    //printf("Gradient Fast\n");
    //srand48(3+part);
    int c,cp,bcp,d,y,t,pex,pexcomp,totsz,sumszx[maxcomp],sumszy[maxcomp];//max 10 components
    ftype *x;
    double n,scr,norm,val,ptrc,wscr,bwscr=-1.0;
    totsz=0;
    sumszx[0]=0;
    sumszy[0]=0;
    int *pares,*pexarray,kk;
    //pares   =malloc(sizeof(int)*k);
    //pexarray=malloc(sizeof(int)*k);
    for (c=0;c<numcomp;c++)
    {
        wxtot+=compx[c];
        totsz+=compy[c];
        sumszx[c+1]=wxtot;
        sumszy[c+1]=totsz;
    }
    //regularization=bwscr
    //float loss=0.0;
    for (c=0;c<totsz;c++)
    {
       // #pragma omp parallel for private(scr,pex,x,y,wx)
        //for (kk=0;kk<k;kk++)
        //{          
        pex=c;//pexarray[kk];
        wx=compx[comp[pex]];
        x=ptrsamplescomp[comp[pex]]+(pex-sumszy[comp[pex]])*wx;
        y=label[pex];
        //printf("c %d ",c);
        //printf("C %d ",comp[pex]);
        scr=score_d(x,w+sumszx[comp[pex]],wx);
        if (scr*y<1.0)
            addmul_d(gr+sumszx[comp[pex]],x,-(double)y,wx);
    }
    mul_d(gr,C,wxtot);
    bwscr=-1.0;
    double scrs[maxcomp],pc[maxcomp],Z=0;
    for (cp=0;cp<numcomp;cp++)
    {   
        wscr=score2_d(w+sumszx[cp],w+sumszx[cp],valreg,compx[cp]-1,sizereg[cp]);
        scrs[cp]=wscr;
        if (wscr>bwscr)
        {
            bwscr=wscr;
            bcp=cp;
        }
    }
    for (cp=0;cp<numcomp;cp++)
    {
        double a=exp(beta*(scrs[cp]-bwscr));   
        pc[cp]=a;
        Z+=a;
    }
    for (cp=0;cp<numcomp;cp++)
        addmul_d2(gr+sumszx[cp],w+sumszx[cp],pc[cp]/Z,compx[cp]-1);
       //add_d2(gr+sumszx[bcp],w+sumszx[bcp],compx[bcp]-1);
}


void fast_pegasos_comp_parall2(ftype *w,int numcomp,int *compx,int *compy,ftype **ptrsamplescomp,int totsamples,int *label,int *comp,ftype C,int iter,int part,int k,int numthr,int *sizereg,ftype valreg,ftype lb)
{
    //assume the last is bias and it is not regularized
    int wx=0,wxtot=0,wcx;
    #ifdef _OPENMP
    omp_set_num_threads(numthr);
    #endif
    printf("k=%d\n",k);
    srand48(3+part);
    int c,cp,bcp,d,y,t,pex,pexcomp,totsz,sumszx[maxcomp],sumszy[maxcomp];//max 10 components
    ftype *x,n,scr,norm,val,ptrc,wscr,bwscr=-1.0;
    totsz=0;
    sumszx[0]=0;
    sumszy[0]=0;
    int *pares,*pexarray,kk;
    pares   =malloc(sizeof(int)*k);
    pexarray=malloc(sizeof(int)*k);
    for (c=0;c<numcomp;c++)
    {
        wxtot+=compx[c];
        totsz+=compy[c];
        sumszx[c+1]=wxtot;
        sumszy[c+1]=totsz;
    }
    for (c=0;c<iter;c++)
    {
        t=c+part*iter+1;
        n=1.0/(t);
        //only the component l2_max*/
        bwscr=-1.0;
        //printf("I am Here!!!!\n");
        for (cp=0;cp<numcomp;cp++)
        {   
            //wscr=score(w+sumszx[cp],w+sumszx[cp],compx[cp]-1);//skip bias
            wscr=score2(w+sumszx[cp],w+sumszx[cp],valreg,compx[cp]-1,sizereg[cp]);
            //just a test
            //wscr=score(w+sumszx[cp],w+sumszx[cp],compx[cp]-sizereg[cp]);
            //printf("Wscore(%d)=%f\n",cp,wscr);
            if (wscr>bwscr)
            {
                bwscr=wscr;
                bcp=cp;
            }
        }
        //printf("Regularize Component %d Valreg:%f Sizereg:%d \n",bcp,valreg,sizereg[bcp]);
        //not regularize pairwise
        //reg(w+sumszx[bcp],n,valreg,compx[bcp]-sizereg[bcp],0);//0.01    
        //|w-w_0|
        //reg(w+sumszx[bcp],n,valreg,compx[bcp],sizereg[bcp]);//0.01    
        //|w|
        //printf("Now, I am Here!!!!\n");    
        reg(w+sumszx[bcp],n,valreg,compx[bcp]-1,sizereg[cp]);
        //mul22(w+sumszx[bcp],1-n,valreg,compx[bcp],sizereg[bcp]);//0.01    
        //all the vector
        //mul(w,1-n*lambda,wxtot);
        for (kk=0;kk<k;kk++)
        {
            pexarray[kk]=(int)(drand48()*(totsamples-0.5));
        }
        //printf("here2!!!\n");
        #pragma omp parallel for private(scr,pex,x,y,wx)
        for (kk=0;kk<k;kk++)
        {          
            pex=pexarray[kk];
            wx=compx[comp[pex]];
            x=ptrsamplescomp[comp[pex]]+(pex-sumszy[comp[pex]])*wx;
            //printf("here2.3!!!\n");
            y=label[pex];
            //printf("Y %d ",y);
            //printf("C %d ",comp[pex]);
            scr=score(x,w+sumszx[comp[pex]],wx);
            //printf("here2.5!!!\n");
            if (scr*y<1.0)
            {
                pares[kk]=pex;
            }
            else
            {
                pares[kk]=-1;
            }
        }
        //printf("here3!!!\n");
        for (kk=0;kk<k;kk++)
        {
            if (pares[kk]!=-1)
            {
                //addmul(w,ex+pares[kk]*wx,(float)(label[pares[kk]])*n/(float)k,wx);            
                pex=pares[kk];
                wx=compx[comp[pex]];
                x=ptrsamplescomp[comp[pex]]+(pex-sumszy[comp[pex]])*wx;
                addmul(w+sumszx[comp[pex]],x,(float)(label[pex])*n*C*totsamples/(float)k,wx);            
            }
        }
        for (cp=0;cp<numcomp;cp++)
            limit(w+sumszx[cp],lb,compx[cp]-1,sizereg[cp]);//0.01    
        //printf("After limit!!!!\n");
        /*if (scr*y<1.0)
        {
            addmul(w+sumszx[comp[pex]],x,C*y*n*totsamples,wx);            
        }*/
    }
    printf("N:%g t:%d\n",n,t);
    free(pares);
    free(pexarray);
}


void fast_pegasos_comp_parall_old(ftype *w,int numcomp,int *compx,int *compy,ftype **ptrsamplescomp,int totsamples,int *label,int *comp,ftype C,int iter,int part,int k,int numthr)
{
    int wx=0,wxtot=0,wcx;
    srand48(3+part);
    #ifdef _OPENMP
    omp_set_num_threads(numthr);
    #endif
    int c,cp,bcp,d,y,t,pex,pexcomp,totsz,sumszx[maxcomp],sumszy[maxcomp];//max 10 components
    ftype *x,n,scr,norm,val,ptrc,wscr,bwscr=-1.0;
    totsz=0;
    sumszx[0]=0;
    sumszy[0]=0;
    int *pares,*pexarray,kk;
    pares   =malloc(sizeof(int)*k);
    pexarray=malloc(sizeof(int)*k);
    for (c=0;c<numcomp;c++)
    {
        wxtot+=compx[c];
        totsz+=compy[c];
        sumszx[c+1]=wxtot;
        sumszy[c+1]=totsz;
    }
    for (c=0;c<iter;c++)
    {
        t=c+part*iter+1;
        pex=(int)(drand48()*(totsamples-0.5));
        //printf("S: %d\n",pex);
        //wx=compx[comp[pex]];
        //x=ptrsamplescomp[comp[pex]]+(pex-sumszy[comp[pex]])*wx;
        //y=label[pex];
        //printf("Y %d ",y);
        n=1.0/(t);
        //printf("C %d ",comp[pex]);
        //scr=score(x,w+sumszx[comp[pex]],wx);
        //only the component l2_max
        bwscr=-1.0;
        for (cp=0;cp<numcomp;cp++)
        {   
            wscr=score(w+sumszx[cp],w+sumszx[cp],compx[cp]);
            //printf("Wscore(%d)=%f\n",cp,wscr);
            if (wscr>bwscr)
            {
                bwscr=wscr;
                bcp=cp;
            }
        }
        //printf("Regularize Component %d \n",bcp);
        mul(w+sumszx[bcp],1-n,compx[bcp]);    
        //all the vector
        //mul(w,1-n*lambda,wxtot);
        for (kk=0;kk<k;kk++)
        {
            pexarray[kk]=(int)(drand48()*(totsamples-0.5));
        }
        #pragma omp parallel for private(scr,pex,x,y,wx)
        for (kk=0;kk<k;kk++)
        {
            pex=pexarray[kk];
            wx=compx[comp[pex]];
            x=ptrsamplescomp[comp[pex]]+(pex-sumszy[comp[pex]])*wx;
            //x=ex+pex*wx;
            y=label[pex];
            //scr=score(x,w,wx);
            scr=score(x,w+sumszx[comp[pex]],wx);
            if (scr*y<1.0)
            {
                pares[kk]=pex;
            }
            else
            {
                pares[kk]=-1;
            }
        }
        for (kk=0;kk<k;kk++)
        {
            if (pares[kk]!=-1)
                pex=pares[kk];
                wx=compx[comp[pex]];
                x=ptrsamplescomp[comp[pex]]+(pex-sumszy[comp[pex]])*wx;
                addmul(w+sumszx[comp[pex]],x,(float)(label[pex])*C*n/(float)k*totsamples,wx);            
        }
        //if (scr*y<1.0)
        //{
        //    addmul(w+sumszx[comp[pex]],x,y*n,wx);            
        //}
    }
    printf("N:%g t:%d\n",n,t);
}

ftype objective_comp(ftype *w,int numcomp,int *compx,int *compy,ftype **ptrsamplescomp,int *label,ftype lambda,ftype *errpos,ftype *errneg,ftype *reg)
{
    int c,y,l,totsz,wxtot,wx;
    ftype val,err=0,*x,totloss;
    int sumszx[maxcomp],sumszy[maxcomp];//max 10 components
    wxtot=0;
    totsz=0;
    sumszx[0]=0;
    sumszy[0]=0;
    errpos[0]=0.0;
    errneg[0]=0.0;
    //printf("numcomp:%d \n",numcomp);
    //printf("compx:%d compy:%d \n",compx[0],compy[0]);
    for (c=0;c<numcomp;c++)
    {
        wxtot+=compx[c];
        totsz+=compy[c];
        sumszx[c+1]=wxtot;
        sumszy[c+1]=totsz;
        //printf("sumx:%d sumy:%d \n",sumszx[c+1],sumszy[c+1]);
    }
    //printf("wxtot:%d totsz:%d \n",wxtot,totsz);
    for (l=0;l<numcomp;l++)
    {
        printf("NumComp: %d\n",l);
        for (c=0;c<compy[l];c++)
        {
            y=label[sumszy[l]+c];
            printf("y %d",y);
            wx=compx[l];
            x=ptrsamplescomp[l]+c*wx;
            //printf("computing score\n");
            val=score(w+sumszx[l],x,wx);
            if (val*y<1)
            {
                //printf("update\n");
                err=err+1-y*val;
                //printf("y %d",y);
                if (y>0)
                {
                    //printf("joder!!!");
                    errpos[0]=errpos[0]+1-y*val;
                }
                else
                {
                    //printf("tio!!!");
                    errneg[0]=errneg[0]+1-y*val;
                }
            }
        } 
    }
    err=err/totsz;
    errpos[0]=errpos[0]/totsz;
    errneg[0]=errneg[0]/totsz;
    reg[0]=lambda/2.0*score(w,w,wx);
    printf("lambda/2*|w|**2=%f Loss=%f \n", *reg, err);
    printf("Pos Loss=%f Neg Loss=%f \n", errpos[0], errneg[0]);
    totloss=err+(*reg);
    return totloss;
}

void fast_pegasos_noproj(ftype *w,int wx,ftype *ex,int exy,ftype *label,ftype lambda,int iter,int part)
{
    srand48(3);
    int c,y,t,pex;
    ftype *x,n,scr,norm,val;
    printf("Parts:%d \n Lambda:%g\n",part,lambda);
    //#pragma omp parallel for
    for (c=0;c<iter;c++)
    {
        t=c+part*iter+1;
        pex=(int)(drand48()*(exy-0.5));
        x=ex+pex*wx;
        y=label[pex];
        n=1.0/(lambda*t);
        scr=score(x,w,wx);
        //printf("rnd:%d y=%d scr=%g eta=%g\n",pex,y,scr,n);
        mul(w,1-n*lambda,wx);
        if (scr*y<1.0)
        {
            //mul(x,y*n,wx)
            addmul(w,x,y*n,wx);            
        }
        //printf("W0:%g",w[0]);
        /*norm=sqrt(score(w,w,wx));
        val=1/(sqrt(lambda)*(norm+0.0001));
        if (val<1.0)
            mul(w,val,wx);*/
    }
    printf("N:%g t:%d\n",n,t);
}

ftype objective(ftype *w,int wx,ftype *ex, int exy,ftype *label,ftype lambda)
{
    int c,y;
    ftype val,err=0,errpos=0,errneg=0,norm;
    for (c=0;c<exy;c++)
    {
        y=label[c];
        val=score(w,ex+c*wx,wx);
        if (val*y<1)
        {
            err+=1-y*val;
            if (y>0)
                errpos+=err;
            else
                errneg+=err;
        }
    } 
    err=err/exy;
    errpos=errpos/exy;
    errneg=errneg/exy;
    norm=lambda/2.0*score(w,w,wx);
    printf("lambda/2*|w|**2=%g Loss=%g \n", norm, err);
    printf("Pos Loss=%g Neg Loss=%g \n", errpos, errneg);
    return norm+err;
}


