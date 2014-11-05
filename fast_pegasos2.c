#include <stdio.h>
#include <stdlib.h>
#include <math.h> //for sqrt

//w is the weight vector to estimate with size wx
//ex is the example matrix of size exy x wx

// gcc -fPIC -c fast_pegasos.c -O3
//gcc -shared -Wl,-soname,libfast_pegasos.so -o libfast_pegasos.so  fast_pegasos.o

//#define ftype float
#define ftype double

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

static void reg3(ftype *a,ftype b,ftype *reg,ftype *zero,ftype *mul,int len)
{
    int c;
    for (c=0;c<len;c++)//normal part
    {
        a[c]=a[c]-(a[c]-zero[c])*b*mul[c]*reg[c];//no need slow regularization
        //a[c]=a[c]-(a[c]-zero[c])*b*reg[c];
    }
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

static inline limit3(ftype *a,ftype *lim,int len)
{
    int c;
    for (c=0;c<len;c++)//regularize at d instead of 0
    {
        a[c]= (a[c]<lim[c]) ? lim[c] : a[c];//limit the minimum pairwise cost
    }
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

static inline addmul3(ftype *a,ftype *b,ftype c,ftype *mul,int len)
{
    int cn;
    for (cn=0;cn<len;cn++)
    {
        a[cn]=a[cn]+b[cn]*c*mul[cn];
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

static inline ftype score_float(float *x,ftype *w,int len)
{
    int c;
    ftype scr=0;
    for (c=0;c<len;c++)
    {
        scr+=(double)x[c]*w[c];
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

static inline ftype score3(ftype *x,ftype *w,ftype *reg,ftype *zero,int len)
{
    int c;
    ftype scr=0;
    for (c=0;c<len;c++)//normal part
    {
        scr+=(x[c]-zero[c])*(w[c]-zero[c])*reg[c];
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
/*
void fast_grad_new(ftype *w,int numcomp,int *compx,int *compy,ftype **ptrsamplescomp,int totsamples,int *label,int *comp,ftype C,int k,int numthr,ftype *reg,ftype *zero,ftype *grad)
{
    // grad has the same dimensionality as w
    int wx=0,wxtot=0,wcx;
    #ifdef _OPENMP
    omp_set_num_threads(numthr);
    #endif
    int wc,c,cp,bcp,d,y,t,pex,pexcomp,totsz,sumszx[maxcomp],sumszy[maxcomp];//max 100 components
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
    bwscr=-1.0;
    //printf("So far:0\n");
    for (cp=0;cp<numcomp;cp++)
    {   
        wscr=score3(w+sumszx[cp],w+sumszx[cp],reg+sumszx[cp],zero+sumszx[cp],compx[cp]);
        if (wscr>bwscr)
        {
            bwscr=wscr;
            bcp=cp;
        }
    }
    for (wc=sumszx[bcp];wc<sumszx[bcp]+compx[bcp];wc++)
        grad[wc]=(w[wc]-zero[wc])*reg[wc];
    //printf("So far:1\n");
    #pragma omp parallel for private(scr,pex,x,y,wx)
    for (c=0;c<totsz;c++)
    {
        pex=c;
        wx=compx[comp[pex]];
        x=ptrsamplescomp[comp[pex]]+(pex-sumszy[comp[pex]])*wx;
        y=label[pex];
        scr=score(x,w+sumszx[comp[pex]],wx);
        if (scr*y<1.0)
        {
            //wx=compx[comp[pex]];
            //x=ptrsamplescomp[comp[pex]]+(pex-sumszy[comp[pex]])*wx;
            if (y>0)
                for (wc=0;wc<wx;wc++)
                    grad[wc+sumszx[comp[pex]]]-=C*x[wc];
            else
                for (wc=0;wc<wx;wc++)
                    grad[wc+sumszx[comp[pex]]]+=C*x[wc];
        }
    }
    //printf("So far:2\n");
}

void fast_objective_new(ftype *w,int numcomp,int *compx,int *compy,ftype **ptrsamplescomp,int totsamples,int *label,int *comp,ftype C,int k,int numthr,ftype *reg,ftype *zero,ftype *ret)
{
    ftype *rreg=ret, *rposl=ret+1, *rnegl=ret+2;
    *rposl=0,*rnegl=0,*rreg=0;
    int wx=0,wxtot=0,wcx;
    #ifdef _OPENMP
    omp_set_num_threads(numthr);
    #endif
    int c,cp,bcp,d,y,t,pex,pexcomp,totsz,sumszx[maxcomp],sumszy[maxcomp];//max 100 components
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
    bwscr=-1.0;
    for (cp=0;cp<numcomp;cp++)
    {   
        wscr=score3(w+sumszx[cp],w+sumszx[cp],reg+sumszx[cp],zero+sumszx[cp],compx[cp]);
        if (wscr>bwscr)
        {
            bwscr=wscr;
            bcp=cp;
        }
    }
    *rreg=0.5*bwscr;
    #pragma omp parallel for private(scr,pex,x,y,wx)
    for (c=0;c<totsz;c++)
    {
        pex=c;
        wx=compx[comp[pex]];
        x=ptrsamplescomp[comp[pex]]+(pex-sumszy[comp[pex]])*wx;
        y=label[pex];
        scr=score(x,w+sumszx[comp[pex]],wx);
        if (scr*y<1.0)
        {
            //wx=compx[comp[pex]];
            //x=ptrsamplescomp[comp[pex]]+(pex-sumszy[comp[pex]])*wx;
            if (y>0)
                *rposl+=1-(scr*y);
            else
                *rnegl+=1-(scr*y);
        }
    }
    *rposl=*rposl*C;
    *rnegl=*rnegl*C;
    printf("Reg:%f Posl:%f Negl:%f Tot:%f \n",*rreg,*rposl,*rnegl,*rreg+*rposl+*rnegl);
}
*/
//use only this, the others are deprecated!
/*
void fast_objgrad_new(ftype *w,int numcomp,int *compx,int *compy,ftype **ptrsamplescomp,int totsamples,int *label,int *comp,ftype C,int k,int numthr,ftype *reg,ftype *zero,ftype *ret,ftype *grad)
{
    ftype *rreg=ret, *rposl=ret+1, *rnegl=ret+2;
    *rposl=0,*rnegl=0,*rreg=0;
    int wx=0,wxtot=0,wcx;
    #ifdef _OPENMP
    omp_set_num_threads(numthr);
    #endif
    int wc,c,cp,bcp,d,y,t,pex,pexcomp,totsz,sumszx[maxcomp],sumszy[maxcomp];//max 100 components
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
    bwscr=-1.0;
    //printf("So far:0\n");
    for (cp=0;cp<numcomp;cp++)
    {   
        wscr=score3(w+sumszx[cp],w+sumszx[cp],reg+sumszx[cp],zero+sumszx[cp],compx[cp]);
        if (wscr>bwscr)
        {
            bwscr=wscr;
            bcp=cp;
        }
    }
    *rreg=0.5*bwscr;
    for (wc=sumszx[bcp];wc<sumszx[bcp]+compx[bcp];wc++)
        grad[wc]=(w[wc]-zero[wc])*reg[wc];
    //check if this pragma is safe, I do not think so...
    //#pragma omp parallel for private(scr,pex,x,y,wx) 
    for (c=0;c<totsz;c++)
    {
        pex=c;
        wx=compx[comp[pex]];
        x=ptrsamplescomp[comp[pex]]+(pex-sumszy[comp[pex]])*wx;
        y=label[pex];
        scr=score(x,w+sumszx[comp[pex]],wx);
        if (scr*y<1.0)
        {
            //wx=compx[comp[pex]];
            //x=ptrsamplescomp[comp[pex]]+(pex-sumszy[comp[pex]])*wx;
            if (y>0)
            {
                *rposl+=1-(scr*y);
                for (wc=0;wc<wx;wc++)
                    grad[wc+sumszx[comp[pex]]]-=C*x[wc];
            }
            else
            {
                *rnegl+=1-(scr*y);
                for (wc=0;wc<wx;wc++)
                    grad[wc+sumszx[comp[pex]]]+=C*x[wc];
            }
        }
    }
    *rposl=*rposl*C;
    *rnegl=*rnegl*C;
    //printf("So far:2\n");
}


void fast_objgrad_parall_new(ftype *w,int numcomp,int *compx,int *compy,ftype **ptrsamplescomp,int totsamples,int *label,int *comp,ftype C,int k,int numthr,ftype *reg,ftype *zero,ftype *ret,ftype *grad)
{
    ftype *rreg=ret, *rposl=ret+1, *rnegl=ret+2;
    *rposl=0,*rnegl=0,*rreg=0;
    int wx=0,wxtot=0,wcx;
    #ifdef _OPENMP
    omp_set_num_threads(numthr);
    #endif
    int th_id,wc,c,cp,bcp,d,y,t,pex,pexcomp,totsz,sumszx[maxcomp],sumszy[maxcomp];//max 100 components
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
    bwscr=-1.0;
    //printf("So far:0\n");
    for (cp=0;cp<numcomp;cp++)
    {   
        wscr=score3(w+sumszx[cp],w+sumszx[cp],reg+sumszx[cp],zero+sumszx[cp],compx[cp]);
        if (wscr>bwscr)
        {
            bwscr=wscr;
            bcp=cp;
        }
    }
    *rreg=0.5*bwscr;
    for (wc=sumszx[bcp];wc<sumszx[bcp]+compx[bcp];wc++)
        grad[wc]=(w[wc]-zero[wc])*reg[wc];

    ftype *agrad = malloc(numthr*wxtot*sizeof(ftype));
    ftype *aposl = malloc(numthr*sizeof(ftype));
    ftype *anegl = malloc(numthr*sizeof(ftype));
    for (c=0;c<numthr;c++)
    {
        aposl[c]=0;
        anegl[c]=0;
        for (wc=0;wc<wxtot;wc++)
            agrad[c*wxtot+wc]=0;
    }    
    //printf("Tot Samples %d \n",totsz);
    //check if this pragma is safe, I do not think so...
    #pragma omp parallel for private(scr,pex,x,y,wx,wc,th_id) //schedule(dynamic)
    for (c=0;c<totsz;c++)
    {
        pex=c;
        wx=compx[comp[pex]];
        x=ptrsamplescomp[comp[pex]]+(pex-sumszy[comp[pex]])*wx;
        y=label[pex];
        scr=score(x,w+sumszx[comp[pex]],wx);
        th_id = omp_get_thread_num();
        //printf("Thread %d\n",th_id);
        if (scr*y<1.0)
        {
            //wx=compx[comp[pex]];
            //x=ptrsamplescomp[comp[pex]]+(pex-sumszy[comp[pex]])*wx;
            if (y>0)
            {
                aposl[th_id]+=1-(scr*y);
                for (wc=0;wc<wx;wc++)
                    agrad[th_id*wxtot+wc+sumszx[comp[pex]]]-=C*x[wc];
            }
            else
            {
                anegl[th_id]+=1-(scr*y);
                for (wc=0;wc<wx;wc++)
                    agrad[th_id*wxtot+wc+sumszx[comp[pex]]]+=C*x[wc];
            }
        }
    }
    for (c=0;c<numthr;c++)
    {
        *rposl+=aposl[c]*C;
        *rnegl+=anegl[c]*C;
        for (wc=0;wc<wxtot;wc++)
            grad[wc]+=agrad[c*wxtot+wc];
    }
    free(agrad);
    free(aposl);
    free(anegl);
    //printf("So far:2\n");
}
*/

void fast_objgrad_parall_float(ftype *w,int numcomp,int *compx,int *compy,float **ptrsamplescomp,int totsamples,int *label,int *comp,ftype C,int k,int numthr,ftype *reg,ftype *zero,ftype *ret,ftype *grad)
{
    ftype *rreg=ret, *rposl=ret+1, *rnegl=ret+2;
    *rposl=0,*rnegl=0,*rreg=0;
    int wx=0,wxtot=0,wcx;
    #ifdef _OPENMP
    omp_set_num_threads(numthr);
    #endif
    int th_id,wc,c,cp,bcp,d,y,t,pex,pexcomp,totsz,sumszx[maxcomp],sumszy[maxcomp];//max 100 components
    ftype n,scr,norm,val,ptrc,wscr,bwscr=-1.0;
    float *x;
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
    bwscr=-1.0;
    //printf("So far:0\n");
    for (cp=0;cp<numcomp;cp++)
    {   
        wscr=score3(w+sumszx[cp],w+sumszx[cp],reg+sumszx[cp],zero+sumszx[cp],compx[cp]);
        if (wscr>bwscr)
        {
            bwscr=wscr;
            bcp=cp;
        }
    }
    *rreg=0.5*bwscr;
    for (wc=sumszx[bcp];wc<sumszx[bcp]+compx[bcp];wc++)
        grad[wc]=(w[wc]-zero[wc])*reg[wc];

    ftype *agrad = malloc(numthr*wxtot*sizeof(ftype));
    ftype *aposl = malloc(numthr*sizeof(ftype));
    ftype *anegl = malloc(numthr*sizeof(ftype));
    for (c=0;c<numthr;c++)
    {
        aposl[c]=0;
        anegl[c]=0;
        for (wc=0;wc<wxtot;wc++)
            agrad[c*wxtot+wc]=0;
    }    
    //printf("Tot Samples %d \n",totsz);
    //check if this pragma is safe, I do not think so...
    #pragma omp parallel for private(scr,pex,x,y,wx,wc,th_id) //schedule(dynamic)
    for (c=0;c<totsz;c++)
    {
        pex=c;
        wx=compx[comp[pex]];
        x=ptrsamplescomp[comp[pex]]+(pex-sumszy[comp[pex]])*wx;
        y=label[pex];
        scr=score_float(x,w+sumszx[comp[pex]],wx);
        th_id = omp_get_thread_num();
        //printf("Thread %d\n",th_id);
        if (scr*y<1.0)
        {
            //wx=compx[comp[pex]];
            //x=ptrsamplescomp[comp[pex]]+(pex-sumszy[comp[pex]])*wx;
            if (y>0)
            {
                aposl[th_id]+=1-(scr*y);
                for (wc=0;wc<wx;wc++)
                    agrad[th_id*wxtot+wc+sumszx[comp[pex]]]-=C*(double)x[wc];
            }
            else
            {
                anegl[th_id]+=1-(scr*y);
                for (wc=0;wc<wx;wc++)
                    agrad[th_id*wxtot+wc+sumszx[comp[pex]]]+=C*(double)x[wc];
            }
        }
    }
    for (c=0;c<numthr;c++)
    {
        *rposl+=aposl[c]*C;
        *rnegl+=anegl[c]*C;
        for (wc=0;wc<wxtot;wc++)
            grad[wc]+=agrad[c*wxtot+wc];
    }
    free(agrad);
    free(aposl);
    free(anegl);
    //printf("So far:2\n");
}

void fast_objgrad_float(ftype *w,int numcomp,int *compx,int *compy,float **ptrsamplescomp,int totsamples,int *label,int *comp,ftype C,int k,int numthr,ftype *reg,ftype *zero,ftype *ret,ftype *grad)
{
    ftype *rreg=ret, *rposl=ret+1, *rnegl=ret+2;
    *rposl=0,*rnegl=0,*rreg=0;
    int wx=0,wxtot=0,wcx;
    int th_id,wc,c,cp,bcp,d,y,t,pex,pexcomp,totsz,sumszx[maxcomp],sumszy[maxcomp];//max 100 components
    ftype n,scr,norm,val,ptrc,wscr,bwscr=-1.0;
    float *x;
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
    bwscr=-1.0;
    //printf("So far:0\n");
    for (cp=0;cp<numcomp;cp++)
    {   
        wscr=score3(w+sumszx[cp],w+sumszx[cp],reg+sumszx[cp],zero+sumszx[cp],compx[cp]);
        if (wscr>bwscr)
        {
            bwscr=wscr;
            bcp=cp;
        }
    }
    *rreg=0.5*bwscr;
    for (wc=sumszx[bcp];wc<sumszx[bcp]+compx[bcp];wc++)
        grad[wc]=(w[wc]-zero[wc])*reg[wc];
    for (c=0;c<totsz;c++)
    {
        pex=c;
        wx=compx[comp[pex]];
        x=ptrsamplescomp[comp[pex]]+(pex-sumszy[comp[pex]])*wx;
        y=label[pex];
        scr=score_float(x,w+sumszx[comp[pex]],wx);
        if (scr*y<1.0)
        {
            if (y>0)
            {
                *rposl+=1-(scr*y);
                for (wc=0;wc<wx;wc++)
                    grad[wc+sumszx[comp[pex]]]-=C*(double)x[wc];
            }
            else
            {
                *rnegl+=1-(scr*y);
                for (wc=0;wc<wx;wc++)
                    grad[wc+sumszx[comp[pex]]]+=C*(double)x[wc];
            }
        }
    }
    *rposl*=C;
    *rnegl*=C;
}

