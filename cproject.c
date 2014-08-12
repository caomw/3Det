/*
def project(res,numpy.ndarray[long,ndim=2] pty,numpy.ndarray[long,ndim=2] ptx):
    """
    compute the correlation with angles ax and ay
    assumes a part of 4x4 hogs
    """
    szy=res.shape[0]
    szx=res.shape[1]
    hy=res.shape[2]
    hx=res.shape[3]
    #szz=res.shape[4]
    cdef:
        int c=0,d=0,py,pym,px,pxm
        float spty=pty.shape[0]
        float sptx=ptx.shape[0]
        #numpy.ndarray res2=numpy.zeros((hy+szy,hx+szx),dtype=numpy.float32)
    res2=numpy.zeros((hy+szy,hx+szx),dtype=numpy.float32)
    for py in range(pty.shape[1]):
            for pym in range(pty.shape[0]):
                for px in range(ptx.shape[1]):
                    for pxm in range(ptx.shape[0]):
                        res2[szy-py:szy-py+hy,szx-px:szx-px+hx]=res2[szy-py:szy-py+hy,szx-px:szx-px+hx]+res[pty[pym,py],ptx[pxm,px]]/spty/sptx
                        #for c in range(hy):
                        #    for d in range(hx):
                        #        res2[szy-py:szy-py+hy,szx-px:szx-px+hx]=res2[szy-py:szy-py+hy,szx-px:szx-px+hx]+res[pty[pym,py],ptx[pxm,px],c,d]/spty/sptx
    return res2
*/
//#include <xmmintrin.h>
//#_mm_setcsr( _mm_getcsr() | (1<<15) | (1<<6) );

void cproject(int res0,int res1,int res2,int res3,float *res,int pty0,int pty1, int *pty,int ptx0,int ptx1,int *ptx, float *ret)
{
    int py,px,pym,pxm,f,g;
    //printf("%d %d %d %d %d %d /",pty1,pty0,ptx1,ptx0,res2,res3);
    for (py=0;py<pty1;py++)
    {
        for (pym=0;pym<pty0;pym++)
        {
            for (px=0;px<ptx1;px++)
            {
                for (pxm=0;pxm<ptx0;pxm++)
                {
                    for (f=0;f<res2;f++)
                    {
                        for (g=0;g<res3;g++)
                        {
                            ret[(f+res0-py)*(res3+res1)+(g+res1-px)]+=res[pty[pym*pty1+py]*(res1*res2*res3)+ptx[pxm*ptx1+px]*res2*res3+f*res3+g]/(float)pty0/(float)ptx0;
                            //printf("%f ",ret[(f+res0-py)*(res3+res1)+(g+res1-px)]);
                        }
                    }
                }
            }
        }
    }
}

/*
nposy=-minym+mm.y*cos(ppglangy[gly]/180.0*numpy.pi)-hsize/2.0*(cos(angy/180.0*numpy.pi))-mm.z*sin(angy/180.0*numpy.pi)
nposx=-minxm+mm.x*cos(ppglangx[glx]/180.0*numpy.pi)-hsize/2.0*(cos(angx/180.0*numpy.pi))-mm.z*sin(angx/180.0*numpy.pi)
*/

#include<math.h>

//    res=[0,0]#numpy.zeros(2,dtype=v.dtype)
//    res[0] = v[0]*cos(a) - v[1]*sin(a)
//    res[1] = v[0]*sin(a) + v[1]*cos(a)

void getproj_old(float minx,float miny,int glangx,int glangy,int glangz,int angx,int angy,float x,float y,float z,int hsize,float *projx, float *projy)
{
    float co=cos(glangz/180.0*M_PI),si=sin(glangz/180.0*M_PI);
    //float xr=x*co-y*si;
    //float yr=x*si+y*co;
    float xr=x*cos(glangx/180.0*M_PI)-hsize/2.0*(cos(angx/180.0*M_PI))-z*sin(angx/180.0*M_PI);
    float yr=y*cos(glangy/180.0*M_PI)-hsize/2.0*(cos(angy/180.0*M_PI))-z*sin(angy/180.0*M_PI);
    //printf("angz %d co %f si %f \n",glangz,co,si);
    float xp=xr*co-yr*si;
    float yp=xr*si+yr*co;
    *projx=-minx+xp;
    *projy=-miny+yp;
    //*projx=-minx+x*cos(glangx/180.0*M_PI)-hsize/2.0*(cos(angx/180.0*M_PI))-z*sin(angx/180.0*M_PI);
    //*projy=-miny+y*cos(glangy/180.0*M_PI)-hsize/2.0*(cos(angy/180.0*M_PI))-z*sin(angy/180.0*M_PI);
}

void getproj_last(float minx,float miny,int glangx,int glangy,int glangz,int angx,int angy,float x,float y,float z,float lz,int hsize,float *projx, float *projy)
{
    float co=cos(glangz/180.0*M_PI),si=sin(glangz/180.0*M_PI);
    //float co=cos(glangz/180.0*M_PI),si=sin(glangz/180.0*M_PI);
    //float xr=x*co-y*si;
    //float yr=x*si+y*co;
    //float xr=x*cos(glangx/180.0*M_PI)+y*sin(glangy/180.0*M_PI)+z*sin(glangx/180.0*M_PI)-hsize/2.0*(cos(angx/180.0*M_PI))-lz*sin(angx/180.0*M_PI);
    float xr=x*cos(glangx/180.0*M_PI)+z*sin(glangx/180.0*M_PI)-hsize/2.0*(cos(angx/180.0*M_PI))-lz*sin(angx/180.0*M_PI);
    //float yr=-x*sin(glangx/180.0*M_PI)+y*cos(glangy/180.0*M_PI)+z*sin(glangy/180.0*M_PI)-hsize/2.0*(cos(angy/180.0*M_PI))-lz*sin(angy/180.0*M_PI);
    float yr=y*cos(glangy/180.0*M_PI)+z*sin(glangy/180.0*M_PI)-hsize/2.0*(cos(angy/180.0*M_PI))-lz*sin(angy/180.0*M_PI);
    //printf("angz %d co %f si %f \n",glangz,co,si);
    float xp=xr*co-yr*si;
    float yp=xr*si+yr*co;
    *projx=-minx+xp;
    *projy=-miny+yp;
    //*projx=-minx+x*cos(glangx/180.0*M_PI)-hsize/2.0*(cos(angx/180.0*M_PI))-z*sin(angx/180.0*M_PI);
    //*projy=-miny+y*cos(glangy/180.0*M_PI)-hsize/2.0*(cos(angy/180.0*M_PI))-z*sin(angy/180.0*M_PI);
}

void rotate(float vx,float vy,int a,float *res)
{
    float co=cos(a/180.0*M_PI),si=sin(a/180.0*M_PI);
    res[0] = vx*co - vy*si;
    res[1] = vx*si + vy*co;
}    

/*
def rotatex(v,a):
    res=rotate([v[1],v[2]],a)
    return [v[0],res[0],res[1]]

def rotatey(v,a):
    res=rotate([v[0],v[2]],a)
    return [res[0],v[1],res[1]]

def rotatez(v,a):
    res=rotate([v[0],v[1]],a)
    return [res[0],res[1],v[2]]
*/

void getproj(float minx,float miny,int glangx,int glangy,int glangz,int angx,int angy,float x,float y,float z,float lz,int hsize,float *projx, float *projy)
{
    //printf("Hsize%d",hsize);
    //printf("before x=%f,y+%f,z=%f    ",x,y,z);
    float res[2],tmpx,tmpy,tmpz,yr,xr;
    rotate(-hsize/2.0,lz,angx,res);//rotatey
    //rotate(0.0,lz,angx,res);//rotatey
    tmpx=res[0];tmpy=-hsize/2.0;tmpz=res[1];
    //tmpx=res[0];tmpy=-hsize/2.0;tmpz=res[1];
    rotate(tmpy,tmpz,angy,res);//rotatex
    tmpx=tmpx;tmpy=res[0],tmpz=res[1];
    //printf("middle x=%f,y+%f,z=%f    ",tmpx,tmpy,tmpz);
    tmpx+=x;tmpy+=y;tmpz+=z;
    rotate(tmpx,tmpz,glangx,res);//rotate y
    tmpx=res[0];tmpy=tmpy;tmpz=res[1];
    rotate(tmpy,tmpz,glangy,res);//rotate x
    tmpx=tmpx;tmpy=res[0];tmpz=res[1];
    xr=tmpx;yr=tmpy;
    float co=cos(glangz/180.0*M_PI),si=sin(glangz/180.0*M_PI);
    //float co=cos(glangz/180.0*M_PI),si=sin(glangz/180.0*M_PI);
    //float xr=x*co-y*si;
    //float yr=x*si+y*co;
    //float xr=x*cos(glangx/180.0*M_PI)+y*sin(glangy/180.0*M_PI)+z*sin(glangx/180.0*M_PI)-hsize/2.0*(cos(angx/180.0*M_PI))-lz*sin(angx/180.0*M_PI);
    //float xr=x*cos(glangx/180.0*M_PI)+z*sin(glangx/180.0*M_PI)-hsize/2.0*(cos(angx/180.0*M_PI))-lz*sin(angx/180.0*M_PI);
    //float yr=-x*sin(glangx/180.0*M_PI)+y*cos(glangy/180.0*M_PI)+z*sin(glangy/180.0*M_PI)-hsize/2.0*(cos(angy/180.0*M_PI))-lz*sin(angy/180.0*M_PI);
    //float yr=y*cos(glangy/180.0*M_PI)+z*sin(glangy/180.0*M_PI)-hsize/2.0*(cos(angy/180.0*M_PI))-lz*sin(angy/180.0*M_PI);
    //printf("angz %d co %f si %f \n",glangz,co,si);
    float xp=xr*co-yr*si;
    float yp=xr*si+yr*co;
    *projx=-minx+xp;
    *projy=-miny+yp;
    //printf("x=%f,y+%f,z=%f \n",xp,yp,tmpz);
    //*projx=-minx+x*cos(glangx/180.0*M_PI)-hsize/2.0*(cos(angx/180.0*M_PI))-z*sin(angx/180.0*M_PI);
    //*projy=-miny+y*cos(glangy/180.0*M_PI)-hsize/2.0*(cos(angy/180.0*M_PI))-z*sin(angy/180.0*M_PI);
}

/*
res[gly,glx][maxmy-posy:maxmy-posy+hsy+hsize,maxmx-posx:maxmx-posx+hsx+hsize]+=(1-disty)*(1-distx)*scr
                res[gly,glx][maxmy-posy:maxmy-posy+hsy+hsize,maxmx-(posx+1):maxmx-(posx+1)+hsx+hsize]+=(1-disty)*(distx)*scr
                res[gly,glx][maxmy-(posy+1):maxmy-(posy+1)+hsy+hsize,maxmx-(posx):maxmx-(posx)+hsx+hsize]+=(disty)*(1-distx)*scr
                res[gly,glx][maxmy-(posy+1):maxmy-(posy+1)+hsy+hsize,maxmx-(posx+1):maxmx-(posx+1)+hsx+hsize]+=(disty)*(distx)*scr
*/

void interpolate(int res0,int res1,int res2,int res3,int res4,float *res,int glx,int gly,int glz,int maxmx,int maxmy, int posx, int posy,int hsx,int hsy,int hsize, float distx,float disty,float *scr)
{
    int c,d;
    for (c=0;c<hsy+hsize-1;c++)
    {
        for (d=0;d<hsx+hsize-1;d++)
        {
            /*res[gly*(res1*res2*res3)+glx*(res2*res3)+(maxmy-posy+c)*res3+(maxmx-posx+d)]+=(1-disty)*(1-distx)*scr[c*(hsx+hsize)+d];
            res[gly*(res1*res2*res3)+glx*(res2*res3)+(maxmy-posy+c)*res3+(maxmx-posx+1+d)]+=(1-disty)*(distx)*scr[c*(hsx+hsize)+d];
            res[gly*(res1*res2*res3)+glx*(res2*res3)+(maxmy-posy+1+c)*res3+(maxmx-posx+d)]+=(disty)*(1-distx)*scr[c*(hsx+hsize)+d];
            res[gly*(res1*res2*res3)+glx*(res2*res3)+(maxmy-posy+1+c)*res3+(maxmx-posx+1+d)]+=(disty)*(distx)*scr[c*(hsx+hsize)+d];*/
           res[gly*(res1*res2*res3*res4)+glx*(res2*res3*res4)+glz*(res3*res4)+(maxmy-posy+c)*res4+(maxmx-posx+d)]+=(1-disty)*(1-distx)*scr[c*(hsx+hsize)+d]+(1-disty)*(distx)*scr[c*(hsx+hsize)+d+1]+(disty)*(1-distx)*scr[(c+1)*(hsx+hsize)+d]+(disty)*(distx)*scr[(c+1)*(hsx+hsize)+d+1];     
        }
    }
}

