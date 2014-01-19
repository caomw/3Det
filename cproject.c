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

void getproj(float minx,float miny,int glangx,int glangy,int glangz,int angx,int angy,float x,float y,float z,int hsize,float *projx, float *projy)
{
    float co=cos(glangz/180.0*M_PI),si=sin(glangz/180.0*M_PI);
    float xr=x*co-y*si,yr=x*si+y*co;
    *projx=-minx+xr*cos(glangx/180.0*M_PI)-hsize/2.0*(cos(angx/180.0*M_PI))-z*sin(angx/180.0*M_PI);
    *projy=-miny+yr*cos(glangy/180.0*M_PI)-hsize/2.0*(cos(angy/180.0*M_PI))-z*sin(angy/180.0*M_PI);
}

/*
res[gly,glx][maxmy-posy:maxmy-posy+hsy+hsize,maxmx-posx:maxmx-posx+hsx+hsize]+=(1-disty)*(1-distx)*scr
                res[gly,glx][maxmy-posy:maxmy-posy+hsy+hsize,maxmx-(posx+1):maxmx-(posx+1)+hsx+hsize]+=(1-disty)*(distx)*scr
                res[gly,glx][maxmy-(posy+1):maxmy-(posy+1)+hsy+hsize,maxmx-(posx):maxmx-(posx)+hsx+hsize]+=(disty)*(1-distx)*scr
                res[gly,glx][maxmy-(posy+1):maxmy-(posy+1)+hsy+hsize,maxmx-(posx+1):maxmx-(posx+1)+hsx+hsize]+=(disty)*(distx)*scr
*/

void interpolate(int res0,int res1,int res2,int res3,float *res,int glx,int gly,int maxmx,int maxmy, int posx, int posy,int hsx,int hsy,int hsize, float distx,float disty,float *scr)
{
    int c,d;
    for (c=1;c<hsy+hsize;c++)
    {
        for (d=1;d<hsx+hsize;d++)
        {
            /*res[gly*(res1*res2*res3)+glx*(res2*res3)+(maxmy-posy+c)*res3+(maxmx-posx+d)]+=(1-disty)*(1-distx)*scr[c*(hsx+hsize)+d];
            res[gly*(res1*res2*res3)+glx*(res2*res3)+(maxmy-posy+c)*res3+(maxmx-posx+1+d)]+=(1-disty)*(distx)*scr[c*(hsx+hsize)+d];
            res[gly*(res1*res2*res3)+glx*(res2*res3)+(maxmy-posy+1+c)*res3+(maxmx-posx+d)]+=(disty)*(1-distx)*scr[c*(hsx+hsize)+d];
            res[gly*(res1*res2*res3)+glx*(res2*res3)+(maxmy-posy+1+c)*res3+(maxmx-posx+1+d)]+=(disty)*(distx)*scr[c*(hsx+hsize)+d];*/
           res[gly*(res1*res2*res3)+glx*(res2*res3)+(maxmy-posy+c)*res3+(maxmx-posx+d)]+=(1-disty)*(1-distx)*scr[c*(hsx+hsize)+d]+(1-disty)*(distx)*scr[c*(hsx+hsize)+d-1]+(disty)*(1-distx)*scr[(c-1)*(hsx+hsize)+d]+(disty)*(distx)*scr[(c-1)*(hsx+hsize)+d-1];     
        }
    }
}

