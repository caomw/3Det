#include <math.h>
#include <stdlib.h>
//#include <sys/types.h>

/*
 * Generalized distance transforms.
 * We use a simple nlog(n) divide and conquer algorithm instead of the
 * theoretically faster linear method, for no particular reason except
 * that this is a bit simpler and I wanted to test it out.
 *
 * The code is a bit convoluted because dt1d can operate either along
 * a row or column of an array.  
 */

#define INF 1E20

#define ftype float
static inline int square(int x) { return x*x; }

inline ftype quadric(ftype x, ftype y,ftype axx, ftype axy, ftype ayy, ftype bx,ftype by)
{
    return axx*(x-bx)*(x-bx)+2*axy*(x-bx)*(y-by)+ayy*(y-by)*(y-by);
}

void rotatec(ftype *imgin,int x,int y,int nch,ftype rads,ftype *imgout,int nx,int ny)
{
    int rows=x,r,l,r0,c0,r1,c1,ch,m;
    int cols=y;
    r0=rows/2;
	c0=cols/2;
    for (r=0;r<nx;r++)
	{
		for(l=0;l<ny;l++)
		{
            r1=(int)(r0+((r-nx/2)*cos(rads))-((l-ny/2)*sin(rads)));
     	    c1=(int)(c0+((r-nx/2)*sin(rads))+((l-ny/2)*cos(rads)));           
                //m=sqrt((r-nx)*(r-nx)+(l-ny)*(l-ny));
    	    if ((c1>=y) || (r1>=x) || (c1<0) || (r1<0))
            {
                c1 = (c1 >= y ? y-1 : c1);
                r1 = (r1 >= x ? x-1 : r1);
                c1 = (c1 < 0 ? 0 : c1);
                r1 = (r1 < 0 ? 0 : r1);
                for(ch=0;ch<nch;ch++)
                    imgout[l*nx*nch+r*nch+ch]=imgin[c1*x*nch+r1*nch+ch];
                //for(ch=0;ch<nch;ch++)
               	//    imgout[l*x*nch+r*nch+ch]=0;
                        //printf("in:(%d,%d)out(%d,%d)\n",l,r,c1,r1);
            }
            else
            {
                for(ch=0;ch<nch;ch++)
                    imgout[l*nx*nch+r*nch+ch]=imgin[c1*x*nch+r1*nch+ch];
            }
		}
	}   
}

void rotatec_bi(ftype *imgin,int x,int y,int nch,ftype rads,ftype *imgout,int nx,int ny)
{
    int rows=x,r,l,r0,c0,r1,c1,ch,m;
    ftype rr,cc,dr,dc;
    int cols=y;
    r0=rows/2;
	c0=cols/2;
    for (r=0;r<nx;r++)
	{
		for(l=0;l<ny;l++)
		{
            rr=(r0+((r-nx/2)*cos(rads))-((l-ny/2)*sin(rads)));
            r1=floor(rr);
            dr=rr-r1;
     	    cc=(c0+((r-nx/2)*sin(rads))+((l-ny/2)*cos(rads)));           
            c1=floor(cc);
            dc=cc-c1;
                //m=sqrt((r-nx)*(r-nx)+(l-ny)*(l-ny));
    	    if ((c1>=y-1) || (r1>=x-1) || (c1<0) || (r1<0))
            {
                c1 = (c1 >= y-1 ? y-2 : c1);
                r1 = (r1 >= x-1 ? x-2 : r1);
                c1 = (c1 < 0 ? 0 : c1);
                r1 = (r1 < 0 ? 0 : r1);
                for(ch=0;ch<nch;ch++)
                    imgout[l*nx*nch+r*nch+ch]=imgin[c1*x*nch+r1*nch+ch]*(1-dr)*(1-dc)+imgin[(c1+1)*x*nch+r1*nch+ch]*(1-dr)*dc+imgin[c1*x*nch+(r1+1)*nch+ch]*dr*(1-dc)+imgin[(c1+1)*x*nch+(r1+1)*nch+ch]*dr*dc;
                //for(ch=0;ch<nch;ch++)
               	//    imgout[l*x*nch+r*nch+ch]=0;
                        //printf("in:(%d,%d)out(%d,%d)\n",l,r,c1,r1);
            }
            else
            {
                for(ch=0;ch<nch;ch++)
                    imgout[l*nx*nch+r*nch+ch]=imgin[c1*x*nch+r1*nch+ch]*(1-dr)*(1-dc)+imgin[(c1+1)*x*nch+r1*nch+ch]*(1-dr)*dc+imgin[c1*x*nch+(r1+1)*nch+ch]*dr*(1-dc)+imgin[(c1+1)*x*nch+(r1+1)*nch+ch]*dr*dc;
            }
		}
	}   
}

//refine DT in a 3x3 window to make deformations and scr be coherent 
void refine(ftype *img,int sy,int sx,int* defy, int* defx,ftype ay,ftype ax,ftype axy,ftype *dst)
{
    //found small problem at borders: defy points one maxima defx another one...
    int dd=1;
    ftype scr;
    int dy,dx,bny,bnx,py,px,ny,nx;
    for (py=0;py<sy;py++)
    {
        for (px=0;px<sx;px++)
        {
            for (ny=-dd;ny<=dd;ny++)
                for (nx=-dd;nx<=dd;nx++)
                {
                    dy=defy[py*sx+px]+ny;dx=defx[py*sx+px]+nx;
                    dy= dy < 0 ? 0 : dy;dx= dx < 0 ? 0 : dx;
                    dy= dy >= sy ? sy-1 : dy; dx = dx>=sx ? sx-1 : dx;
                    scr=img[dy*sx+dx]-(dy-py)*(dy-py)*ay-(dx-px)*(dx-px)*ax+2*(dy-py)*(dx-px)*axy;
                    //printf("SCR %f DY%d DX%d \n",scr,dy-py,dx-px);
                    if (scr>dst[py*sx+px])
                    {
                        dst[py*sx+px]=scr;
                        bny=dy;bnx=dx;    
                    }
                }
            //if (py==17 && px==31)
                //printf("%d def:%f dy%d dx%d",px,-(bny-py)*(bny-py)*ay-(bnx-px)*(bnx-px)*ax+2*(bny-py)*(bnx-px)*axy,bny-py,bnx-px);
            defy[py*sx+px]=bny;
            defx[py*sx+px]=bnx;
            if (dst[py*sx+px]<img[py*sx+px])//round problem --> do not deform
            {
                dst[py*sx+px]=img[py*sx+px];
                defy[py*sx+px]=py;
                defx[py*sx+px]=px;
            }
        }
        //printf("dy%d \n",py);
    }           
}


//2D distnace transform
void dt2D_helper(ftype *src, ftype *dst, int *ptrx,int *ptry,int dimx, int dimy,int sx1,int sy1,int sx2,int sy2,int dx1,int dy1,int dx2,int dy2, ftype axx, ftype axy, ftype ayy, ftype bx,ftype by) 
{
    ftype aux;
    //quadric placed in p and checking in d
    //int x,y;
    //dtype p,px,py,pxy;
    //scrd=scr[dy*dimx+dx]+quadric(px-dx,py-dy,axx,axy,ayy,bx,by);
    //px=scr[y*dimx+(x+1)]+quadric(x,y,axx,axy,ayy,bx,by);
    //py=scr[(y+1)*dimx+x]+quadric(x,y,axx,axy,ayy,bx,by);
    if ((dx2 >= dx1) && (dy2 >= dy1))
    {
       int dy = (dy1+dy2) >> 1;
       int dx = (dx1+dx2) >> 1;
       int sx = sx1;
       int sy = sy1;
       int px,py;
        //printf("s1=(%d,%d) s2=(%d,%d) d1=(%d,%d) d2=(%d,%d)\n",sx1,sy1,sx2,sy2,dx1,dy1,dx2,dy2);
       for (px = sx1; px <= sx2; px++)
        {
        for (py = sy1; py <= sy2; py++)
        {
        //printf("%f-%f ",src[sy*dimx+sx] - quadric(dx-sx,dy-sy,axx,axy,ayy,bx,by),src[py*dimx+px] - quadric(dx-px,dy-py,axx,axy,ayy,bx,by));
         if (src[sy*dimx+sx] - quadric(dx-sx,dy-sy,axx,axy,ayy,bx,by) < src[py*dimx+px] - quadric(dx-px,dy-py,axx,axy,ayy,bx,by))
         {
            sx = px;sy = py;
         }
        }
        }
        //printf("s=(%d,%d)\n",sx,sy);
        aux = src[sy*dimx+sx] - quadric(dx-sx,dy-sy,axx,axy,ayy,bx,by);
        if (aux>=dst[dy*dimx+dx]) 
        {dst[dy*dimx+dx]=aux;
       //if (dst[dy*dimx+dx]<0) printf("Neg:(%d,%d)=%f\n",sx,sy,dst[dy*dimx+dx]<0);
       ptrx[dy*dimx+dx] = sx;
       ptry[dy*dimx+dx] = sy;}
       dt2D_helper(src, dst, ptrx,ptry, dimx,dimy, sx1, sy1, sx, sy2, dx1,dy1, dx-1,dy2, axx,axy,ayy, bx,by);
       dt2D_helper(src, dst, ptrx,ptry, dimx,dimy, sx ,sy1, sx2, sy2, dx+1,dy1, dx2,dy2, axx,axy,ayy, bx,by);
       dt2D_helper(src, dst, ptrx,ptry, dimx,dimy, sx1, sy1, sx2, sy, dx1,dy1, dx2,dy-1, axx,axy,ayy, bx,by);
       dt2D_helper(src, dst, ptrx,ptry, dimx,dimy, sx1 ,sy, sx2, sy2, dx1,dy+1, dx2,dy2, axx,axy,ayy, bx,by);
       //dt2D_helper(src, dst, ptrx,ptry, dimx,dimy, sx1, sy1, sx, sy, dx1,dy1, dx-1,dy-1, axx,axy,ayy, by,bx);
       //dt2D_helper(src, dst, ptrx,ptry, dimx,dimy, sx ,sy, sx2, sy2, dx+1,dy+1, dx2,dy2, axx,axy,ayy, by,bx);
       //dt2D_helper(src, dst, ptrx,ptry, dimx,dimy, sx1, sy, sx, sy2, dx1,dy+1, dx-1,dy2, axx,axy,ayy, by,bx);
       //dt2D_helper(src, dst, ptrx,ptry, dimx,dimy, sx ,sy1, sx2, sy, dx+1,dy1, dx2,dy-1, axx,axy,ayy, by,bx);
        //dt2D_helper(src, dst, ptrx,ptry, dimx,dimy, sx1, sy1, sx, sy, dx1,dy1, dx-1,dy, axx,axy,ayy, by,bx);
        //dt2D_helper(src, dst, ptrx,ptry, dimx,dimy, sx ,sy, sx2, sy2, dx+1,dy, dx2,dy2, axx,axy,ayy, by,bx);
        //dt2D_helper(src, dst, ptrx,ptry, dimx,dimy, sx1, sy, sx, sy2, dx1,dy+1, dx,dy2, axx,axy,ayy, by,bx);
        //dt2D_helper(src, dst, ptrx,ptry, dimx,dimy, sx ,sy1, sx2, sy, dx,dy1, dx2,dy-1, axx,axy,ayy, by,bx);
      //lines
       //dt2D_helper(src, dst, ptrx,ptry, dimx,dimy, sx1, sy, sx, sy, dx1,dy, dx-1,dy, axx,axy,ayy, by,bx);
       //dt2D_helper(src, dst, ptrx,ptry, dimx,dimy, sx, sy, sx2, sy, dx+1,dy, dx2,dy, axx,axy,ayy, by,bx);
       //dt2D_helper(src, dst, ptrx,ptry, dimx,dimy, sx ,sy1, sx, sy, dx,dy1, dx,dy-1, axx,axy,ayy, by,bx);
       //dt2D_helper(src, dst, ptrx,ptry, dimx,dimy, sx ,sy, sx, sy2, dx,dy+1, dx,dy2, axx,axy,ayy, by,bx);

    }
}

void dt2D(ftype *src, ftype *M,int *Iy,int *Ix, int dimy ,int dimx, ftype ayy, ftype axx, ftype axy, ftype by, ftype bx)
{
  int y,x;
  //ftype *tmpM = (ftype *)malloc(dimy*dimx* sizeof(ftype));
  //int *tmpIx = (int *)malloc(dimy*dimx* sizeof(int));
  //int *tmpIy = (int *)malloc(dimy*dimx* sizeof(int));
  //for (y = 0; y < dimy; y++)
    //dt1d(src+y*dimx, tmpM+y*dimx, tmpIy+y*dimx, 1, dimx, ax, bx);
    //for (x = 0; x < dimx; x++)
  dt2D_helper(src, M, Ix,Iy,dimx,dimy,0,0,dimx-1,dimy-1,0,0, dimx-1,dimy-1,axx,axy,ayy,bx,by);
        //dt1d(tmpM+x, M+x, tmpIx+x, dimx, dimy, ay, by);
  /*for (y = 0; y < dimy; y++) {
    for (x = 0; x < dimx; x++) {
      int p = y*dimx+x;
      Iy[p] = tmpIx[p];
      Ix[p] = tmpIy[tmpIx[p]*dimx+x];
    }
  }*/
  //free(tmpM);
  //free(tmpIx);
  //free(tmpIy);
}

// dt helper function
void dt_helper(ftype *src, ftype *dst, int *ptr, int step, 
	       int s1, int s2, int d1, int d2, ftype a, ftype b) {
 if (d2 >= d1) {
   int d = (d1+d2) >> 1;
   int s = s1;
   int p;
    //printf("s1=(%d) s2=(%d) d1=(%d) d2=(%d)\n",s1,s2,d1,d2);
   for (p = s1; p <= s2; p++)
     if (src[s*step] - a*square(d-s) - b*(d-s) < 
	 src[p*step] - a*square(d-p) - b*(d-p))
	s = p;
    //printf("s=(%d)\n",s);
   dst[d*step] = src[s*step] - a*square(d-s) - b*(d-s);
   ptr[d*step] = s;
   dt_helper(src, dst, ptr, step, s1, s, d1, d-1, a, b);
   dt_helper(src, dst, ptr, step, s, s2, d+1, d2, a, b);
 }
}

// dt of 1d array
void dt1d(ftype *src, ftype *dst, int *ptr, int step, int n, 
	  ftype a, ftype b) {
  dt_helper(src, dst, ptr, step, 0, n-1, 0, n-1, a, b);
}


void dtpy(ftype *src, ftype *M,ftype *Iy,ftype *Ix, int dimy ,int dimx, 
	  ftype ay, ftype ax, ftype by, ftype bx)
{
  int y,x;
  ftype *tmpM = (ftype *)malloc(dimy*dimx* sizeof(ftype));
  int *tmpIx = (int *)malloc(dimy*dimx* sizeof(int));
  int *tmpIy = (int *)malloc(dimy*dimx* sizeof(int));

  //printf("where is the error, here?");
  for (y = 0; y < dimy; y++)
    dt1d(src+y*dimx, tmpM+y*dimx, tmpIy+y*dimx, 1, dimx, ax, bx);
  //printf("or here?");
  for (x = 0; x < dimx; x++)
    dt1d(tmpM+x, M+x, tmpIx+x, dimx, dimy, ay, by);
  //printf("or here?\n");
  for (y = 0; y < dimy; y++) {
    for (x = 0; x < dimx; x++) {
      int p = y*dimx+x;
      Iy[p] = tmpIx[p];
      Ix[p] = tmpIy[tmpIx[p]*dimx+x];
    }
  }
  free(tmpM);
  free(tmpIx);
  free(tmpIy);
}

/* dt of 1d function using squared distance */
void fdt1D(float *f,float *d,int *p, int n,float a,float b) 
{
  int *v = (int *)malloc(n* sizeof(int)),q;
  float *z = (ftype *)malloc((n+1)* sizeof(ftype)),s;
  int k = 0;
  //for (q = 0; q <= n-1; q++) printf("f%f ",f[q]);
  //for (k=0;k<n;k++) v[k]=0;
  //k=0;
  v[0] = 0;
  z[0] = -INF;
  z[1] = +INF;
  for (q = 1; q < n; q++) {
    s  = ((-f[q]+a*square(q)-b*q)-(-f[v[k]]+a*square(v[k])-b*v[k]))/(2.0*a*q-2.0*a*v[k]);
    while (s <= z[k]) {
      k--;
      s  = ((-f[q]+a*square(q)-b*q)-(-f[v[k]]+a*square(v[k])-b*v[k]))/(2.0*a*q-2.0*a*v[k]);
    }
    k++;
    v[k] = q;
    z[k] = s;
    z[k+1] = +INF;
  }

  k = 0;
  for (q = 0; q < n; q++) {
    while (z[k+1] < q)
      k++;
    d[q] = - (a*square(q-v[k]) - b*(q-v[k]) - f[v[k]]);
    //printf("(v%d,f%f,d%d) ",v[k],f[v[k]],square(q-v[k]));
    p[q] = v[k];
  }

  free(v);
  free(z);
}

/* dt of 2d function using squared distance */
void fdtpy(ftype *im,ftype *dst,ftype *fy,ftype *fx,int dimy,int dimx,ftype ay,ftype ax,ftype by,ftype bx) 
{
  int x,y,pt;
  float *f = (ftype *)malloc(dimx*dimy* sizeof(ftype));
  int *ffx = (int *)malloc(dimx* sizeof(int));
  int *ffy = (int *)malloc(dimy* sizeof(int));
  ftype *yy = (ftype *)malloc(dimx*dimy* sizeof(ftype));
  float *d = (ftype *)malloc(dimy* sizeof(ftype));
  // transform along columns
  for ( x = 0; x < dimx; x++) {
    for ( y = 0; y < dimy; y++) {
      f[y] = im[y*dimx+x];
    }
    fdt1D(f,d,ffy, dimy,ay,by);
    for (y = 0; y < dimy; y++) {
      dst[y*dimx+x] = d[y];
      yy[y*dimx+x] = (float)ffy[y];
    }
  }

  // transform along rows
  for (y = 0; y < dimy; y++) {
    for (x = 0; x < dimx; x++) {
      f[x] = dst[y*dimx+x];
    }
    fdt1D(f,d,ffx, dimx,ax,bx);
    for (x = 0; x < dimx; x++) {
      dst[y*dimx+x] = d[x];
      fx[y*dimx+x] = (float)ffx[x];
    }
  }
   for (y = 0; y < dimy; y++) {
        for (x = 0; x < dimx; x++) {
          pt = y*dimx+x;
          fy[pt] = yy[(int)yy[pt]*dimx+(int)(fx[pt])];
          //Ix[p] = tmpIy[tmpIx[p]*dimx+x];
    }
  }
  free(ffx);
  free(ffy);
  free(f);
  free(d);
  free(yy);
}

#define IMSZ 5000 //maximum image size
static int v[IMSZ];
static float z[IMSZ];//

/* dt of 1d function using squared distance */
void fdt1Ds(float *f,float *d,float *p,int step, int n,float a,float b) 
{
  //int *v = (int *)malloc(n* sizeof(int)),q;
  //float *z = (ftype *)malloc((n+1)* sizeof(ftype)),s;
  float s;
  int q,k = 0;
  //for (q = 0; q <= n-1; q++) printf("f%f ",f[q]);
  //for (k=0;k<n;k++) v[k]=0;
  //k=0;
  v[0] = 0;
  z[0] = -INF;
  z[1] = +INF;
  for (q = 1; q < n; q++) {
    s  = ((-f[q*step]+a*square(q)-b*q)-(-f[v[k]*step]+a*square(v[k])-b*v[k]))/(2.0*a*q-2.0*a*v[k]);
    while (s <= z[k]) {
      k--;
      s  = ((-f[q*step]+a*square(q)-b*q)-(-f[v[k]*step]+a*square(v[k])-b*v[k]))/(2.0*a*q-2.0*a*v[k]);
    }
    k++;
    v[k] = q;
    z[k] = s;
    z[k+1] = +INF;
  }

  k = 0;
  for (q = 0; q < n; q++) {
    while (z[k+1] < q)
      k++;
    d[q*step] = - (a*square(q-v[k]) - b*(q-v[k]) - f[v[k]*step]);
    //printf("(v%d,f%f,d%d) ",v[k],f[v[k]],square(q-v[k]));
    p[q*step] = v[k];
  }

  //free(v);
  //free(z);
}

/* dt of 2d function using squared distance */
void ffdtpy(ftype *im,ftype *dst,ftype *fy,ftype *fx,int dimy,int dimx,ftype ay,ftype ax,ftype by,ftype bx) 
{
  int x,y,pt;
  float *f = (ftype *)malloc(dimx*dimy* sizeof(ftype));
  ftype *yy = (ftype *)malloc(dimx*dimy* sizeof(ftype));
  // transform along columns
  for ( x = 0; x < dimx; x++) 
    fdt1Ds(im+x,f+x,yy+x,dimx,dimy,ay,by);
  // transform along rows
  for (y = 0; y < dimy; y++) {
    fdt1Ds(f+y*dimx,dst+y*dimx,fx+y*dimx,1,dimx,ax,bx);
  }
   for (y = 0; y < dimy; y++) {
        for (x = 0; x < dimx; x++) {
          pt = y*dimx+x;
          fy[pt] = yy[(int)yy[pt]*dimx+(int)(fx[pt])];
    }
  }
  free(yy);
  free(f);
}



// matlab entry point
// [M, Ix, Iy] = dt(vals, ax, bx, ay, by)
/*void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { 
  if (nrhs != 5)
    mexErrMsgTxt("Wrong number of inputs"); 
  if (nlhs != 3)
    mexErrMsgTxt("Wrong number of outputs");
  if (mxGetClassID(prhs[0]) != mxftype_CLASS)
    mexErrMsgTxt("Invalid input");

  const int *dims = mxGetDimensions(prhs[0]);
  ftype *vals = (ftype *)mxGetPr(prhs[0]);
  ftype ax = mxGetScalar(prhs[1]);
  ftype bx = mxGetScalar(prhs[2]);
  ftype ay = mxGetScalar(prhs[3]);
  ftype by = mxGetScalar(prhs[4]);
  
  mxArray *mxM = mxCreateNumericArray(2, dims, mxftype_CLASS, mxREAL);
  mxArray *mxIx = mxCreateNumericArray(2, dims, mxINT32_CLASS, mxREAL);
  mxArray *mxIy = mxCreateNumericArray(2, dims, mxINT32_CLASS, mxREAL);
  ftype *M = (ftype *)mxGetPr(mxM);
  int32_t *Ix = (int32_t *)mxGetPr(mxIx);
  int32_t *Iy = (int32_t *)mxGetPr(mxIy);

  ftype *tmpM = (ftype *)mxCalloc(dims[0]*dims[1], sizeof(ftype));
  int32_t *tmpIx = (int32_t *)mxCalloc(dims[0]*dims[1], sizeof(int32_t));
  int32_t *tmpIy = (int32_t *)mxCalloc(dims[0]*dims[1], sizeof(int32_t));

  for (int x = 0; x < dims[1]; x++)
    dt1d(vals+x*dims[0], tmpM+x*dims[0], tmpIy+x*dims[0], 1, dims[0], ay, by);

  for (int y = 0; y < dims[0]; y++)
    dt1d(tmpM+y, M+y, tmpIx+y, dims[0], dims[1], ax, bx);

  // get argmins and adjust for matlab indexing from 1
  for (int x = 0; x < dims[1]; x++) {
    for (int y = 0; y < dims[0]; y++) {
      int p = x*dims[0]+y;
      Ix[p] = tmpIx[p]+1;
      Iy[p] = tmpIy[tmpIx[p]*dims[0]+y]+1;
    }
  }

  mxFree(tmpM);
  mxFree(tmpIx);
  mxFree(tmpIy);
  plhs[0] = mxM;
  plhs[1] = mxIx;
  plhs[2] = mxIy;
}*/
