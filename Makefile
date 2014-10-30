CC = gcc
CP = g++
#CC = icc

CFLAGS = -O3 -g -march=nocona -fomit-frame-pointer -fopenmp -mfpmath=sse -msse -ffast-math
#CFLAGS = -O3 -march=nocona -ffast-math -fomit-frame-pointer -fopenmp
#CFLAGS = -lm -msse2 -O2 -march=nocona -ffast-math -fomit-frame-pointer -fopenmp 

#OMPFLAGS = -fopenmp

#CC=icc
#CFLAGS = -xP -fast
#OMPFLAGS = -openmp

LIB_TARGETS = libresize.so libexcorr.so libhog.so libfastpegasos.so libfastpegasos2.so libcrf2.so libfastDP.so cproject.so libdt.so
all:	$(LIB_TARGETS)

libcrf2.so: ./MRF2.1/myexample2.cpp Makefile
	$(CP) $(CFLAGS) -shared -Wl,-soname=libcrf2.so -DUSE_64_BIT_PTR_CAST -fPIC ./MRF2.1/myexample2.cpp ./MRF2.1/GCoptimization.cpp ./MRF2.1/maxflow.cpp ./MRF2.1/graph.cpp ./MRF2.1/LinkedBlockList.cpp ./MRF2.1/TRW-S.cpp ./MRF2.1/BP-S.cpp ./MRF2.1/ICM.cpp ./MRF2.1/MaxProdBP.cpp ./MRF2.1/mrf.cpp ./MRF2.1/regions-maxprod.cpp -o libcrf2.so

libfastDP.so: ./fastDP/src/myFast_PD.cpp ./fastDP/src/Fast_PD.h ./fastDP/src/maxflow.cpp Makefile
	$(CP) $(CFLAGS) -shared -Wl,-soname=libfastDP.so -DUSE_64_BIT_PTR_CAST -fPIC ./fastDP/src/graph.cpp ./fastDP/src/maxflow.cpp ./fastDP/src/myFast_PD.cpp -o libfastDP.so

libexcorr.so: excorr.c Makefile
	$(CC) $(CFLAGS) -shared -Wl,-soname=libexcorr.so -fPIC excorr.c -o libexcorr.so #libmyrmf.so.1.0.1

libdt.so: dt.c Makefile
	$(CC) $(CFLAGS) -shared -Wl,-soname=libdt.so -fPIC dt.c -o libdt.so #libmyrmf.so.1.0.1

#libdt2.so: dt2D.c Makefile
#	$(CC) $(CFLAGS) -shared -Wl,-soname=libdt2.so -fPIC dt2D.c -o libdt2.so #libmyrmf.so.1.0.1

cproject.so: cproject.c Makefile
	$(CC) $(CFLAGS) -shared -Wl,-soname=cproject.so -fPIC cproject.c -o cproject.so 

libfastpegasos.so: fast_pegasos.c Makefile
	$(CC) $(CFLAGS) -shared -Wl,-soname=libfastpegasos.so -fPIC -lc fast_pegasos.c -o libfastpegasos.so

libfastpegasos2.so: fast_pegasos2.c Makefile
	$(CC) $(CFLAGS) -shared -Wl,-soname=libfastpegasos2.so -fPIC -lc fast_pegasos2.c -o libfastpegasos2.so

libresize.so:	resize.c Makefile
	$(CC) $(CFLAGS) -shared -Wl,-soname=libresize.so -fPIC resize.c -o libresize.so

libhog.so:	features2.c Makefile
	$(CC) $(CFLAGS) -shared -Wl,-soname=libhog.so -fPIC features2.c -o libhog.so

clean:
	rm -f *.o *.pyc $(EXE_TARGETS) $(LIB_TARGETS)


