CC=g++
FLAGS= -O2 -Wall -DLIB_03 -fopenmp
#FLAGS= -g -DLIB_03 -fopenmp
objs=PLSACluster.o iPLSA.o Schedule.o CBScheduler.o AlignAlloc.o time.o extend_mkl.o

CC_FLAGS=${CC} ${FLAGS}

PLSACluster : $(objs) Makefile
	$(CC_FLAGS) $(objs) -o PLSACluster
PLSACluster.o : PLSACluster.cpp iPLSA.hpp
	$(CC_FLAGS) -c PLSACluster.cpp
iPLSA.o : iPLSA.cpp iPLSA.hpp
	$(CC_FLAGS) -c iPLSA.cpp
Schedule.o : Schedule.cpp Schedule.hpp
	$(CC_FLAGS) -c Schedule.cpp
CBScheduler.o : CBScheduler.cpp CBScheduler.hpp
	$(CC_FLAGS) -c CBScheduler.cpp
AlignAlloc.o : AlignAlloc.cpp AlignAlloc.h
	$(CC_FLAGS) -c AlignAlloc.cpp
time.o : time.cpp time.h
	$(CC_FLAGS) -c time.cpp
extend_mkl.o : extend_mkl.cpp extend_mkl.h
	$(CC_FLAGS) -c extend_mkl.cpp

clean:
	-rm $(objs) PLSACluster
