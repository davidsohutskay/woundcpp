##=================##
## MAKE FILE
##=================##

##—————————————————##
## flags
##—————————————————##
CC = g++ 
## NOTE: you need a path to Eigen headers! 
INCPATH = -I/Users/abuganza/Dropbox/Software -Iinclude -I/usr/local/include
##LIBPATH = -L/opt/local/lib -llapack -lcblas -latlas
CFLAGS = -std=c++11 -O3

##—————————————————##
## rule to make all
##—————————————————##
%.o: %.cpp
	$(CC) -c $(INCPATH) $(CFLAGS) $<

##—————————————————##
## core code
##—————————————————##
objs/wound.o: src/wound.cpp
	$(CC) $(INCPATH) $(LIBPATH) -c $(CFLAGS) src/wound.cpp
	mv wound.o objs

objs/solver.o: src/solver.cpp
	$(CC) $(INCPATH) $(LIBPATH) -c $(CFLAGS) src/solver.cpp
	mv solver.o objs

objs/myMeshGenerator.o: meshing/myMeshGenerator.cpp
	$(CC) $(INCPATH) $(LIBPATH) -c $(CFLAGS) meshing/myMeshGenerator.cpp
	mv myMeshGenerator.o objs

##—————————————————##
## tests
##—————————————————##


##—————————————————##
## results, full
##—————————————————##
apps/results_circle_wound: src/results_circle_wound.cpp objs/wound.o objs/solver.o objs/myMeshGenerator.o
	$(CC) $(INCPATH) $(LIBPATH) $(CFLAGS) $^ -o results_circle_wound 
	mv results_circle_wound apps

##—————————————————##
## clean and remove
##—————————————————##
		
clean:
	rm $(wildcard objs/*.o)