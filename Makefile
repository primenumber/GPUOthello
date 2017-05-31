NVCC=nvcc
NVCCOPT=-std=c++11 --compiler-options -march=native -arch=sm_61 -m64 -O3 -rdc=true 
OBJS=main.o solver.o board.o

.SUFFIXES: .cpp .c .cu .o

cosolve: $(OBJS)
	nvcc -o cosolve $(NVCCOPT) $(OBJS)

.cu.o:
	$(NVCC) $(NVCCOPT) -c $< -o $@

check: cosolve
	./cosolve prob54 output54
	diff -s ans54 output54
