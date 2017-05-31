NVCCOPT=-std=c++11 --compiler-options -march=native -arch=sm_61 -m64 -O3 -rdc=true 

cosolve: main.cu
	nvcc -o cosolve $(NVCCOPT) main.cu

cosolve.cubin: main.cu
	nvcc -o cosolve.cubin -cubin $(NVCCOPT) main.cu

check: cosolve
	./cosolve /data/othello/board54_true output54
	diff -s ../issen/output54 output54
