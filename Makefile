cosolve: main.cu
	nvcc -o cosolve -std=c++11 -arch=sm_61 -m64 -O3 -rdc=true main.cu
