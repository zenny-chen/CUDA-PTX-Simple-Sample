nvcc --gpu-architecture=sm_52 --device-c test.ptx
nvcc --gpu-architecture=sm_52 main.cu --device-c
nvcc main.obj test.obj -o test.exe
del main.obj test.obj test.exp test.lib
