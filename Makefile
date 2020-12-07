#!/usr/bin/make

managed: 2DConvolution.cu
	nvcc -Wno-deprecated-gpu-targets 2DConvolution.cu -lcudart -o 2dconv

unmanaged: 2DConvolution.cu
	nvcc -Wno-deprecated-gpu-targets -DUNMANAGED 2DConvolution.cu -lcudart -o 2dconv

clean:
	rm -rf 2dconv *.txt
