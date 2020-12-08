#!/usr/bin/make

transferManaged: transferBandwidthManaged.cu
	nvcc -g -Wno-deprecated-gpu-targets transferBandwidthManaged.cu -lcudart -o transferManaged

transfer: transferBandwidth.cu
	nvcc -g -Wno-deprecated-gpu-targets transferBandwidth.cu -lcudart -o transfer

managed: 2DConvolution.cu
	nvcc -g -Wno-deprecated-gpu-targets 2DConvolution.cu -lcudart -o 2dconv

unmanaged: 2DConvolution.cu
	nvcc -g -Wno-deprecated-gpu-targets -DUNMANAGED 2DConvolution.cu -lcudart -o 2dconv

clean:
	rm -rf 2dconv transfer *.txt
