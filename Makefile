#!/usr/bin/make

EXES = 2DConvolution 2mm 3DConvolution

all: $(EXES) transferManaged transfer

%:  %.cu
	nvcc -g -Wno-deprecated-gpu-targets $^ -lcudart -o $@
	nvcc -g -Wno-deprecated-gpu-targets -DUNMANAGED $^ -lcudart -o $@-unmanaged

transferManaged: transferBandwidthManaged.cu
		nvcc -g -Wno-deprecated-gpu-targets transferBandwidthManaged.cu -lcudart -o transferManaged

transfer: transferBandwidth.cu
		nvcc -g -Wno-deprecated-gpu-targets transferBandwidth.cu -lcudart -o transfer
clean:
	rm -rf $(EXES) *-unmanaged transferManaged transfer
