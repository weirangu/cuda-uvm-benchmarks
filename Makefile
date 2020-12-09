#!/usr/bin/make

EXES = 2DConvolution 2mm 3DConvolution

all: $(EXES)

%:  %.cu
	nvcc -g -Wno-deprecated-gpu-targets $^ -lcudart -o $@
	nvcc -g -Wno-deprecated-gpu-targets -DUNMANAGED $^ -lcudart -o $@-unmanaged

clean:
	rm -rf $(EXES) *-unmanaged
